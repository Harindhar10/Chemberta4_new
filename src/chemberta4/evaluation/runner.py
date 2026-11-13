"""The resumable generation loop both tasks share.

A run appends one JSON object per finished item to a JSONL checkpoint and
flushes as it goes, so an interrupted run loses nothing: restarting reads the
checkpoint back, skips what is already there and generates only the rest.

Records that failed at the API are deliberately *not* written. They are left
out of the checkpoint so the next run retries them, which is what makes a run
that hit a context-length error or a dropped connection self-healing.

Requests go out on a thread pool. The work is network-bound — vLLM does the
batching server-side — so threads are the right tool and ``workers`` is just
how many requests are in flight.

Alongside the checkpoint, each finished *group* of items (a ChemLLMBench
subtask, a MoleculeNet dataset/k/seed, a chunk of retrosynthesis molecules) is
also written to its own ``tasks/<group>.jsonl`` the moment its last record
lands. The checkpoint stays the one file that resume and scoring read.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read every record of a JSONL file; a missing file reads as empty.

    A truncated final line, which a killed run can leave behind, is skipped
    rather than raising.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial last line from an interrupted write
    return records


def generate(
    *,
    items: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    out_path: str | Path,
    workers: int = 64,
    keep: Callable[[dict[str, Any]], bool] = lambda rec: True,
    status_of: Callable[[dict[str, Any]], str] = lambda rec: str(rec.get("status", "ok")),
    label: str = "",
    progress_every: int = 50,
    group_of: Callable[[Any], str] | None = None,
    on_group_done: Callable[[str], Any] | None = None,
) -> Counter:
    """Run ``worker`` over ``items``, appending each result to ``out_path``.

    Parameters
    ----------
    items:
        Work units, already filtered down to what still needs doing.
    worker:
        Called with one item; returns the record to store.
    keep:
        Decides whether a record is written. Return ``False`` for an API
        failure so the item is retried on the next run.
    status_of:
        Pulls a short status out of a record, for the tallies printed at the end.
    label:
        Prefix for progress lines, e.g. ``"[llm]"``.
    group_of, on_group_done:
        Optional. ``group_of`` names the group an item belongs to; once the
        last pending item of a group has been written (or dropped) and flushed,
        ``on_group_done(group)`` is called from the main thread. Groups finish
        roughly in the order the items are given.

    Returns
    -------
    collections.Counter
        How many records of each status, including ``"dropped"`` for records
        ``keep`` rejected.
    """
    counts: Counter = Counter()
    if not items:
        return counts

    tag = f"{label} " if label else ""
    track = group_of is not None and on_group_done is not None
    remaining: Counter = Counter(group_of(item) for item in items) if track else Counter()
    started = time.time()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out_path, "a") as fout:
        futures = {pool.submit(worker, item): item for item in items}
        for n, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if not keep(record):
                counts["dropped"] += 1
            else:
                counts[status_of(record)] += 1
                fout.write(json.dumps(record) + "\n")
                fout.flush()
            if track:
                group = group_of(futures[future])
                remaining[group] -= 1
                if remaining[group] == 0:
                    on_group_done(group)
            if progress_every and n % progress_every == 0:
                print(f"  {tag}{n}/{len(items)}")

    elapsed = time.time() - started
    tally = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(
        f"{tag}{len(items)} items in {elapsed / 60:.2f} min "
        f"({elapsed / len(items):.2f} s/item): {tally}"
    )
    if counts["dropped"]:
        n = counts["dropped"]
        print(f"{tag}{n} item{'s' if n > 1 else ''} failed at the API and "
              f"{'were' if n > 1 else 'was'} not saved; re-run to retry")
    return counts


def group_filename(group: str) -> str:
    """A filesystem-safe file name for a group.

    >>> group_filename("Buchwald-Hartwig_k0_random")
    'Buchwald-Hartwig_k0_random.jsonl'
    >>> group_filename("a/b c")
    'a_b_c.jsonl'
    """
    return re.sub(r"[^\w.-]", "_", group) + ".jsonl"


def write_group(
    out_dir: str | Path,
    group: str,
    records: Iterable[dict[str, Any]],
    expected_keys: set[Any],
    key_of: Callable[[dict[str, Any]], Any],
    *,
    report_missing: bool = True,
) -> Path | None:
    """Write ``out_dir/<group>.jsonl`` if every expected key has a record.

    API failures and records outside ``expected_keys`` are ignored; for a key
    stored more than once the last record wins. The file is written to a
    temporary name and renamed, so a killed process never leaves half a file.
    An incomplete group is not written and ``None`` is returned.
    """
    chosen: dict[Any, dict[str, Any]] = {}
    for rec in records:
        if rec.get("status") == "api_error":
            continue
        key = key_of(rec)
        if key in expected_keys:
            chosen[key] = rec
    missing = len(expected_keys) - len(chosen)
    if missing:
        if report_missing:
            print(f"[tasks] {group}: {missing} missing; written after a re-run retries them")
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / group_filename(group)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for rec in chosen.values():
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


class GroupWriter:
    """Writes each finished group of a run to ``<out_dir>/<group>.jsonl``.

    ``expected`` maps every group of the run to the keys it must contain. On
    a group's completion the checkpoint is re-read, so records from earlier
    runs of the same run directory count too.

    >>> import tempfile
    >>> d = Path(tempfile.mkdtemp())
    >>> _ = (d / "results.jsonl").write_text('{"id": 1}\\n{"id": 2}\\n')
    >>> w = GroupWriter(d / "results.jsonl", d / "tasks", {"a": {1}, "b": {2, 3}},
    ...                 key_of=lambda r: r["id"])
    >>> w("a").name  # doctest: +ELLIPSIS
    [tasks] a: wrote .../tasks/a.jsonl
    'a.jsonl'
    >>> w.write_complete()
    ['a']
    """

    def __init__(
        self,
        checkpoint: str | Path,
        out_dir: str | Path,
        expected: dict[str, set[Any]],
        key_of: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.out_dir = Path(out_dir)
        self.expected = expected
        self.key_of = key_of

    @classmethod
    def from_items(
        cls,
        checkpoint: str | Path,
        out_dir: str | Path,
        items: Iterable[Any],
        *,
        group_of: Callable[[Any], str],
        item_key: Callable[[Any], Any],
        key_of: Callable[[dict[str, Any]], Any],
    ) -> "GroupWriter":
        """Build ``expected`` from every item of the run, done or not."""
        expected: dict[str, set[Any]] = defaultdict(set)
        for item in items:
            expected[group_of(item)].add(item_key(item))
        return cls(checkpoint, out_dir, dict(expected), key_of)

    def __call__(self, group: str) -> Path | None:
        path = write_group(
            self.out_dir, group, load_jsonl(self.checkpoint), self.expected[group], self.key_of
        )
        if path is not None:
            print(f"[tasks] {group}: wrote {path}")
        return path

    def write_complete(self) -> list[str]:
        """(Re)write every group the checkpoint already completes.

        Run before generating, so groups that finished before a restart, or
        before group files existed, get their file too. Incomplete groups are
        skipped silently. Returns the groups written.
        """
        records = load_jsonl(self.checkpoint)
        return [
            group
            for group, keys in self.expected.items()
            if write_group(
                self.out_dir, group, records, keys, self.key_of, report_missing=False
            ) is not None
        ]


def pending(items: Iterable[Any], done: set[Any], key: Callable[[Any], Any]) -> list[Any]:
    """The items whose ``key`` is not already in ``done``."""
    return [item for item in items if key(item) not in done]
