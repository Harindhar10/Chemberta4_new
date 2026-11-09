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
"""

from __future__ import annotations

import json
import time
from collections import Counter
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
    started = time.time()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out_path, "a") as fout:
        futures = [pool.submit(worker, item) for item in items]
        for n, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if not keep(record):
                counts["dropped"] += 1
            else:
                counts[status_of(record)] += 1
                fout.write(json.dumps(record) + "\n")
                fout.flush()
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


def pending(items: Iterable[Any], done: set[Any], key: Callable[[Any], Any]) -> list[Any]:
    """The items whose ``key`` is not already in ``done``."""
    return [item for item in items if key(item) not in done]
