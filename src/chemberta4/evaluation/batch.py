"""Run pending items through a provider's Batch API, resumably.

The same command does everything, whatever state the run is in:

1. Items not yet in the checkpoint and not already in a submitted batch are
   sent as new batches. Their ids go into ``<run_dir>/batches.json`` (the
   manifest) before anything else happens, so no batch is ever sent twice.
2. With ``wait`` (the default) it polls until every open batch has finished,
   writing each result into the same checkpoint the real-time path writes.
   Without it, it returns after submitting; running the command again picks
   up the same batches.

A request that failed (errored, expired, or missing from the results) is not
checkpointed, exactly like an ``api_error`` in real time, so the next run
sends it again. A batch's results are collected once; the manifest then marks
it done.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from chemberta4.evaluation.backends.base import Response

#: Requests per batch: under both providers' count and size limits for these prompts.
CHUNK = 10_000


def batch_id_safe(custom_id: str) -> str:
    """A custom_id both providers accept (``[A-Za-z0-9_-]``, at most 64 chars).

    >>> batch_id_safe("Buchwald-Hartwig-k0-random-r12")
    'Buchwald-Hartwig-k0-random-r12'
    >>> batch_id_safe("a.b c")
    'a_b_c'
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", custom_id)[:64]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def save_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(path)


def run_batch(
    *,
    items: Sequence[Any],
    backend: Any,
    custom_id: Callable[[Any], str],
    messages_of: Callable[[Any], list[dict[str, Any]]],
    finish: Callable[[Any, list[dict[str, Any]], Response], dict[str, Any]],
    out_path: str | Path,
    manifest_path: str | Path,
    keep: Callable[[dict[str, Any]], bool] = lambda rec: True,
    status_of: Callable[[dict[str, Any]], str] = lambda rec: str(rec.get("status", "ok")),
    group_of: Callable[[Any], str] | None = None,
    on_group_done: Callable[[str], Any] | None = None,
    wait: bool = True,
    poll_s: float = 60.0,
) -> Counter:
    """Submit what is pending, then (with ``wait``) collect every open batch.

    ``items`` are the items still missing from the checkpoint. ``finish`` turns
    an item, its messages and the response into the record to store, exactly
    as the real-time worker does. Returns status counts as ``runner.generate``;
    without ``wait``, ``counts["pending"]`` is the number of requests still in
    open batches.
    """
    out_path, manifest_path = Path(out_path), Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {batch_id_safe(custom_id(item)): item for item in items}
    if len(by_id) != len(items):
        raise ValueError("custom ids are not unique after sanitizing")

    manifest = load_manifest(manifest_path)
    in_flight = {cid for b in manifest if not b["collected"] for cid in b["custom_ids"]}
    new = [cid for cid in by_id if cid not in in_flight]
    for start in range(0, len(new), CHUNK):
        chunk = new[start : start + CHUNK]
        requests = [(cid, backend.request(messages_of(by_id[cid]))) for cid in chunk]
        batch_id = backend.submit(requests)
        manifest.append({"id": batch_id, "custom_ids": chunk, "collected": False})
        save_manifest(manifest_path, manifest)
        print(f"[batch] submitted {batch_id} ({len(chunk)} requests)")

    open_batches = [b for b in manifest if not b["collected"]]
    print(f"[batch] {len(open_batches)} open batch(es), {len(by_id)} requests pending")
    counts: Counter = Counter()
    first_error = ""
    if not wait:
        if open_batches:
            counts["pending"] = sum(len(b["custom_ids"]) for b in open_batches)
            print("[batch] --no-wait: re-run the same command to collect and score the results")
        return counts

    track = group_of is not None and on_group_done is not None
    remaining = Counter(group_of(item) for item in items) if track else Counter()
    with open(out_path, "a") as fout:
        while open_batches:
            for entry in list(open_batches):
                if not backend.finished(entry["id"]):
                    continue
                responses = backend.results(entry["id"])
                for cid in entry["custom_ids"]:
                    item = by_id.get(cid)
                    if item is None:  # no longer part of this run (e.g. a smaller --limit)
                        continue
                    response = responses.get(cid) or Response(
                        text="missing from batch results", status=502
                    )
                    record = finish(item, messages_of(item), response)
                    if keep(record):
                        counts[status_of(record)] += 1
                        fout.write(json.dumps(record) + "\n")
                    else:
                        counts["dropped"] += 1
                        first_error = first_error or response.text
                    fout.flush()
                    if track:
                        group = group_of(item)
                        remaining[group] -= 1
                        if remaining[group] == 0:
                            on_group_done(group)
                entry["collected"] = True
                save_manifest(manifest_path, manifest)
                open_batches.remove(entry)
                print(f"[batch] collected {entry['id']}")
            if open_batches:
                time.sleep(poll_s)

    tally = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"[batch] done: {tally or 'nothing collected'}")
    if counts["dropped"]:
        print(f"[batch] {counts['dropped']} request(s) failed and were not saved; re-run to retry")
        print(f"[batch] first error: {first_error[:500]}")
    return counts
