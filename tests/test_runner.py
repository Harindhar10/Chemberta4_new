"""Tests for the resumable generation loop.

Resumability is the property that makes a long run survivable, and it is easy
to break in a way that is invisible until a run is interrupted. These tests
cover the three rules that matter:

- a finished item is appended to the checkpoint immediately, not at the end;
- an item that failed at the API is left out, so a re-run retries it;
- a truncated final line, which a killed process can leave behind, is skipped
  on read rather than raising.
"""

from __future__ import annotations

import json

from chemberta4.evaluation.runner import generate, load_jsonl, pending


def test_load_jsonl_missing_file_is_empty(tmp_path) -> None:
    assert load_jsonl(tmp_path / "nope.jsonl") == []


def test_load_jsonl_skips_a_truncated_last_line(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "part')  # killed mid-write
    assert [r["id"] for r in load_jsonl(path)] == [1, 2]


def test_generate_writes_one_record_per_item(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    counts = generate(
        items=[1, 2, 3],
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=path,
        workers=2,
        progress_every=0,
    )
    assert counts["ok"] == 3
    assert sorted(r["id"] for r in load_jsonl(path)) == [1, 2, 3]


def test_generate_drops_failures_so_they_are_retried(tmp_path) -> None:
    path = tmp_path / "out.jsonl"

    def worker(n: int) -> dict:
        return {"id": n, "status": "api_error" if n == 2 else "ok"}

    counts = generate(
        items=[1, 2, 3],
        worker=worker,
        out_path=path,
        workers=2,
        keep=lambda rec: rec["status"] != "api_error",
        status_of=lambda rec: rec["status"],
        progress_every=0,
    )
    assert counts["dropped"] == 1
    stored = {r["id"] for r in load_jsonl(path)}
    assert stored == {1, 3}, "the failed item must not be checkpointed"

    # A second pass sees item 2 as still outstanding and finishes it.
    todo = pending([1, 2, 3], stored, key=lambda n: n)
    assert todo == [2]
    generate(
        items=todo,
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=path,
        workers=1,
        progress_every=0,
    )
    assert sorted(r["id"] for r in load_jsonl(path)) == [1, 2, 3]


def test_generate_appends_to_an_existing_checkpoint(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"id": 0, "status": "ok"}) + "\n")
    generate(
        items=[1],
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=path,
        workers=1,
        progress_every=0,
    )
    assert sorted(r["id"] for r in load_jsonl(path)) == [0, 1]


def test_generate_on_empty_input_does_nothing(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    assert generate(items=[], worker=lambda n: {}, out_path=path, workers=1) == {}
    assert not path.exists()


def test_pending_filters_by_key() -> None:
    items = [("bbbp", 0), ("bbbp", 5), ("hiv", 0)]
    done = {("bbbp", 0)}
    assert pending(items, done, key=lambda item: item) == [("bbbp", 5), ("hiv", 0)]


# --------------------------------------------------------------------------
# Group files: tasks/<group>.jsonl written as each group finishes
# --------------------------------------------------------------------------
from chemberta4.evaluation.runner import GroupWriter, write_group  # noqa: E402


def _writer(tmp_path, items):
    return GroupWriter.from_items(
        tmp_path / "out.jsonl",
        tmp_path / "tasks",
        items,
        group_of=lambda n: "even" if n % 2 == 0 else "odd",
        item_key=lambda n: n,
        key_of=lambda rec: rec["id"],
    )


def test_group_file_is_written_once_its_last_record_is_flushed(tmp_path) -> None:
    items = [1, 2, 3, 4, 5]
    writer = _writer(tmp_path, items)
    seen: list[tuple[str, list[int]]] = []

    def on_done(group: str) -> None:
        path = writer(group)
        seen.append((group, sorted(r["id"] for r in load_jsonl(path))))

    generate(
        items=items,
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=tmp_path / "out.jsonl",
        workers=3,
        progress_every=0,
        group_of=lambda n: "even" if n % 2 == 0 else "odd",
        on_group_done=on_done,
    )
    assert sorted(seen) == [("even", [2, 4]), ("odd", [1, 3, 5])]


def test_group_with_a_dropped_record_is_not_written(tmp_path) -> None:
    items = [1, 2, 3, 4]
    writer = _writer(tmp_path, items)
    generate(
        items=items,
        worker=lambda n: {"id": n, "status": "api_error" if n == 3 else "ok"},
        out_path=tmp_path / "out.jsonl",
        workers=2,
        keep=lambda rec: rec["status"] != "api_error",
        progress_every=0,
        group_of=lambda n: "even" if n % 2 == 0 else "odd",
        on_group_done=writer,
    )
    assert (tmp_path / "tasks" / "even.jsonl").exists()
    assert not (tmp_path / "tasks" / "odd.jsonl").exists()


def test_group_counts_records_from_an_earlier_run(tmp_path) -> None:
    (tmp_path / "out.jsonl").write_text('{"id": 1, "status": "ok"}\n')
    writer = _writer(tmp_path, [1, 3])
    assert writer.write_complete() == []  # 3 is still missing
    generate(
        items=[3],
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=tmp_path / "out.jsonl",
        progress_every=0,
        group_of=lambda n: "odd",
        on_group_done=writer,
    )
    assert [r["id"] for r in load_jsonl(tmp_path / "tasks" / "odd.jsonl")] == [1, 3]


def test_write_group_keeps_only_its_keys_and_the_last_duplicate(tmp_path) -> None:
    records = [
        {"id": 1, "v": "old"},
        {"id": 2, "v": "other group"},
        {"id": 1, "v": "new"},
        {"id": 9, "status": "api_error"},
    ]
    path = write_group(tmp_path, "g", records, {1}, key_of=lambda r: r["id"])
    assert load_jsonl(path) == [{"id": 1, "v": "new"}]
    assert not list(tmp_path.glob("*.tmp")), "the temporary file must be renamed away"


def test_generate_without_groups_is_unchanged(tmp_path) -> None:
    counts = generate(
        items=[1, 2],
        worker=lambda n: {"id": n, "status": "ok"},
        out_path=tmp_path / "out.jsonl",
        progress_every=0,
        on_group_done=lambda g: (_ for _ in ()).throw(AssertionError("called")),
    )
    assert counts["ok"] == 2
