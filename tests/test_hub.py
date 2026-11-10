"""Tests for publishing runs to the Hugging Face Hub.

Many runs share one dataset card, so each push edits it in place. These tests
check those edits without touching the network:

- re-pushing a run replaces its section and viewer config instead of
  duplicating them, and other runs' entries survive;
- a failed upload returns quietly and leaves the local files alone.
"""

from __future__ import annotations

import sys
import types

import pandas as pd

from chemberta4.evaluation import hub


def card_after(*pushes: tuple[str, str]) -> str:
    text = None
    for run_id, section in pushes:
        text = hub.update_card(
            text,
            repo_id="me/evals",
            task="moleculenet",
            run_id=run_id,
            raw_file="results.jsonl",
            section=section,
        )
    return text


def test_new_card_has_template_header_and_config() -> None:
    header, body = hub.split_card(card_after(("A", "first")))
    assert header["configs"] == [
        {"config_name": "moleculenet--A", "data_files": "moleculenet/A/results.jsonl"}
    ]
    assert 'load_dataset("me/evals"' in body
    assert "first" in body


def test_repush_replaces_its_own_entries_and_keeps_others() -> None:
    header, body = hub.split_card(card_after(("A", "old A"), ("B", "B"), ("A", "new A")))
    assert [c["config_name"] for c in header["configs"]] == ["moleculenet--A", "moleculenet--B"]
    assert "new A" in body and "old A" not in body and "B" in body
    assert body.count("<!-- run:moleculenet/A -->") == 1
    assert body.index("new A") < body.index("<!-- run:moleculenet/B -->")  # stays in place


def test_existing_header_fields_survive() -> None:
    text = "---\nlicense: mit\n---\n# my card\n"
    out = hub.update_card(
        text, repo_id="me/evals", task="retrosynthesis", run_id="r",
        raw_file="generations.jsonl", section="s",
    )
    header, body = hub.split_card(out)
    assert header["license"] == "mit"
    assert body.startswith("# my card")


def test_moleculenet_metrics_mean_and_std_across_seeds() -> None:
    metrics = pd.DataFrame({
        "dataset": ["bbbp", "bbbp", "delaney"],
        "k": [0, 0, 5],
        "seed": [0, 1, 0],
        "metric": ["roc_auc", "roc_auc", "rmse"],
        "value": [0.6, 0.8, 1.5],
        "parse_fail_rate": [0.0, 0.1, 0.0],
    })
    lines = hub.moleculenet_metrics_md(metrics).splitlines()
    assert lines[2].startswith("| bbbp | 0 | roc_auc | 0.7 | 0.1414 | 2 | 0.05 |")
    assert lines[3] == "| delaney | 5 | rmse | 1.5 |  | 1 | 0 |"


def test_retrosynthesis_metrics_table() -> None:
    view = {"all_correct": 10.0, "any_correct": 20.0, "maxfrag": 15.0}
    md = hub.retrosynthesis_metrics_md({
        "top1": view, "topk": view, "n_scored": 10,
        "parse_failures_pct": 0.0, "invalid_smiles_pct": 5.0, "mean_proposals": 3.2,
    })
    assert "| top-1 | 10 | 20 | 15 |" in md
    assert "| 10 | 0 | 5 | 3.2 |" in md


def test_push_failure_returns_false_and_keeps_files(tmp_path, monkeypatch) -> None:
    (tmp_path / "results.jsonl").write_text('{"id": 1}\n')

    class BrokenApi:
        def __init__(self, **_):
            pass

        def repo_info(self, *a, **k):
            raise RuntimeError("network down")

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = BrokenApi
    fake.CommitOperationAdd = object
    fake.hf_hub_download = lambda *a, **k: None
    utils = types.ModuleType("huggingface_hub.utils")
    utils.EntryNotFoundError = type("EntryNotFoundError", (Exception,), {})
    utils.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)

    ok = hub.push_run(
        repo_id="me/evals", task="moleculenet", run_id="A", run_dir=tmp_path,
        config={"model": "m"}, metrics_md="", token="t",
    )
    assert ok is False
    assert (tmp_path / "results.jsonl").read_text() == '{"id": 1}\n'
