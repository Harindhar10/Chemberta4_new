"""Tests for the ChemLLMBench port.

The family modules are copied from the reference repository, so the tests
below pin the behaviour this repo relies on:

- the copied parsers, normalisers and metrics behave as the reference does on
  known cases;
- a run checkpoints, resumes and scores through the same glue as MoleculeNet;
- the W&B summary keys for MoleculeNet did not change when ``log_run`` was
  generalised to ChemLLMBench frames.

Everything here uses only the committed data, so no network is needed.
Prompt parity with the reference loaders is checked separately against a
reference checkout (see CHEMLLMBENCH.md).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from chemberta4.evaluation.backends.base import ModelBackend, Response
from chemberta4.evaluation.runner import generate, load_jsonl, pending
from chemberta4.tasks.chemllmbench import run
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from chemberta4.tasks.chemllmbench.registry import TASK_GROUPS, TASKS, resolve


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def test_every_task_is_in_a_group_and_all_lists_them_all() -> None:
    grouped = {k for name, keys in TASK_GROUPS.items() if name != "all" for k in keys}
    assert grouped == set(TASKS) == set(TASK_GROUPS["all"])


def test_resolve_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown ChemLLMBench task"):
        resolve("bbbp,nope")


# --------------------------------------------------------------------------
# Copied parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"answer": "CCO"}', "CCO"),
        ('```json\n{"answer": "Yes"}\n```', "Yes"),
        ('<think>{"answer": "wrong"}</think>{"answer": "right"}', "right"),
        ('{"answer": "C1=CC=CC=C1', "C1=CC=CC=C1"),  # truncated, no closing quote
        ("no json here", None),
    ],
)
def test_extract_json_answer(text: str, expected: str | None) -> None:
    assert extract_json_answer(text) == expected


def test_strip_thinking_drops_an_unclosed_trace() -> None:
    assert strip_thinking("answer first <think> and then it ran out") == "answer first"


def test_formula_key_is_order_independent_and_case_sensitive() -> None:
    from chemberta4.tasks.chemllmbench.name_prediction import _formula_key

    assert _formula_key("H6C2O") == _formula_key("C2H6O")
    assert _formula_key("CO") != _formula_key("Co")


def test_retrosynthesis_normalise_sorts_fragments() -> None:
    from chemberta4.tasks.chemllmbench.retrosynthesis import _normalise

    assert _normalise("OCC.CC(=O)O") == _normalise("CC(O)=O.CCO")
    assert _normalise("CCO.not_smiles") == ""


# --------------------------------------------------------------------------
# Copied metrics
# --------------------------------------------------------------------------
def test_property_score_counts_parse_failures_as_wrong() -> None:
    pytest.importorskip("sklearn")
    from chemberta4.tasks.chemllmbench import property_prediction as pp

    metrics = pp.score([1, 0, 1, 0], [1, 0, -1, 1], "BBBP")
    assert metrics["accuracy"] == 0.5
    assert metrics["invalid_predictions"] == 1


def test_reaction_score_reports_invalid_smiles() -> None:
    from chemberta4.tasks.chemllmbench import reaction_prediction as rp

    metrics = rp.score(["CCO", "c1ccccc1"], ["OCC", "garbage("], "forward")
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["invalid_smiles_percent"] == 50.0


def test_selection_parser_reads_a_labelled_answer() -> None:
    from chemberta4.tasks.chemllmbench import reaction_selection as rs

    assert rs.parse("Optimal solvent: CO.", "Solvent_selection") == "CO"


# --------------------------------------------------------------------------
# Run glue: requests, checkpoint, resume, scoring
# --------------------------------------------------------------------------
class EchoBackend(ModelBackend):
    """Answers every request with the reference answer, except a few rows."""

    model = "fake/echo"

    def __init__(self, answers: dict[str, str], fail: set[str] = frozenset()) -> None:
        self.answers, self.fail, self.calls = answers, fail, 0

    def generate(self, messages, **overrides) -> Response:
        self.calls += 1
        user = messages[1]["content"]
        if user in self.fail:
            return Response(text="boom", status=502)
        return Response(text=json.dumps({"answer": self.answers[user]}))


def _answers(requests) -> dict[str, str]:
    """The ground truth for each prompt, in the form each parser expects."""
    out = {}
    for r in requests:
        truth = r.ground_truth
        out[r.user] = {1: "Yes", 0: "No"}.get(truth, truth) if isinstance(truth, int) else truth
    return out


def test_requests_respect_limit_and_zero_shot_only_tasks() -> None:
    requests = run.build_requests(["reactant_selection", "iupac_to_formula"], [0, 2], limit=3)
    by_task = pd.Series([(r.task, r.k, r.strategy) for r in requests]).value_counts().to_dict()
    assert by_task == {
        ("reactant_selection", 0, "none"): 3,  # built once, whatever the shots
        ("iupac_to_formula", 0, "random"): 3,
        ("iupac_to_formula", 2, "random"): 3,
    }


def test_run_checkpoints_retries_api_errors_and_scores(tmp_path) -> None:
    pytest.importorskip("sklearn")
    tasks = ["iupac_to_formula", "Buchwald-Hartwig", "reactant_selection"]
    requests = run.build_requests(tasks, [0], limit=4)
    failing = {requests[0].user}
    backend = EchoBackend(_answers(requests), fail=failing)
    out = tmp_path / "results.jsonl"

    def go(todo, be):
        return generate(
            items=todo,
            worker=lambda r: run.generate_one(be, r, run_id="t"),
            out_path=out,
            workers=2,
            keep=lambda rec: rec["status"] != "api_error",
            progress_every=0,
        )

    counts = go(requests, backend)
    assert counts["dropped"] == 1 and counts["ok"] == len(requests) - 1

    # The API failure was not stored, so the re-run sends exactly that request.
    todo = pending(requests, run.done_keys(load_jsonl(out)), run.resume_key)
    assert [r.user for r in todo] == [requests[0].user]
    go(todo, EchoBackend(_answers(requests)))
    assert pending(requests, run.done_keys(load_jsonl(out)), run.resume_key) == []

    metrics, summary = run.evaluate_run(tmp_path, keys={run.resume_key(r) for r in requests})
    head = metrics[metrics["primary"]].set_index("task")["value"].to_dict()
    assert head == {"iupac_to_formula": 1.0, "reactant_selection": 1.0, "Buchwald-Hartwig": 1.0}
    assert summary["iupac_to_formula"]["k0_random"]["correct"] == 4
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "predictions" / "reactant_selection_k0_none.csv").exists()


def test_parse_failure_is_kept_and_counted(tmp_path) -> None:
    request = run.build_requests(["smiles_to_formula"], [0], limit=1)[0]

    class Empty(ModelBackend):
        def generate(self, messages, **overrides) -> Response:
            return Response(text="", reasoning="ran out of tokens")

    rec = run.generate_one(Empty(), request, run_id="t")
    assert rec["status"] == "truncated" and rec["prediction"] == ""
    (tmp_path / "results.jsonl").write_text(json.dumps(rec) + "\n")
    metrics, _ = run.evaluate_run(tmp_path)
    values = metrics.set_index("metric")["value"]
    assert values["accuracy"] == 0.0 and values["parse_fail_rate"] == 1.0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def test_wandb_summary_keys_unchanged_for_moleculenet() -> None:
    from chemberta4.evaluation.reporting import summary_keys

    df = pd.DataFrame({
        "dataset": ["bbbp", "bbbp"], "k": [5, 5], "seed": [0, 1],
        "metric": ["roc_auc", "roc_auc"], "value": [0.6, 0.8],
        "parse_fail_rate": [0.0, 0.1], "fill_rate": [0.0, 0.1],
    })
    keys = summary_keys(df)
    assert keys["bbbp/k5/s0/roc_auc"] == 0.6
    assert keys["bbbp/k5/s1/parse_fail_rate"] == 0.1
    assert keys["bbbp/k5/roc_auc_mean"] == pytest.approx(0.7)
    assert "bbbp/k5/roc_auc_std" in keys


# --------------------------------------------------------------------------
# Prompt parity with the reference checkout (skipped when it is absent)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference(request):
    import importlib
    import sys

    from chemberta4.tasks.chemllmbench.data import SOURCE_DIR

    if not (SOURCE_DIR / "main.py").exists():
        pytest.skip(f"no ChemLLMBench checkout at {SOURCE_DIR} (set CHEMLLMBENCH_SRC_DIR)")
    sys.path.insert(0, str(SOURCE_DIR))
    request.addfinalizer(lambda: sys.path.remove(str(SOURCE_DIR)))
    return lambda name: importlib.import_module(name)


@pytest.mark.parametrize(
    "key, strategy, k",
    [
        ("smiles_to_iupac", "random", 0),
        ("iupac_to_smiles", "random", 8),
        ("smiles_to_formula", "scaffold", 4),
        ("Buchwald-Hartwig", "random", 10),
        ("Suzuki-Miyaura", "random", 0),
    ],
)
def test_prompts_match_reference(reference, key, strategy, k) -> None:
    task = TASKS[key]
    if task.module == "name_prediction":
        ref = reference("Name_Prediction.data").load_dataset(task.subtask, strategy, k)
    else:
        ref = reference("Yield_prediction.data").load_function(task.subtask, strategy, k)
    system, user, _ = task.family().load(task.subtask, strategy, k)
    assert (system, user) == (ref[0], ref[1])


@pytest.mark.parametrize("key", ["reactant_selection", "ligand_selection", "solvent_selection"])
def test_selection_prompts_match_reference(reference, key) -> None:
    from chemberta4.tasks.chemllmbench.data import SOURCE_DIR, SHIPPED

    task = TASKS[key]
    family = task.family()
    csv = SOURCE_DIR / SHIPPED[family.TASK_MAP[task.subtask]]
    system, user, df = reference("Reaction_Selection.data").load_dataset(str(csv))
    assert family.load(task.subtask) == (system, user, df["ground_truth"].astype(str).tolist())
