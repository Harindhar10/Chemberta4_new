"""Tests for the depth-1 ``autosolve`` mode.

They pin down what is easy to get subtly wrong and what moves reported numbers:

- the hallucination gate *ranks* (best first, flagged demoted, not dropped) and
  drops only the unassessable, so ``proposals[0]`` is the least suspicious;
- in autosolve mode that gate runs after the retry loop, so ``llm_pipeline`` is
  never asked to apply it, while pipeline mode still passes the flag through;
- the shared AZ checkpoint is not re-searched, and a transient AZ failure is not
  written into it;
- a missing AZ config or aizynthfinder fails with a message that says what to do.

AiZynthFinder, the LLM server and the chemistry scorer are all stubbed, so none of
this needs AZ model files or a GPU.
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("rdkit", reason="the autosolve modules need RDKit")
pytest.importorskip("litellm", reason="the copied llm module, imported by autosolve and cli, needs litellm")

from chemberta4 import cli  # noqa: E402
from chemberta4.tasks.retrosynthesis import autosolve, az  # noqa: E402
from chemberta4.tasks.retrosynthesis import run as task  # noqa: E402

PRODUCT = "CC(=O)Oc1ccccc1C(=O)O"
A, B, C, D = ["CCO"], ["CCN"], ["CCC"], ["CCCl"]

# What the stubbed chemistry scorer says about each candidate: score, penalty.
SCORES = {"CCO": (80, 0), "CCC": (60, 5), "CCN": (10, 30), "CCCl": None}


def fake_score(reactants: str, product: str, weights=None) -> dict:
    entry = SCORES[reactants]
    if entry is None:
        return {"unassessable": True, "message": "stub"}
    score, penalty = entry
    return {"score": score, "penalty_total": penalty, "severity": "low"}


@pytest.fixture
def stub_score(monkeypatch):
    monkeypatch.setattr(autosolve, "calculate_hallucination_score", fake_score)


# --------------------------------------------------------------------------
# rank_hallucination
# --------------------------------------------------------------------------
def test_rank_puts_accepted_first_by_score_then_demoted(stub_score):
    # reject_below is 40: 80 and 60 are accepted, 10 is demoted.
    assert autosolve.rank_hallucination(PRODUCT, [B, C, A]) == [A, C, B]


def test_rank_demotes_flagged_instead_of_dropping(stub_score):
    assert autosolve.rank_hallucination(PRODUCT, [B]) == [B]


def test_rank_drops_unassessable_and_unparseable(stub_score):
    assert autosolve.rank_hallucination(PRODUCT, [D, ["not_a_smiles"], A]) == [A]


def test_rank_ties_break_on_fewer_penalties(stub_score, monkeypatch):
    scores = {"CCO": (50, 9), "CCC": (50, 2)}
    monkeypatch.setattr(
        autosolve,
        "calculate_hallucination_score",
        lambda r, p, w=None: {"score": scores[r][0], "penalty_total": scores[r][1]},
    )
    assert autosolve.rank_hallucination(PRODUCT, [A, C]) == [C, A]


def test_rank_treats_a_bare_string_as_one_pathway(stub_score):
    # Iterated per character this would become "C", "C", "O" -- three pathways.
    assert autosolve.rank_hallucination(PRODUCT, "CCO") == [A]


def test_canonicalize_pathways_keeps_shape_and_unparseable():
    out = autosolve.canonicalize_pathways([["C(O)C", "not_a_smiles"], "OCC"])
    assert out == [["CCO", "not_a_smiles"], "CCO"]


# --------------------------------------------------------------------------
# hybrid_views
# --------------------------------------------------------------------------
BASE = {"mol_no": 0, "input": "P", "output": "R", "model_id": "m"}


LLM = {"proposals": [["B"]]}


def az_rec(solved: bool, n_routes: int, proposals: list) -> dict:
    return {"solved": solved, "n_routes": n_routes, "proposals": proposals}


def test_hybrid_uses_az_and_not_the_llm_when_az_solved():
    views = autosolve.hybrid_views(az_rec(True, 2, [["A"]]), LLM, BASE)
    assert views["hybrid"]["source"] == "az"
    assert views["hybrid"]["proposals"] == [["A"]]
    assert views["llm"]["proposals"] == [["B"]]  # the llm view is still reported


def test_hybrid_keeps_an_empty_az_answer_instead_of_falling_back():
    # A basic or in-stock target: AZ "solves" it with a route that has no reaction.
    # AutoSolver returns that route, so it is scored as unanswered, not rescued.
    hybrid = autosolve.hybrid_views(az_rec(True, 1, []), LLM, BASE)["hybrid"]
    assert hybrid["source"] == "az"
    assert hybrid["proposals"] == []
    assert hybrid["status"] == 504


def test_hybrid_falls_back_to_the_llm_only_when_az_did_not_solve():
    views = autosolve.hybrid_views(az_rec(False, 0, []), LLM, BASE)
    assert views["hybrid"]["source"] == "llm"
    assert views["hybrid"]["proposals"] == [["B"]]


def test_hybrid_falls_back_when_az_claims_solved_but_returned_no_routes():
    # AutoSolver tests ``az_solved and az_routes``; an empty route list is falsy.
    assert autosolve.hybrid_views(az_rec(True, 0, []), LLM, BASE)["hybrid"]["source"] == "llm"


def test_hybrid_ignores_az_proposals_when_unsolved():
    views = autosolve.hybrid_views(az_rec(False, 3, [["A"]]), LLM, BASE)
    assert views["az"]["proposals"] == []
    assert views["hybrid"]["source"] == "llm"


def test_hybrid_status_is_504_only_when_the_answer_is_empty():
    views = autosolve.hybrid_views(az_rec(False, 0, []), {"proposals": []}, BASE)
    assert views["hybrid"]["status"] == 504
    assert views["az"]["status"] == views["llm"]["status"] == 504


# --------------------------------------------------------------------------
# az_first_step_proposals
# --------------------------------------------------------------------------
def route(*reactants: str) -> dict:
    return {
        "type": "mol",
        "smiles": "T",
        "children": [
            {
                "type": "reaction",
                "children": [{"type": "mol", "smiles": r, "children": []} for r in reactants],
            }
        ],
    }


def test_only_the_top_az_route_is_used():
    # AutoSolver answers with az_routes[0]; later routes must not add proposals.
    assert autosolve.az_first_step_proposals([route("OCC", "CC(=O)O"), route("CCCl")]) == [
        ["OCC", "CC(=O)O"]
    ]


def test_a_reaction_free_top_route_gives_no_proposal_even_if_a_later_route_has_one():
    in_stock = {"type": "mol", "smiles": "O", "in_stock": True, "children": []}
    assert autosolve.az_first_step_proposals([in_stock, route("OCC")]) == []


def test_no_routes_gives_no_proposal():
    assert autosolve.az_first_step_proposals([]) == []
    assert autosolve.az_first_step_proposals(None) == []


# --------------------------------------------------------------------------
# run_az_record
# --------------------------------------------------------------------------
def test_run_az_record_reports_a_failure_instead_of_raising():
    def boom(smiles, model):
        raise RuntimeError("no model")

    rec = autosolve.run_az_record(1, PRODUCT, "USPTO", run_az=boom)
    assert rec["solved"] is False
    assert rec["proposals"] == []
    assert rec["error"] == "RuntimeError: no model"


# --------------------------------------------------------------------------
# generate_one: where the hallucination gate runs
# --------------------------------------------------------------------------
def stub_pipeline(monkeypatch, pathways: list) -> list[dict]:
    """Replace ``llm_pipeline``; return the list its calls are recorded in."""
    import chemberta4.deepretro.llm as llm

    calls: list[dict] = []

    def fake_pipeline(smiles, **kwargs):
        calls.append({"smiles": smiles, **kwargs})
        return list(pathways), ["e"] * len(pathways), [0.1] * len(pathways)

    monkeypatch.setattr(llm, "llm_pipeline", fake_pipeline)
    return calls


@pytest.fixture
def captured(monkeypatch):
    return stub_pipeline(monkeypatch, [B, A, C])


ROW = {"input": PRODUCT, "output": "OC(=O)c1ccccc1O.CC(=O)O"}


def test_autosolve_keeps_the_gate_out_of_the_retry_loop(captured, stub_score):
    rec = task.generate_one(
        ROW, 0, model="m", mode="autosolve", stability_check=True, hallucination_check=True
    )
    assert captured[0]["hallucination_check"] is False
    assert captured[0]["stability_check"] is True
    assert rec["proposals"] == [A, C, B]  # ranked, not filtered
    assert rec["status"] == 200


def test_autosolve_without_the_flag_leaves_the_order_alone(captured, stub_score):
    rec = task.generate_one(ROW, 0, model="m", mode="autosolve", hallucination_check=False)
    assert rec["proposals"] == [B, A, C]


def test_pipeline_mode_still_passes_the_gate_into_the_loop(captured):
    task.generate_one(ROW, 0, model="m", mode="pipeline", hallucination_check=True)
    assert captured[0]["hallucination_check"] is True


def test_autosolve_canonicalizes_the_target_before_the_llm_sees_it(captured):
    # AutoSolver strips and canonicalizes the target first, so the prompt, the
    # validity check and the gate all use one spelling.
    rec = task.generate_one({"input": " OCC ", "output": "x"}, 0, model="m", mode="autosolve")
    assert captured[0]["smiles"] == "CCO"
    assert rec["input"] == " OCC "  # the record keeps what the dataset said


def test_pipeline_mode_sends_the_target_as_given(captured):
    task.generate_one({"input": "OCC", "output": "x"}, 0, model="m", mode="pipeline")
    assert captured[0]["smiles"] == "OCC"


def test_autosolve_keeps_the_first_k_pathways(monkeypatch):
    stub_pipeline(monkeypatch, [A, B, C, D])
    rec = task.generate_one(ROW, 0, model="m", mode="autosolve", top_k=2)
    assert rec["proposals"] == [A, B]


def test_autosolve_defaults_to_three_pathways(monkeypatch):
    stub_pipeline(monkeypatch, [A, B, C, D])
    assert len(task.generate_one(ROW, 0, model="m", mode="autosolve")["proposals"]) == 3


def test_autosolve_slices_first_then_skips_empty_pathways(monkeypatch):
    # solve_multiple takes pathways[:k] and only then skips those without reactants,
    # so an empty pathway inside the window costs a slot.
    stub_pipeline(monkeypatch, [[], A, B, C])
    rec = task.generate_one(ROW, 0, model="m", mode="autosolve", top_k=3)
    assert rec["proposals"] == [A, B]


def test_pipeline_mode_is_not_capped(monkeypatch):
    stub_pipeline(monkeypatch, [A, B, C, D])
    assert len(task.generate_one(ROW, 0, model="m", mode="pipeline")["proposals"]) == 4


def test_the_k_cap_applies_after_ranking(monkeypatch, stub_score):
    # B scores worst, so it is demoted past the window of two.
    stub_pipeline(monkeypatch, [B, A, C])
    rec = task.generate_one(
        ROW, 0, model="m", mode="autosolve", hallucination_check=True, top_k=2
    )
    assert rec["proposals"] == [A, C]


# --------------------------------------------------------------------------
# AZ module and phase
# --------------------------------------------------------------------------
def test_basic_molecule_is_not_searched():
    solved, routes = az.run_az("CCO")
    assert solved is True
    assert autosolve.az_first_step_proposals(routes) == []


def test_missing_config_fails_with_a_message_that_says_what_to_set(monkeypatch):
    monkeypatch.delenv("AZ_MODELS_PATH", raising=False)
    monkeypatch.delenv("AZ_MODEL_CONFIG_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="AZ_MODELS_PATH"):
        az.resolve_config("USPTO")


def test_config_is_read_from_the_model_folder(monkeypatch, tmp_path):
    (tmp_path / "USPTO").mkdir()
    (tmp_path / "USPTO" / "config.yml").write_text("")
    monkeypatch.setenv("AZ_MODELS_PATH", str(tmp_path))
    monkeypatch.delenv("AZ_MODEL_CONFIG_PATH", raising=False)
    assert az.resolve_config("USPTO") == str(tmp_path / "USPTO" / "config.yml")


def test_missing_aizynthfinder_names_the_extra(monkeypatch):
    monkeypatch.setattr(az, "resolve_config", lambda model=None: "config.yml")
    monkeypatch.setattr(az, "_FINDER_CACHE", {})
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "aizynthfinder.aizynthfinder", None)
    with pytest.raises(ImportError, match=r"chemberta4\[az\]"):
        az.run_az("c1ccccc1CC(=O)NCCN")


def test_az_phase_does_not_search_what_is_already_checkpointed(tmp_path, monkeypatch):
    # No config is set: if the phase tried to search, resolve_config would raise.
    monkeypatch.delenv("AZ_MODELS_PATH", raising=False)
    monkeypatch.delenv("AZ_MODEL_CONFIG_PATH", raising=False)
    jsonl = tmp_path / "az.jsonl"
    jsonl.write_text("".join(json.dumps({"mol_no": i}) + "\n" for i in (0, 1)))
    rows = [(0, {"input": "CCO"}), (1, {"input": "CCN"})]
    cli.run_az_phase(rows, jsonl, "USPTO", workers=1)
    assert len(jsonl.read_text().splitlines()) == 2


def test_az_phase_fails_before_spawning_workers_when_config_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("AZ_MODELS_PATH", raising=False)
    monkeypatch.delenv("AZ_MODEL_CONFIG_PATH", raising=False)
    jsonl = tmp_path / "az.jsonl"
    with pytest.raises(FileNotFoundError):
        cli.run_az_phase([(0, {"input": "c1ccccc1CC(=O)NCCN"})], jsonl, "USPTO", workers=1)
    assert not jsonl.exists() or jsonl.read_text() == ""


def test_run_name_for_existing_modes_is_unchanged():
    assert task.run_name("d", "m/Model") == "d_Model_raw"
    # The autosolve-only settings must not leak into the other modes' names.
    assert (
        task.run_name(
            "d", "m/Model", mode="pipeline", az_model="X",
            stability_check=False, hallucination_check=False, top_k=9,
        )
        == "d_Model_pipeline"
    )


def test_default_autosolve_run_name_is_short():
    name = task.run_name("d", "m/Model", mode="autosolve", az_model="Pistachio_100+")
    assert name == "d_Model_autosolve_Pistachio_100+"


def test_run_name_names_every_setting_that_differs_from_deepretro():
    # Otherwise a run with another setting would resume this run's checkpoint.
    name = task.run_name(
        "d", "m/Model", mode="autosolve", az_model="USPTO",
        stability_check=False, hallucination_check=False, top_k=5,
    )
    assert name == "d_Model_autosolve_USPTO_nostab_nohall_k5"
    assert task.run_name("d", "m/Model", mode="autosolve", top_k=1).endswith("_k1")


# --------------------------------------------------------------------------
# Defaults resolved by scripts/evaluate.py
# --------------------------------------------------------------------------
def resolve_args(argv: list[str], config: dict | None = None):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--task", "retrosynthesis", "--model", "m", *argv])
    return module.resolve(args, config or {})


def test_autosolve_defaults_to_deepretros_checks_model_and_k():
    args = resolve_args(["--mode", "autosolve"])
    assert args.stability_check is True
    assert args.hallucination_check is True
    assert args.az_model == "Pistachio_100+"
    assert args.top_k == 3


@pytest.mark.parametrize("mode", ["raw", "pipeline"])
def test_other_modes_keep_the_checks_off_by_default(mode):
    args = resolve_args(["--mode", mode])
    assert args.stability_check is False
    assert args.hallucination_check is False


def test_the_checks_can_be_switched_off_in_autosolve():
    args = resolve_args(["--mode", "autosolve", "--no-stability-check", "--no-hallucination-check"])
    assert args.stability_check is False
    assert args.hallucination_check is False


def test_the_checks_can_still_be_switched_on_in_pipeline():
    args = resolve_args(["--mode", "pipeline", "--stability-check", "--hallucination-check"])
    assert args.stability_check is True
    assert args.hallucination_check is True


def test_a_config_value_beats_the_mode_default():
    args = resolve_args(["--mode", "autosolve"], {"stability_check": False})
    assert args.stability_check is False
    assert args.hallucination_check is True
