"""Scoring tests for single-step retrosynthesis.

These pin down the rules that decide whether a proposal counts as correct,
because they are easy to get subtly wrong and every reported number depends on
them:

- a prediction must match the ground truth as a *set of the same size*, so
  proposing a superset or a subset is wrong even if every molecule appears;
- SMILES are compared after RDKit canonicalization, not as strings;
- MaxFrag only asks about the largest precursor, which is the looser metric
  from Tetko et al. (2020);
- a proposal containing an unparseable SMILES scores zero rather than raising.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="scoring needs RDKit")

from chemberta4.deepretro.one_step_eval import (  # noqa: E402
    canon,
    max_frag,
    score_proposal,
    score_record,
)

ETHANOL = "CCO"
ACETIC = "CC(=O)O"


def test_canon_normalizes_equivalent_smiles() -> None:
    assert canon("OCC") == canon("CCO") == ETHANOL
    assert canon("not a molecule") is None


def test_exact_match_is_order_independent() -> None:
    any_c, all_c, maxfrag, _, _, valid = score_proposal(["OCC", ACETIC], [ETHANOL, ACETIC])
    assert (any_c, all_c, maxfrag, valid) == (1, 1, 1, True)


def test_missing_a_reactant_is_not_correct() -> None:
    # Right molecule, wrong count: the cardinality check must reject it.
    any_c, all_c, _, _, missed, valid = score_proposal([ETHANOL], [ETHANOL, ACETIC])
    assert (any_c, all_c) == (0, 0)
    assert missed == [ACETIC]
    assert valid is True


def test_extra_reactant_is_not_correct() -> None:
    any_c, all_c, _, not_common, _, _ = score_proposal(
        [ETHANOL, ACETIC, "CO"], [ETHANOL, ACETIC]
    )
    assert (any_c, all_c) == (0, 0)
    assert not_common == [canon("CO")]


def test_invalid_smiles_scores_zero_without_raising() -> None:
    any_c, all_c, maxfrag, not_common, _, valid = score_proposal(["Q@#"], [ETHANOL])
    assert (any_c, all_c, maxfrag) == (0, 0, 0)
    assert valid is False
    assert not_common == ["Q@#"]


def test_max_frag_picks_the_most_heavy_atoms() -> None:
    assert max_frag([ETHANOL, "CCCCO"]) == "CCCCO"
    assert max_frag([]) is None


def test_maxfrag_can_be_right_while_exact_match_is_wrong() -> None:
    # The largest precursor matches but a second one is missing: this is the
    # gap MaxFrag accuracy is meant to expose.
    _, all_c, maxfrag, _, _, _ = score_proposal(["CCCCO"], ["CCCCO", ETHANOL])
    assert all_c == 0
    assert maxfrag == 1


def test_score_record_counts_a_parse_failure_as_incorrect() -> None:
    row = score_record(
        {"mol_no": 0, "input": "CCOC(C)=O", "output": f"{ETHANOL}.{ACETIC}",
         "proposals": [], "status": 504}
    )
    assert row["parse_status"] == 504
    assert row["all_correct"] == 0
    assert row["n_proposals"] == 0
    # The molecule stays in the denominator; it is not dropped.
    assert row["missed"] == [ETHANOL, ACETIC]


def test_score_record_reports_first_hit_rank_for_topk() -> None:
    row = score_record(
        {
            "mol_no": 1,
            "input": "CCOC(C)=O",
            "output": f"{ETHANOL}.{ACETIC}",
            "proposals": [["CO", "CC(=O)Cl"], ["OCC", ACETIC]],
        }
    )
    assert row["all_correct"] == 0, "top-1 is the wrong proposal"
    assert row["all_correct_topk"] == 1, "the second proposal is right"
    assert row["first_hit_rank"] == 2
