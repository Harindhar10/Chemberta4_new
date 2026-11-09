"""Guard the prompts against silent drift.

Scores are only comparable across runs if the prompt is the same, and this
repository holds *copies* of prompts that live elsewhere. Two kinds of drift
would invalidate comparisons without any test failing:

1. the MoleculeNet prompt builder diverging from the DFS ``molnet`` package
   that produced the earlier Claude/GPT results;
2. the copied retrosynthesis prompt diverging from DeepRetro's after a
   re-sync.

Both checks need the original source trees, so each skips when its tree is
absent. They are meant to run on the machine that has the checkouts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DFS = Path.home() / "Documents" / "DFS" / "DFS"
DEEPRETRO = DFS / "DeepRetro"

CASES = [
    ("bbbp", 0, 0, 0),
    ("bbbp", 5, 0, 3),
    ("hiv", 20, 1, 7),
    ("delaney", 5, 2, 1),
]


def _load_original_molnet():
    """Import the DFS ``molnet`` package, or skip."""
    if not (DFS / "molnet" / "prompts.py").exists():
        pytest.skip(f"no molnet package at {DFS}")
    sys.path.insert(0, str(DFS))
    try:
        return importlib.import_module("molnet.prompts")
    except ImportError as exc:
        pytest.skip(f"cannot import molnet.prompts: {exc}")


@pytest.mark.parametrize(("dataset", "k", "seed", "row"), CASES)
def test_moleculenet_prompt_matches_original(dataset: str, k: int, seed: int, row: int) -> None:
    """Our copy must build byte-identical messages to the DFS molnet package."""
    pytest.importorskip("pandas")
    original = _load_original_molnet()
    from chemberta4.tasks.moleculenet import prompts as copied

    try:
        expected = original.build_messages(dataset, "test", k, row, seed)
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"original splits unavailable: {exc}")

    actual = copied.build_messages(dataset, "test", k, row, seed)
    assert actual == expected


def test_moleculenet_examples_are_nested_by_k() -> None:
    """The 5-shot examples must be a prefix-free subset of the 20-shot ones.

    The draw does not depend on k, so a k=5 run and a k=20 run share examples.
    If this breaks, shot counts stop being comparable within a seed.
    """
    pytest.importorskip("pandas")
    from chemberta4.tasks.moleculenet.prompts import select_examples

    try:
        five = select_examples("bbbp", 5, 0)
        twenty = select_examples("bbbp", 20, 0)
    except FileNotFoundError as exc:
        pytest.skip(f"splits not prepared: {exc}")

    assert set(five["smiles"]) <= set(twenty["smiles"])


def test_copied_retro_prompt_matches_deepretro() -> None:
    """The copied prompt must still match the DeepRetro checkout's."""
    source = DEEPRETRO / "deepretro" / "utils" / "variables.py"
    if not source.exists():
        pytest.skip(f"no DeepRetro checkout at {DEEPRETRO}")

    spec = importlib.util.spec_from_file_location("_dr_variables", source)
    assert spec and spec.loader
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    from chemberta4.deepretro import prompts as copied

    for name in ["SYS_PROMPT", "USER_PROMPT", "SYS_PROMPT_V4", "USER_PROMPT_V4",
                 "SYS_PROMPT_OPENAI", "USER_PROMPT_OPENAI", "SYS_PROMPT_DEEPSEEK",
                 "USER_PROMPT_DEEPSEEK", "USER_PROMPT_DEEPSEEK_V4", "ADDON_PROMPT_7_MEMBER"]:
        assert getattr(copied, name) == getattr(original, name), (
            f"{name} drifted from DeepRetro; scores before and after this change "
            f"are not comparable. Update DEEPRETRO.md if the change is intended."
        )


def test_ether0_prompt_asks_for_reaction_smiles() -> None:
    """The ether0 prompt must name the molecule and the format it wants."""
    from chemberta4.tasks.retrosynthesis.run import ether0_messages

    content = ether0_messages("CC(=O)Oc1ccccc1C(=O)O")[0]["content"]
    assert "CC(=O)Oc1ccccc1C(=O)O" in content
    assert "reaction smiles" in content.lower()
