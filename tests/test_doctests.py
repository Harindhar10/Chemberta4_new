"""Run the doctests carried by our own modules and by the copied ones.

The copied DeepRetro files came with doctests, and those are the regression
tests for the copied code: if a re-sync changes behaviour, they fail here.

Modules are loaded one at a time and skipped when an optional dependency is
missing, so the suite is useful on a machine with only part of the stack
installed instead of failing at collection.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "chemberta4.evaluation.thinking",
    "chemberta4.evaluation.hub",
    "chemberta4.tasks.retrosynthesis.run",
    "chemberta4.tasks.retrosynthesis.az",
    "chemberta4.tasks.retrosynthesis.autosolve",
    "chemberta4.tasks.chemllmbench.data",
    "chemberta4.tasks.chemllmbench.registry",
    "chemberta4.tasks.chemllmbench.run",
    "chemberta4.evaluation.reporting",
    "chemberta4.deepretro.one_step_eval",
    "chemberta4.deepretro.llm_helpers",
    "chemberta4.deepretro.llm_interface",
    "chemberta4.deepretro.llm_trace",
    "chemberta4.deepretro.utils_molecule",
    "chemberta4.deepretro.vllm_metrics",
    "chemberta4.deepretro.stability_checker",
    "chemberta4.deepretro.hallucination_checker",
    "chemberta4.deepretro.hallucination_weights",
]


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name}: {exc}")

    results = doctest.testmod(
        module,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
        verbose=False,
        report=False,
    )
    assert results.failed == 0, f"{results.failed} of {results.attempted} doctests failed in {name}"
