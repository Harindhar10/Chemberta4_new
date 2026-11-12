"""The ChemLLMBench task keys, groups and headline metrics.

Keys, display names, groups and the primary-metric order are those of the
reference ``main.py`` (``TASK_REGISTRY``, ``TASK_GROUPS``,
``PRIMARY_METRIC_KEYS``), so results line up with its leaderboard columns.
Each key points at a family module in this package and the sub-task name that
module expects.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class Task:
    key: str
    display: str
    module: str  # family module in chemberta4.tasks.chemllmbench
    subtask: str  # the name the family module's load/parse/score take
    zero_shot_only: bool = False

    def family(self) -> ModuleType:
        return importlib.import_module(f"chemberta4.tasks.chemllmbench.{self.module}")


_TASKS = [
    # Name Prediction
    Task("smiles_to_iupac", "Name Prediction – SMILES → IUPAC", "name_prediction", "SMILES_TO_IUPAC"),
    Task("iupac_to_smiles", "Name Prediction – IUPAC → SMILES", "name_prediction", "IUPAC_TO_SMILES"),
    Task("smiles_to_formula", "Name Prediction – SMILES → Formula", "name_prediction", "SMILES_TO_FORMULA"),
    Task("iupac_to_formula", "Name Prediction – IUPAC → Formula", "name_prediction", "IUPAC_TO_FORMULA"),
    # Property Prediction
    Task("bbbp", "Property Prediction – BBBP", "property_prediction", "BBBP"),
    Task("bace", "Property Prediction – BACE", "property_prediction", "BACE"),
    Task("tox21", "Property Prediction – Tox21", "property_prediction", "Tox21"),
    Task("hiv", "Property Prediction – HIV", "property_prediction", "HIV"),
    Task("clintox", "Property Prediction – ClinTox", "property_prediction", "ClinTox"),
    # Reaction Prediction
    Task("reaction_prediction", "Forward Reaction Prediction", "reaction_prediction", "forward"),
    # Reaction Component Selection (zero-shot only)
    Task("reactant_selection", "Reaction Selection – Reactant (zero-shot)",
         "reaction_selection", "Reactant_selection", zero_shot_only=True),
    Task("ligand_selection", "Reaction Selection – Ligand (zero-shot)",
         "reaction_selection", "Ligand_selection", zero_shot_only=True),
    Task("solvent_selection", "Reaction Selection – Solvent (zero-shot)",
         "reaction_selection", "Solvent_selection", zero_shot_only=True),
    # Retrosynthesis
    Task("retrosynthesis", "Retrosynthesis", "retrosynthesis", "retrosynthesis"),
    # Molecule Design
    Task("mol_caption", "Molecule Caption (SMILES → Description)", "molecule_design", "MOL_CAPTION"),
    Task("mol_design", "Molecule Design (Description → SMILES)", "molecule_design", "MOL_DESIGN"),
    # Yield Prediction
    Task("Buchwald-Hartwig", "Reaction Yield Prediction – Buchwald-Hartwig",
         "yield_prediction", "Buchwald-Hartwig"),
    Task("Suzuki-Miyaura", "Reaction Yield Prediction – Suzuki-Miyaura",
         "yield_prediction", "Suzuki-Miyaura"),
]

TASKS: dict[str, Task] = {t.key: t for t in _TASKS}

TASK_GROUPS: dict[str, list[str]] = {
    "all": list(TASKS),
    "name_prediction": ["smiles_to_iupac", "iupac_to_smiles", "smiles_to_formula", "iupac_to_formula"],
    "property_prediction": ["bbbp", "bace", "tox21", "hiv", "clintox"],
    "reaction_prediction": ["reaction_prediction"],
    "reaction_component_selection": ["reactant_selection", "ligand_selection", "solvent_selection"],
    "retrosynthesis": ["retrosynthesis"],
    "molecule_design": ["mol_caption", "mol_design"],
    "yield_prediction": ["Buchwald-Hartwig", "Suzuki-Miyaura"],
}

# Tried in order to pick the one number reported per task.
PRIMARY_METRIC_KEYS = ("accuracy", "exact_match", "top1_accuracy", "exact", "bleu4")


def resolve(names: list[str] | str | None) -> list[str]:
    """Task keys for a list of task keys and group names, case-insensitive.

    Groups expand in place and duplicates are dropped, keeping first order.

    >>> resolve("property_prediction,bbbp,HIV")
    ['bbbp', 'bace', 'tox21', 'hiv', 'clintox']
    >>> resolve(["buchwald-hartwig"])
    ['Buchwald-Hartwig']
    """
    if names is None:
        names = ["all"]
    if isinstance(names, str):
        names = [n for n in names.split(",") if n.strip()]
    tasks_by_lower = {k.lower(): k for k in TASKS}
    groups_by_lower = {k.lower(): k for k in TASK_GROUPS}
    keys: list[str] = []
    for name in names:
        low = name.strip().lower()
        if low in groups_by_lower:
            keys.extend(TASK_GROUPS[groups_by_lower[low]])
        elif low in tasks_by_lower:
            keys.append(tasks_by_lower[low])
        else:
            raise ValueError(
                f"Unknown ChemLLMBench task or group {name!r}. "
                f"Tasks: {list(TASKS)}. Groups: {list(TASK_GROUPS)}."
            )
    return list(dict.fromkeys(keys))


def primary_metric(metrics: dict) -> str | None:
    """The headline metric name for a metrics dict (0.0 counts as present).

    >>> primary_metric({"bleu2": 0.1, "bleu4": 0.0})
    'bleu4'
    >>> primary_metric({"fcd": 3.0}) is None
    True
    """
    for key in PRIMARY_METRIC_KEYS:
        if metrics.get(key) is not None:
            return key
    return None
