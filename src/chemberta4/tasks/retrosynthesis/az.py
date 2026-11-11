"""AiZynthFinder access for the ``autosolve`` retrosynthesis mode.

A native re-implementation, not a copy, of the AZ calls in DeepRetro's
``deepretro/utils/az.py`` (commit bf9acf0): same stock and policy selection, same
"basic molecule" shortcut. Three things differ on purpose:

* Paths come from ``AZ_MODELS_PATH`` / ``AZ_MODEL_CONFIG_PATH`` as given.
  DeepRetro prefixes its own project root, which would point nowhere here.
* There is no disk cache. The shared checkpoint written by the autosolve run
  (see :func:`chemberta4.cli.run_az_phase`) already memoizes per molecule.
* ``BASIC_MOLECULES`` is not restored; :func:`is_basic_molecule` covers it.

aizynthfinder is imported lazily so everything else works without it.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from rdkit import Chem
from rdkit.Chem import rdqueries

# One finder per config path and process. Building a finder loads the policy
# networks and the ZINC stock, so reuse it across molecules, as AZ's own CLI does.
_FINDER_CACHE: dict[str, Any] = {}


def is_basic_molecule(smiles: str) -> bool:
    """Whether ``smiles`` is too small to be worth searching (< 5 atoms or < 5 carbons).

    Examples
    --------
    >>> is_basic_molecule("CCO")
    True
    >>> is_basic_molecule("C1CCCCC1")
    False
    >>> is_basic_molecule("invalid_smiles!!")
    False
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    carbons = len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6)))
    return mol.GetNumAtoms() < 5 or carbons < 5


def resolve_config(az_model: str | None = None) -> str:
    """Path to the AZ ``config.yml`` for ``az_model``, else ``AZ_MODEL_CONFIG_PATH``.

    Raises
    ------
    FileNotFoundError
        If neither the model folder's config nor the fallback exists.

    Examples
    --------
    >>> resolve_config("no-such-model")  # doctest: +SKIP
    Traceback (most recent call last):
    FileNotFoundError: ...
    """
    models_path = os.getenv("AZ_MODELS_PATH")
    if az_model and models_path:
        candidate = os.path.join(models_path, az_model, "config.yml")
        if os.path.isfile(candidate):
            return candidate
    fallback = os.getenv("AZ_MODEL_CONFIG_PATH")
    if fallback and os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"No AiZynthFinder config for model {az_model!r}. Set AZ_MODELS_PATH so that "
        "<AZ_MODELS_PATH>/<az-model>/config.yml exists, or set AZ_MODEL_CONFIG_PATH. "
        "A relative path is resolved against the current directory. The public USPTO "
        "model: python -m aizynthfinder.tools.download_public_data az_models/USPTO "
        "(Pistachio models are private; see README.md)."
    )


def _get_finder(config: str) -> Any:
    finder = _FINDER_CACHE.get(config)
    if finder is not None:
        return finder
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
    except ImportError as exc:
        raise ImportError(
            f"autosolve mode could not import AiZynthFinder ({type(exc).__name__}: {exc}). "
            "If it is not installed: pip install 'chemberta4[az]'"
        ) from exc
    finder = AiZynthFinder(configfile=config)
    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    _FINDER_CACHE[config] = finder
    return finder


def run_az(smiles: str, az_model: str = "Pistachio_100+") -> tuple[bool, Sequence[dict[str, Any]]]:
    """Search one target. Returns ``(solved, routes)`` as DeepRetro's ``run_az`` does.

    A basic molecule is reported solved with a single reaction-free route and
    no search, so its first-step proposals are empty.

    Examples
    --------
    >>> run_az("CCO")
    (True, [{'type': 'mol', 'smiles': 'CCO', 'is_chemical': True, 'in_stock': True, 'children': []}])
    """
    if is_basic_molecule(smiles):
        route = {
            "type": "mol",
            "smiles": smiles,
            "is_chemical": True,
            "in_stock": True,
            "children": [],
        }
        return True, [route]
    finder = _get_finder(resolve_config(az_model))
    finder.target_smiles = smiles
    finder.tree_search()
    finder.build_routes()
    solved = bool(finder.extract_statistics()["is_solved"])
    routes = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
    return solved, routes
