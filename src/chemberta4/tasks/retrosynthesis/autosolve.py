"""Depth-1 autosolve: AiZynthFinder first, the LLM as fallback, scored as three views.

Native re-implementations of the pieces DeepRetro (commit bf9acf0) keeps in
``deepretro/utils/one_step_eval.py`` (``az_first_step_proposals`` :214,
``run_az_record`` :268, ``hybrid_views`` :305) and
``deepretro/algorithms/pipeline_checks.py`` (``hallucination_checker``). They are
not copies: the copied files are marked "do not edit by hand", and these need the
AZ module of this package rather than DeepRetro's. Behaviour is kept the same.

The hallucination gate here differs from ``filter_hallucination_pathways`` in the
copied ``llm.py``, which drops a pathway whose severity is not low or medium. This
gate keeps every assessable pathway, puts those scoring at or above
``reject_below`` first (best score first) and demotes the rest. Because the order
of proposals decides top-1, the two are not interchangeable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from chemberta4.deepretro.hallucination_checker import calculate_hallucination_score
from chemberta4.deepretro.hallucination_weights import DEFAULT_WEIGHTS, HallucinationWeights
from chemberta4.deepretro.llm_helpers import Pathway
from chemberta4.deepretro.one_step_eval import score_record, summarize, write_summary
from chemberta4.deepretro.utils_molecule import canonicalize, is_valid_smiles

VIEWS = ("hybrid", "az", "llm")


def az_first_step_proposals(routes: Any) -> list[Pathway]:
    """Reactants of the first reaction of AiZynthFinder's top route.

    ``AutoSolver`` answers with ``az_routes[0]`` and nothing else, so only the first
    route counts. AZ routes are trees (``mol`` -> ``reaction`` -> ``mol`` ...) that
    run all the way to stock; the single-step answer is the first reaction's
    reactants. A top route with no reaction (a basic or in-stock target) gives no
    proposal.

    Examples
    --------
    >>> route = {"type": "mol", "smiles": "CCOC(C)=O", "children": [
    ...     {"type": "reaction", "children": [
    ...         {"type": "mol", "smiles": "OCC", "children": []},
    ...         {"type": "mol", "smiles": "CC(=O)O", "children": []}]}]}
    >>> other = {"type": "mol", "smiles": "CCOC(C)=O", "children": [
    ...     {"type": "reaction", "children": [
    ...         {"type": "mol", "smiles": "CCCl", "children": []}]}]}
    >>> az_first_step_proposals([route, other])
    [['OCC', 'CC(=O)O']]
    >>> az_first_step_proposals([{"type": "mol", "smiles": "O", "in_stock": True}, route])
    []
    """
    route = next(iter(routes or []), None)
    if not isinstance(route, dict):
        return []
    reaction = next(
        (c for c in route.get("children") or [] if c.get("type") == "reaction"), None
    )
    if reaction is None:
        return []
    reactants = [
        c["smiles"]
        for c in reaction.get("children") or []
        if c.get("type") == "mol" and c.get("smiles")
    ]
    return [reactants] if reactants else []


def run_az_record(
    mol_no: int,
    smiles: str,
    az_model: str,
    run_az: Callable[[str, str], tuple[bool, Sequence[Any]]] | None = None,
) -> dict[str, Any]:
    """Search one molecule and summarize it as a checkpoint record.

    Top-level, not a closure, so it can run in a worker process. Any AZ failure is
    caught and recorded as an unsolved molecule with ``error`` set, so one bad
    molecule or a missing model cannot stop a run. ``run_az`` is injectable for
    tests; workers use the real one.

    Examples
    --------
    >>> route = {"type": "mol", "smiles": "CCOC(C)=O", "children": [
    ...     {"type": "reaction", "children": [
    ...         {"type": "mol", "smiles": "OCC", "children": []}]}]}
    >>> rec = run_az_record(3, "CCOC(C)=O", "Pistachio_100+", run_az=lambda s, m: (True, [route]))
    >>> rec["solved"], rec["proposals"], rec["error"]
    (True, [['OCC']], None)
    >>> def boom(s, m): raise RuntimeError("no model")
    >>> run_az_record(3, "CCOC(C)=O", "Pistachio_100+", run_az=boom)["error"]
    'RuntimeError: no model'
    """
    if run_az is None:
        from chemberta4.tasks.retrosynthesis.az import run_az
    started = time.time()
    try:
        solved, routes = run_az(smiles, az_model)
        return {
            "mol_no": mol_no,
            "solved": bool(solved),
            "n_routes": len(routes),
            "proposals": az_first_step_proposals(routes) if solved else [],
            "error": None,
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # AZ is external; one bad molecule must not stop the run
        return {
            "mol_no": mol_no,
            "solved": False,
            "n_routes": 0,
            "proposals": [],
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 2),
        }


def hybrid_views(
    az_rec: dict[str, Any], llm_rec: dict[str, Any], base: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The ``az``, ``llm`` and ``hybrid`` records for one molecule.

    ``hybrid`` is exactly what ``AutoSolver`` does: if AZ solved the target and
    returned a route (``solved and n_routes > 0``) the answer is that route's first
    step and the LLM is not consulted, **even when the step is empty**. A basic or
    in-stock target comes back from AZ as a route with no reaction, so it scores as
    unanswered rather than falling through to the LLM. Only when AZ did not solve
    the target is the LLM's answer used. Each record has the ``proposals`` shape
    that ``score_record`` accepts.

    Examples
    --------
    >>> base = {"mol_no": 0, "input": "P", "output": "R", "model_id": "m"}
    >>> llm = {"proposals": [["B"]]}
    >>> solved = {"solved": True, "n_routes": 2, "proposals": [["A"]]}
    >>> views = hybrid_views(solved, llm, base)
    >>> views["hybrid"]["source"], views["hybrid"]["proposals"]
    ('az', [['A']])
    >>> basic = {"solved": True, "n_routes": 1, "proposals": []}
    >>> hybrid = hybrid_views(basic, llm, base)["hybrid"]
    >>> hybrid["source"], hybrid["proposals"], hybrid["status"]
    ('az', [], 504)
    >>> unsolved = {"solved": False, "n_routes": 0, "proposals": []}
    >>> hybrid_views(unsolved, llm, base)["hybrid"]["source"]
    'llm'
    """
    az_proposals = az_rec["proposals"] if az_rec.get("solved") else []
    llm_proposals = llm_rec.get("proposals") or []
    use_az = bool(az_rec.get("solved")) and az_rec.get("n_routes", 0) > 0
    answer = az_proposals if use_az else llm_proposals
    return {
        "az": {**base, "proposals": az_proposals, "status": 200 if az_proposals else 504},
        "llm": {**base, "proposals": llm_proposals, "status": 200 if llm_proposals else 504},
        "hybrid": {
            **base,
            "proposals": answer,
            "status": 200 if answer else 504,
            "source": "az" if use_az else "llm",
        },
    }


def canonicalize_pathways(pathways: Sequence[Any]) -> list[Any]:
    """Canonicalize every reactant, leaving what RDKit cannot parse untouched.

    Applied before the gate so the proposals and the checker see one spelling per
    molecule. The shape of each pathway (a bare string or a list) is preserved.

    Examples
    --------
    >>> canonicalize_pathways([["C(O)C", "not_a_smiles"], "OCC"])
    [['CCO', 'not_a_smiles'], 'CCO']
    """
    out: list[Any] = []
    for pathway in pathways:
        if isinstance(pathway, str):
            out.append(canonicalize(pathway))
        elif isinstance(pathway, Sequence):
            out.append([canonicalize(r) if isinstance(r, str) else r for r in pathway])
        else:
            out.append(pathway)
    return out


def rank_hallucination(
    product: str,
    pathways: Sequence[Pathway | str],
    weights: HallucinationWeights | None = None,
) -> list[Pathway]:
    """Order pathways least suspicious first; drop the invalid and unassessable.

    Pathways scoring at or above ``reject_below`` come first, best score first, then
    ascending penalty total. Those below it follow, ordered the same way, so a
    flagged pathway stays available as a fallback instead of being deleted. An
    unparseable, empty, no-op or carbon-radical pathway is dropped. A bare string
    is one pathway, not a sequence of single-character pathways.

    Examples
    --------
    >>> aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    >>> good, bad = ["OC(=O)c1ccccc1O", "CC(=O)Cl"], ["CCO"]
    >>> rank_hallucination(aspirin, [bad, good])[0]
    ['OC(=O)c1ccccc1O', 'CC(=O)Cl']
    >>> rank_hallucination(aspirin, [["not_a_smiles"]])
    []
    >>> one = "OC(=O)c1ccccc1O.CC(=O)Cl"
    >>> rank_hallucination(aspirin, one) == [[one]]
    True
    """
    if isinstance(pathways, str):
        pathways = [pathways]
    threshold = (weights or DEFAULT_WEIGHTS).reject_below
    accepted: list[tuple[int, int, Pathway]] = []
    demoted: list[tuple[int, int, Pathway]] = []
    for pathway in pathways:
        reactants = list(pathway) if isinstance(pathway, list) else [pathway]
        joined = ".".join(reactants)
        if not is_valid_smiles(joined):
            continue
        report = calculate_hallucination_score(joined, product, weights)
        if report.get("unassessable"):
            continue
        entry = (report["score"], report.get("penalty_total", 0), reactants)
        (accepted if report["score"] >= threshold else demoted).append(entry)
    order = lambda e: (-e[0], e[1])  # noqa: E731 - best score, then fewest penalties
    return [p for _, _, p in sorted(accepted, key=order)] + [
        p for _, _, p in sorted(demoted, key=order)
    ]


def score_views(
    df: pd.DataFrame,
    llm_records: dict[int, dict[str, Any]],
    az_records: dict[int, dict[str, Any]],
    *,
    model: str,
    az_model: str,
    run_dir: str,
    summary_csv: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Score the hybrid, az and llm views, write them, and return the summary.

    Writes ``results_{hybrid,az,llm}.csv`` plus ``summary.json`` / ``summary.txt``
    into ``run_dir`` and one row per view into ``summary_csv``. Only molecules present
    in *both* record sets are scored; the rest are reported, never silently dropped
    from the denominator. The AZ checkpoint is shared across runs and can cover more
    molecules than this run, so callers pass only this run's records.

    Returns
    -------
    dict
        The combined summary: ``coverage``, ``az_answered_subset`` and ``views``.
    """
    both = sorted(set(az_records) & set(llm_records))
    missing = len(set(az_records) ^ set(llm_records))
    if missing:
        print(f"{missing} molecules are in only one checkpoint and are left out; re-run to fill them")
    if not both:
        print("nothing to score")
        return {}

    rows: dict[str, list[dict[str, Any]]] = {v: [] for v in VIEWS}
    sources: dict[int, str] = {}
    for i in both:
        row = df.loc[i]
        base = {"mol_no": i, "input": row["input"], "output": row["output"], "model_id": model}
        views = hybrid_views(az_records[i], llm_records[i], base)
        sources[i] = views["hybrid"]["source"]
        for v in VIEWS:
            rows[v].append(score_record(views[v]))

    scored = {v: pd.DataFrame(rows[v]).set_index("mol_no") for v in VIEWS}
    scored["hybrid"]["source"] = pd.Series(sources)
    az_meta = pd.DataFrame([az_records[i] for i in both]).set_index("mol_no")
    scored["az"] = scored["az"].join(az_meta[["solved", "n_routes", "error", "seconds"]])
    for v in VIEWS:
        scored[v].to_csv(os.path.join(run_dir, f"results_{v}.csv"))

    n = len(both)
    view_summaries: dict[str, dict[str, Any]] = {}
    reports = []
    for v in VIEWS:
        view_summaries[v], text = summarize(scored[v], {**meta, "view": v})
        reports.append(f"##### view: {v} #####\n{text}")

    # Head to head where AZ answered: would the LLM have done better there?
    az_answered = [i for i in both if sources[i] == "az"]
    subset: dict[str, dict[str, float]] = {}
    if az_answered:
        for v in ("az", "llm"):
            sub = scored[v].loc[az_answered]
            subset[v] = {
                "top1_all_correct": 100 * sub.all_correct.mean(),
                "topk_all_correct": 100 * sub.all_correct_topk.mean(),
                "top1_maxfrag": 100 * sub.maxfrag_correct.mean(),
            }
    coverage = {
        "az_solved_pct": 100 * sum(bool(az_records[i]["solved"]) for i in both) / n,
        "az_answered_pct": 100 * len(az_answered) / n,
        "az_errors": sum(bool(az_records[i].get("error")) for i in both),
        "az_mean_seconds": float(az_meta["seconds"].mean()),
    }
    summary = {
        **meta,
        "n_scored": n,
        "coverage": coverage,
        "az_answered_subset": subset,
        "views": view_summaries,
    }

    lines = [
        f"model: {model}   az_model: {az_model}   scored {n} molecules",
        f"AZ solved {coverage['az_solved_pct']:.1f}%, answered (has a first step) "
        f"{coverage['az_answered_pct']:.1f}%, {coverage['az_errors']} AZ errors, "
        f"{coverage['az_mean_seconds']:.1f} s/molecule",
    ]
    if subset:
        lines.append(
            f"On the {len(az_answered)} molecules AZ answered -- top-1 all_correct: "
            f"AZ {subset['az']['top1_all_correct']:.1f}% vs LLM {subset['llm']['top1_all_correct']:.1f}%"
        )
    report = "\n".join(lines) + "\n\n" + "\n\n".join(reports)
    print(report)
    write_summary(summary, report, run_dir, summary_csv, views=view_summaries)
    print(f"\nwrote {run_dir}/summary.json, summary.txt and results_{{hybrid,az,llm}}.csv")
    return summary
