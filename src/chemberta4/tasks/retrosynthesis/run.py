"""Run single-step retrosynthesis against a vLLM-served model.

Two modes, which answer different questions.

``raw``
    One completion per molecule. The response text is stored as it arrives and
    parsed at scoring time. This measures the model on its own.

``pipeline``
    One ``llm_pipeline`` per molecule: temperature retries when a response does
    not parse, ``validity_check`` on every proposed SMILES, and the optional
    stability and hallucination filters. This measures what DeepRetro would
    actually hand to its multi-step search, so it is the number to quote for
    the system rather than the model.

``autosolve``
    DeepRetro's depth-1 loop (``AutoSolver``): AiZynthFinder first, the LLM as
    fallback. The stability and hallucination checks default to **on**, as in
    ``AutoSolver``. The LLM half differs from ``pipeline`` in four ways, all taken
    from ``AutoSolver``: the target is canonicalized before the prompt; the
    hallucination gate runs once after the retry loop and *ranks* pathways instead of
    dropping them; the answer is capped at the first ``top_k`` pathways; and when AZ
    solves a target its top route is the answer and the LLM's is not used. See
    :mod:`chemberta4.tasks.retrosynthesis.autosolve`. The AZ half and the three
    scored views (hybrid, az, llm) are in :mod:`chemberta4.cli`.

``--prompt ether0`` swaps DeepRetro's prompt for the one futurehouse/ether0 was
trained on, which asks for a single reaction SMILES. Use it to give ether0 a
fair run. Its answer parses to one proposal, so only top-1 means anything there.

Generation is resumable; see :mod:`chemberta4.evaluation.runner`.
"""

from __future__ import annotations

from typing import Any

ETHER0_RETRO_PROMPT = (
    "Suggest a commercially feasible one-step route to synthesize {smiles}. "
    "Answer with reaction smiles format "
    "(e.g., CC=O.O=C1CCC1Cl>[Mg2+].CCOCC>CC(O)C1CCC1=O)"
)

MODES = ("raw", "pipeline", "autosolve")
PROMPTS = ("deepretro", "ether0")


def ether0_messages(smiles: str) -> list[dict[str, str]]:
    """Chat messages asking ether0 for a one-step synthesis of ``smiles``.

    Examples
    --------
    >>> ether0_messages("CCO")[0]["role"]
    'user'
    >>> "CCO" in ether0_messages("CCO")[0]["content"]
    True
    """
    return [{"role": "user", "content": ETHER0_RETRO_PROMPT.format(smiles=smiles)}]


def prompt_messages(prompt: str, smiles: str) -> list[dict[str, str]] | None:
    """Explicit messages for ``prompt``, or ``None`` to use DeepRetro's own.

    Examples
    --------
    >>> prompt_messages("deepretro", "CCO") is None
    True
    >>> len(prompt_messages("ether0", "CCO"))
    1
    """
    return ether0_messages(smiles) if prompt == "ether0" else None


def generate_one(
    row: Any,
    mol_no: int,
    *,
    model: str,
    mode: str = "raw",
    prompt: str = "deepretro",
    thinking: bool = False,
    max_output_tokens: int = 1024,
    stability_check: bool = False,
    hallucination_check: bool = False,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run one molecule and return its record.

    Imported lazily because the copied LLM modules pull in litellm, rdkit and
    structlog; a ``--help`` or a data-preparation run should not pay for that.
    ``top_k`` caps the proposals in ``autosolve`` mode only.

    ``raw`` records carry ``raw`` (the response text). ``pipeline`` and ``autosolve``
    records carry the parsed ``proposals`` plus, for tracing:

    ``attempts``
        One entry per ``llm_pipeline`` try (``attempt``, ``temperature``, ``model``,
        ``status``, ``raw``), including the failed retries.
    ``explanations``, ``confidence``
        As ``llm_pipeline`` returned them, aligned with its pathways.
    ``llm_pathways`` (``autosolve`` only)
        Those pathways before canonicalization, ranking and the ``top_k`` cap, which
        reorder and drop, so ``explanations`` and ``confidence`` do not line up with
        ``proposals``.
    """
    from chemberta4.deepretro.llm import call_LLM, llm_pipeline

    smiles = row["input"]
    messages = prompt_messages(prompt, smiles)
    record = {
        "mol_no": mol_no,
        "model_id": model,
        "thinking": thinking,
        "mode": mode,
        "prompt": prompt,
        "input": smiles,
        "output": row["output"],
    }

    if mode == "raw":
        # Temperature 1.0 is DeepRetro's own setting for this evaluation.
        status, text = call_LLM(
            smiles,
            model=model,
            messages=messages,
            temperature=1.0,
            enable_thinking=thinking,
            max_output_tokens=max_output_tokens,
        )
        return {**record, "call_status": status, "raw": text}

    if mode == "autosolve":
        from chemberta4.deepretro.utils_molecule import canonicalize
        from chemberta4.tasks.retrosynthesis.autosolve import (
            canonicalize_pathways,
            rank_hallucination,
        )

        # AutoSolver rewrites the target canonically before anything else, so the
        # prompt, the validity check and the gate all see one spelling. The record
        # keeps ``input`` as given, so scoring and the dataset join do not change.
        target = canonicalize(smiles.strip())
        messages = prompt_messages(prompt, target)
        # AutoSolver keeps the hallucination gate out of the retry loop: a rejection
        # there cannot trigger another attempt. So the loop gets validity and
        # stability only, and the gate runs once on what comes back.
        attempts: list[dict[str, Any]] = []
        pathways, explanations, confidence = llm_pipeline(
            target,
            model=model,
            messages=messages,
            enable_thinking=thinking,
            max_output_tokens=max_output_tokens,
            stability_check=stability_check,
            hallucination_check=False,
            trace=attempts,
        )
        # ``explanations`` and ``confidence`` line up with these pathways, not with
        # the ranked, capped ``proposals`` below.
        llm_pathways = list(pathways)
        pathways = canonicalize_pathways(pathways)
        if hallucination_check:
            pathways = rank_hallucination(target, pathways)
        # solve_multiple keeps the first k pathways, then skips any with no reactants.
        pathways = [p for p in pathways[:top_k] if p]
        return {
            **record,
            "proposals": pathways,
            "status": 200 if pathways else 504,
            "attempts": attempts,
            "llm_pathways": llm_pathways,
            "explanations": explanations,
            "confidence": confidence,
        }

    attempts = []
    pathways, explanations, confidence = llm_pipeline(
        smiles,
        model=model,
        messages=messages,
        enable_thinking=thinking,
        max_output_tokens=max_output_tokens,
        stability_check=stability_check,
        hallucination_check=hallucination_check,
        trace=attempts,
    )
    return {
        **record,
        "proposals": pathways,
        "status": 200 if pathways else 504,
        "attempts": attempts,
        "explanations": explanations,
        "confidence": confidence,
    }


def keep_record(record: dict[str, Any]) -> bool:
    """Whether to checkpoint a record.

    A failed API call is left out so the next run retries that molecule. A
    model that answered but produced nothing parseable *is* kept: that is a
    result about the model, not a transport failure, and it counts as incorrect.

    Examples
    --------
    >>> keep_record({"call_status": 200, "raw": "..."})
    True
    >>> keep_record({"call_status": 504})
    False
    >>> keep_record({"proposals": [], "status": 504})
    True
    """
    return record.get("call_status", 200) == 200


def run_name(
    data_slug: str,
    model: str,
    *,
    mode: str = "raw",
    prompt: str = "deepretro",
    thinking: bool = False,
    az_model: str | None = None,
    stability_check: bool = True,
    hallucination_check: bool = True,
    top_k: int = 3,
) -> str:
    """Directory name for a run, so differing settings never share a checkpoint.

    The last three arguments matter in ``autosolve`` mode only, where their
    defaults are DeepRetro's; a setting that differs from DeepRetro's is named.

    Examples
    --------
    >>> run_name("uspto_50k_test_250", "hosted_vllm/Qwen/Qwen3-8B")
    'uspto_50k_test_250_Qwen3-8B_raw'
    >>> run_name("d", "m/Model", mode="pipeline", thinking=True, prompt="ether0")
    'd_Model_think_pipeline_ether0prompt'
    >>> run_name("d", "m/Model", mode="autosolve", az_model="Pistachio_100+")
    'd_Model_autosolve_Pistachio_100+'
    >>> run_name("d", "m/Model", mode="autosolve", az_model="USPTO",
    ...          stability_check=False, hallucination_check=False, top_k=5)
    'd_Model_autosolve_USPTO_nostab_nohall_k5'
    """
    model_slug = model.rstrip("/").split("/")[-1]
    name = f"{data_slug}_{model_slug}{'_think' if thinking else ''}_{mode}"
    if mode == "autosolve":
        if az_model:
            name += f"_{az_model}"
        if not stability_check:
            name += "_nostab"
        if not hallucination_check:
            name += "_nohall"
        if top_k != 3:
            name += f"_k{top_k}"
    if prompt != "deepretro":
        name += f"_{prompt}prompt"
    return name
