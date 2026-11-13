"""Run MoleculeNet property prediction against a backend.

One request per (dataset, k, seed, molecule). The prompt, the in-context
examples and the sampling of ``--limit`` molecules all come from the copied
``molnet`` code, so a run here is directly comparable with the earlier
Claude/GPT and DeepRetro runs over the same datasets.

Unlike the closed-source API eval there is no JSON-schema-constrained decoding:
a vLLM server is asked for a JSON object and sometimes returns something else.
An answer that does not parse is not dropped. It is filled with the train-set
prior at scoring time and surfaces as ``parse_fail_rate`` and ``fill_rate``, so
every model is scored over the same molecules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from chemberta4.evaluation.backends.base import ModelBackend
from chemberta4.tasks.moleculenet.prompts import (
    build_messages,
    load_split,
    parse_response,
    system_text,
)

# One work unit: everything needed to rebuild the request from scratch.
Request = tuple[str, int, int, int, str]


def build_requests(
    datasets: list[str],
    shots: list[int],
    seeds: list[int],
    split: str = "test",
    limit: int | None = None,
) -> list[Request]:
    """Every ``(dataset, k, seed, row, custom_id)`` a run will send.

    ``limit`` samples molecules per dataset *per seed*, so each seed is an
    independent pass over a different sample, as the earlier API eval did.
    """
    requests: list[Request] = []
    for seed in seeds:
        for dataset in datasets:
            test = load_split(dataset, split)
            rows = test.index
            if limit:
                rows = test.sample(n=min(limit, len(test)), random_state=seed).index
            requests += [
                (dataset, k, seed, int(row), f"{dataset}-k{k}-s{seed}-r{int(row)}")
                for k in shots
                for row in rows
            ]
    return requests


def resume_key(request: Request) -> tuple[str, int, int, int]:
    """What identifies a request in the checkpoint: everything but the id."""
    return request[0], request[1], request[2], request[3]


def done_keys(records: list[dict[str, Any]]) -> set[tuple[str, int, int, int]]:
    """Keys of records worth keeping, so a re-run regenerates only the rest.

    Keyed on the columns rather than ``custom_id`` so records written before
    seeds existed still match: they count as seed 0. API failures are excluded,
    which is what lets a re-run retry them.
    """
    return {record_key(rec) for rec in records if rec.get("status") != "api_error"}


def record_key(rec: dict[str, Any]) -> tuple[str, int, int, int]:
    """The :func:`resume_key` of a stored record; a missing seed counts as 0."""
    return rec["dataset"], int(rec["k"]), int(rec.get("seed") or 0), int(rec["row"])


def generate_one(
    backend: ModelBackend,
    request: Request,
    *,
    run_id: str,
    split: str = "test",
    thinking: bool = False,
    thinking_effort: str | None = None,
) -> dict[str, Any]:
    """Send one request and return its record.

    The status tells the three failure modes apart, because they need different
    handling: ``api_error`` is retried on the next run, ``truncated`` means the
    token cap ended the reasoning before any answer, and ``parse_error`` means
    the model answered but not in JSON.
    """
    dataset, k, seed, row, custom_id = request
    messages = build_messages(dataset, split, k, row, seed)
    # The system prompt is built as text parts for prompt caching on the
    # Anthropic API; a vLLM server takes string content only.
    messages = [{"role": "system", "content": system_text(messages)}, messages[1]]

    response = backend.generate(
        messages,
        metadata={
            "session_id": run_id,
            "trace_name": f"{dataset}/k{k}",
            "generation_name": "predict",
            "tags": [dataset, f"k={k}", f"seed={seed}", "local"],
            "trace_metadata": {"dataset": dataset, "k": k, "seed": seed, "row": row},
        },
    )

    if not response.ok:
        predictions, status, error = {}, "api_error", response.text
    else:
        predictions, error = parse_response(dataset, response.text)
        if error is None:
            status = "ok"
        elif response.reasoning and not response.text.strip():
            status = "truncated"
        else:
            status = "parse_error"

    usage = response.usage
    return {
        "run_id": run_id,
        "model": getattr(backend, "model", "unknown"),
        "provider": "local",
        "mode": "local",
        "thinking": thinking,
        "thinking_effort": thinking_effort if thinking else None,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "row": row,
        "custom_id": custom_id,
        "text": response.text,
        "reasoning": response.reasoning,
        "predictions": predictions,
        "status": status,
        "error": error,
        "stop_reason": None,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "reasoning_chars": usage.get("reasoning_chars", 0),
        "cost_usd": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
