"""Run ChemLLMBench against a backend.

One request per (task, k, strategy, test row). The prompts and references come
from the copied family modules, so a run here sees the same prompts the
reference ``main.py`` sent. The only difference is how the model is called:
each prompt goes to the vLLM server through the backend, and each answer is
checkpointed to ``results.jsonl`` as it arrives, so a run can be resumed.

Scoring re-parses the stored raw text with the copied parsers and calls the
copied metric functions. Scoring never reads the stored ``prediction``, so
after a parser fix the old generations can be scored again without generating
them again.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np
import pandas as pd

from chemberta4.evaluation.backends.base import ModelBackend
from chemberta4.evaluation.runner import load_jsonl
from chemberta4.tasks.chemllmbench.registry import TASKS, primary_metric

ZERO_SHOT_STRATEGY = "none"  # strategy recorded for zero-shot-only tasks


class Request(NamedTuple):
    """One work unit, carrying its own prompts so a worker needs nothing else."""

    task: str
    k: int
    strategy: str
    row: int
    system: str
    user: str
    ground_truth: Any

    @property
    def custom_id(self) -> str:
        return f"{self.task}-k{self.k}-{self.strategy}-r{self.row}"


def _plain(value: Any) -> Any:
    """A JSON-safe version of a value: numpy scalars become Python ones and NaN becomes None.

    >>> _plain(np.int64(3)), _plain(np.float64("nan")), _plain("x")
    (3, None, 'x')
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def build_requests(
    tasks: list[str],
    shots: list[int],
    strategy: str = "random",
    limit: int | None = None,
) -> list[Request]:
    """Every request a run will send.

    A task whose data cannot be loaded is skipped with a message. This happens,
    for example, when a download fails or MOL_DESIGN is asked for scaffold
    sampling, which the reference refuses. The other tasks still run, as they
    did under the reference ``main.py``.

    Zero-shot-only tasks (reaction selection) are built once with ``k=0``,
    whatever ``shots`` says. ``limit`` keeps the first N rows of each test set:
    the test sets are already fixed samples, so the first N rows are a subset
    of the rows the reference scored.
    """
    requests: list[Request] = []
    for key in tasks:
        task = TASKS[key]
        family = task.family()
        configs = (
            [(0, ZERO_SHOT_STRATEGY)] if task.zero_shot_only else [(k, strategy) for k in shots]
        )
        for k, strat in configs:
            try:
                system, user, y_true = family.load(task.subtask, strat, k)
            except Exception as exc:
                print(f"[skip] {key} k={k} {strat}: {type(exc).__name__}: {exc}")
                continue
            n = len(user) if limit is None else min(limit, len(user))
            requests += [
                Request(key, k, strat, row, system[row], user[row], _plain(y_true[row]))
                for row in range(n)
            ]
    return requests


def resume_key(request: Request) -> tuple[str, int, str, int]:
    """The (task, k, strategy, row) that identifies a request in the checkpoint."""
    return request.task, request.k, request.strategy, request.row


def record_key(rec: dict[str, Any]) -> tuple[str, int, str, int]:
    return rec["task"], int(rec["k"]), rec["strategy"], int(rec["row"])


def done_keys(records: Iterable[dict[str, Any]]) -> set[tuple[str, int, str, int]]:
    """Keys of the stored records, excluding API failures so a re-run retries them."""
    return {record_key(rec) for rec in records if rec.get("status") != "api_error"}


def _is_parse_failure(prediction: Any) -> bool:
    """The copied parsers return "" (string tasks) or -1 (Yes/No tasks) on failure."""
    return prediction == "" or prediction == -1 or prediction is None


def generate_one(
    backend: ModelBackend,
    request: Request,
    *,
    run_id: str,
    thinking: bool = False,
    thinking_effort: str | None = None,
) -> dict[str, Any]:
    """Send one request and return its record.

    The status separates three failures:

    - ``api_error``: not kept, so the next run retries it.
    - ``truncated``: the token cap ended the reasoning before any answer.
    - ``parse_error``: the model answered, but the copied parser found nothing
      in the answer.
    """
    task = TASKS[request.task]
    messages = [
        {"role": "system", "content": request.system},
        {"role": "user", "content": request.user},
    ]
    response = backend.generate(
        messages,
        metadata={
            "session_id": run_id,
            "trace_name": f"{request.task}/k{request.k}",
            "generation_name": "predict",
            "tags": [request.task, f"k={request.k}", request.strategy, "local"],
            "trace_metadata": {
                "task": request.task, "k": request.k,
                "strategy": request.strategy, "row": request.row,
            },
        },
    )

    prediction: Any = None
    if not response.ok:
        status, error = "api_error", response.text
    else:
        prediction = _plain(task.family().parse(response.text, task.subtask))
        error = None
        if not _is_parse_failure(prediction):
            status = "ok"
        elif response.reasoning and not response.text.strip():
            status = "truncated"
        else:
            status = "parse_error"

    usage = response.usage
    return {
        "run_id": run_id,
        "model": getattr(backend, "model", "unknown"),
        "thinking": thinking,
        "thinking_effort": thinking_effort if thinking else None,
        "task": request.task,
        "k": request.k,
        "strategy": request.strategy,
        "row": request.row,
        "custom_id": request.custom_id,
        "text": response.text,
        "reasoning": response.reasoning,
        "prediction": prediction,
        "ground_truth": request.ground_truth,
        "status": status,
        "error": error,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "reasoning_chars": usage.get("reasoning_chars", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_run(
    run_dir: str | Path,
    keys: set[tuple[str, int, str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score ``run_dir/results.jsonl`` with the copied parsers and metrics.

    Parameters
    ----------
    keys:
        Score only these (task, k, strategy, row) keys, normally the requests
        of the current run. The checkpoint can hold more, for example from an
        earlier run without ``--limit``. ``None`` scores everything in it.

    Returns
    -------
    (metrics, summary)
        ``metrics`` is a long frame with one row per (task, k, strategy,
        metric). ``summary`` is nested as ``{task: {"k<k>_<strategy>": metrics}}``.
        When all three selection tasks were scored at the same setting, the
        summary also gets a ``reaction_selection_macro_avg`` entry, as
        ``selection.py`` reports.

    Per-group prediction CSVs are written to ``run_dir/predictions/``.
    """
    run_dir = Path(run_dir)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in load_jsonl(run_dir / "results.jsonl"):
        if rec.get("status") == "api_error":
            continue
        if keys is not None and record_key(rec) not in keys:
            continue
        groups[(rec["task"], int(rec["k"]), rec["strategy"])].append(rec)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    pred_dir = run_dir / "predictions"
    for (key, k, strategy), recs in sorted(groups.items(), key=lambda kv: (list(TASKS).index(kv[0][0]), kv[0][1], kv[0][2])):
        # The checkpoint is written in completion order; score in row order.
        recs = sorted({int(r["row"]): r for r in recs}.values(), key=lambda r: int(r["row"]))
        task = TASKS[key]
        family = task.family()
        preds = [_plain(family.parse(r["text"], task.subtask)) for r in recs]
        y_true = [r["ground_truth"] for r in recs]
        try:
            metrics = {m: _plain(v) for m, v in family.score(y_true, preds, task.subtask).items()}
        except Exception as exc:  # e.g. the chemllmbench extra is missing for captions
            print(f"[!] could not score {key} k={k} {strategy}: {type(exc).__name__}: {exc}")
            continue
        n_parse_fail = sum(_is_parse_failure(p) for p in preds)
        metrics["parse_fail_rate"] = n_parse_fail / len(recs)
        head = primary_metric(metrics)
        summary.setdefault(key, {})[f"k{k}_{strategy}"] = {
            "display": task.display, "primary_metric": head, **metrics,
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)) or value is None:
                rows.append({
                    "task": key, "k": k, "strategy": strategy, "metric": name,
                    "value": value, "primary": name == head, "n": len(recs),
                })

        pred_dir.mkdir(exist_ok=True)
        pd.DataFrame({
            "row": [r["row"] for r in recs],
            "custom_id": [r.get("custom_id") for r in recs],
            "raw_output": [r["text"] for r in recs],
            "prediction": preds,
            "ground_truth": y_true,
            "status": [r["status"] for r in recs],
        }).to_csv(pred_dir / f"{key}_k{k}_{strategy}.csv", index=False)

    selection = ("reactant_selection", "ligand_selection", "solvent_selection")
    if all(t in summary for t in selection):
        setting = f"k0_{ZERO_SHOT_STRATEGY}"
        accs = [summary[t][setting]["accuracy"] for t in selection]
        summary["reaction_selection_macro_avg"] = {setting: {"accuracy": sum(accs) / len(accs)}}

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return pd.DataFrame(rows), summary


def report(metrics: pd.DataFrame) -> str:
    """A one-line-per-group summary, like the reference's BENCHMARK SUMMARY table."""
    if metrics.empty:
        return "nothing scored"
    head = metrics[metrics["primary"]]
    lines = [f"{'task':<22} {'k':>3} {'strategy':<9} {'metric':<14} {'value':>8}  n"]
    for r in head.itertuples():
        value = "" if r.value is None or pd.isna(r.value) else f"{r.value:.4f}"
        lines.append(f"{r.task:<22} {r.k:>3} {r.strategy:<9} {r.metric:<14} {value:>8}  {r.n}")
    return "\n".join(lines)
