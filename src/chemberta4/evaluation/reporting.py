"""Log a finished run to Weights & Biases.

Optional throughout. Without ``WANDB_API_KEY`` (or ``--wandb-api-key``) a run
simply does not log, and nothing here raises: a failed upload must never lose
scores that are already on disk.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def wandb_available(api_key: str | None = None) -> str | None:
    """The W&B API key to use, or ``None`` when logging is off."""
    return api_key or os.getenv("WANDB_API_KEY") or None


def login(api_key: str) -> None:
    """Log in before generating, so a bad key fails the run immediately.

    Checking at the end would mean losing the upload of a run that already
    spent GPU time.
    """
    import wandb

    wandb.login(key=api_key, relogin=True, verify=True)


def _prefix(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for col in columns:
        value = row[col]
        parts.append(f"k{value}" if col == "k" else f"s{value}" if col == "seed" else str(value))
    return "/".join(parts)


def summary_keys(metrics: pd.DataFrame) -> dict[str, Any]:
    """W&B summary entries for a long metrics frame.

    >>> df = pd.DataFrame({"task": ["bbbp"], "k": [0], "strategy": ["random"],
    ...                    "metric": ["accuracy"], "value": [0.5]})
    >>> summary_keys(df)
    {'bbbp/k0/random/accuracy': 0.5}
    """
    groups = [c for c in ("dataset", "task", "k", "strategy", "seed") if c in metrics.columns]
    out: dict[str, Any] = {}
    for row in metrics.to_dict("records"):
        prefix = _prefix(row, groups)
        out[f"{prefix}/{row['metric']}"] = row["value"]
        for extra in ("parse_fail_rate", "fill_rate"):
            if extra in row:
                out[f"{prefix}/{extra}"] = row[extra]
    if "seed" in groups:
        by = [c for c in groups if c != "seed"] + ["metric"]
        across = metrics.groupby(by)["value"].agg(["mean", "std"])
        for idx, stats in across.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            *keys, metric = idx
            prefix = _prefix(dict(zip(by[:-1], keys)), by[:-1])
            out[f"{prefix}/{metric}_mean"] = stats["mean"]
            out[f"{prefix}/{metric}_std"] = stats["std"]
    return out


def log_run(
    *,
    project: str,
    run_name: str,
    config: dict[str, Any],
    metrics: pd.DataFrame | None = None,
    perf: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
) -> None:
    """Log one run's scores and perf samples to W&B. Never raises.

    ``metrics`` is a long frame with ``metric`` and ``value`` columns: the
    per-(dataset, k, seed) frame from the MoleculeNet scorer, or the
    per-(task, k, strategy) frame from ChemLLMBench. Each row becomes a summary
    key prefixed by whichever of those grouping columns the frame has. When
    there is a ``seed`` column, the mean and standard deviation across seeds are
    added so a multi-seed run reads at a glance.
    """
    try:
        import wandb

        run = wandb.init(project=project, name=run_name, config=config)
        if perf is not None:
            summary, samples = perf
            for sample in samples:
                run.log(
                    {k: v for k, v in sample.items() if k != "t" and v is not None}
                    | {"elapsed_s": sample["t"]}
                )
            run.summary.update({k: v for k, v in summary.items() if v is not None})
        if metrics is not None and not metrics.empty:
            run.log({"metrics": wandb.Table(dataframe=metrics)})
            run.summary.update(summary_keys(metrics))
        run.finish()
    except Exception as exc:
        print(f"W&B logging failed, scores are still on disk: {exc}")
