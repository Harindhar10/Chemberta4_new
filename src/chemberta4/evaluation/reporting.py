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


def log_run(
    *,
    project: str,
    run_name: str,
    config: dict[str, Any],
    metrics: pd.DataFrame | None = None,
    perf: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
) -> None:
    """Log one run's scores and perf samples to W&B. Never raises.

    ``metrics`` is the per-(dataset, k, seed) frame from the MoleculeNet
    scorer. Each row becomes summary keys, and the mean and standard deviation
    across seeds are added so a multi-seed run reads at a glance.
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
            for row in metrics.itertuples():
                prefix = f"{row.dataset}/k{row.k}/s{row.seed}"
                run.summary.update(
                    {
                        f"{prefix}/{row.metric}": row.value,
                        f"{prefix}/parse_fail_rate": row.parse_fail_rate,
                        f"{prefix}/fill_rate": row.fill_rate,
                    }
                )
            across = metrics.groupby(["dataset", "k", "metric"])["value"].agg(["mean", "std"])
            for (dataset, k, metric), stats in across.iterrows():
                run.summary.update(
                    {
                        f"{dataset}/k{k}/{metric}_mean": stats["mean"],
                        f"{dataset}/k{k}/{metric}_std": stats["std"],
                    }
                )
        run.finish()
    except Exception as exc:
        print(f"W&B logging failed, scores are still on disk: {exc}")
