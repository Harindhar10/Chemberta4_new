"""Wire a task, a backend and the run loop together.

``scripts/evaluate.py`` parses arguments; this does the work. Keeping the two
apart means the whole pipeline can be driven from a notebook or a test by
building a namespace, without going through ``sys.argv``.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from chemberta4.evaluation import perf as perf_mod
from chemberta4.evaluation import reporting, runner
from chemberta4.evaluation.backends.vllm_server import VLLMServerBackend, api_base


def make_backend(args: Namespace, *, temperature: float = 0.0) -> VLLMServerBackend:
    """A backend configured from the parsed arguments."""
    return VLLMServerBackend(
        args.model,
        temperature=temperature,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=args.thinking,
        thinking_effort=args.thinking_effort,
    )


def announce(args: Namespace, run_dir: Path, n_total: int) -> None:
    """Print what this run is about to do, before it spends any GPU time."""
    print(f"task        : {args.task}")
    print(f"model       : {args.model}")
    print(f"server      : {api_base()}")
    print(f"thinking    : {args.thinking}" + (f" ({args.thinking_effort})" if args.thinking else ""))
    print(f"max tokens  : {args.max_output_tokens}")
    print(f"workers     : {args.workers}")
    print(f"items       : {n_total}")
    print(f"writing to  : {run_dir}")


# --------------------------------------------------------------------------
# MoleculeNet
# --------------------------------------------------------------------------
def run_moleculenet(args: Namespace) -> None:
    """Generate and score MoleculeNet property predictions."""
    from chemberta4.tasks.moleculenet import run as task
    from chemberta4.tasks.moleculenet.config import DATASETS, DEFAULT_SHOTS
    from chemberta4.tasks.moleculenet.metrics import evaluate_run

    datasets = (args.datasets or ",".join(DATASETS)).split(",")
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}. Choose from {list(DATASETS)}")
    shots = [int(s) for s in (args.shots or ",".join(map(str, DEFAULT_SHOTS))).split(",")]
    seeds = [int(s) for s in str(args.seeds).split(",")]

    # The effort level is in the run name so each level keeps its own
    # checkpoint; sharing one would make the resume skip requests answered at a
    # different level.
    model_slug = args.model.rstrip("/").split("/")[-1]
    run_id = f"{model_slug}{f'_think_{args.thinking_effort}' if args.thinking else ''}"
    run_dir = Path(args.out_root) / "moleculenet" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = reporting.wandb_available(args.wandb_api_key) if args.wandb_project else None
    if args.wandb_project and not api_key:
        print("no W&B key (--wandb-api-key or $WANDB_API_KEY); this run will not be logged")
    if api_key:
        reporting.login(api_key)  # fail on a bad key before spending GPU time

    requests = task.build_requests(datasets, shots, seeds, args.split, args.limit)
    done = task.done_keys(runner.load_jsonl(run_dir / "results.jsonl"))
    todo = runner.pending(requests, done, task.resume_key)
    announce(args, run_dir, len(requests))
    print(f"{len(done)} already done, generating {len(todo)}")

    backend = make_backend(args, temperature=0.0)
    perf_data = None
    with perf_mod.perf_window(args.perf) as perf:
        counts = runner.generate(
            items=todo,
            worker=lambda request: task.generate_one(
                backend,
                request,
                run_id=run_id,
                split=args.split,
                thinking=args.thinking,
                thinking_effort=args.thinking_effort,
            ),
            out_path=run_dir / "results.jsonl",
            workers=args.workers,
            # An API failure is not stored, so the next run retries it. A parse
            # failure is stored: the model answered, and that is a result.
            keep=lambda rec: rec["status"] != "api_error",
            status_of=lambda rec: rec["status"],
        )
        if perf is not None and todo:
            completed = sum(v for k, v in counts.items() if k != "dropped")
            perf_data = (perf_mod.write_perf(perf, run_dir, completed), perf.samples)

    # Records written before --seeds existed have no seed and count as seed 0.
    metrics = evaluate_run(run_dir, args.split, default_seed=0)
    if not metrics.empty:
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        print(f"\nwrote {run_dir / 'metrics.csv'}")

    if args.wandb_project and api_key:
        reporting.log_run(
            project=args.wandb_project,
            run_name=model_slug,
            config={
                "task": "moleculenet",
                "model": args.model,
                "thinking": args.thinking,
                "thinking_effort": args.thinking_effort if args.thinking else None,
                "max_output_tokens": args.max_output_tokens,
                "workers": args.workers,
                "datasets": datasets,
                "shots": shots,
                "seeds": seeds,
                "limit": args.limit,
                "tag": args.tag,
            },
            metrics=metrics,
            perf=perf_data,
        )


# --------------------------------------------------------------------------
# Retrosynthesis
# --------------------------------------------------------------------------
def run_retrosynthesis(args: Namespace) -> None:
    """Generate and score single-step retrosynthesis proposals."""
    from chemberta4.tasks.retrosynthesis import data as retro_data
    from chemberta4.tasks.retrosynthesis import run as task
    from chemberta4.deepretro.one_step_eval import (
        score_record,
        summarize,
        write_summary,
    )

    dataset = args.data or retro_data.DEFAULT_DATASET
    df = retro_data.load(dataset)
    data_slug = Path(dataset).stem
    run_id = task.run_name(
        data_slug, args.model, mode=args.mode, prompt=args.prompt, thinking=args.thinking
    )
    out_root = Path(args.out_root) / "retrosynthesis"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = run_dir / "generations.jsonl"

    rows = [(int(i), row) for i, row in df.iterrows()]
    done = {rec["mol_no"] for rec in runner.load_jsonl(jsonl)}
    todo = [(i, row) for i, row in rows if i not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    announce(args, run_dir, len(rows))
    print(f"mode        : {args.mode}   prompt: {args.prompt}")
    print(f"{len(done)} already done, generating {len(todo)}")

    with perf_mod.perf_window(args.perf) as perf:
        counts = runner.generate(
            items=todo,
            worker=lambda item: task.generate_one(
                item[1],
                item[0],
                model=args.model,
                mode=args.mode,
                prompt=args.prompt,
                thinking=args.thinking,
                max_output_tokens=args.max_output_tokens,
                stability_check=args.stability_check,
                hallucination_check=args.hallucination_check,
            ),
            out_path=jsonl,
            workers=args.workers,
            keep=task.keep_record,
            status_of=lambda rec: "ok" if rec.get("proposals") or rec.get("raw") else "empty",
        )
        if perf is not None and todo:
            completed = sum(v for k, v in counts.items() if k != "dropped")
            perf_mod.write_perf(perf, run_dir, completed)

    records = {rec["mol_no"]: rec for rec in runner.load_jsonl(jsonl)}
    if not records:
        print("nothing to score")
        return
    if args.limit is not None:
        in_run = {i for i, _ in rows[: args.limit]}
        records = {i: r for i, r in records.items() if i in in_run}

    scored = (
        pd.DataFrame([score_record(rec) for rec in records.values()])
        .set_index("mol_no")
        .sort_index()
    )
    scored.to_csv(run_dir / "results.csv")
    meta: dict[str, Any] = {
        "run": run_id,
        "model_id": args.model,
        "mode": args.mode,
        "prompt": args.prompt,
        "thinking": args.thinking,
        "max_new_tokens": args.max_output_tokens,
        "data_csv": str(retro_data.path(data_slug) if not str(dataset).endswith(".csv") else dataset),
        "n_total": int(len(df)),
    }
    summary, report = summarize(scored, meta)
    print(report)
    write_summary(summary, report, str(run_dir), str(out_root / "summary_all.csv"))
    print(f"\nwrote {run_dir}/summary.json, summary.txt and results.csv")
