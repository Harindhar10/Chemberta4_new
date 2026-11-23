"""Wire a task, a backend and the run loop together.

``scripts/evaluate.py`` parses arguments; this does the work. Keeping the two
apart means the whole pipeline can be driven from a notebook or a test by
building a namespace, without going through ``sys.argv``.
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from chemberta4.evaluation import perf as perf_mod
from chemberta4.evaluation import batch, hub, pricing, reporting, runner
from chemberta4.evaluation.backends.base import ModelBackend
from chemberta4.evaluation.backends.closed import make_closed_backend, provider_of
from chemberta4.evaluation.backends.vllm_server import VLLMServerBackend, api_base


def is_closed(args: Namespace) -> bool:
    """Whether the model is Claude or GPT rather than a vLLM-served model."""
    return provider_of(args.model) is not None


def make_backend(args: Namespace, *, temperature: float = 0.0) -> ModelBackend:
    """A backend configured from the parsed arguments.

    ``temperature`` is the task's default for a vLLM model; ``--temperature``
    overrides it. A closed model gets ``--temperature`` only, and none when it
    is unset, since current Claude and GPT reasoning models reject one.
    """
    if is_closed(args):
        try:
            return make_closed_backend(
                args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                enable_thinking=args.thinking,
                thinking_effort=args.thinking_effort,
                batch=not args.realtime,
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
    return VLLMServerBackend(
        args.model,
        temperature=temperature if args.temperature is None else args.temperature,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=args.thinking,
        thinking_effort=args.thinking_effort,
    )


def temperature_used(args: Namespace, default: Any) -> Any:
    """The temperature a run sent, for ``run_config.json``."""
    if args.temperature is not None:
        return args.temperature
    return "provider default" if is_closed(args) else default


def announce(args: Namespace, run_dir: Path, n_total: int) -> None:
    """Print what this run is about to do, before it spends any GPU time."""
    print(f"task        : {args.task}")
    print(f"model       : {args.model}")
    if is_closed(args):
        print(f"provider    : {provider_of(args.model)} "
              f"({'real time' if args.realtime else 'Batch API'})")
    else:
        print(f"server      : {api_base()}")
    print(f"thinking    : {args.thinking}" + (f" ({args.thinking_effort})" if args.thinking else ""))
    print(f"max tokens  : {args.max_output_tokens}")
    print(f"workers     : {args.workers}")
    print(f"items       : {n_total}")
    print(f"writing to  : {run_dir}")


def git_commit() -> str | None:
    """The repo's HEAD commit, or ``None`` outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_config(args: Namespace, run_dir: Path, **task_fields: Any) -> dict[str, Any]:
    """The run's hyperparameters, also written to ``run_config.json``.

    Shared by W&B and the Hub card. The server URL is left out: it names
    private infrastructure and does not change the results.
    """
    config = {
        "task": args.task,
        "model": args.model,
        **task_fields,
        "thinking": args.thinking,
        "thinking_effort": args.thinking_effort if args.thinking else None,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "limit": args.limit,
        "tag": args.tag,
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


#: Molecules per retrosynthesis group file, by ``mol_no``.
RETRO_GROUP_SIZE = 100


def group_files(
    run_dir: Path,
    items: list[Any],
    *,
    group_of: Any,
    item_key: Any,
    key_of: Any,
    checkpoint: Path | None = None,
) -> runner.GroupWriter:
    """The writer of this run's ``tasks/<group>.jsonl`` files.

    Every item of the run, done or not, defines what a group must contain.
    Groups the checkpoint already completes are written straight away, so a
    restarted run (or one that predates group files) has them too.
    """
    writer = runner.GroupWriter.from_items(
        checkpoint or run_dir / "results.jsonl",
        run_dir / "tasks",
        items,
        group_of=group_of,
        item_key=item_key,
        key_of=key_of,
    )
    written = writer.write_complete()
    print(f"group files : {len(written)}/{len(writer.expected)} complete in {run_dir / 'tasks'}")
    return writer


def hub_setup(args: Namespace) -> str | None:
    """The HF token when ``--hf-repo`` is set, checked before any generation."""
    if not getattr(args, "hf_repo", None):
        return None
    token = hub.hub_token(getattr(args, "hf_token", None))
    if not token:
        print("no HF token (--hf-token or $HF_TOKEN); this run will not be pushed")
        return None
    hub.check_access(args.hf_repo, token)  # fail on a bad token before spending GPU time
    return token


def confirm_cost(args: Namespace, backend: Any, to_send: list[Any], messages_of: Any) -> None:
    """Print a cost upper bound for ``to_send`` and ask before spending it.

    ``--max-cost`` refuses a run whose estimate is above it; ``--yes`` skips
    the question. With no price known for the model, ``--max-cost`` refuses
    rather than guess.
    """
    if not to_send:
        return
    chars = sum(len(json.dumps(messages_of(item))) for item in to_send)
    est = pricing.estimate(
        backend.prices, chars, len(to_send), args.max_output_tokens, batch=backend.batch
    )
    if est is None:
        print(f"cost        : unknown ({args.model} is not in litellm's price list)")
        if args.max_cost is not None:
            raise SystemExit("--max-cost is set but no price is known for this model")
    else:
        print(f"cost        : at most ~${sum(est):.2f} for {len(to_send)} requests "
              f"(input ~${est[0]:.2f} before caching, output ≤${est[1]:.2f} "
              f"if every answer hits {args.max_output_tokens} tokens)")
        if args.max_cost is not None and sum(est) > args.max_cost:
            raise SystemExit(f"estimate is above --max-cost {args.max_cost}; not sending")
    if args.yes:
        return
    try:
        answer = input("send? [y/N] ")
    except EOFError:  # no terminal, e.g. under nohup: pass --yes there
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise SystemExit("not sent (pass --yes to skip this question)")


def generate_items(
    args: Namespace,
    backend: Any,
    run_dir: Path,
    *,
    items: list[Any],
    messages_of: Any,
    metadata_of: Any,
    finish: Any,
    custom_id: Any,
    out_path: Path,
    keep: Any,
    status_of: Any,
    group_of: Any,
    on_group_done: Any,
) -> Any:
    """Generate ``items``: concurrently in real time, or through the Batch API.

    ``finish(item, messages, response)`` builds the record in every mode, so a
    batch run stores exactly what a real-time run would.
    """
    def worker(item: Any) -> dict[str, Any]:
        messages = messages_of(item)
        return finish(item, messages, backend.generate(messages, metadata=metadata_of(item)))

    common = dict(
        out_path=out_path, keep=keep, status_of=status_of,
        group_of=group_of, on_group_done=on_group_done,
    )
    if not is_closed(args) or not backend.batch:
        if is_closed(args):
            confirm_cost(args, backend, items, messages_of)
        return runner.generate(items=items, worker=worker, workers=args.workers, **common)

    manifest = run_dir / "batches.json"
    in_flight = {
        cid for b in batch.load_manifest(manifest) if not b["collected"] for cid in b["custom_ids"]
    }
    to_send = [i for i in items if batch.batch_id_safe(custom_id(i)) not in in_flight]
    confirm_cost(args, backend, to_send, messages_of)
    return batch.run_batch(
        items=items,
        backend=backend,
        custom_id=custom_id,
        messages_of=messages_of,
        finish=finish,
        manifest_path=manifest,
        wait=not args.no_wait,
        **common,
    )


def print_cost_summary(checkpoint: Path) -> None:
    """Total cost and prompt-cache hit rate over a closed-model run's records."""
    records = [r for r in runner.load_jsonl(checkpoint) if "cache_read_tokens" in r]
    if not records:
        return
    read = sum(r.get("cache_read_tokens") or 0 for r in records)
    prompt = read + sum(
        (r.get("input_tokens") or 0) + (r.get("cache_write_tokens") or 0) for r in records
    )
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    print(f"\ncost        : ${sum(costs):.2f} over {len(costs)}/{len(records)} priced records")
    if prompt:
        print(f"cache hits  : {read / prompt:.0%} of prompt tokens read from cache")


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
    hf_token = hub_setup(args)

    requests = task.build_requests(datasets, shots, seeds, args.split, args.limit)
    done = task.done_keys(runner.load_jsonl(run_dir / "results.jsonl"))
    todo = runner.pending(requests, done, task.resume_key)
    announce(args, run_dir, len(requests))
    print(f"{len(done)} already done, generating {len(todo)}")

    def group_of(request: Any) -> str:
        dataset, k, seed = request[0], request[1], request[2]
        return f"{dataset}_k{k}_seed{seed}"

    groups = group_files(
        run_dir, requests, group_of=group_of, item_key=task.resume_key, key_of=task.record_key
    )

    backend = make_backend(args, temperature=0.0)
    perf_data = None
    with perf_mod.perf_window(args.perf) as perf:
        counts = generate_items(
            args,
            backend,
            run_dir,
            items=todo,
            group_of=group_of,
            on_group_done=groups,
            messages_of=lambda request: task.request_messages(request, args.split),
            metadata_of=lambda request: task.request_metadata(request, run_id),
            finish=lambda request, messages, response: task.make_record(
                backend,
                request,
                messages,
                response,
                run_id=run_id,
                thinking=args.thinking,
                thinking_effort=args.thinking_effort,
            ),
            custom_id=lambda request: request[4],
            out_path=run_dir / "results.jsonl",
            # An API failure is not stored, so the next run retries it. A parse
            # failure is stored: the model answered, and that is a result.
            keep=lambda rec: rec["status"] != "api_error",
            status_of=lambda rec: rec["status"],
        )
        if perf is not None and todo:
            completed = sum(v for k, v in counts.items() if k != "dropped")
            perf_data = (perf_mod.write_perf(perf, run_dir, completed), perf.samples)
    if counts.get("pending"):
        return  # --no-wait: scored once the batches are collected
    if is_closed(args):
        print_cost_summary(run_dir / "results.jsonl")

    # Records written before --seeds existed have no seed and count as seed 0.
    metrics = evaluate_run(run_dir, args.split, default_seed=0)
    if not metrics.empty:
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        print(f"\nwrote {run_dir / 'metrics.csv'}")

    config = run_config(
        args,
        run_dir,
        temperature=temperature_used(args, 0.0),
        datasets=datasets,
        shots=shots,
        seeds=seeds,
        split=args.split,
    )
    if args.wandb_project and api_key:
        reporting.log_run(
            project=args.wandb_project,
            run_name=model_slug,
            config=config,
            metrics=metrics,
            perf=perf_data,
        )
    if hf_token:
        hub.push_run(
            repo_id=args.hf_repo,
            task="moleculenet",
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            metrics_md=hub.moleculenet_metrics_md(metrics),
            token=hf_token,
        )


# --------------------------------------------------------------------------
# ChemLLMBench
# --------------------------------------------------------------------------
def run_chemllmbench(args: Namespace) -> None:
    """Generate and score the ChemLLMBench tasks."""
    from chemberta4.tasks.chemllmbench import run as task
    from chemberta4.tasks.chemllmbench.registry import resolve

    try:
        tasks = resolve(args.subtasks)
    except ValueError as exc:
        raise SystemExit(str(exc))
    shots = [int(s) for s in str(args.shots if args.shots is not None else "0").split(",")]

    # Same naming as MoleculeNet: k and strategy live in each record, so one run
    # directory holds every shot count and strategy for a model and effort level.
    model_slug = args.model.rstrip("/").split("/")[-1]
    run_id = f"{model_slug}{f'_think_{args.thinking_effort}' if args.thinking else ''}"
    run_dir = Path(args.out_root) / "chemllmbench" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = reporting.wandb_available(args.wandb_api_key) if args.wandb_project else None
    if args.wandb_project and not api_key:
        print("no W&B key (--wandb-api-key or $WANDB_API_KEY); this run will not be logged")
    if api_key:
        reporting.login(api_key)  # fail on a bad key before spending GPU time
    hf_token = hub_setup(args)

    requests = task.build_requests(tasks, shots, args.strategy, args.limit)
    done = task.done_keys(runner.load_jsonl(run_dir / "results.jsonl"))
    todo = runner.pending(requests, done, task.resume_key)
    announce(args, run_dir, len(requests))
    print(f"subtasks    : {', '.join(tasks)}")
    print(f"shots       : {shots}   strategy: {args.strategy}")
    print(f"{len(requests) - len(todo)} already done, generating {len(todo)}")

    def group_of(request: Any) -> str:
        return f"{request.task}_k{request.k}_{request.strategy}"

    groups = group_files(
        run_dir, requests, group_of=group_of, item_key=task.resume_key, key_of=task.record_key
    )

    backend = make_backend(args, temperature=0.0)
    perf_data = None
    with perf_mod.perf_window(args.perf) as perf:
        counts = generate_items(
            args,
            backend,
            run_dir,
            items=todo,
            group_of=group_of,
            on_group_done=groups,
            messages_of=task.request_messages,
            metadata_of=lambda request: task.request_metadata(request, run_id),
            finish=lambda request, messages, response: task.make_record(
                backend,
                request,
                messages,
                response,
                run_id=run_id,
                thinking=args.thinking,
                thinking_effort=args.thinking_effort,
            ),
            custom_id=lambda request: request.custom_id,
            out_path=run_dir / "results.jsonl",
            keep=lambda rec: rec["status"] != "api_error",
            status_of=lambda rec: rec["status"],
        )
        if perf is not None and todo:
            completed = sum(v for k, v in counts.items() if k != "dropped")
            perf_data = (perf_mod.write_perf(perf, run_dir, completed), perf.samples)
    if counts.get("pending"):
        return  # --no-wait: scored once the batches are collected
    if is_closed(args):
        print_cost_summary(run_dir / "results.jsonl")

    metrics, _ = task.evaluate_run(run_dir, keys={task.resume_key(r) for r in requests})
    print()
    print(task.report(metrics))
    if not metrics.empty:
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        print(f"\nwrote {run_dir / 'metrics.csv'}, summary.json and predictions/")

    config = run_config(
        args,
        run_dir,
        temperature=temperature_used(args, 0.0),
        subtasks=tasks,
        shots=shots,
        strategy=args.strategy,
    )
    if args.wandb_project and api_key:
        reporting.log_run(
            project=args.wandb_project,
            run_name=f"chemllmbench_{model_slug}",
            config=config,
            metrics=metrics,
            perf=perf_data,
        )
    if hf_token:
        hub.push_run(
            repo_id=args.hf_repo,
            task="chemllmbench",
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            metrics_md=hub.chemllmbench_metrics_md(metrics),
            token=hf_token,
        )


# --------------------------------------------------------------------------
# Retrosynthesis
# --------------------------------------------------------------------------
def az_cache_path(args: Namespace, data_slug: str) -> Path:
    """Shared AiZynthFinder checkpoint for this dataset and AZ model.

    AZ's answer does not depend on the LLM, so every autosolve run over the same
    data and AZ model reads and extends one file instead of searching again.
    """
    if args.az_cache:
        path = Path(args.az_cache)
    else:
        path = Path(args.out_root) / "az_cache" / f"{data_slug}_{args.az_model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_az_phase(rows: list[tuple[int, Any]], jsonl: Path, az_model: str, workers: int) -> None:
    """Search every molecule that is not yet in the shared AZ checkpoint.

    A molecule whose search raised is *not* checkpointed, so the next run retries
    it. Writing it would freeze a transient failure, such as a missing model file,
    into a cache that every later run reads as "AZ found nothing".
    """
    from chemberta4.deepretro.utils_molecule import canonicalize
    from chemberta4.tasks.retrosynthesis import az
    from chemberta4.tasks.retrosynthesis.autosolve import run_az_record

    done = {rec["mol_no"] for rec in runner.load_jsonl(jsonl)}
    todo = [(i, row) for i, row in rows if i not in done]
    print(f"[az] {len(done)} cached, searching {len(todo)}")
    if not todo:
        return
    # Fail before spawning workers: a missing config would otherwise surface as
    # one error record per molecule.
    print(f"AZ config: {az.resolve_config(az_model)}")

    started = time.time()
    errors: list[str] = []
    # spawn, not fork: by now litellm and langfuse have background threads, and
    # forking a threaded process can deadlock the child.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool, open(
        jsonl, "a"
    ) as fout:
        futures = [
            pool.submit(run_az_record, i, canonicalize(row["input"]), az_model)
            for i, row in todo
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            if rec["error"]:
                errors.append(rec["error"])
            else:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if n % 10 == 0:
                print(f"  [az] {n}/{len(todo)} ({(time.time() - started) / 60:.1f} min)")
    print(f"[az] {len(todo)} molecules in {(time.time() - started) / 60:.2f} min")
    if errors:
        print(
            f"[az] {len(errors)} molecules failed and were not checkpointed "
            f"(retried next run); first error: {errors[0]}"
        )


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
        data_slug,
        args.model,
        mode=args.mode,
        prompt=args.prompt,
        thinking=args.thinking,
        thinking_effort=args.thinking_effort,
        az_model=args.az_model,
        stability_check=args.stability_check,
        hallucination_check=args.hallucination_check,
        top_k=args.top_k,
    )
    out_root = Path(args.out_root) / "retrosynthesis"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = run_dir / "generations.jsonl"
    hf_token = hub_setup(args)

    rows = [(int(i), row) for i, row in df.iterrows()]
    done = {rec["mol_no"] for rec in runner.load_jsonl(jsonl)}
    todo = [(i, row) for i, row in rows if i not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    announce(args, run_dir, len(rows))
    print(f"mode        : {args.mode}   prompt: {args.prompt}")
    if args.mode == "autosolve":
        print(f"az model    : {args.az_model}   workers: {args.az_workers}   top-k: {args.top_k}")
        print(f"checks      : stability={args.stability_check} hallucination={args.hallucination_check}")
    print(f"{len(done)} already done, generating {len(todo)}")

    def group_of(item: tuple[int, Any]) -> str:
        start = item[0] // RETRO_GROUP_SIZE * RETRO_GROUP_SIZE
        return f"mols_{start:05d}-{start + RETRO_GROUP_SIZE - 1:05d}"

    # This run's molecules: what is already stored plus what it will generate.
    # With --limit, the last group can be smaller than RETRO_GROUP_SIZE.
    in_run = [(i, row) for i, row in rows if i in done] + todo
    groups = group_files(
        run_dir,
        in_run,
        checkpoint=jsonl,
        group_of=group_of,
        item_key=lambda item: item[0],
        key_of=lambda rec: rec["mol_no"],
    )

    status_of = lambda rec: "ok" if rec.get("proposals") or rec.get("raw") else "empty"  # noqa: E731
    if is_closed(args) and args.mode == "raw":
        # Claude/GPT raw runs go through the closed backends, batched by default.
        backend = make_backend(args)
        counts = generate_items(
            args,
            backend,
            run_dir,
            items=todo,
            group_of=group_of,
            on_group_done=groups,
            messages_of=lambda item: task.raw_messages(item[1], model=args.model, prompt=args.prompt),
            metadata_of=lambda item: {},
            finish=lambda item, messages, response: task.raw_record(
                item[1],
                item[0],
                messages,
                response,
                model=args.model,
                prompt=args.prompt,
                thinking=args.thinking,
                thinking_effort=args.thinking_effort,
                stability_check=args.stability_check,
                hallucination_check=args.hallucination_check,
            ),
            custom_id=lambda item: f"mol{item[0]}",
            out_path=jsonl,
            keep=task.keep_record,
            status_of=status_of,
        )
        if counts.get("pending"):
            return  # --no-wait: scored once the batches are collected
        print_cost_summary(jsonl)
    else:
        if is_closed(args):
            print("note        : pipeline/autosolve call DeepRetro's own litellm path in "
                  "real time (follow-up calls cannot be batched)")
        with perf_mod.perf_window(args.perf) as perf:
            counts = runner.generate(
                items=todo,
                group_of=group_of,
                on_group_done=groups,
                worker=lambda item: task.generate_one(
                    item[1],
                    item[0],
                    model=args.model,
                    mode=args.mode,
                    prompt=args.prompt,
                    thinking=args.thinking,
                    thinking_effort=args.thinking_effort,
                    max_output_tokens=args.max_output_tokens,
                    stability_check=args.stability_check,
                    hallucination_check=args.hallucination_check,
                    top_k=args.top_k,
                ),
                out_path=jsonl,
                workers=args.workers,
                keep=task.keep_record,
                status_of=status_of,
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

    meta: dict[str, Any] = {
        "run": run_id,
        "model_id": args.model,
        "mode": args.mode,
        "prompt": args.prompt,
        "thinking": args.thinking,
        "thinking_effort": args.thinking_effort if args.thinking else None,
        "max_new_tokens": args.max_output_tokens,
        "data_csv": str(retro_data.path(data_slug) if not str(dataset).endswith(".csv") else dataset),
        "n_total": int(len(df)),
    }
    if args.mode == "autosolve":
        from chemberta4.tasks.retrosynthesis.autosolve import score_views

        az_jsonl = az_cache_path(args, data_slug)
        print(f"AZ cache: {az_jsonl}")
        az_rows = rows[: args.limit] if args.limit is not None else rows
        run_az_phase(az_rows, az_jsonl, args.az_model, args.az_workers)
        # The shared checkpoint can cover more molecules than this run.
        in_run = {i for i, _ in az_rows}
        az_records = {
            rec["mol_no"]: rec
            for rec in runner.load_jsonl(az_jsonl)
            if rec["mol_no"] in in_run
        }
        full = score_views(
            df,
            records,
            az_records,
            model=args.model,
            az_model=args.az_model,
            run_dir=str(run_dir),
            summary_csv=str(out_root / "summary_all.csv"),
            meta={**meta, "az_model": args.az_model},
        )
        if not full:
            return
        # The Hub card and the config below describe the hybrid view, which is
        # what AutoSolver would hand to its search.
        summary = full["views"]["hybrid"]
    else:
        scored = (
            pd.DataFrame([score_record(rec) for rec in records.values()])
            .set_index("mol_no")
            .sort_index()
        )
        scored.to_csv(run_dir / "results.csv")
        summary, report = summarize(scored, meta)
        print(report)
        write_summary(summary, report, str(run_dir), str(out_root / "summary_all.csv"))
        print(f"\nwrote {run_dir}/summary.json, summary.txt and results.csv")

    config = run_config(
        args,
        run_dir,
        # Raw mode samples at DeepRetro's 1.0 (a closed model: what was sent,
        # where the provider default is also 1.0); the pipeline raises the
        # temperature on each retry.
        temperature=(
            (temperature_used(args, 1.0) if is_closed(args) else 1.0)
            if args.mode == "raw"
            else "pipeline schedule"
        ),
        mode=args.mode,
        prompt=args.prompt,
        data=data_slug,
        stability_check=args.stability_check,
        hallucination_check=args.hallucination_check,
        **(
            {"az_model": args.az_model, "top_k": args.top_k}
            if args.mode == "autosolve"
            else {}
        ),
    )
    if hf_token:
        hub.push_run(
            repo_id=args.hf_repo,
            task="retrosynthesis",
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            metrics_md=hub.retrosynthesis_metrics_md(summary),
            token=hf_token,
        )
