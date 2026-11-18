#!/usr/bin/env python
"""Evaluate a vLLM-served model, Claude or GPT on MoleculeNet, single-step retrosynthesis or ChemLLMBench.

Start the server first, in its own environment::

    vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 17408 \
        --enable-prefix-caching [--reasoning-parser qwen3]

then point this at it::

    export HOSTED_VLLM_API_BASE=http://localhost:8000/v1

    python scripts/evaluate.py --task moleculenet \
        --model hosted_vllm/Qwen/Qwen3-8B --shots 0,5 --seeds 0 --limit 50

    python scripts/evaluate.py --task retrosynthesis \
        --model hosted_vllm/Qwen/Qwen3-8B --mode raw --limit 10

    python scripts/evaluate.py --task chemllmbench \
        --model hosted_vllm/Qwen/Qwen3-8B --subtasks property_prediction --shots 0,10

Claude and GPT need no server; the model id names the provider and the key
comes from $ANTHROPIC_API_KEY / $OPENAI_API_KEY. Requests go through the Batch
API (half price) unless --realtime is given::

    python scripts/evaluate.py --task moleculenet \
        --model anthropic/claude-sonnet-5-5 --shots 0,5 --seeds 0 --limit 50

A cost estimate is shown and confirmed before anything is sent (--yes skips
it). --no-wait submits and exits; re-running the same command collects.

Every task is resumable: finished items are read back from the run's JSONL
checkpoint and skipped, so an interrupted run is restarted with the same
command. Items that failed at the API are not checkpointed and are retried.

``--config`` reads defaults from a YAML file; anything given on the command
line wins. ``--perf`` records what the server was doing during generation.
``--hf-repo`` publishes the run to a public dataset repo on the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Python puts this script's directory first on sys.path, where this file would
# shadow the Hugging Face `evaluate` package that ChemLLMBench's caption and
# design metrics import. Nothing here imports a sibling script, so drop it.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_config(path: str | None) -> dict:
    """Read a YAML config, or return an empty dict when none was given."""
    if not path:
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--task", required=True, choices=["moleculenet", "retrosynthesis", "chemllmbench"])
    p.add_argument(
        "--model",
        help="hosted_vllm/<hf id>, anthropic/<claude id> or openai/<gpt id>",
    )
    p.add_argument("--config", help="YAML defaults; command-line flags win")
    p.add_argument("--out-root", default=None, help="default: outputs/ in the repo")

    gen = p.add_argument_group("generation")
    gen.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    gen.add_argument(
        "--thinking-effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="reasoning_effort sent with --thinking; high and max both map to "
        "Qwen3.8's xhigh, its most verbose level",
    )
    gen.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="default: 1024, or 16384 with --thinking",
    )
    gen.add_argument(
        "--workers", type=int, default=None, help="concurrent requests; vLLM batches them"
    )
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="default: 0 for vLLM on MoleculeNet/ChemLLMBench; for Claude/GPT none is "
        "sent unless set (current Claude and GPT reasoning models reject one)",
    )

    closed = p.add_argument_group("Claude / GPT (anthropic/... and openai/... models)")
    closed.add_argument(
        "--realtime",
        action="store_true",
        default=None,
        help="call the API directly instead of the Batch API (full price, immediate)",
    )
    closed.add_argument(
        "--no-wait",
        action="store_true",
        default=None,
        help="submit batches and exit; re-run the same command to collect and score",
    )
    closed.add_argument(
        "--yes", "-y", action="store_true", default=None, help="skip the cost confirmation"
    )
    closed.add_argument(
        "--max-cost", type=float, default=None, help="refuse to send above this estimated $"
    )

    mol = p.add_argument_group("moleculenet")
    mol.add_argument("--datasets", default=None, help="comma-separated; default is all")
    mol.add_argument(
        "--shots",
        default=None,
        help="comma-separated k values; 0 = zero-shot (also used by chemllmbench, default 0)",
    )
    mol.add_argument("--seeds", default=None, help="comma-separated; each seed is a full pass")
    mol.add_argument("--split", default=None)

    retro = p.add_argument_group("retrosynthesis")
    retro.add_argument("--mode", choices=["raw", "pipeline", "autosolve"], default=None)
    retro.add_argument("--prompt", choices=["deepretro", "ether0"], default=None)
    retro.add_argument("--data", default=None, help="dataset name or path to a CSV")
    retro.add_argument(
        "--stability-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: on in autosolve mode (DeepRetro's default), off otherwise",
    )
    retro.add_argument(
        "--hallucination-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: on in autosolve mode (DeepRetro's default), off otherwise",
    )
    retro.add_argument(
        "--az-model",
        default=None,
        help="autosolve mode: AiZynthFinder model folder under $AZ_MODELS_PATH "
        "(default: Pistachio_100+, DeepRetro's default)",
    )
    retro.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="autosolve mode: LLM pathways kept per molecule (default: 3, as DeepRetro's batch runner)",
    )
    retro.add_argument(
        "--az-workers",
        type=int,
        default=None,
        help="autosolve mode: parallel AZ processes, each loads its own model (default: 4)",
    )
    retro.add_argument(
        "--az-cache",
        default=None,
        help="autosolve mode: AZ checkpoint shared across LLM runs "
        "(default: <out-root>/az_cache/<data>_<az-model>.jsonl)",
    )

    bench = p.add_argument_group("chemllmbench")
    bench.add_argument(
        "--subtasks",
        default=None,
        help="comma-separated task keys or groups, e.g. bbbp,name_prediction (default: all)",
    )
    bench.add_argument(
        "--strategy",
        choices=["random", "scaffold"],
        default=None,
        help="few-shot example selection (default: random); selection tasks are zero-shot only",
    )

    obs = p.add_argument_group("observability")
    obs.add_argument(
        "--perf",
        action="store_true",
        default=None,
        help="record server throughput and latency from /metrics into perf.json",
    )
    obs.add_argument("--wandb-project", default=None, help="log the run to this W&B project")
    obs.add_argument("--wandb-api-key", default=None, help="default: $WANDB_API_KEY")
    obs.add_argument("--tag", default=None, help="free-form label for the serving setup")
    obs.add_argument(
        "--hf-repo",
        default=None,
        help="push generations, metrics and hyperparameters to this public "
        "HF dataset repo, e.g. you/chemberta4-evals",
    )
    obs.add_argument("--hf-token", default=None, help="write token; default: $HF_TOKEN")
    return p


def resolve(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """Fill unset flags from the config file, then from the built-in defaults."""
    defaults = {
        "thinking": False,
        "thinking_effort": "medium",
        "workers": 64,
        "split": "test",
        "shots": None,
        "seeds": "0",
        "datasets": None,
        "subtasks": "all",
        "strategy": "random",
        "mode": "raw",
        "prompt": "deepretro",
        "data": None,
        "az_model": "Pistachio_100+",
        "az_workers": 4,
        "top_k": 3,
        "az_cache": None,
        "perf": False,
        "limit": None,
        "max_output_tokens": None,
        "temperature": None,
        "realtime": False,
        "no_wait": False,
        "yes": False,
        "max_cost": None,
        "wandb_project": None,
        "hf_repo": None,
        "tag": None,
    }
    for key, fallback in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, fallback))
    # DeepRetro's AutoSolver runs the stability and hallucination checks by default;
    # the other modes keep them opt-in. The mode is known by now, so the fallback
    # can depend on it. A value set on the command line or in the config still wins.
    for key in ("stability_check", "hallucination_check"):
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, args.mode == "autosolve"))
    if args.model is None:
        args.model = config.get("model")
    if not args.model:
        sys.exit("--model is required (or set `model:` in --config)")
    if args.model.startswith(("anthropic/", "openai/")) and args.perf:
        sys.exit("--perf reads a vLLM server's /metrics; it does not apply to Claude/GPT")
    if args.max_output_tokens is None:
        args.max_output_tokens = 16384 if args.thinking else 1024
    if args.out_root is None:
        args.out_root = config.get("out_root") or str(REPO_ROOT / "outputs")
    return args


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args = resolve(args, load_config(args.config))

    if args.task == "moleculenet":
        from chemberta4.cli import run_moleculenet

        run_moleculenet(args)
    elif args.task == "chemllmbench":
        from chemberta4.cli import run_chemllmbench

        run_chemllmbench(args)
    else:
        from chemberta4.cli import run_retrosynthesis

        run_retrosynthesis(args)


if __name__ == "__main__":
    main()
