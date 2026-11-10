#!/usr/bin/env python
"""Evaluate a vLLM-served model on MoleculeNet or single-step retrosynthesis.

Start the server first, in its own environment::

    vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 17408 \
        --enable-prefix-caching [--reasoning-parser qwen3]

then point this at it::

    export HOSTED_VLLM_API_BASE=http://localhost:8000/v1

    python scripts/evaluate.py --task moleculenet \
        --model hosted_vllm/Qwen/Qwen3-8B --shots 0,5 --seeds 0 --limit 50

    python scripts/evaluate.py --task retrosynthesis \
        --model hosted_vllm/Qwen/Qwen3-8B --mode raw --limit 10

Both tasks are resumable: finished items are read back from the run's JSONL
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
    p.add_argument("--task", required=True, choices=["moleculenet", "retrosynthesis"])
    p.add_argument("--model", help="litellm id, e.g. hosted_vllm/Qwen/Qwen3-8B")
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

    mol = p.add_argument_group("moleculenet")
    mol.add_argument("--datasets", default=None, help="comma-separated; default is all")
    mol.add_argument("--shots", default=None, help="comma-separated k values; 0 = zero-shot")
    mol.add_argument("--seeds", default=None, help="comma-separated; each seed is a full pass")
    mol.add_argument("--split", default=None)

    retro = p.add_argument_group("retrosynthesis")
    retro.add_argument("--mode", choices=["raw", "pipeline"], default=None)
    retro.add_argument("--prompt", choices=["deepretro", "ether0"], default=None)
    retro.add_argument("--data", default=None, help="dataset name or path to a CSV")
    retro.add_argument("--stability-check", action="store_true", default=None)
    retro.add_argument("--hallucination-check", action="store_true", default=None)

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
        "mode": "raw",
        "prompt": "deepretro",
        "data": None,
        "stability_check": False,
        "hallucination_check": False,
        "perf": False,
        "limit": None,
        "max_output_tokens": None,
        "wandb_project": None,
        "hf_repo": None,
        "tag": None,
    }
    for key, fallback in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, fallback))
    if args.model is None:
        args.model = config.get("model")
    if not args.model:
        sys.exit("--model is required (or set `model:` in --config)")
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
    else:
        from chemberta4.cli import run_retrosynthesis

        run_retrosynthesis(args)


if __name__ == "__main__":
    main()
