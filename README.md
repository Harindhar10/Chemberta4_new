# chemberta4

Evaluation of open-source LLMs on two chemistry tasks, served locally by vLLM:

- **MoleculeNet** — molecular property prediction, few-shot, scored by ROC-AUC
  (classification) and RMSE (regression) over 11 datasets.
- **Single-step retrosynthesis** — USPTO-50k products, scored by top-1 and
  top-k exact match plus MaxFrag accuracy.

Both tasks use the same prompts and metrics as the earlier Claude/GPT and
DeepRetro runs, so a model evaluated here is directly comparable with those
results. The DeepRetro code this depends on is copied into
`src/chemberta4/deepretro/`, not installed — see [DEEPRETRO.md](DEEPRETRO.md).

## Install

From the repo root:

```sh
python3 -m venv /venv/chemberta

uv pip install --python /venv/chemberta/bin/python --upgrade pip

uv pip install --python /venv/chemberta/bin/python \
    vllm \
    -e '.[score,wandb,hub,dev]'
```

vLLM and the evaluation code share one environment. `score` pulls in deepchem,
needed to build the MoleculeNet splits and to compute ROC-AUC and RMSE.
`wandb` is optional, and `hub` is only needed to publish results to the
Hugging Face Hub.

## Prepare data

```sh
python scripts/prepare_data.py --task moleculenet      # DeepChem scaffold splits
python scripts/prepare_data.py --task retrosynthesis   # copies USPTO-50k test sets
```

Both write to `data/processed/`, which is gitignored. To reuse splits that
already exist instead of rebuilding them:

```sh
export CHEMBERTA4_MOLNET_DATA=~/Documents/DFS/DFS/molnet/datasets
```

The retrosynthesis sets are copied from a DeepRetro checkout; point
`DEEPRETRO_DATA_DIR` at its `data/` directory if it is not in the default place.

## Run

Start the server, then point the evaluation at it.

```sh
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 17408 \
    --enable-prefix-caching

export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
```

```sh
# MoleculeNet: 50 molecules per dataset, zero- and 5-shot, one seed
python scripts/evaluate.py --task moleculenet \
    --model hosted_vllm/Qwen/Qwen3-8B --shots 0,5 --seeds 0 --limit 50

# Retrosynthesis: 10 molecules, raw mode
python scripts/evaluate.py --task retrosynthesis \
    --model hosted_vllm/Qwen/Qwen3-8B --mode raw --limit 10
```

Results land in `outputs/<task>/<run>/`: `results.jsonl` or
`generations.jsonl` (the checkpoint), `results.csv`, `metrics.csv` or
`summary.json`, and `perf.json` with `--perf`.

### Resuming

Every run is resumable. Finished items are read back from the checkpoint and
skipped, so an interrupted run is restarted with the same command. Items that
failed at the API are deliberately not checkpointed, so they are retried; a run
that hit a context-length error or a dropped connection heals itself.

### Thinking models

```sh
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --max-model-len 32768

python scripts/evaluate.py --task moleculenet --model hosted_vllm/Qwen/Qwen3-8B \
    --thinking --thinking-effort medium --max-output-tokens 16384
```

With `--reasoning-parser` the trace comes back in its own field; without one it
is split off the inline `<think>` block, which matters because the answer-JSON
regex would otherwise match braces inside the reasoning. Each effort level
writes to its own run directory, so levels never share a checkpoint.

A reasoning trace that uses the whole token budget leaves no answer. Those
records are marked `truncated` rather than `parse_error`, so the two causes stay
distinguishable in the summary.

### Retrosynthesis modes

| Mode | What it measures |
| --- | --- |
| `raw` | One completion per molecule, parsed at scoring time. The model on its own. |
| `pipeline` | DeepRetro's `llm_pipeline`: temperature retries, `validity_check`, and optional `--stability-check` and `--hallucination-check`. What the multi-step search would receive. |
| `autosolve` | DeepRetro's `AutoSolver` at depth 1: AiZynthFinder first, the LLM only when AZ does not solve the target. Scores three views: `hybrid` (what `AutoSolver` returns), `az` and `llm`. Needs `pip install .[az]` and `AZ_MODELS_PATH`. |

`--prompt ether0` swaps in the prompt futurehouse/ether0 was trained on, which
asks for a single reaction SMILES. Only top-1 is meaningful with it.

`autosolve` uses DeepRetro's defaults: the stability and hallucination checks are **on** (`--no-stability-check` / `--no-hallucination-check` turn them off), the AZ model is `Pistachio_100+`, and the LLM keeps the first `--top-k 3` pathways. The hallucination check *ranks* pathways (best first, flagged ones demoted, not dropped) once after the retry loop; in `pipeline` it drops them inside the loop and stays off unless asked for. The two give different top-1, so compare like with like. If AZ solves a target, its route is the answer even when that route has no reaction (a basic target), exactly as in `AutoSolver`. `DEEPRETRO.md` lists the rules and where this code deliberately differs.

AZ results are cached in `<out-root>/az_cache/<data>_<az-model>.jsonl` and shared across LLM runs over the same data, so a second model does not repeat the search.

### Getting the AiZynthFinder models

`autosolve` needs `<AZ_MODELS_PATH>/<az-model>/config.yml`. Set `AZ_MODELS_PATH` in the shell
or in `.env`. A relative path is resolved against the directory you run from, normally the
repo root, so run every command from the same place; an absolute path also works.

**USPTO (public):**

```bash
pip install -e ".[az]"
export AZ_MODELS_PATH=az_models   # git-ignored
python -m aizynthfinder.tools.download_public_data "$AZ_MODELS_PATH/USPTO"
python scripts/evaluate.py --task retrosynthesis --mode autosolve --az-model USPTO ...
```

The target must be the `USPTO` subfolder, so that the model name matches the folder.

**Pistachio_25 / 50 / 100 / 100+ (private):** these are DeepRetro's own models and are not
public. DeepRetro's `data/download_pistachio_models.py` fetches them from a private S3 bucket
with `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. It copies the USPTO `config.yml` into each
folder and edits its paths, so download USPTO first. Without that access, `autosolve`'s default
`--az-model Pistachio_100+` fails with "No AiZynthFinder config"; pass `--az-model USPTO`.

The public USPTO templates come from USPTO data, so AZ may have seen these test reactions.
Its `az` and `hybrid` scores with `USPTO` are likely inflated; prefer a Pistachio model for a
fair comparison.

### Scoring without generating

Scoring reads the checkpoint, so re-running the same command after the
generation finishes re-scores without sending a request. That is how to apply a
metric change to results you already have.

## Publishing to the Hub

```sh
export HF_TOKEN=hf_...          # a write token

python scripts/evaluate.py --task moleculenet --model hosted_vllm/Qwen/Qwen3-8B \
    --shots 0,5 --limit 50 --hf-repo Harindhar10/chemberta4-evals
```

`--hf-repo` pushes the finished run to a **public** dataset repo, which is
created on first use. Every run shares that one repo:

- The run's files (raw JSONL, scores, `run_config.json`, `perf.json`) go under
  `<task>/<run_id>/`.
- The dataset card gets a section for the run with its hyperparameters and
  metrics. For MoleculeNet that is the mean and std across seeds.
- Each run is a dataset-viewer config, so the generations can be browsed on
  the Hub or loaded directly:

```python
from datasets import load_dataset
ds = load_dataset("Harindhar10/chemberta4-evals", "moleculenet--Qwen3-8B")
```

Pushing the same run again replaces its files and its card section in one
commit. The token is checked before generation starts. A failed upload only
prints a warning, because the scores are already on disk. Re-running the
command re-scores and re-pushes without sending a request.

`run_id` is the local run-folder name. Settings not in that name share a run:
`--shots`, `--seeds` and `--limit` for MoleculeNet, and the thinking effort for
retrosynthesis. Two such runs replace each other locally and on the Hub.

## Configs

```sh
python scripts/evaluate.py --task moleculenet \
    --config configs/evaluation/open_source/vllm_example.yaml
```

Command-line flags win over the file. The server URL is not in the config; it
comes from `HOSTED_VLLM_API_BASE`, so one config works wherever the server runs.

## Tests

```sh
pytest
```

Three groups: the doctests carried by our modules and the copied ones;
scoring tests that pin down the exact-match, MaxFrag and parse-failure rules;
and prompt-parity tests that fail if the copied MoleculeNet prompt drifts from
the DFS `molnet` package or the copied retrosynthesis prompt drifts from
DeepRetro. The parity tests skip when those checkouts are absent.

## Layout

```
src/chemberta4/
  cli.py                      task + backend + run loop, wired together
  tasks/moleculenet/          config, data, prompts, metrics, run
  tasks/retrosynthesis/       data, run
  evaluation/
    backends/base.py          ModelBackend.generate() -> Response
    backends/vllm_server.py   the vLLM OpenAI-compatible server
    runner.py                 resumable threaded generation loop
    thinking.py               splits a reasoning trace from the answer
    perf.py                   vLLM /metrics during a run
    reporting.py              W&B
    hub.py                    Hugging Face Hub dataset repo and card
  deepretro/                  copied from DeepRetro; see DEEPRETRO.md
scripts/                      prepare_data.py, evaluate.py
```

## Not here yet

ChemLLMBench (17 tasks in 7 families), closed-source backends for Claude and
GPT, and training. The repository map artifact covers the intended shape.
