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

```sh
pip install -e '.[score,wandb,dev]'
```

`score` pulls in deepchem, needed to build the MoleculeNet splits and to
compute ROC-AUC and RMSE. `wandb` is optional. vLLM is **not** a dependency
here: serve the model from its own environment, since vLLM pins versions that
conflict with the scoring stack.

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

Start the server in its own environment, then point the evaluation at it.

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

`--prompt ether0` swaps in the prompt futurehouse/ether0 was trained on, which
asks for a single reaction SMILES. Only top-1 is meaningful with it.

There is no `autosolve` mode here; AiZynthFinder stays in DeepRetro.

### Scoring without generating

Scoring reads the checkpoint, so re-running the same command after the
generation finishes re-scores without sending a request. That is how to apply a
metric change to results you already have.

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
  deepretro/                  copied from DeepRetro; see DEEPRETRO.md
scripts/                      prepare_data.py, evaluate.py
```

## Not here yet

ChemLLMBench (17 tasks in 7 families), closed-source backends for Claude and
GPT, and training. The repository map artifact covers the intended shape.
