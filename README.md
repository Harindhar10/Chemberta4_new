# chemberta4

**chemberta4 is a library for benchmarking open-source models on chemistry, and for
mid- and post-training them.** Today it evaluates models on three benchmarks:
MoleculeNet, single-step retrosynthesis and ChemLLMBench. More is on the way: ChemEval and
model training (see [Roadmap](#roadmap)).

**How it works, in three steps:**

1. Serve an open-source model locally with [vLLM](https://github.com/vllm-project/vllm).
2. Run `scripts/evaluate.py`. It sends chemistry questions to the model and scores the
   answers.
3. Read the scores in `outputs/<task>/<run>/`.

---

## The tasks

### MoleculeNet: molecular property prediction

The model gets a molecule (as a SMILES string) and predicts one of its properties. Examples:
"does this compound cross the blood-brain barrier?" or "how soluble is it in water?". There
are 11 datasets:

| Dataset | Type | What is predicted |
| --- | --- | --- |
| `bbbp` | classification | Blood-brain barrier penetration |
| `bace_classification` | classification | Whether a compound inhibits human BACE-1 |
| `clintox` | classification | FDA-approved drugs vs. candidates that failed clinical trials for toxicity |
| `hiv` | classification | Inhibition of HIV replication (NCI AIDS antiviral screen) |
| `tox21` | classification | Toxicity in nuclear-receptor and stress-response assays |
| `sider` | classification | Side effects of marketed drugs, grouped by organ class |
| `delaney` | regression | Aqueous solubility (ESOL) |
| `freesolv` | regression | Hydration free energy in water |
| `lipo` | regression | Lipophilicity (octanol/water distribution at pH 7.4) |
| `bace_regression` | regression | Binding affinity to BACE-1 |
| `clearance` | regression | In vitro intrinsic clearance in human liver microsomes |

- **Few-shot:** `--shots 0,5` means the model is tested with 0 examples in the prompt
  (zero-shot) and again with 5 worked examples (5-shot).
- **Seeds:** each seed is a full pass with a different random choice of examples. Results
  are averaged across seeds.
- **Scores:** classification uses **ROC-AUC** (higher is better). Regression uses **RMSE**
  (lower is better).

### Single-step retrosynthesis

The model gets a product molecule and must predict the reactants that make it. The test
molecules come from **USPTO-50k**, a standard set of reactions taken from US patents.

- **Top-1 / top-k exact match:** whether the model's first answer (or any of its first k
  answers) matches the true reactants exactly.
- **MaxFrag:** a looser score where only the largest reactant has to match
  ([Tetko et al., 2020](https://doi.org/10.1038/s41467-020-19266-y)).

### ChemLLMBench: 18 tasks in 7 families

A broad chemistry benchmark, ported from the ChemLLMBench reference implementation. Every
task has a fixed test set of 100 items. Few-shot examples are drawn either at random or by
scaffold similarity.

| Group | Task keys | Headline metric |
| --- | --- | --- |
| `name_prediction` | `smiles_to_iupac`, `iupac_to_smiles`, `smiles_to_formula`, `iupac_to_formula` | accuracy |
| `property_prediction` | `bbbp`, `bace`, `tox21`, `hiv`, `clintox` | accuracy (Yes/No) |
| `reaction_prediction` | `reaction_prediction` (USPTO forward) | top-1 accuracy |
| `retrosynthesis` | `retrosynthesis` (USPTO-50k) | top-1 accuracy |
| `reaction_component_selection` | `reactant_selection`, `ligand_selection`, `solvent_selection` | accuracy (zero-shot only) |
| `molecule_design` | `mol_caption` (SMILES → text), `mol_design` (text → SMILES) | BLEU-4 / exact match |
| `yield_prediction` | `Buchwald-Hartwig`, `Suzuki-Miyaura` | accuracy (high yield or not) |

- **Selecting tasks:** `--subtasks` takes task keys and group names, comma-separated
  (default: `all`).
- **Few-shot:** `--shots 0,10` sets the number of examples (default `0`), and
  `--strategy random|scaffold` sets how they are chosen. The reference refuses scaffold
  sampling for `mol_design`, so that combination is skipped with a message. The yield
  tasks fall back to random.
- **Secondary metrics:** these go into `metrics.csv`. Examples are F1 for property
  prediction, invalid-SMILES % for reactions, and ROUGE, METEOR, Tanimoto similarities,
  validity and FCD for molecule design.
- **The `chemllmbench` extra** is needed to score `mol_caption` and `mol_design`
  (`pip install -e '.[chemllmbench]'`). FCD also needs `fcd` to be importable, otherwise
  it is NaN.

### Comparable with earlier results

MoleculeNet and retrosynthesis use the same prompts and metrics as the earlier Claude/GPT
and DeepRetro runs, so a model evaluated here can be compared directly with those results.
The DeepRetro code this needs is copied into `src/chemberta4/deepretro/`, not installed. See
[DEEPRETRO.md](DEEPRETRO.md). In the same way, ChemLLMBench's prompts, parsers and metrics
are copied from its reference implementation (see [CHEMLLMBENCH.md](CHEMLLMBENCH.md)), so
scores line up with that repository's leaderboard.

## Glossary

| Term | Meaning |
| --- | --- |
| **vLLM** | A fast server that runs open-source LLMs behind an OpenAI-compatible API. |
| **SMILES** | A text format for writing a molecule, e.g. `CCO` is ethanol. |
| **DeepRetro** | An earlier retrosynthesis project. Its prompts and pipeline are reused here. |
| **AiZynthFinder (AZ)** | A template-based retrosynthesis tool. Used in `autosolve` mode. |
| **Thinking model** | A model that writes out its reasoning before it answers (e.g. Qwen3 with thinking on). |
| **Checkpoint** | The JSONL file of finished answers. It is what makes runs resumable. |

---

## Quickstart

### 1. Install

From the repo root:

```sh
python3 -m venv /venv/chemberta

uv pip install --python /venv/chemberta/bin/python --upgrade pip

uv pip install --python /venv/chemberta/bin/python \
    vllm \
    -e '.[score,chemllmbench,wandb,hub,dev]'
```

vLLM and the evaluation code share one environment. The optional extras:

- `score`: installs deepchem. It is needed to build the MoleculeNet splits and to compute
  ROC-AUC and RMSE.
- `chemllmbench`: scikit-learn plus the caption and design metrics for ChemLLMBench.
- `wandb`: optional logging to Weights & Biases.
- `hub`: only needed to publish results to the Hugging Face Hub.
- `dev`: tests.

Copy [.env.example](.env.example) to `.env` and fill in what you need (see
[Environment variables](#environment-variables)).

### 2. Prepare the data

```sh
python scripts/prepare_data.py --task moleculenet      # DeepChem scaffold splits
python scripts/prepare_data.py --task retrosynthesis   # copies USPTO-50k test sets
python scripts/prepare_data.py --task chemllmbench     # downloads ChemLLMBench sources
```

MoleculeNet writes to `data/processed/`, which is gitignored.

- The retrosynthesis and ChemLLMBench test sets are committed. For ChemLLMBench,
  `prepare_data` downloads the files its loaders fetch from the web into
  `data/raw/chemllmbench/` (gitignored). These are the DeepChem CSVs, ChEBI-20 and the
  USPTO training pickles (about 470 MB, used only for few-shot reaction prompts). A run
  that needs a file it does not have downloads it itself.

- To reuse MoleculeNet splits that already exist instead of rebuilding them:
  `export CHEMBERTA4_MOLNET_DATA=~/Documents/DFS/DFS/molnet/datasets`
- The retrosynthesis sets are copied from a DeepRetro checkout. If it is not in the default
  place, point `DEEPRETRO_DATA_DIR` at its `data/` directory.

### 3. Start the model server

```sh
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 17408 \
    --enable-prefix-caching

export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
```

### 4. Run an evaluation

```sh
# MoleculeNet: 50 molecules per dataset, zero- and 5-shot, one seed
python scripts/evaluate.py --task moleculenet \
    --model hosted_vllm/Qwen/Qwen3-8B --shots 0,5 --seeds 0 --limit 50

# Retrosynthesis: 10 molecules, raw mode
python scripts/evaluate.py --task retrosynthesis \
    --model hosted_vllm/Qwen/Qwen3-8B --mode raw --limit 10

# ChemLLMBench: every task, zero- and 10-shot, random examples
python scripts/evaluate.py --task chemllmbench \
    --model hosted_vllm/Qwen/Qwen3-8B --shots 0,10 --strategy random
```

### 5. Read the results

Each run writes to `outputs/<task>/<run>/`:

| File | Contents |
| --- | --- |
| `results.jsonl` / `generations.jsonl` | Every raw model answer. This is the checkpoint. |
| `tasks/<group>.jsonl` | The same records, one file per finished group: ChemLLMBench `<subtask>_k<k>_<strategy>`, MoleculeNet `<dataset>_k<k>_seed<seed>`, retrosynthesis `mols_00000-00099` (100 molecules each). Each file is written as soon as its group's last answer lands. |
| `results.csv` | Per-molecule results as a table. |
| `predictions/` | ChemLLMBench: one CSV of raw answers and parsed predictions per task, k and strategy. |
| `metrics.csv` / `summary.json` | The final scores. |
| `perf.json` | Server throughput and latency (only with `--perf`). |

## Common options

| Flag | What it does |
| --- | --- |
| `--limit N` | Evaluate only the first N molecules (good for a quick test). |
| `--datasets bbbp,hiv` | MoleculeNet: run only these datasets (default: all). |
| `--shots 0,5,20` | MoleculeNet: number of examples in the prompt. |
| `--seeds 0,1,2` | MoleculeNet: repeat with different example draws. |
| `--subtasks bbbp,name_prediction` | ChemLLMBench: task keys or groups (default: all). |
| `--strategy random\|scaffold` | ChemLLMBench: how few-shot examples are chosen. |
| `--mode raw\|pipeline\|autosolve` | Retrosynthesis: how answers are produced (see below). |
| `--prompt deepretro\|ether0` | Retrosynthesis: which prompt to use. |
| `--thinking` | Turn on reasoning for thinking models. |
| `--workers N` | Number of requests sent at once (vLLM batches them). |
| `--config file.yaml` | Load defaults from a YAML file. |

Run `python scripts/evaluate.py --help` for every option.

---

## Guides

### Resuming and re-scoring

- **Every run is resumable.** Finished items are read back from the checkpoint and skipped.
  To restart an interrupted run, run the same command again.
- **Failed requests are retried.** Items that failed at the API are deliberately not
  checkpointed, so a run that hit a context-length error or a dropped connection heals
  itself on the next run.
- **Re-scoring is free.** Scoring reads the checkpoint, so re-running a finished command
  re-scores without sending any request. Use this to apply a metric change to results you
  already have.
- **Finished groups are saved as their own files.** When the last item of a group is
  saved, its records are also written to `tasks/<group>.jsonl`. A group with a failed
  request is written after a re-run retries that request. A restarted run also writes
  any group the checkpoint already completes. The checkpoint stays what resume and
  scoring read.

### Thinking models

```sh
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --max-model-len 32768

python scripts/evaluate.py --task moleculenet --model hosted_vllm/Qwen/Qwen3-8B \
    --thinking --thinking-effort medium --max-output-tokens 16384
```

- With `--reasoning-parser`, vLLM returns the reasoning in its own field. Without it, the
  reasoning is split off the inline `<think>` block. This matters because otherwise the
  answer parser could match braces inside the reasoning.
- Each `--thinking-effort` level (`low`, `medium`, `high`, `max`) writes to its own run
  directory, so levels never share a checkpoint.
- If the reasoning uses up the whole token budget, there is no answer. Those records are
  marked `truncated`, not `parse_error`, so you can tell the two apart in the summary.

### Retrosynthesis modes

| Mode | What it measures |
| --- | --- |
| `raw` | One answer per molecule, parsed at scoring time. The model on its own. |
| `pipeline` | DeepRetro's `llm_pipeline`: temperature retries, `validity_check`, and optional `--stability-check` and `--hallucination-check`. What the multi-step search would receive. |
| `autosolve` | DeepRetro's `AutoSolver` at depth 1: AiZynthFinder first, and the LLM only when AZ does not solve the target. Scores three views: `hybrid` (what `AutoSolver` returns), `az` and `llm`. Needs `pip install .[az]` and `AZ_MODELS_PATH`. |

`--prompt ether0` swaps in the prompt that futurehouse/ether0 was trained on. It asks for a
single reaction SMILES, so only top-1 is meaningful with it.

**`autosolve` defaults** (same as DeepRetro):

- The stability and hallucination checks are **on**. Turn them off with
  `--no-stability-check` / `--no-hallucination-check`.
- The AZ model is `Pistachio_100+`.
- The LLM keeps the first `--top-k 3` pathways.

**How the hallucination check differs between modes:**

- In `autosolve`, it *ranks* pathways once after the retry loop: best first, flagged ones
  moved down, not removed.
- In `pipeline`, it *drops* flagged pathways inside the loop, and is off unless you ask
  for it.
- The two give different top-1 scores, so compare like with like.

If AZ solves a target, its route is the answer, even when that route has no reaction (a
basic target), exactly as in `AutoSolver`. [DEEPRETRO.md](DEEPRETRO.md) lists the rules and
where this code deliberately differs.

AZ results are cached in `<out-root>/az_cache/<data>_<az-model>.jsonl` and shared across LLM
runs on the same data, so a second model does not repeat the search.

### Getting the AiZynthFinder models

`autosolve` needs `<AZ_MODELS_PATH>/<az-model>/config.yml`. Set `AZ_MODELS_PATH` in the
shell or in `.env`. A relative path is resolved against the directory you run from (normally
the repo root), so run every command from the same place. An absolute path also works.

**USPTO (public):**

```bash
pip install -e ".[az]"
export AZ_MODELS_PATH=az_models   # git-ignored
python -m aizynthfinder.tools.download_public_data "$AZ_MODELS_PATH/USPTO"
python scripts/evaluate.py --task retrosynthesis --mode autosolve --az-model USPTO ...
```

The download target must be the `USPTO` subfolder, so that the model name matches the folder.

**Pistachio_25 / 50 / 100 / 100+ (private):**

- These are DeepRetro's own models and are not public.
- DeepRetro's `data/download_pistachio_models.py` fetches them from a private S3 bucket
  using `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
- It copies the USPTO `config.yml` into each folder and edits its paths, so download USPTO
  first.
- Without that access, the default `--az-model Pistachio_100+` fails with
  "No AiZynthFinder config". Pass `--az-model USPTO` instead.

> **Note:** the public USPTO templates come from USPTO data, so AZ may have seen these test
> reactions. Its `az` and `hybrid` scores with `USPTO` are likely inflated. Prefer a
> Pistachio model for a fair comparison.

### Using a config file

```sh
python scripts/evaluate.py --task moleculenet \
    --config configs/evaluation/open_source/vllm_example.yaml
```

- Command-line flags override the file.
- The server URL is not in the config. It comes from `HOSTED_VLLM_API_BASE`, so one config
  works wherever the server runs.
- See [vllm_example.yaml](configs/evaluation/open_source/vllm_example.yaml) for every
  setting with comments.

### Publishing to the Hugging Face Hub

```sh
export HF_TOKEN=hf_...          # a write token

python scripts/evaluate.py --task moleculenet --model hosted_vllm/Qwen/Qwen3-8B \
    --shots 0,5 --limit 50 --hf-repo harindhar10/chemberta4-evals
```

`--hf-repo` pushes the finished run to a **public** dataset repo, created on first use. All
runs share that one repo:

- The run's files (raw JSONL, scores, `run_config.json`, `perf.json`) go under
  `<task>/<run_id>/`.
- The dataset card gets a section for the run with its hyperparameters and metrics. For
  MoleculeNet that is the mean and std across seeds.
- Each run is a dataset-viewer config, so the generations can be browsed on the Hub or
  loaded directly:

```python
from datasets import load_dataset
ds = load_dataset("harindhar10/chemberta4-evals", "moleculenet--Qwen3-8B")
```

Good to know:

- Pushing the same run again replaces its files and its card section in one commit.
- The token is checked before generation starts.
- A failed upload only prints a warning, because the scores are already on disk.
  Re-running the command re-scores and re-pushes without sending a request.
- `run_id` is the local run-folder name. Settings not in that name share a run: `--shots`,
  `--seeds` and `--limit` for MoleculeNet, and the thinking effort for retrosynthesis. Two
  such runs replace each other, locally and on the Hub.

---

## Environment variables

| Variable | Used for |
| --- | --- |
| `HOSTED_VLLM_API_BASE` | URL of the vLLM server, e.g. `http://localhost:8000/v1`. Required. |
| `CHEMBERTA4_MOLNET_DATA` | Reuse existing MoleculeNet splits from this folder. |
| `DEEPRETRO_DATA_DIR` | DeepRetro's `data/` folder, if not in the default place. |
| `CHEMBERTA4_CHEMLLMBENCH_DATA` / `CHEMBERTA4_CHEMLLMBENCH_RAW` | Override the ChemLLMBench committed / downloaded data folders. |
| `CHEMLLMBENCH_SRC_DIR` | ChemLLMBench reference checkout, for the parity tests and `--copy-shipped`. |
| `AZ_MODELS_PATH` | Folder holding the AiZynthFinder models (`autosolve`). |
| `AZ_MODEL_CONFIG_PATH` | Fallback AZ config, used only when the one above is missing. |
| `HF_TOKEN` | Hugging Face write token, for `--hf-repo`. |
| `WANDB_API_KEY` | Weights & Biases key, for `--wandb-project`. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional Langfuse tracing. |

## Tests

```sh
pytest
```

There are three groups:

- **Doctests** in our modules and in the copied DeepRetro modules.
- **Scoring tests** that pin down the exact-match, MaxFrag and parse-failure rules.
- **Prompt-parity tests** that fail if the copied MoleculeNet prompt drifts from the DFS
  `molnet` package, the copied retrosynthesis prompt drifts from DeepRetro, or the
  ChemLLMBench prompts drift from the reference checkout. They are skipped when those
  checkouts are absent.

## Project layout

```
src/chemberta4/
  cli.py                      wires together the task, the backend and the run loop
  tasks/moleculenet/          config, data, prompts, metrics, run
  tasks/retrosynthesis/       data, run, autosolve, AiZynthFinder wrapper
  tasks/chemllmbench/         copied task families, registry, run; see CHEMLLMBENCH.md
  evaluation/
    backends/base.py          ModelBackend.generate() -> Response
    backends/vllm_server.py   client for the vLLM OpenAI-compatible server
    runner.py                 resumable, threaded generation loop
    thinking.py               splits a reasoning trace from the answer
    perf.py                   reads vLLM /metrics during a run
    reporting.py              Weights & Biases logging
    hub.py                    Hugging Face Hub dataset repo and card
  deepretro/                  copied from DeepRetro; see DEEPRETRO.md
scripts/                      prepare_data.py, evaluate.py
configs/                      example YAML configs
tests/                        pytest suite
```

## Roadmap

Coming next:

- **ChemEval**.
- **Closed-source backends** for Claude and GPT.
- **Model training**: mid- and post-training of open-source models.

The repository map artifact covers the intended shape.
