# The copied ChemLLMBench code

`src/chemberta4/tasks/chemllmbench/` holds the prompts, in-context-example
samplers, answer parsers and metrics of the ChemLLMBench reference
implementation. They are copied word for word, so a run here sends the same
prompts and scores answers the same way as the reference, and its results can
be compared with the reference leaderboard (`results/leaderboard.csv` in the
reference checkout). What changed is how the model is called: each prompt is
sent to a `vllm serve` endpoint, and each answer is written to the resumable
checkpoint every other task uses.

**Source:** `~/Documents/DFS/DFS/Chemberata4_ChemLLMBench`, branch `my-changes`,
commit `a3c6447`. Its working tree was clean and matched `origin/my-changes`.

## What was copied

Each family module is the reference `data.py`, followed by the part of the
reference runner that holds the parser, the normalisers and the metrics. Both
are copied whole, apart from the `__main__` block. The line range is in each
file's header.

| Module | Reference files | Task keys |
| --- | --- | --- |
| `name_prediction.py` | `Name_Prediction/data.py`, `name_pred.py` 32–160 | `smiles_to_iupac`, `iupac_to_smiles`, `smiles_to_formula`, `iupac_to_formula` |
| `property_prediction.py` | `Property_Prediction/data.py`, `Property_prediction.py` 43–114 | `bbbp`, `bace`, `tox21`, `hiv`, `clintox` |
| `reaction_prediction.py` | `Reaction_prediction/data.py`, `reaction_pred.py` 18–102 | `reaction_prediction` |
| `retrosynthesis.py` | `Retrosynthesis/data.py`, `retro.py` 16–102 | `retrosynthesis` |
| `reaction_selection.py` | `Reaction_Selection/data.py`, `selection.py` 15–123 | `reactant_selection`, `ligand_selection`, `solvent_selection` |
| `molecule_design.py` | `Molecule_Design/data.py`, `Mol_design_and_mol_caption.py` 18–180 | `mol_caption`, `mol_design` |
| `yield_prediction.py` | `Yield_prediction/data.py`, `yield_pred.py` 20–81 | `Buchwald-Hartwig`, `Suzuki-Miyaura` |
| `parsing.py` | `utils.py`: `strip_thinking`, `extract_json_answer` | (shared) |
| `registry.py` | `main.py`: `TASK_REGISTRY`, `TASK_GROUPS`, `PRIMARY_METRIC_KEYS` | (all) |

Not copied:

- the reference `utils.py` model classes (`HuggingFaceModel`, which runs vLLM
  in-process, and `LiteLLMModel`, for closed-source APIs), together with their
  sample cache and Langfuse setup;
- each runner's `run_benchmark`, `_build_model`, result-file writing and
  `argparse`;
- `main.py`'s leaderboard and summary files.

## What was changed

Only these edits, each applied by exact string replacement:

1. **Imports.** `from utils import ...` became
   `from chemberta4.tasks.chemllmbench.parsing import ...`. The runner imports
   the copied section uses (`re`, `RDLogger`, `sklearn.metrics`, `MACCSkeys`)
   are listed at the start of that section. `evaluate` and `Levenshtein` are
   imported inside the two functions that use them, so the module loads
   without the optional `chemllmbench` extra.
2. **Data paths.**
   - Files the reference kept next to its code now come from
     `data/processed/chemllmbench/`. `BH_dataset .npz` lost the space in its
     name.
   - Files the reference fetched from URLs on every run (the DeepChem CSVs and
     ChEBI-20) are now read through `fetch_url`, which downloads each one once
     into `data/raw/chemllmbench/`.
   - The USPTO Box pickles are saved there too, instead of in the current
     directory.

Each module also ends with a short **adapter** that is not in the reference:
`load(subtask, strategy, k)`, `parse(raw, subtask)` and
`score(y_true, y_pred, subtask)`. They call the copied functions exactly as
the reference `run_benchmark` did. There is one deliberate shortcut: zero-shot
reaction prediction and retrosynthesis build their prompts from the committed
test CSV alone, without the training pickle. The prompts are identical either
way, because `_create_prompt` never reads the training set when `k == 0`.

## Checking parity

`tests/test_chemllmbench.py` compares the prompts for the name, yield and
selection tasks with the reference loaders whenever a checkout is present
(`CHEMLLMBENCH_SRC_DIR`). When the port was made, all 41 (task, strategy, k)
settings were checked against the reference loaders, network-sourced tasks
included. Prompts and references matched exactly.

## Re-syncing

Because the copies are line ranges plus a few string edits, a re-sync starts
with a diff:

```sh
cd ~/Documents/DFS/DFS/Chemberata4_ChemLLMBench
git diff a3c6447..HEAD -- '*/data.py' '*/*_pred*.py' '*/retro.py' '*/selection.py' \
    Molecule_Design/Mol_design_and_mol_caption.py utils.py main.py
```

Then copy the changed functions across, re-run `pytest`, and update the
commit above. If the shipped data changed:
`python scripts/prepare_data.py --task chemllmbench --copy-shipped`.
