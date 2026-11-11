# The copied DeepRetro code

`src/chemberta4/deepretro/` holds files copied from the DeepRetro repository
rather than installed as a dependency, so chemberta4 runs without a DeepRetro
checkout on `sys.path`. (Copying a dependency's source into a repository like
this is usually called *vendoring*, if you want to search for the practice.)

**Source:** `~/Documents/DFS/DFS/DeepRetro`, commit `bf9acf0` (2026-10-30),
working tree clean for the copied paths.

## What was copied

| Copied file | DeepRetro path | Used for |
| --- | --- | --- |
| `llm.py` | `deepretro/utils/llm.py` | `call_LLM`, `llm_pipeline`, `validate_split_json` |
| `llm_interface.py` | `deepretro/utils/llm_interface.py` | `LLMRequest`, `create_llm_interface`, `split_thinking` |
| `llm_helpers.py` | `deepretro/utils/llm_helpers.py` | `Pathway`, prompt-mode helpers |
| `llm_trace.py` | `deepretro/utils/llm_trace.py` | Langfuse tracing: `record_llm_call`, `langfuse_metadata` |
| `utils_molecule.py` | `deepretro/utils/utils_molecule.py` | `canonicalize`, `validity_check` |
| `prompts.py` | `deepretro/utils/variables.py` | the retrosynthesis prompts and model lists |
| `vllm_metrics.py` | `deepretro/utils/vllm_metrics.py` | `PerfRun`, `log_to_wandb` |
| `one_step_eval.py` | `deepretro/utils/one_step_eval.py` | scoring: exact match, MaxFrag, summaries |
| `stability_checker.py` | `deepretro/algorithms/stability_checker.py` | `check_molecule_stability` (pipeline mode) |
| `hallucination_checker.py` | `deepretro/algorithms/hallucination_checker.py` | `calculate_hallucination_score` (pipeline mode) |
| `hallucination_weights.py` | `deepretro/algorithms/hallucination_weights.py` | weights for the above |

This is the complete transitive closure of `raw` and `pipeline` retrosynthesis
modes, found by following every `from deepretro...` import in each file.

## What was changed

Nothing but the following. No reformatting, no renames, no logic edits.

1. **Import paths.** `deepretro.utils.X` and `deepretro.algorithms.X` became
   `chemberta4.deepretro.X`, flattening the two subpackages into one.
   `deepretro.utils.variables` became `chemberta4.deepretro.prompts`.
   Six doctest imports of the form `from deepretro.algorithms import <symbol>`
   now name the module that defines the symbol, since the package re-exports do
   not exist here.
2. **`prompts.py` was trimmed** from 1425 to 666 lines, keeping only the twelve
   names the copied code imports: `SYS_PROMPT`, `USER_PROMPT`, their `_V4`,
   `_OPENAI` and `_DEEPSEEK` variants, `ADDON_PROMPT_7_MEMBER`,
   `OPENAI_MODELS` and `DEEPSEEK_MODELS`. Dropped: reaction-class encodings,
   scalability tables, `BASIC_MOLECULES`, agent `TOOLS`, AiZynthFinder model
   lists and `ERROR_MAP`. Nothing in chemberta4 reaches them.
3. **`one_step_eval.py` lost its autosolve helpers** — `az_first_step_proposals`,
   `run_az_record` and `hybrid_views`, together with the lazy
   `from deepretro.utils.az import run_az`. chemberta4 has no `autosolve` mode,
   so AiZynthFinder is never called. `write_summary` keeps its `views`
   parameter, which is generic.

Each file repeats its provenance in a three-line header comment.

## Re-syncing

Because only import lines changed, a re-sync is a diff:

```sh
cd ~/Documents/DFS/DFS/DeepRetro
git diff bf9acf0..HEAD -- deepretro/utils/llm.py deepretro/utils/llm_interface.py \
    deepretro/utils/llm_helpers.py deepretro/utils/llm_trace.py \
    deepretro/utils/utils_molecule.py deepretro/utils/vllm_metrics.py \
    deepretro/utils/one_step_eval.py deepretro/utils/variables.py \
    deepretro/algorithms/stability_checker.py \
    deepretro/algorithms/hallucination_checker.py \
    deepretro/algorithms/hallucination_weights.py
```

Apply anything relevant by hand, then update the commit named above here and in
each file header.

Prompts and parsers are the part that matters for comparability: if DeepRetro
changes `SYS_PROMPT` or the response parser, scores produced before and after a
re-sync are not directly comparable. The tests in `tests/test_prompts.py` assert
the copied prompt still matches DeepRetro's when a checkout is present, so
drift is visible rather than silent.

## Re-implemented, not copied: the `autosolve` mode

The depth-1 `autosolve` mode needs AiZynthFinder and a hallucination gate that the
copied files do not have. Those live in `src/chemberta4/tasks/retrosynthesis/` as
chemberta4's own code, because the copied files are marked "do not edit by hand".
They follow DeepRetro at commit `bf9acf0`:

| chemberta4 | DeepRetro |
| --- | --- |
| `az.py` `run_az`, `_get_finder`, `is_basic_molecule` | `deepretro/utils/az.py` |
| `autosolve.py` `az_first_step_proposals`, `run_az_record`, `hybrid_views` | the helpers removed from the copy of `deepretro/utils/one_step_eval.py`, **changed to follow `AutoSolver.single_step`/`solve_multiple`** (see below) rather than the eval script |
| `autosolve.py` `rank_hallucination` | `deepretro/algorithms/pipeline_checks.py` `hallucination_checker` (`rank_only=True`) |
| `autosolve.py` `canonicalize_pathways` | `AutoSolver._canonicalize_pathways` |

A re-sync diff of those DeepRetro files shows what to port by hand.

### Equivalence to `AutoSolver` at depth 1

The `hybrid` view is meant to reproduce what `AutoSolver` returns for a target at depth
1. These are the rules it follows, with the DeepRetro defaults they come from:

- **Checks default on.** `stability_check=True` (which also raises the retry budget
  from 6 to 15 attempts) and the heuristic hallucination gate, both on in `autosolve`
  and off in `raw`/`pipeline`. `--no-stability-check` / `--no-hallucination-check`
  turn them off; a value in the config file overrides the mode default.
- **Validity** always runs, inside `llm_pipeline`.
- **Target canonicalized first.** `AutoSolver` strips and canonicalizes the target
  before the prompt, the validity check and the gate. The record's `input` keeps the
  dataset's spelling.
- **Gate after the retry loop, ranking.** `llm_pipeline` is called with
  `hallucination_check=False`; the gate then runs once and ranks pathways (score at
  or above `reject_below` first, flagged ones demoted, not dropped). In `pipeline` the
  gate still drops pathways inside the loop, unchanged.
- **AZ solved means AZ answers.** When AZ solves a target and returns a route, the
  answer is `routes[0]`'s first step and the LLM is not used, **even when that route
  has no reaction** (a basic or in-stock target). Such a target scores as unanswered.
- **LLM answer capped at `top_k`** (default 3, the batch runner's `--top-k`): the
  first `k` ranked pathways, then any with no reactants dropped, as `solve_multiple`.
- **AZ model default** `Pistachio_100+`.

### Where it deliberately differs

- **Paths.** `AZ_MODELS_PATH` and `AZ_MODEL_CONFIG_PATH` are used as given. DeepRetro
  prefixes its own project root.
- **No disk cache, no `BASIC_MOLECULES` list.** The shared AZ checkpoint replaces the
  cache; `is_basic_molecule` (under 5 atoms or 5 carbons) covers the list for
  drug-sized targets.
- **AZ failures are not checkpointed.** `AutoSolver` turns an AZ exception into an
  unsolved leaf and does *not* try the LLM; DeepRetro's eval script writes an errored
  record, which later runs read as "AZ found nothing". Here the molecule is retried next
  run and left out of the current scoring (reported as "in only one checkpoint"), so
  `az_errors` in the coverage block counts only records already cached.
- **Thinking and token limit** are the model's serving settings, not part of the loop.
  chemberta4 runs non-thinking with `--max-output-tokens` for every mode, whereas
  `AutoSolver` defaults `enable_thinking=True` for hosted APIs.
- **ML checker not ported.** Only the heuristic gate is implemented (no `deepchem` or
  `xgboost`), and nothing past depth 1: no recursion, agent mode or metadata.
- **All molecules go through the LLM**, including those AZ solves, so the `llm` view
  covers the whole set. `AutoSolver` would skip the LLM for them; the `hybrid` answer
  is the same either way, only the LLM cost differs.
