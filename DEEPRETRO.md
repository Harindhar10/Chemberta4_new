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
