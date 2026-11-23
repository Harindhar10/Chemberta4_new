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

At copy time, nothing but the following: no reformatting, no renames, no logic
edits. The logic edits made since then are listed under
[Edits since the copy](#edits-since-the-copy).

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

### Edits since the copy

These are chemberta4's own changes to the copied files. A re-sync must keep them.
Files not listed here (`llm_trace.py`, `utils_molecule.py`, `vllm_metrics.py`,
`one_step_eval.py` and the checkers) have only the changes above.

| Commit | File | Edit |
| --- | --- | --- |
| `062ed83` | `llm.py` | `llm_pipeline` takes an optional `trace` list |
| `2691b3a` | `llm.py` | `llm_pipeline` takes and forwards `thinking_effort` |
| `bc06e56` | `llm_helpers.py` | gpt-6 counts as a reasoning model; reasoning models always get `temperature=1` |
| `3768fdc` | `llm_helpers.py` | thinking off sends the lowest `reasoning_effort` to OpenAI reasoning models |
| `3768fdc` | `llm_interface.py` | warn when a response is empty because the token budget ran out |
| `04985f5` | `llm_helpers.py` | OpenAI `reasoning_effort` is exempt from `drop_params` via `allowed_openai_params` |
| (uncommitted) | `prompts.py` | new `USER_PROMPT_OPENAI_NO_PREAMBLE`; `USER_PROMPT_OPENAI` itself is unchanged |
| (uncommitted) | `llm_interface.py` | OpenAI uses `USER_PROMPT_OPENAI_NO_PREAMBLE`; the 7-ring addon goes in the user message only |

1. **Per-attempt trace (`062ed83`).** `llm_pipeline(..., trace=None)` appends one
   dict per attempt, with `attempt`, `temperature`, `model`, `status` and `raw`
   (the response text, `None` if there was none). `status` is the failing status,
   204 if the response parsed but every pathway was filtered out, else 200. The
   return value is unchanged. The logged `temperature` is the one the retry loop
   requested, not necessarily the one sent: see edit 3.
2. **`thinking_effort` forwarded (`2691b3a`).** Upstream `llm_pipeline` has no
   `thinking_effort` parameter, so the effort always fell back to the request
   default (`"medium"`). It is now a parameter (default `"medium"`) passed into
   the `LLMRequest`.
3. **gpt-6 and `temperature=1` (`bc06e56`).** `looks_like_openai_reasoning_model`
   also matches `gpt-6` (for `openai/gpt-6-luna`). `build_completion_params` sends
   `temperature=1` to every model with `requires_temperature_one`, whether or not
   thinking is on. Upstream does that only with thinking on
   (`requires_temperature_one and enable_thinking`), and these models reject any
   other temperature either way. The retry loop's temperature sweep therefore
   has no effect on them.
4. **Lowest reasoning effort with thinking off (`3768fdc`).** New helper
   `openai_lowest_reasoning_effort` returns `"none"` for gpt-5.1+ and gpt-6,
   `"minimal"` for base gpt-5 and `"low"` for o1/o3/o4. When thinking is off,
   `build_completion_params` sends it as `reasoning_effort` to OpenAI reasoning
   models. Upstream sends nothing, and OpenAI then reasons at `"medium"`: with a
   small `--max-output-tokens`, the hidden reasoning used up the whole budget and
   gpt-6-luna returned empty content on every attempt. That empty content is
   logged as status 502 ("no JSON found" from `OpenAILLM.parse_response`), not an
   HTTP error. Anthropic is unchanged: Claude does not think unless asked.
   Verified 2026-11-23 with curl: gpt-6-luna accepts `"none"`, with or without
   `temperature=1`.
5. **Empty-response warning (`3768fdc`).** `LLMInterface.call` logs a warning
   with the token limit and reasoning-token count when the content is empty and
   `finish_reason == "length"`, so this failure shows in the logs.
6. **`reasoning_effort` survives `drop_params` (`04985f5`).** `llm.py` sets
   `litellm.drop_params = True`. A LiteLLM whose model map predates a model (the
   remote box's system LiteLLM, for gpt-6-luna) treats `reasoning_effort` as
   unsupported and strips it silently, so OpenAI reasoned at `"medium"` anyway
   and edit 4 had no effect. Whenever `build_completion_params` sets
   `reasoning_effort` for an OpenAI model, it also sets
   `allowed_openai_params=["reasoning_effort"]`, which exempts it from the drop.
7. **No duplicated prompt text (uncommitted).** Upstream `USER_PROMPT_OPENAI`
   opens with the same three paragraphs as `SYS_PROMPT_OPENAI`, so OpenAI models
   received them twice. `USER_PROMPT_OPENAI` is kept unchanged, so it still
   matches DeepRetro. A new `USER_PROMPT_OPENAI_NO_PREAMBLE`, sliced from it at
   "Perform a single-step", is what both OpenAI `PROMPT_CONFIG` entries now use.
   `build_messages` also stopped appending the 7-membered-ring addon to the
   system message: it now appears only in the user message, for every family.
   As a result, OpenAI scores from before and after this edit are not strictly
   comparable. To go back, point the OpenAI entries at `USER_PROMPT_OPENAI` and
   restore `sys_prompt + addon`.

## Re-syncing

Only import lines and the [edits above](#edits-since-the-copy) differ from
DeepRetro, so a re-sync is a diff, followed by re-applying those edits:

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
