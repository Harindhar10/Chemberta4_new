"""Publish a finished run to a public dataset repo on the Hugging Face Hub.

Optional throughout, like the W&B reporting next to it. Without a token
(``--hf-token`` or ``HF_TOKEN``) nothing is pushed, and :func:`push_run` never
raises: a failed upload must never lose scores that are already on disk.

One repo holds every run. Each run's files go under ``<task>/<run_id>/`` and
the run gets two entries in the dataset card: a viewer config pointing at its
raw generations, and a section with its hyperparameters and metrics. A second
push of the same run replaces both rather than adding to them. Other runs'
entries are left as they were.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# Files from a run directory that are published. Anything else in it (stray
# notes, editor files) stays local.
RUN_FILES = (
    "results.jsonl",
    "generations.jsonl",
    "results.csv",
    "metrics.csv",
    "summary.json",
    "summary.txt",
    "perf.json",
    "run_config.json",
)
RAW_FILES = ("results.jsonl", "generations.jsonl")

CARD_TEMPLATE = """\
# chemberta4 evaluations

Raw generations and scores from evaluating open-source LLMs, served by vLLM, on
MoleculeNet property prediction, USPTO-50k single-step retrosynthesis and
ChemLLMBench.

Each run lives under `<task>/<run_id>/`:

- `results.jsonl` / `generations.jsonl`: one record per request, holding the
  model's raw output, any reasoning trace, and the parsed prediction.
- `metrics.csv` / `summary.json`: the scores.
- `run_config.json`: the evaluation hyperparameters.

Each run is also a dataset config:

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", "<task>--<run_id>")
```

## Runs
"""

_SECTION_RE = r"<!-- run:{key} -->\n.*?<!-- /run -->\n?"


# --------------------------------------------------------------------------
# Pure helpers: card editing and rendering, no network
# --------------------------------------------------------------------------
def split_card(text: str) -> tuple[dict[str, Any], str]:
    """Split a card into its YAML header and its markdown body.

    >>> split_card("---\\na: 1\\n---\\nbody\\n")
    ({'a': 1}, 'body\\n')
    >>> split_card("no header")
    ({}, 'no header')
    """
    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end():]


def join_card(header: dict[str, Any], body: str) -> str:
    """The inverse of :func:`split_card`."""
    return f"---\n{yaml.safe_dump(header, sort_keys=False)}---\n{body}"


def upsert_config(
    configs: list[dict[str, Any]], name: str, data_files: str
) -> list[dict[str, Any]]:
    """Add the viewer config ``name``, or replace the one already there.

    >>> upsert_config([{"config_name": "a", "data_files": "x"}], "a", "y")
    [{'config_name': 'a', 'data_files': 'y'}]
    >>> [c["config_name"] for c in upsert_config([{"config_name": "a"}], "b", "z")]
    ['a', 'b']
    """
    entry = {"config_name": name, "data_files": data_files}
    out = [c for c in configs if c.get("config_name") != name]
    for i, c in enumerate(configs):
        if c.get("config_name") == name:
            out.insert(i, entry)
            return out
    return out + [entry]


def upsert_section(body: str, key: str, text: str) -> str:
    """Replace the run section marked ``key`` in ``body``, or append it.

    >>> b = upsert_section("# t\\n", "a", "one")
    >>> b = upsert_section(b, "b", "two")
    >>> b = upsert_section(b, "a", "ONE")
    >>> print(b)
    # t
    <BLANKLINE>
    <!-- run:a -->
    ONE
    <!-- /run -->
    <BLANKLINE>
    <!-- run:b -->
    two
    <!-- /run -->
    <BLANKLINE>
    """
    block = f"<!-- run:{key} -->\n{text.strip()}\n<!-- /run -->\n"
    pattern = re.compile(_SECTION_RE.format(key=re.escape(key)), re.DOTALL)
    if pattern.search(body):
        return pattern.sub(lambda _: block, body, count=1)
    return body.rstrip("\n") + "\n\n" + block


def md_table(rows: Iterable[Sequence[Any]], columns: Sequence[str]) -> str:
    """A GitHub-flavoured markdown table.

    >>> print(md_table([("a", 1)], ["key", "value"]))
    | key | value |
    | --- | --- |
    | a | 1 |
    """

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def moleculenet_metrics_md(metrics: Any) -> str:
    """Mean and standard deviation across seeds for each (dataset, k).

    ``metrics`` is the frame :func:`evaluate_run` returns, one row per
    (dataset, k, seed). The grouping matches the W&B summary keys.
    """
    if metrics is None or metrics.empty:
        return "_No scored results._"
    grouped = metrics.groupby(["dataset", "k", "metric"], sort=True)
    rows = []
    for (dataset, k, metric), g in grouped:
        std = g["value"].std() if len(g) > 1 else None
        rows.append((
            dataset, int(k), metric, float(g["value"].mean()), std, len(g),
            float(g["parse_fail_rate"].mean()) if "parse_fail_rate" in g else None,
        ))
    return md_table(rows, ["dataset", "k", "metric", "mean", "std", "seeds", "parse fail rate"])


def chemllmbench_metrics_md(metrics: Any) -> str:
    """The headline metric of each (task, k, strategy) group.

    ``metrics`` is the long frame from the ChemLLMBench ``evaluate_run``; only
    the rows flagged ``primary`` are shown, with the parse failure rate.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"task": ["bbbp"] * 2, "k": [0, 0], "strategy": ["random"] * 2,
    ...     "metric": ["accuracy", "parse_fail_rate"], "value": [0.5, 0.1],
    ...     "primary": [True, False], "n": [10, 10]})
    >>> print(chemllmbench_metrics_md(df))
    | task | k | strategy | metric | value | n | parse fail rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | bbbp | 0 | random | accuracy | 0.5 | 10 | 0.1 |
    """
    if metrics is None or metrics.empty:
        return "_No scored results._"
    fail = {
        (r.task, r.k, r.strategy): r.value
        for r in metrics[metrics["metric"] == "parse_fail_rate"].itertuples()
    }
    rows = [
        (r.task, int(r.k), r.strategy, r.metric, r.value, int(r.n), fail.get((r.task, r.k, r.strategy)))
        for r in metrics[metrics["primary"]].itertuples()
    ]
    return md_table(rows, ["task", "k", "strategy", "metric", "value", "n", "parse fail rate"])


def retrosynthesis_metrics_md(summary: dict[str, Any]) -> str:
    """Top-1 and top-k accuracy (%) from the :func:`summarize` dict."""
    rows = [
        (view, s["all_correct"], s["any_correct"], s["maxfrag"])
        for view, s in (("top-1", summary["top1"]), ("top-k", summary["topk"]))
    ]
    table = md_table(rows, ["view", "all correct %", "any correct %", "MaxFrag %"])
    extra = md_table(
        [(
            summary.get("n_scored"),
            summary.get("parse_failures_pct"),
            summary.get("invalid_smiles_pct"),
            summary.get("mean_proposals"),
        )],
        ["molecules", "parse failures %", "invalid SMILES %", "mean proposals"],
    )
    return f"{table}\n\n{extra}"


def run_section(
    task: str, run_id: str, config: dict[str, Any], metrics_md: str, files: Sequence[str]
) -> str:
    """The card section for one run: hyperparameters, metrics, file links."""
    params = md_table(
        [(k, v) for k, v in config.items() if v is not None], ["hyperparameter", "value"]
    )
    links = ", ".join(f"[`{f}`]({task}/{run_id}/{f})" for f in files)
    return (
        f"### {task} / {run_id}\n\n"
        f"Config: `{config_name(task, run_id)}`. Files: {links}\n\n"
        f"<details><summary>Hyperparameters</summary>\n\n{params}\n\n</details>\n\n"
        f"{metrics_md}\n"
    )


def config_name(task: str, run_id: str) -> str:
    """The viewer config name for a run.

    >>> config_name("moleculenet", "Qwen3-8B")
    'moleculenet--Qwen3-8B'
    """
    return f"{task}--{run_id}"


def update_card(
    text: str | None,
    *,
    repo_id: str,
    task: str,
    run_id: str,
    raw_file: str | None,
    section: str,
) -> str:
    """The card ``text`` (``None`` for a new repo) with this run upserted."""
    header, body = split_card(text) if text else ({}, CARD_TEMPLATE.format(repo_id=repo_id))
    header.setdefault("pretty_name", "chemberta4 evaluations")
    header.setdefault("tags", ["chemistry", "llm-evaluation", "moleculenet", "retrosynthesis"])
    if raw_file:
        header["configs"] = upsert_config(
            header.get("configs") or [], config_name(task, run_id), f"{task}/{run_id}/{raw_file}"
        )
    return join_card(header, upsert_section(body, f"{task}/{run_id}", section))


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
def hub_token(token: str | None = None) -> str | None:
    """The HF token to use, or ``None`` when publishing is off."""
    return token or os.getenv("HF_TOKEN") or None


def check_access(repo_id: str, token: str) -> None:
    """Check the token and create the public repo before generating.

    Raises on a bad token or a repo the token cannot write to, so the run
    stops before it spends GPU time on results it could not publish.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"hub         : https://huggingface.co/datasets/{repo_id} (as {user})")


def push_run(
    *,
    repo_id: str,
    task: str,
    run_id: str,
    run_dir: str | Path,
    config: dict[str, Any],
    metrics_md: str,
    token: str,
    attempts: int = 3,
) -> bool:
    """Upload a run's files and its card section in one commit. Never raises.

    The commit is made against the revision the card was read from, so two
    runs pushing at once cannot silently drop each other's card edits: the
    loser's commit is rejected and it re-reads the card and tries again.
    """
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

        api = HfApi(token=token)
        run_dir = Path(run_dir)
        files = [f for f in RUN_FILES if (run_dir / f).exists()]
        raw_file = next((f for f in RAW_FILES if f in files), None)
        section = run_section(task, run_id, config, metrics_md, files)

        for attempt in range(1, attempts + 1):
            sha = api.repo_info(repo_id, repo_type="dataset").sha
            try:
                card_path = hf_hub_download(
                    repo_id, "README.md", repo_type="dataset", revision=sha, token=token
                )
                current = Path(card_path).read_text()
            except EntryNotFoundError:
                current = None
            card = update_card(
                current, repo_id=repo_id, task=task, run_id=run_id,
                raw_file=raw_file, section=section,
            )
            operations = [
                CommitOperationAdd(f"{task}/{run_id}/{f}", str(run_dir / f)) for f in files
            ] + [CommitOperationAdd("README.md", card.encode())]
            try:
                api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=f"{task}: {run_id}",
                    repo_type="dataset",
                    parent_commit=sha,
                )
                break
            except HfHubHTTPError as exc:
                if attempt == attempts:
                    raise
                print(f"hub push conflicted ({exc}); retrying {attempt}/{attempts - 1}")
        print(f"pushed to https://huggingface.co/datasets/{repo_id}/tree/main/{task}/{run_id}")
        return True
    except Exception as exc:
        print(f"Hub upload failed, scores are still on disk: {exc}")
        return False
