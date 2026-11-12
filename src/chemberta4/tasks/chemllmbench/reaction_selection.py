# Copied from Chemberata4_ChemLLMBench @ a3c6447 (branch my-changes):
#   Reaction_Selection/data.py + Reaction_Selection/selection.py (lines 15-123)
# Only data paths and imports changed; see CHEMLLMBENCH.md.
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path

import ast
import pandas as pd


LIGAND_CSV = "ligand_sample.csv"
REAGENT_CSV = "reagent_sample.csv"
SOLVENT_CSV = "solvent_sample.csv"


TASK_MAP = {
    "Reactant_selection":REAGENT_CSV,
    "Ligand_selection":LIGAND_CSV,
    "Solvent_selection":SOLVENT_CSV
}

def _load_prompts(df):
    system_prompts = []
    user_prompts = []

    for task in df["task"]:
        # Everything before the first actual example
        split_marker = "Given the rest of reaction components:"

        system_prompt, user_prompt = task.split(split_marker, 1)

        system_prompts.append(system_prompt.strip() + " Wrap your response in JSON: {\"answer\": \"<your answer>\"}.")

        # Add the marker back because it is part of the actual task
        user_prompts.append(
            split_marker + user_prompt
        )

    return system_prompts, user_prompts


def _create_ground_truth(df):
    df["yield_details"] = df["yield_details"].apply(ast.literal_eval)

    df["ground_truth"] = df["yield_details"].apply(
        lambda x: max(x, key=lambda y: y[0][1])[0][0]
    )


def load_dataset(name):
    df = pd.read_csv(name)

    _create_ground_truth(df)

    system_prompts, user_prompts = _load_prompts(df)

    return system_prompts, user_prompts, df


# ── Runner section (parsing, normalisation, metrics) ────────────────────────
import re
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

TASKS = ["Reactant_selection", "Ligand_selection", "Solvent_selection"]

# Label the model is asked to answer with (used for parsing)
LABEL_MAP = {
    "Reactant_selection": "Optimal reactant",
    "Ligand_selection": "Optimal ligand",
    "Solvent_selection": "Optimal solvent",
}

# Any component label a model might prefix its answer with
_COMPONENT = r"(?:reactant|reagent|ligand|solvent|base)s?"


# ── Parsing ─────────────────────────────────────────────────────────────────
def _clean(text):
    """Strip markdown fences, bold markers, quotes and backticks."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("**", "")
    return text.strip().strip("`'\" \n")


def _strip_token(tok):
    """Trim wrapping quotes/backticks and trailing punctuation from a token."""
    tok = tok.strip("`'\",;")
    # a SMILES never ends with '.', so a trailing period is sentence punctuation
    return tok.rstrip(".").strip("`'\",;")


def _parse_prediction(raw_text, task):
    """Extract the predicted SMILES string from the model's raw output.

    Handles, e.g.:
        "Optimal reactant: Brc1ccc2ncccc2c1"
        "Reactant: Brc1ccc2ncccc2c1"        (label differs from the template)
        "Reactant:Brc1ccc2ncccc2c1"         (no space after the colon)
        "The optimal solvent is: CO."       (sentence form)
        "Brc1ccc2ncccc2c1"                  (no label at all)
        "Reactant: X\\nExplanation ..."      (extra text after the answer)
    """
    if not raw_text:
        return ""
    text = _clean(str(raw_text))
    if not text:
        return ""

    text = strip_thinking(text)
    if not text:
        return ""
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    # 1) Task-specific label, e.g. "Optimal solvent: CO"
    label = re.escape(LABEL_MAP[task].strip())
    m = re.search(rf"{label}s?\s*(?:is\s*)?[:=]\s*(\S+)", text, re.IGNORECASE)
    if m:
        return _strip_token(m.group(1))

    # 2) Generic "Optimal <anything>: X"
    m = re.search(r"optimal\s+\w+\s*(?:is\s*)?[:=]\s*(\S+)", text, re.IGNORECASE)
    if m:
        return _strip_token(m.group(1))

    # 3) Any component label at the start of a line: "Reactant: X", "Solvent:X"
    m = re.search(rf"^\s*{_COMPONENT}\s*(?:is\s*)?[:=]\s*(\S+)",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        return _strip_token(m.group(1))

    # 4) No label: first token of the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _strip_token(line.split()[0])
    return ""


# ── Normalisation ───────────────────────────────────────────────────────────
def _normalise(smiles):
    """Canonical SMILES if RDKit can parse it, else the stripped raw string
    (some organometallic ligands fail RDKit sanitisation)."""
    if not smiles:
        return ""
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


# ── Metrics ─────────────────────────────────────────────────────────────────
def _compute_accuracy(y_true, y_pred):
    """Accuracy = fraction of predictions matching the ground truth."""
    n = len(y_true)
    correct_flags = [
        bool(p) and p == t
        for t, p in zip((_normalise(t) for t in y_true),
                        (_normalise(p) for p in y_pred))
    ]
    num_correct = sum(correct_flags)
    metrics = {
        "accuracy": num_correct / n if n else 0.0,
        "correct": num_correct,
        "total": n,
    }
    return metrics, correct_flags


# ── chemberta4 adapter (not in the reference) ───────────────────────────────
# What run.py calls: the prompts and references exactly as the reference
# run_benchmark built them, its parser, and its metrics.
SUBTASKS = TASKS


def load(subtask, strategy=None, k=0):
    """(system_prompts, user_prompts, y_true). Zero-shot only: strategy and k are ignored."""
    system_prompts, user_prompts, df = load_dataset(str(DATA_DIR / TASK_MAP[subtask]))
    df = df.reset_index(drop=True)
    return system_prompts, user_prompts, df["ground_truth"].astype(str).tolist()


def parse(raw, subtask):
    return _parse_prediction(raw, subtask)


def score(y_true, y_pred, subtask):
    return _compute_accuracy(y_true, y_pred)[0]
