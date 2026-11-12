# Copied from Chemberata4_ChemLLMBench @ a3c6447 (branch my-changes):
#   Yield_prediction/data.py + Yield_prediction/yield_pred.py (lines 20-81)
# Only data paths and imports changed; see CHEMLLMBENCH.md.
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path

from warnings import warn
import pandas as pd
import random
import numpy as np
import os
from pathlib import Path
# ---------------------------------------------------------------------------
# Constants & Prompts
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

BH_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Buchwald-Hartwig reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Buchwald-Hartwig reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
In comparison to the traditional Buchwald-Hartwig reaction, this study involved the addition of additives and aimed to evaluate the effects of isoxazole agment additives as Glorius fragment additives on the amination of various aryl and heteroaryl halides. 
You will be provided with several examples molecules. Please answer with only Yes or No. Wrap your response in JSON: {"answer": "<Yes/No>"}.
"""

SUZUKI_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Suzuki-Miyaura reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Suzuki-Miyaura reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
You will be provided with several examples molecules. Please answer with only Yes or No. Wrap your response in JSON: {"answer": "<Yes/No>"}.
"""

# Local file paths (assumes the files are in the same folder as this script)

URL = DATA_DIR

BH_TRAIN_FILE = URL/"BH_dataset.npz"
BH_TEST_FILE = URL/ "BH_sample_100_test.npz"

SUZUKI_TRAIN_FILE = URL/"Suzuki.npz"
SUZUKI_TEST_FILE = URL/"SU_sample_100_test.npz"

SUZUKI_MAP = {
    "train": SUZUKI_TRAIN_FILE,
    "test": SUZUKI_TEST_FILE,
    "prompt": SUZUKI_PROMPT
}

BH_MAP = {
    "train": BH_TRAIN_FILE,
    "test": BH_TEST_FILE,
    "prompt": BH_PROMPT
}

TASK_MAP = {
    "Buchwald-Hartwig": BH_MAP,
    "Suzuki-Miyaura": SUZUKI_MAP
}

# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def _extract_npz_to_df(file_path, split):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find '{file_path}'. Please make sure you downloaded it to the same directory as this script.")
        
    npz = np.load(file_path, allow_pickle=True)

    if split == "train":
        if "data" in npz.files:  # Buchwald-Hartwig format
            data = npz['data'][2]
            df = pd.DataFrame(data)
            df.rename(columns={"yld": "yeild", "rsmi": "reaction"}, inplace=True)
            cols = list(df.columns)
            cols[0], cols[1] = cols[1], cols[0]
            df = df[cols]
        elif "data_df" in npz.files:  # Suzuki-Miyaura format: (reaction, numeric yield)
            data = npz['data_df']
            df = pd.DataFrame({"reaction": data[:, 0], "yeild": data[:, 1].astype(float)})
        else:
            raise KeyError(f"'{file_path}' has neither 'data' nor 'data_df'; found {npz.files}")
        df['yeild'] = np.where(df['yeild'] > 70, 'Yes', 'No')
        return df
    else:
        data = npz['data_df']
        df = pd.DataFrame({
            "reaction": data[:, 0],
            "yeild": data[:, 1]
        })
        return df

# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _format_examples(rows):
    """Return a list of (reaction, yeild) tuples."""
    return list(zip(rows["reaction"], rows["yeild"]))

def _random_sample_examples(df, k):
    """Randomly sample *k* training examples for in-context learning."""
    sampled = df.sample(k, random_state=SEED)
    return _format_examples(sampled)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(base_prompt, examples):
    """Combine the base task prompt with formatted ICL examples."""
    system_prompt = base_prompt.strip()
    if not examples:
        return system_prompt

    system_prompt += "\n\n" + "\n\n".join(
        f"reaction: {inp}\nHigh yeilding: {out}"
        for inp, out in examples
    )
    return system_prompt

def _create_prompt(sample_strategy, df_train, df_test, base_prompt, k):
    """Build parallel lists of (system_prompts, user_prompts)."""
    test_inputs = df_test["reaction"].tolist()

    # User prompt: present the input and ask for the output
    user_prompts = [f"reaction: {inp}\nyeild:" for inp in test_inputs]

    # Zero-shot — no examples
    if k == 0:
        system_prompt = _build_system_prompt(base_prompt, examples=None)
        return [system_prompt] * len(test_inputs), user_prompts

    # Random — same examples for every test molecule
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(
            base_prompt, _random_sample_examples(df_train, k)
        )
        return [system_prompt] * len(test_inputs), user_prompts
        
    else:
        warn(f"Only support random sampling strategy for now, falling back to random sampling")
        system_prompt = _build_system_prompt(
            base_prompt, _random_sample_examples(df_train, k)
        )
        return [system_prompt] * len(test_inputs), user_prompts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_function(name, sample_strategy="random", k=0):
    """
    Load the Yield-Prediction dataset and return prompts + test data.

    Parameters
    ----------
    name : str
        Task name — either 'Buchwald-Hartwig' or 'Suzuki-Miyaura'.
    sample_strategy : str, default "random"
        How to choose ICL examples. Only "random" is supported.
    k : int, default 0
        Number of in-context examples (0 = zero-shot).

    Returns
    -------
    system_prompts : list[str]
    user_prompts   : list[str]
    df_test        : pd.DataFrame
    """
    if name not in TASK_MAP:
        raise ValueError(
            f"Unknown task '{name}'. Choose from {list(TASK_MAP.keys())}"
        )

    task_config = TASK_MAP[name]

    # Load from local dataframes
    df_train = _extract_npz_to_df(task_config["train"], "train")
    df_test = _extract_npz_to_df(task_config["test"], "test")

    # Generate prompts
    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy,
        df_train=df_train,
        df_test=df_test,
        base_prompt=task_config["prompt"],
        k=k,
    )

    return system_prompts, user_prompts, df_test


# ── Runner section (parsing, normalisation, metrics) ────────────────────────
import re
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from sklearn.metrics import accuracy_score

SUPPORTED_DATASETS = ["Buchwald-Hartwig", "Suzuki-Miyaura"]

def _parse_prediction(raw_text):
    """Extract the first Yes/No answer relevant to the target task.
    Returns 1 (Yes), 0 (No), or -1 (parse failure)."""
    if not raw_text:
        return -1

    text = raw_text.strip()
    text = strip_thinking(text)
    if not text:
        return -1
    ans = extract_json_answer(text, key="answer")

    if ans is not None:
        m = re.search(r"\b(yes|no)\b", ans.lower())
        if m:
            return 1 if m.group(1) == "yes" else 0

    # Try to find "yeild: Yes/No" or "yield: Yes/No" pattern first
    pattern = re.compile(
        r"y(?:ei|ie)ld\s*:\s*(Yes|No)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0

    # Fallback: first standalone Yes or No in the text
    fallback = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if fallback:
        return 1 if fallback.group(1).lower() == "yes" else 0

    return -1


def _compute_metrics(y_true, y_pred):
    valid = np.array(y_pred) != -1
    total = len(y_true)
    n_valid = int(valid.sum())
    n_invalid = total - n_valid

    # For metric computation, replace invalid predictions with wrong answer
    y_pred_clean = []
    for yt, yp in zip(y_true, y_pred):
        if yp == -1:
            y_pred_clean.append(1 - yt)  # guaranteed wrong
        else:
            y_pred_clean.append(yp)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_clean)

    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "total": total,
        "valid_predictions": n_valid,
        "invalid_predictions": n_invalid,
    }

    return metrics


# ── chemberta4 adapter (not in the reference) ───────────────────────────────
# What run.py calls: the prompts and references exactly as the reference
# run_benchmark built them, its parser, and its metrics.
SUBTASKS = SUPPORTED_DATASETS


def load(subtask, strategy, k):
    """(system_prompts, user_prompts, y_true); labels are 1 (high yield) or 0."""
    system_prompts, user_prompts, df_test = load_function(name=subtask, sample_strategy=strategy, k=k)
    return system_prompts, user_prompts, df_test["yeild"].map({"Yes": 1, "No": 0}).tolist()


def parse(raw, subtask):
    return _parse_prediction(raw)


def score(y_true, y_pred, subtask):
    return _compute_metrics(y_true, y_pred)
