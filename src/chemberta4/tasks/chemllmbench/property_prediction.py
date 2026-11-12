# Copied from Chemberata4_ChemLLMBench @ a3c6447 (branch my-changes):
#   Property_Prediction/data.py + Property_Prediction/Property_prediction.py (lines 43-114)
# Only data paths and imports changed; see CHEMLLMBENCH.md.
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path

BBBP_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. \nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically penetration/non-penetration to the brain-blood barrier, based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether it has penetrative property (Yes) or not (No). The task is to predict the binary label for a given molecule, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"


BACE_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing wether it can inhibit(Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit(No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. You will be provided with task template, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}. \n"


Tox21_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is toxic(Yes) or Not toxic(No), based on the SMILES string representation of each molecule. A template will be provided. The task is to predict the binary label for a given molecule SMILES, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"

HIV_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically inhibit of HIV replication based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether a molecule can inhibit (Yes) or cannot inhibit (No) HIV replication. The task is to precisely predict the binary label for a given molecule, considering its properties and its potential to impede HIV replication, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"

ClinTox_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule.. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"












import random
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

SEED = 42
SAMPLE_SIZE  = 100
random.seed(SEED)

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
Tox21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
HIV_URL =   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
ClinTox_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"


DATASETS = {
    "BBBP": BBBP_URL,
    "BACE": BACE_URL,
    "Tox21": Tox21_URL,
    "HIV": HIV_URL,
    "ClinTox": ClinTox_URL
}

PROMPTS = {
    "BBBP":BBBP_prompt,
    "BACE":BACE_prompt,
    "Tox21":Tox21_prompt,
    "HIV":HIV_prompt,
    "ClinTox":ClinTox_prompt
}

TASK_DESCRIPTION = {
    "BBBP":"Penetration",
    "BACE":"BACE-1 Inhibit",
    "Tox21":"Toxic",
    "HIV":"Inhibit",
    "ClinTox":"Clinically-trail-toxic"
}

# Per-dataset primary label column
LABEL_COL_MAP = {
    "BBBP": "p_np",
    "BACE": "Class",
    "Tox21": "Toxic",          # aggregated column created in _preprocess
    "HIV": "HIV_active",
    "ClinTox": "CT_TOX",
}

# # Tox21 assay columns used for aggregation
TOX21_ASSAY_COLS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
                    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _preprocess(df, name):
    """Dataset-specific preprocessing. Creates a unified label column where needed."""
    if name == "Tox21":
        # If ANY assay flags toxic (1), the molecule is toxic
        df["Toxic"] = (df[TOX21_ASSAY_COLS].fillna(0).max(axis=1) >= 1).astype(int)
    return df


def _rename_mol_column(df):
    """Renames the 'mol' column to 'smiles' in a pandas DataFrame."""
    return df.rename(columns={"mol": "smiles"})

def _filter_smiles(df):
    """Drop rows with invalid SMILES in place and return the same df."""
    mask = np.fromiter(
        (isinstance(s, str) and Chem.MolFromSmiles(s) is not None
         for s in df["smiles"]),
        dtype=bool,
        count=len(df),
    )
    df.drop(index=df.index[~mask], inplace=True)
    return df


def _format_examples(rows, name):
    """Turn df rows into a list of example tuples.
    - ClinTox: (smiles, label, fda_label)
    - Others:  (smiles, label)
    HIV's activity class (CI/CM/CA) is deliberately left out: it defines HIV_active.
    """
    label_col = LABEL_COL_MAP[name]
    labels = ["Yes" if c == 1 else "No" for c in rows[label_col]]

    if name == "ClinTox":
        fda_labels = ["Yes" if c == 1 else "No" for c in rows["FDA_APPROVED"]]
        return list(zip(rows["smiles"], labels, fda_labels))
    else:
        return list(zip(rows["smiles"], labels))


def _random_sample_examples(df, k, name):
    """k/2 positive + k/2 negative random examples (same for every test molecule)."""
    label_col = LABEL_COL_MAP[name]
    positive_examples = df[df[label_col] == 1].sample(int(k/2), random_state=SEED)
    negative_examples = df[df[label_col] == 0].sample(int(k/2), random_state=SEED)

    if name.lower() == "hiv":
        positive_examples = df[df[label_col] == 1].sample(int(0.2*k), random_state=SEED)
        negative_examples = df[df[label_col] == 0].sample(int(0.8*k), random_state=SEED)

    return _format_examples(pd.concat([positive_examples, negative_examples]), name)


def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    return _fp_gen.GetFingerprint(scaffold)


def _scaffold_sample_examples(df_train, train_fps, test_smiles, k, name):
    """k training molecules whose scaffolds are most similar to the test molecule
    (ChemLLMBench-style scaffold sampling). Different for every test molecule."""
    test_fp = _scaffold_fingerprint(test_smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    top_k = np.argsort(similarities)[::-1][:k]
    return _format_examples(df_train.iloc[top_k], name)


def _build_system_prompt(name, examples):
    """Base prompt, plus a dataset-specific examples block."""
    system_prompt = PROMPTS[name]
    if not examples:
        return system_prompt

    if name == "ClinTox":
        system_prompt += "\n" + "\n".join(
            f"SMILES: {smiles}\nFDA-approved: {fda}\n{TASK_DESCRIPTION[name]}: {label}"
            for smiles, label, fda in examples
        )
    else:
        system_prompt += "\n" + "\n".join(
            f"SMILES: {smiles}\n{TASK_DESCRIPTION[name]}: {label}"
            for smiles, label in examples
        )
    return system_prompt


def _create_prompt(sample_strategy, df_train, df_test, name, k):
    """Returns (system_prompts, user_prompts), one entry per test molecule."""
    test_smiles = df_test["smiles"].tolist()

    # Build dataset-specific user prompts
    if name == "ClinTox":
        fda_labels = ["Yes" if c == 1 else "No" for c in df_test["FDA_APPROVED"]]
        user_prompts = [
            f"SMILES: {s}\nFDA-approved: {fda}\n{TASK_DESCRIPTION[name]}:"
            for s, fda in zip(test_smiles, fda_labels)
        ]
    else:
        user_prompts = [f"SMILES: {s}\n{TASK_DESCRIPTION[name]}:" for s in test_smiles]

    # zero-shot: base prompt only
    if k == 0:
        system_prompt = _build_system_prompt(name, examples=None)
        return [system_prompt] * len(test_smiles), user_prompts

    # random: same examples for all test molecules -> one shared system prompt
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(name, _random_sample_examples(df_train, k, name))
        return [system_prompt] * len(test_smiles), user_prompts

    # scaffold: examples depend on the test molecule -> system prompt per molecule
    if sample_strategy == "scaffold":
        train_fps = [_scaffold_fingerprint(s) for s in df_train["smiles"]]  # computed once
        system_prompts = [
            _build_system_prompt(name, _scaffold_sample_examples(df_train, train_fps, s, k, name))
            for s in test_smiles
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")


def load_dataset(name, sample_strategy, k):
    df = pd.read_csv(fetch_url(DATASETS[name]))


    if name.lower() == "bace":
        df = _rename_mol_column(df)

    _preprocess(df, name)
    _filter_smiles(df)

    df_test = df.sample(SAMPLE_SIZE, random_state=SEED)

    df.drop(df_test.index, inplace=True)
    df.reset_index(drop=True, inplace=True)   # so iloc/positions line up with train_fps

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy, df_train=df, df_test=df_test, name=name, k=k
    )

    return system_prompts, user_prompts, df_test


# ── Runner section (parsing, normalisation, metrics) ────────────────────────
import re
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from sklearn.metrics import accuracy_score, f1_score

# ── Supported datasets ──────────────────────────────────────────────────────
SUPPORTED_DATASETS = ["BBBP", "BACE", "Tox21", "HIV", "ClinTox"]

def _parse_prediction(raw_text, dataset_name):
    """Extract the first Yes/No answer relevant to the target task.
    Returns 1 (Yes), 0 (No), or -1 (parse failure)."""
    if not raw_text:
        return -1

    text = raw_text.strip()
    text = strip_thinking(text)
    if not text:
        return -1
    ans = extract_json_answer(text)

    if ans is not None:
        m = re.search(r"\b(yes|no)\b", ans.lower())
        if m:
            return 1 if m.group(1) == "yes" else 0

    task_key = TASK_DESCRIPTION[dataset_name]

    # Try to find "<TaskKey>: Yes/No" pattern first
    pattern = re.compile(
        rf"{re.escape(task_key)}\s*:\s*(Yes|No)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0

    # Fallback: first standalone Yes or No in the text
    fallback = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if fallback:
        return 1 if fallback.group(1).lower() == "yes" else 0

    return -1


# ── Helper: compute metrics ─────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred):
    """Compute standard classification metrics. Invalid predictions (-1) are
    treated as incorrect (mapped to the minority class complement)."""

    # Mask for valid predictions
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
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
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
    """(system_prompts, user_prompts, y_true) for one dataset; labels are 0/1."""
    system_prompts, user_prompts, df_test = load_dataset(name=subtask, sample_strategy=strategy, k=k)
    return system_prompts, user_prompts, df_test[LABEL_COL_MAP[subtask]].astype(int).tolist()


def parse(raw, subtask):
    return _parse_prediction(raw, subtask)


def score(y_true, y_pred, subtask):
    return _compute_metrics(y_true, y_pred)
