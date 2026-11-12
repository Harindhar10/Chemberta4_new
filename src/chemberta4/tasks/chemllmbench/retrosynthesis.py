# Copied from Chemberata4_ChemLLMBench @ a3c6447 (branch my-changes):
#   Retrosynthesis/data.py + Retrosynthesis/retro.py (lines 16-102)
# Only data paths and imports changed; see CHEMLLMBENCH.md.
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path

import os
from pathlib import Path
import random
import requests
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
# from .Prompts import RETROSYNTHESIS as RETRO_PROMPT



RETRO_PROMPT = """
You are an expert chemist. Given the product SMILES, your task is to predict the reactants SMILES using your experienced chemical Retrosynthesis knowledge. 
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the reactants. The output contains the reactants and reagents which are split by '.'. The reactants smiles must be valid and chemically reasonable. Wrap your response in JSON: {"answer": "<your answer>"}.
"""

# ----------------------------- config -----------------------------
SEED = 42
SAMPLE_SIZE = 100
_DATA_DIR = DATA_DIR
TEST_SET_CSV = str(_DATA_DIR / "uspto50k_retro_test.csv")   # if this file exists it is used as the test set

random.seed(SEED)

SHARED_NAME = "7eci3nd9vy0xplqniitpk02rbg9q2zcq"
FILES = {"retrosynthesis": "854847820319"}

MAPPING_COL_NAMES = {
    "reactants": "reactants_smiles",
    "products": "products_smiles",
}

INPUT_COL = "products_smiles"     # what the model is given
TARGET_COL = "reactants_smiles"   # what the model must predict

INPUT_LABEL = "Product"
TARGET_LABEL = "Reactants"

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ----------------------------- download / load -----------------------------
def download_box_file(file_id, out_path):
    """Download a file from a public Box shared link (atomic, cached)."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Using cached file: {out_path}")
        return out_path

    url = "https://az.app.box.com/index.php"
    params = {
        "rm": "box_download_shared_file",
        "shared_name": SHARED_NAME,
        "file_id": f"f_{file_id}",
    }
    tmp_path = out_path + ".part"

    with requests.get(url, params=params, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError("Got an HTML page instead of the file. "
                               "Download it manually from the browser instead.")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    os.replace(tmp_path, out_path)
    print(f"Downloaded: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def load_raw_dataset(name):
    path = download_box_file(FILES[name], box_pickle_path(name))
    obj = pd.read_pickle(path)
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


# ----------------------------- preprocessing (in place) -----------------------------
def _safe_smiles(m):
    return Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else None


def convert_all_mols(df, mol_cols=("reactants_mol", "products_mol")):
    """Mol -> canonical SMILES for all columns, then drop the Mol columns."""
    for col in mol_cols:
        df[col.removesuffix("_mol")] = df[col].map(_safe_smiles)
    df.drop(columns=list(mol_cols), inplace=True)


def drop_columns(df, cols):
    df.drop(columns=list(cols), inplace=True)


def rename_columns(df, mapping):
    df.rename(columns=mapping, inplace=True)


def filter_invalid_smiles(df, cols=(INPUT_COL, TARGET_COL)):
    """Drop rows where any of the given SMILES columns is missing/invalid (in place)."""
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        mask &= np.fromiter(
            (isinstance(s, str) and Chem.MolFromSmiles(s) is not None for s in df[col]),
            dtype=bool, count=len(df),
        )
    df.drop(index=df.index[~mask], inplace=True)
    df.reset_index(drop=True, inplace=True)


def split_train_test(df, split_col="set", train_label="train", test_label="test",
                     drop_split_col=True):
    cols = [c for c in df.columns if not (drop_split_col and c == split_col)]
    labels = df[split_col]
    train_df = df.loc[labels == train_label, cols].reset_index(drop=True)
    test_df = df.loc[labels == test_label, cols].reset_index(drop=True)
    return train_df, test_df


def load_or_sample_test_set(test_df, csv_path=TEST_SET_CSV, n=SAMPLE_SIZE, seed=SEED):
    """Use the original test-set CSV if present, otherwise sample n rows (not saved)."""
    if os.path.exists(csv_path):
        print(f"Using existing test set: {csv_path}")
        return pd.read_csv(csv_path)

    n = min(n, len(test_df))
    print(f"No test set CSV found, sampling {n} rows randomly (seed={seed})")
    return test_df.sample(n=n, random_state=seed).reset_index(drop=True)


# ----------------------------- ICL example sampling -----------------------------
def _format_examples(rows):
    """List of (products_smiles, reactants_smiles) tuples."""
    return list(zip(rows[INPUT_COL], rows[TARGET_COL]))


def _random_sample_examples(df_train, k):
    """k random examples, shared by every test molecule."""
    return _format_examples(df_train.sample(k, random_state=SEED))


def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    return _fp_gen.GetFingerprint(scaffold)


def _scaffold_sample_examples(df_train, train_fps, test_smiles, k):
    """k train examples whose product scaffolds are most similar to the test product."""
    test_fp = _scaffold_fingerprint(test_smiles)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
    top_k = np.argsort(sims)[::-1][:k]
    return _format_examples(df_train.iloc[top_k])


# ----------------------------- prompt building -----------------------------
def _build_system_prompt(examples):
    """Base prompt plus an examples block."""
    system_prompt = RETRO_PROMPT
    if not examples:
        return system_prompt
    system_prompt += "\n" + "\n\n".join(
        f"{INPUT_LABEL}: {inp}\n{TARGET_LABEL}: {tgt}" for inp, tgt in examples
    )
    return system_prompt


def _create_prompt(sample_strategy, df_train, df_test, k):
    """Returns (system_prompts, user_prompts), one entry per test molecule."""
    test_smiles = df_test[INPUT_COL].tolist()
    user_prompts = [f"Products: {s}\nReactants:" for s in test_smiles]

    # zero-shot
    if k == 0:
        return [_build_system_prompt(None)] * len(test_smiles), user_prompts

    # random: one shared system prompt
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(_random_sample_examples(df_train, k))
        return [system_prompt] * len(test_smiles), user_prompts

    # scaffold: a different system prompt per test molecule
    if sample_strategy == "scaffold":
        train_fps = [_scaffold_fingerprint(s) for s in df_train[INPUT_COL]]  # computed once
        system_prompts = [
            _build_system_prompt(_scaffold_sample_examples(df_train, train_fps, s, k))
            for s in test_smiles
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")


# ----------------------------- main entry point -----------------------------
def load_retro_dataset(sample_strategy="random", k=5, name="retrosynthesis"):
    """Returns (system_prompts, user_prompts, df_test).

    sample_strategy: 'random' or 'scaffold' (ignored when k == 0)
    k: number of in-context examples (0 = zero-shot)
    """
    df = load_raw_dataset(name)
    drop_columns(df, ["reaction_type"])
    convert_all_mols(df)
    rename_columns(df, MAPPING_COL_NAMES)
    filter_invalid_smiles(df)

    train_df, test_df = split_train_test(df)
    df_test = load_or_sample_test_set(test_df)

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy, df_train=train_df, df_test=df_test, k=k
    )
    return system_prompts, user_prompts, df_test


# ── Runner section (parsing, normalisation, metrics) ────────────────────────
import re
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings for invalid SMILES


# ── Parsing ─────────────────────────────────────────────────────────────────
def _clean(text):
    """Strip markdown fences, quotes and backticks."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip().strip("`'\" \n")


def _parse_prediction(raw_text):
    """Extract the predicted reactant SMILES (a single whitespace-free token)."""
    if not raw_text:
        return ""
    text = _clean(raw_text)
    text = strip_thinking(text)
    if not text:
        return ""
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    # Preferred: text after the target label, e.g. "Reactants: CCO.CC(=O)O"
    match = re.search(rf"{re.escape(TARGET_LABEL)}\s*:\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip("`'\"")

    # Fallback: model ignored the label -> first token of first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0].strip("`'\"")
    return ""


# ── Validity / normalisation ────────────────────────────────────────────────
def _is_invalid_prediction(smiles):
    """True if the string is empty or any '.'-separated fragment fails to parse."""
    if not smiles:
        return True
    for frag in smiles.split("."):
        if not frag or Chem.MolFromSmiles(frag) is None:
            return True
    return False


def _normalise(smiles):
    """Canonicalise every fragment, sort them, and re-join with '.'.

    Returns "" if the string is empty or any fragment is invalid, so an invalid
    prediction can never match a reference.
    """
    if not smiles:
        return ""
    canon = []
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag) if frag else None
        if mol is None:
            return ""
        canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon))


# ── Metrics ─────────────────────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred):
    """Top-1 accuracy and percentage of invalid predicted SMILES.

    y_true: list of reference reactant SMILES
    y_pred: list of parsed predicted reactant SMILES
    """
    n = len(y_true)
    num_invalid = sum(_is_invalid_prediction(p) for p in y_pred)

    norm_true = [_normalise(t) for t in y_true]
    norm_pred = [_normalise(p) for p in y_pred]
    num_correct = sum(1 for t, p in zip(norm_true, norm_pred) if p and p == t)

    return {
        "top1_accuracy": num_correct / n if n else 0.0,
        "invalid_smiles_percent": 100.0 * num_invalid / n if n else 0.0,
        "correct": num_correct,
        "invalid": num_invalid,
        "total": n,
    }


# ── chemberta4 adapter (not in the reference) ───────────────────────────────
# What run.py calls: the prompts and references exactly as the reference
# run_benchmark built them, its parser, and its metrics.
SUBTASKS = ("retrosynthesis",)


def load(subtask, strategy, k):
    """(system_prompts, user_prompts, y_true).

    Zero-shot prompts only need the shipped test CSV, so the training pickle is
    not downloaded for them; the prompts are identical either way, because
    _create_prompt never reads the training set when k == 0.
    """
    if k == 0 and os.path.exists(TEST_SET_CSV):
        df_test = pd.read_csv(TEST_SET_CSV)
        system_prompts, user_prompts = _create_prompt(
            sample_strategy=strategy, df_train=None, df_test=df_test, k=0
        )
    else:
        system_prompts, user_prompts, df_test = load_retro_dataset(sample_strategy=strategy, k=k)
    return system_prompts, user_prompts, df_test[TARGET_COL].astype(str).tolist()


def parse(raw, subtask):
    return _parse_prediction(raw)


def score(y_true, y_pred, subtask):
    return _compute_metrics(y_true, y_pred)
