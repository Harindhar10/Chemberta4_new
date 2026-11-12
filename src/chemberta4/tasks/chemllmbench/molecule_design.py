# Copied from Chemberata4_ChemLLMBench @ a3c6447 (branch my-changes):
#   Molecule_Design/data.py + Molecule_Design/Mol_design_and_mol_caption.py (lines 18-180)
# Only data paths and imports changed; see CHEMLLMBENCH.md.
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path

MOL_CAPTION = """
You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n\
Please strictly follow the format, no other information can be provided. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

MOL_DESIGN = """
"You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES \
string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""











import pandas as pd
import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

FILES = {
    "train": "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data/train.txt",
    "test": "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data/test.txt"
}

TASK_COL_MAP = {
    "MOL_CAPTION": ("smiles", "description"),
    "MOL_DESIGN":  ("description", "smiles"),
}

PROMPTS = {
    "MOL_CAPTION": MOL_CAPTION,
    "MOL_DESIGN":  MOL_DESIGN,
}

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _canonicalise_smiles(df):
    """Convert all SMILES to canonical form, dropping invalid entries."""
    df.dropna(subset=["smiles"], inplace=True)
    
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df.dropna(subset=["mol"], inplace=True)
    
    df["smiles"] = df["mol"].apply(Chem.MolToSmiles)
    df.drop(columns=["mol"], inplace=True)
    
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Example formatting
# ---------------------------------------------------------------------------

def _format_examples(rows, name):
    """Return a list of (input_value, output_value) tuples for the given task."""
    from_col, to_col = TASK_COL_MAP[name]
    return list(zip(rows[from_col], rows[to_col]))


def _random_sample_examples(df, k, name):
    """Randomly sample *k* training examples for in-context learning."""
    sampled = df.sample(k, random_state=SEED)
    return _format_examples(sampled, name)

def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return _fp_gen.GetFingerprint(scaffold)

def _scaffold_sample_examples(df_train, train_fps, test_smiles, k, name):
    """Select the *k* training molecules whose scaffolds are most
    Tanimoto-similar to the test molecule's scaffold."""
    test_fp = _scaffold_fingerprint(test_smiles)
    
    # Filter out any None fingerprints in case of scaffold errors
    valid_indices = [i for i, fp in enumerate(train_fps) if fp is not None]
    valid_fps = [train_fps[i] for i in valid_indices]
    
    if test_fp is None or not valid_fps:
        # Fallback to random if scaffold extraction fails
        return _random_sample_examples(df_train, k, name)
        
    similarities = DataStructs.BulkTanimotoSimilarity(test_fp, valid_fps)
    top_k_valid_idx = np.argsort(similarities)[::-1][:k]
    top_k_original_idx = [valid_indices[i] for i in top_k_valid_idx]
    
    return _format_examples(df_train.iloc[top_k_original_idx], name)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(name, examples):
    """Combine the base task prompt with formatted ICL examples."""
    system_prompt = PROMPTS[name]
    if not examples:
        return system_prompt

    from_col, to_col = TASK_COL_MAP[name]
    system_prompt += "\n\n" + "\n\n".join(
        f"{from_col}: {inp}\n{to_col}: {out}"
        for inp, out in examples
    )
    return system_prompt

def _create_prompt(sample_strategy, df_train, df_test, name, k):
    """Build parallel lists of (system_prompts, user_prompts)."""
    from_col, to_col = TASK_COL_MAP[name]
    test_inputs = df_test[from_col].tolist()

    # User prompt: present the input and ask for the output
    user_prompts = [f"{from_col}: {inp}\n{to_col}:" for inp in test_inputs]

    # Zero-shot — no examples
    if k == 0:
        system_prompt = _build_system_prompt(name, examples=None)
        return [system_prompt] * len(test_inputs), user_prompts

    # Random — same examples for every test molecule
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(
            name, _random_sample_examples(df_train, k, name)
        )
        return [system_prompt] * len(test_inputs), user_prompts

    # Scaffold — examples depend on the test molecule
    if sample_strategy == "scaffold":
        if from_col != "smiles":
            raise ValueError(
                f"Scaffold sampling requires a SMILES input column, "
                f"but task '{name}' uses '{from_col}'. "
                f"Use 'random' sampling instead."
            )
            
        train_fps = [_scaffold_fingerprint(s) for s in df_train["smiles"]]
        system_prompts = [
            _build_system_prompt(
                name,
                _scaffold_sample_examples(df_train, train_fps, s, k, name),
            )
            for s in test_inputs
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(name, sample_strategy="random", k=0):
    """Load the ChEBI-20 dataset and return prompts + test data."""
    if name not in TASK_COL_MAP:
        raise ValueError(
            f"Unknown task '{name}'. Choose from {list(TASK_COL_MAP.keys())}"
        )

    # 1) Load train, val, test directly into DataFrames
    dfs = {}
    for split, url in FILES.items():
        df = pd.read_csv(fetch_url(url), sep='\t')
        # Standardize column names to match TASK_COL_MAP
        df.rename(columns={"SMILES": "smiles", "CID": "cid"}, inplace=True)
        
        # Filter invalid SMILES and canonicalise
        dfs[split] = _canonicalise_smiles(df)

    df_train = dfs["train"]
    df_test = dfs["test"]

    if len(df_test) >= 100:
        df_test = df_test.sample(n=100, random_state=SEED).reset_index(drop=True)
    else:
        print(f"Warning: Test set has only {len(df_test)} valid samples. Using all available.")

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy,
        df_train=df_train,
        df_test=df_test,
        name=name,
        k=k,
    )

    return system_prompts, user_prompts, df_test


# ── Runner section (parsing, normalisation, metrics) ────────────────────────
import re
from chemberta4.tasks.chemllmbench.data import DATA_DIR, fetch_url, box_pickle_path
from chemberta4.tasks.chemllmbench.parsing import extract_json_answer, strip_thinking
from rdkit.Chem import MACCSkeys

# ── Supported tasks ─────────────────────────────────────────────────────────
SUPPORTED_TASKS = list(TASK_COL_MAP.keys())

# ── Metric loaders (cached so each is loaded once) ──────────────────────────
_METRIC_CACHE = {}


def _get_metric(name):
    if name not in _METRIC_CACHE:
        import evaluate
        _METRIC_CACHE[name] = evaluate.load(name)
    return _METRIC_CACHE[name]


_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── Parsing ─────────────────────────────────────────────────────────────────
def _parse_prediction(raw_text, task_name):
    if not raw_text:
        return ""

    text = raw_text.strip()
    text = strip_thinking(text)
    if not text:
        return ""

    ans = extract_json_answer(text)
    if ans:
        return ans.strip()

    
    _, tgt_col = TASK_COL_MAP[task_name]
    match = None

    if task_name == "MOL_CAPTION":
        # captions can be multi-line -> capture everything after the label
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(.+)", text,
                          re.IGNORECASE | re.DOTALL)
    elif task_name == "MOL_DESIGN":
        # SMILES contain no whitespace -> first token only
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(\S+)", text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback: model ignored the label -> first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0] if task_name == "MOL_DESIGN" else line
    return ""


# ── Chemistry helpers ───────────────────────────────────────────────────────
def _fp_similarities(mol_a, mol_b):
    """Tanimoto similarity of (MACCS, RDK, Morgan) fingerprints."""
    maccs = DataStructs.TanimotoSimilarity(
        MACCSkeys.GenMACCSKeys(mol_a), MACCSkeys.GenMACCSKeys(mol_b))
    rdk = DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(mol_a), Chem.RDKFingerprint(mol_b))
    morgan = DataStructs.TanimotoSimilarity(
        _morgan_gen.GetFingerprint(mol_a), _morgan_gen.GetFingerprint(mol_b))
    return maccs, rdk, morgan


def _compute_fcd(valid_pred_smiles, true_smiles):
    """FCD between valid predictions and ground truth. NaN if unavailable."""
    try:
        from fcd import get_fcd , load_ref_model
        model = load_ref_model()
        if len(valid_pred_smiles) < 2 or len(true_smiles) < 2:
            return float("nan")
        return float(get_fcd(valid_pred_smiles, true_smiles,model))
    except Exception as e:
        print(f"[!] FCD could not be computed: {e}")
        return float("nan")


# ── Metrics ─────────────────────────────────────────────────────────────────
def _caption_metrics(y_true, y_pred):
    bleu = _get_metric("bleu")
    rouge = _get_metric("rouge")
    meteor = _get_metric("meteor")

    refs = [[t] for t in y_true]  # HF bleu expects a list of reference lists

    bleu2 = bleu.compute(predictions=y_pred, references=refs, max_order=2)["bleu"]
    bleu4 = bleu.compute(predictions=y_pred, references=refs, max_order=4)["bleu"]

    r = rouge.compute(predictions=y_pred, references=y_true, use_stemmer=True)
    m = meteor.compute(predictions=y_pred, references=y_true)["meteor"]

    return {
        "bleu2": float(bleu2),
        "bleu4": float(bleu4),
        "rouge1": float(r["rouge1"]),
        "rouge2": float(r["rouge2"]),
        "rougeL": float(r["rougeL"]),
        "meteor": float(m),
        "total": len(y_true),
    }


def _design_metrics(y_true, y_pred):
    n = len(y_true)

    # BLEU on character-level SMILES tokens
    bleu = _get_metric("bleu").compute(
        predictions=y_pred,
        references=[[t] for t in y_true],
        tokenizer=lambda s: list(s),
    )["bleu"]

    # Levenshtein over all samples (invalid/empty predictions still count)
    import Levenshtein
    lev = float(np.mean([Levenshtein.distance(t, p) for t, p in zip(y_true, y_pred)]))

    exact, valid = 0, 0
    maccs_s, rdk_s, morgan_s = [], [], []
    valid_pred_can, true_can = [], []

    for t, p in zip(y_true, y_pred):
        mol_t = Chem.MolFromSmiles(t) if t else None
        mol_p = Chem.MolFromSmiles(p) if p else None
        if mol_t is not None:
            true_can.append(Chem.MolToSmiles(mol_t))
        if mol_p is None:
            continue
        valid += 1
        valid_pred_can.append(Chem.MolToSmiles(mol_p))
        if mol_t is None:
            continue
        if Chem.MolToSmiles(mol_t) == Chem.MolToSmiles(mol_p):
            exact += 1
        m, r, mo = _fp_similarities(mol_t, mol_p)
        maccs_s.append(m)
        rdk_s.append(r)
        morgan_s.append(mo)

    mean_or_zero = lambda xs: float(np.mean(xs)) if xs else 0.0

    return {
        "bleu": float(bleu),
        "exact": exact / n,
        "levenshtein": lev,
        "validity": valid / n,
        "maccs_fts": mean_or_zero(maccs_s),   # averaged over valid predictions only
        "rdk_fts": mean_or_zero(rdk_s),
        "morgan_fts": mean_or_zero(morgan_s),
        "fcd": _compute_fcd(valid_pred_can, true_can),
        "valid_predictions": valid,
        "total": n,
    }


def _compute_metrics(y_true, y_pred, task_name):
    """y_true: list of references, y_pred: list of parsed predictions."""
    if task_name == "MOL_CAPTION":
        return _caption_metrics(y_true, y_pred)
    elif task_name == "MOL_DESIGN":
        return _design_metrics(y_true, y_pred)
    raise ValueError(f"No metrics defined for task: {task_name!r}")


# ── chemberta4 adapter (not in the reference) ───────────────────────────────
# What run.py calls: the prompts and references exactly as the reference
# run_benchmark built them, its parser, and its metrics.
SUBTASKS = SUPPORTED_TASKS


def load(subtask, strategy, k):
    """(system_prompts, user_prompts, y_true) for MOL_CAPTION or MOL_DESIGN."""
    system_prompts, user_prompts, df_test = load_dataset(name=subtask, sample_strategy=strategy, k=k)
    _, to_col = TASK_COL_MAP[subtask]
    return system_prompts, user_prompts, df_test[to_col].astype(str).tolist()


def parse(raw, subtask):
    return _parse_prediction(raw, subtask)


def score(y_true, y_pred, subtask):
    return _compute_metrics(y_true, y_pred, subtask)
