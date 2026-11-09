"""The single-step retrosynthesis test sets.

Each CSV has an ``input`` column (the product SMILES) and an ``output`` column
(the ground-truth reactants, dot-separated). Everything else is carried along
and ignored.

The sets themselves come from the DeepRetro repository, which built them from
USPTO-50k; ``prepare`` copies one in rather than re-deriving it, so a run here
scores the same molecules as the runs recorded there.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(
    os.getenv("CHEMBERTA4_RETRO_DATA") or REPO_ROOT / "data" / "processed" / "retrosynthesis"
)

# Where to copy from, and what each file is. The 250-molecule set is the one
# DeepRetro's own single-step evaluations use.
SOURCE_DIR = Path(
    os.getenv("DEEPRETRO_DATA_DIR") or Path.home() / "Documents" / "DFS" / "DFS" / "DeepRetro" / "data"
)
DATASETS = {
    "uspto_50k_test_250": "250 USPTO-50k products with reaction type and cluster id",
    "USPTO-50k_500": "500 USPTO-50k products",
    "USPTO-50k_100": "100 USPTO-50k products",
}
DEFAULT_DATASET = "uspto_50k_test_250"

REQUIRED_COLUMNS = ("input", "output")


def path(dataset: str = DEFAULT_DATASET) -> Path:
    """Where ``dataset`` lives once prepared."""
    return DATA_DIR / f"{dataset}.csv"


def load(dataset: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Read a test set, by name or by path, checking it has the columns we need.

    The index is the molecule number used in checkpoints and results, so it is
    left as the row position and never reset after filtering.
    """
    csv = Path(dataset) if Path(dataset).suffix == ".csv" else path(str(dataset))
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run: python scripts/prepare_data.py --task retrosynthesis"
        )
    df = pd.read_csv(csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} is missing column(s) {missing}; found {list(df.columns)}")
    return df


def prepare(datasets: list[str] | None = None, source_dir: Path | None = None) -> list[Path]:
    """Copy the test sets in from DeepRetro's ``data/`` directory."""
    source_dir = Path(source_dir or SOURCE_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name in datasets or [DEFAULT_DATASET]:
        src = source_dir / f"{name}.csv"
        if not src.exists():
            print(f"skipped {name}: {src} not found")
            continue
        dst = path(name)
        shutil.copyfile(src, dst)
        df = pd.read_csv(dst)
        print(f"{name}: {len(df)} molecules -> {dst}")
        written.append(dst)
    if not written:
        print(
            f"Nothing copied. Point DEEPRETRO_DATA_DIR at a checkout's data/ "
            f"directory (tried {source_dir})."
        )
    return written
