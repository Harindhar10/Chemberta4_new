"""Where the ChemLLMBench data lives, and how it gets there.

Two kinds of file:

- **Shipped**: the test sets and small training sets the reference repository
  keeps next to its code (``llm_test.csv``, the ``*_sample.csv`` selection
  sets, the USPTO test CSVs and the yield ``.npz`` files). They are committed
  under ``data/processed/chemllmbench/`` so a run scores exactly the rows the
  reference scored.
- **Downloaded**: the DeepChem property CSVs, ChEBI-20 and the USPTO training
  pickles. The reference fetched them at run time; here they are fetched once
  into ``data/raw/chemllmbench/`` (gitignored) and read from there afterwards.
  ``python scripts/prepare_data.py --task chemllmbench`` fetches them up front;
  otherwise the first run that needs one fetches it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(
    os.getenv("CHEMBERTA4_CHEMLLMBENCH_DATA") or REPO_ROOT / "data" / "processed" / "chemllmbench"
)
RAW_DIR = Path(
    os.getenv("CHEMBERTA4_CHEMLLMBENCH_RAW") or REPO_ROOT / "data" / "raw" / "chemllmbench"
)

# The reference checkout, used only by ``prepare(copy_shipped=True)`` to
# re-copy the shipped files.
SOURCE_DIR = Path(
    os.getenv("CHEMLLMBENCH_SRC_DIR")
    or Path.home() / "Documents" / "DFS" / "DFS" / "Chemberata4_ChemLLMBench"
)

# Shipped file -> its path in the reference checkout.
SHIPPED = {
    "llm_test.csv": "Name_Prediction/llm_test.csv",
    "reagent_sample.csv": "Reaction_Selection/reagent_sample.csv",
    "ligand_sample.csv": "Reaction_Selection/ligand_sample.csv",
    "solvent_sample.csv": "Reaction_Selection/solvent_sample.csv",
    "uspto_test.csv": "Reaction_prediction/uspto_test.csv",
    "uspto50k_retro_test.csv": "Retrosynthesis/uspto50k_retro_test.csv",
    "BH_dataset.npz": "Yield_prediction/BH_dataset .npz",  # the space is dropped here
    "BH_sample_100_test.npz": "Yield_prediction/BH_sample_100_test.npz",
    "Suzuki.npz": "Yield_prediction/Suzuki.npz",
    "SU_sample_100_test.npz": "Yield_prediction/SU_sample_100_test.npz",
}


def url_cache_path(url: str) -> Path:
    """Local path a URL is cached at: its file name plus a short hash of the URL.

    >>> p = url_cache_path("https://x.org/a/tox21.csv.gz")
    >>> p.name.startswith("tox21_"), p.name.endswith(".csv.gz"), p.parent == RAW_DIR
    (True, True, True)
    """
    name = Path(urlparse(url).path).name or "download"
    stem, dot, ext = name.partition(".")
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    return RAW_DIR / f"{stem}_{digest}{dot}{ext}"


def fetch_url(url: str) -> str:
    """Download ``url`` once into ``RAW_DIR`` and return the local path.

    The copied loaders called ``pd.read_csv(url)`` on every run; they now read
    this local copy instead. The write is atomic, so an interrupted download
    never leaves a truncated file that later runs would trust.
    """
    path = url_cache_path(url)
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp, path)
    print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)


def box_pickle_path(name: str) -> str:
    """Where the USPTO training pickle ``name`` is cached.

    The reference wrote it to ``<name>.pkl`` in the current directory.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return str(RAW_DIR / f"{name}.pkl")


def prepare(copy_shipped: bool = False, families: list[str] | None = None) -> None:
    """Fetch every downloaded file, so later runs need no network.

    Parameters
    ----------
    copy_shipped:
        Also re-copy the committed files from the reference checkout
        (``CHEMLLMBENCH_SRC_DIR``). Only needed after the reference changes.
    families:
        Restrict downloads to these families (module names, e.g.
        ``property_prediction``). Default: all of them.
    """
    if copy_shipped:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for dst, src in SHIPPED.items():
            source = SOURCE_DIR / src
            if not source.exists():
                print(f"skipped {dst}: {source} not found")
                continue
            shutil.copyfile(source, DATA_DIR / dst)
            print(f"copied {source} -> {DATA_DIR / dst}")

    missing = [f for f in SHIPPED if not (DATA_DIR / f).exists()]
    if missing:
        print(f"missing shipped files in {DATA_DIR}: {missing} (re-run with --copy-shipped)")

    from chemberta4.tasks.chemllmbench import molecule_design, property_prediction
    from chemberta4.tasks.chemllmbench import reaction_prediction, retrosynthesis

    wanted = set(families or ["property_prediction", "molecule_design",
                              "reaction_prediction", "retrosynthesis"])
    if "property_prediction" in wanted:
        for url in property_prediction.DATASETS.values():
            fetch_url(url)
    if "molecule_design" in wanted:
        for url in molecule_design.FILES.values():
            fetch_url(url)
    # The USPTO pickles are only read for few-shot prompts; zero-shot runs use
    # the shipped test CSVs alone.
    for module in (reaction_prediction, retrosynthesis):
        if module.__name__.rsplit(".", 1)[-1] not in wanted:
            continue
        for name, file_id in module.FILES.items():
            try:
                module.download_box_file(file_id, box_pickle_path(name))
            except Exception as exc:  # Box sometimes serves an HTML page instead
                print(f"could not fetch the {name} training pickle: {exc}")
