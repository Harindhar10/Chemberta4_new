#!/usr/bin/env python
"""Write the datasets an evaluation needs into ``data/processed/``.

    python scripts/prepare_data.py --task moleculenet
    python scripts/prepare_data.py --task moleculenet --datasets bbbp,hiv
    python scripts/prepare_data.py --task retrosynthesis
    python scripts/prepare_data.py --task chemllmbench

MoleculeNet builds DeepChem scaffold splits, which needs the ``score`` extra
(``pip install -e '.[score]'``). Retrosynthesis copies the USPTO-50k test sets
in from a DeepRetro checkout; point ``DEEPRETRO_DATA_DIR`` at its ``data/``
directory if it is not in the default place. ChemLLMBench downloads the files
its loaders fetch from the web (DeepChem CSVs, ChEBI-20, the USPTO training
pickles) into ``data/raw/chemllmbench/``; its test sets are already committed.

Both are idempotent: re-running overwrites with the same content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", required=True, choices=["moleculenet", "retrosynthesis", "chemllmbench"])
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated subset; default is every dataset the task knows",
    )
    parser.add_argument("--data-dir", default=None, help="override the output directory")
    parser.add_argument(
        "--copy-shipped",
        action="store_true",
        help="chemllmbench: also re-copy the committed files from $CHEMLLMBENCH_SRC_DIR",
    )
    args = parser.parse_args(argv)

    if args.task == "moleculenet":
        from chemberta4.tasks.moleculenet import data
        from chemberta4.tasks.moleculenet.config import DATA_DIR, DATASETS

        names = args.datasets or ",".join(DATASETS)
        out = args.data_dir or str(DATA_DIR)
        data.main(["--datasets", names, "--data_dir", out])
        return

    if args.task == "chemllmbench":
        from chemberta4.tasks.chemllmbench import data as bench_data

        families = args.datasets.split(",") if args.datasets else None
        bench_data.prepare(copy_shipped=args.copy_shipped, families=families)
        return

    from chemberta4.tasks.retrosynthesis import data as retro_data

    names = args.datasets.split(",") if args.datasets else [retro_data.DEFAULT_DATASET]
    retro_data.prepare(names)


if __name__ == "__main__":
    main()
