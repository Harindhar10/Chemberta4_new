"""Write DeepChem scaffold splits of the MoleculeNet datasets as CSVs.

    python scripts/prepare_data.py --task moleculenet
    python scripts/prepare_data.py --task moleculenet --datasets bbbp,hiv

Needs the `score` extra (deepchem). Copied from the DFS `molnet` package.

Each dataset becomes <data_dir>/<name>/{train,valid,test}.csv with a `smiles` column and
one column per task.
"""
import argparse
from pathlib import Path

import deepchem as dc
import pandas as pd

from .config import DATA_DIR, DATASETS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', default=','.join(DATASETS), help='comma-separated MoleculeNet dataset names')
    p.add_argument('--data_dir', default=str(DATA_DIR), help='output directory')
    args = p.parse_args(argv)

    for name in args.datasets.split(','):
        # DummyFeaturizer keeps the raw SMILES strings as X.
        tasks, splits, _ = getattr(dc.molnet, f'load_{name}')(
            featurizer=dc.feat.DummyFeaturizer(), transformers=[], splitter='scaffold', reload=False)
        out_dir = Path(args.data_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, ds in zip(['train', 'valid', 'test'], splits):
            df = pd.DataFrame({'smiles': ds.X, **{t: ds.y[:, i] for i, t in enumerate(tasks)}})
            df.to_csv(out_dir / f'{split}.csv', index=False)
            print(f'{name} {split}: {len(df)} molecules')


if __name__ == '__main__':
    main()
