from __future__ import annotations

import os

import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/mPancreas'
GEO = 'GSE132188'
LABEL_COL = 'Celltype'
SPLIT_COL = 'day'
TEST_DAY = '15.5'

_H5AD_URL = (
    'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE132188'
    '&format=file&file=GSE132188%5Fadata%2Eh5ad%2Eh5'
)


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    h5ad_path = os.path.join(data_dir, 'mPancreas.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up mPancreas ({GEO})[/bold]')
    adata = read_h5ad_or_download(h5ad_path, url=_H5AD_URL)

    if LABEL_COL not in adata.obs.columns:
        fine = adata.obs['clusters_fig6_fine_final'].astype(str)
        broad = adata.obs['clusters_fig6_broad_final'].astype(str)
        use_broad = fine.isin(['Acinar', 'Ductal', 'Tip', 'Trunk'])
        adata.obs[LABEL_COL] = fine.where(~use_broad, broad)
        adata = adata[adata.obs[LABEL_COL] != 'Other/Doublet'].copy()

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    test_mask = split_vals == TEST_DAY
    train_mask = ~test_mask

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )

if __name__ == "__main__":
    setup()
