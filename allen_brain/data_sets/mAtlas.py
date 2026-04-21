from __future__ import annotations

import os

import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/mAtlas'
GEO = 'GSE132042'
LABEL_COL = 'cell_ontology_class'
SPLIT_COL = 'age'
TRAIN_AGE = '18m'

FIGSHARE_URL = 'https://ndownloader.figshare.com/files/23937842'


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    h5ad_path = os.path.join(data_dir, 'mAtlas.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up mAtlas ({GEO})[/bold]')
    adata = read_h5ad_or_download(h5ad_path, url=FIGSHARE_URL)

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = split_vals == TRAIN_AGE
    test_mask = ~train_mask

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
