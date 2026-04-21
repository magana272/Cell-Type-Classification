"""hPancreas — human pancreas (multiple studies), TOSICA benchmark.

Source: TOSICA demo data (figshare.com/projects/TOSICA_demo/158489)
  Baron (GSE84133) + Muraro (GSE85241) → train
  Xin (GSE81608) + Segerstolpe (E-MTAB-5061) + Lawlor (GSE86473) → test

Train (10,600 cells):
  Alpha            3136
  Beta             2966
  Ductal           1290
  Acinar           1144
  Delta             793
  PSC               524
  PP                356
  Endothelial       273
  Macrophage         52
  Mast               25
  Epsilon            21
  Schwann            13
  T_cell              7

Test (4,218 cells):
  Alpha            2011
  Beta             1006
  Ductal            414
  PP                282
  Acinar            209
  Delta             188
  PSC                73
  Endothelial        16
  Epsilon             7
  Mast                7
  MHC class II        5

14 classes, 3000 HVGs.
"""
from __future__ import annotations

import os

import anndata as ad
import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
)

DATA_DIR = 'data/hPancreas'
LABEL_COL = 'Celltype'
SPLIT_COL = 'split'

# TOSICA demo h5ad files (original study annotations, 3000 HVGs)
_TRAIN_URL = 'https://ndownloader.figshare.com/files/39010169'
_TEST_URL = 'https://ndownloader.figshare.com/files/39010166'

# Normalize TOSICA lowercase labels → Title Case
CELLTYPE_MAP = {
    'alpha': 'Alpha', 'beta': 'Beta', 'delta': 'Delta',
    'epsilon': 'Epsilon', 'acinar': 'Acinar', 'ductal': 'Ductal',
    'endothelial': 'Endothelial', 'macrophage': 'Macrophage',
    'schwann': 'Schwann', 'mast': 'Mast', 't_cell': 'T_cell',
    # Already correct case:
    'PP': 'PP', 'PSC': 'PSC', 'MHC class II': 'MHC class II',
}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hPancreas dataset."""
    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print('[bold]Setting up hPancreas (TOSICA benchmark)[/bold]')

    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'demo_train.h5ad')
    test_path = os.path.join(data_dir, 'demo_test.h5ad')

    from allen_brain.cell_data.cell_download import download_h5ad

    if not os.path.exists(train_path):
        download_h5ad(_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        download_h5ad(_TEST_URL, test_path)

    adata_train = ad.read_h5ad(train_path)
    adata_test = ad.read_h5ad(test_path)

    console.print(f'  train: {adata_train.n_obs:,} cells, '
                  f'{adata_train.obs[LABEL_COL].nunique()} types')
    console.print(f'  test:  {adata_test.n_obs:,} cells, '
                  f'{adata_test.obs[LABEL_COL].nunique()} types')

    # Normalize cell-type labels to Title Case
    for adata_part in (adata_train, adata_test):
        raw = adata_part.obs[LABEL_COL].astype(str)
        adata_part.obs[LABEL_COL] = raw.map(CELLTYPE_MAP).fillna(raw)

    # Align to shared gene set (should already match for TOSICA demo)
    shared_genes = adata_train.var_names.intersection(adata_test.var_names)
    adata_train = adata_train[:, shared_genes].copy()
    adata_test = adata_test[:, shared_genes].copy()

    # Tag split for condition_split_and_save
    adata_train.obs[SPLIT_COL] = 'train'
    adata_test.obs[SPLIT_COL] = 'test'
    adata = ad.concat([adata_train, adata_test], join='inner')
    adata.obs_names_make_unique()

    train_mask = (adata.obs[SPLIT_COL] == 'train').values
    test_mask = (adata.obs[SPLIT_COL] == 'test').values

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
