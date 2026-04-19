"""hArtery — human artery (GSE159677).

Train: state = HEA
  T	2769
  Fib	1648
  EC	2843
  SMC	1080
  Myeloid	1166
  B	439
  NK	394
  MSC	388
  Plasma	155
  Mast	78
Test:  state = DIS
  T	13323
  Fib	2816
  EC	3117
  SMC	2801
  Myeloid	8941
  B	1169
  NK	1110
  MSC	857
  Plasma	814
  Mast	451
Samples (6)
Less... Less...           
GSM4837523	Patient 1 AC scRNA-seq
GSM4837524	Patient 1 PA scRNA-seq
GSM4837525	Patient 2 AC scRNA-seq
GSM4837526	Patient 2 PA scRNA-seq
GSM4837527	Patient 3 AC scRNA-seq
GSM4837528	Patient 3 PA scRNA-seq



GSM4837523_02dat20190515tisCARconDIS_featurebcmatrixfiltered.tar.gz	
GSM4837523_02dat20190515tisCARconDIS_moleculeinfo.h5

GSM4837524_01dat20190515tisCARconHEA_featurebcmatrixfiltered.tar.gz	
GSM4837524_01dat20190515tisCARconHEA_moleculeinfo.h5


"""
from __future__ import annotations

import os

import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/hArtery'
GEO = 'GSE159677'
LABEL_COL = 'Celltype'
SPLIT_COL = 'state'
TRAIN_VALUES = {'HEA'}
TEST_VALUES = {'DIS'}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hArtery dataset."""
    h5ad_path = os.path.join(data_dir, 'hArtery.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up hArtery ({GEO})[/bold]')
    adata = read_h5ad_or_download(h5ad_path, accession=GEO)

    # Map GEO metadata → state when built from raw 10X matrices
    if SPLIT_COL not in adata.obs.columns:
        adata.obs[SPLIT_COL] = adata.obs['location'].map({
            'atherosclerotic core': 'DIS',
            'proximal adjacent': 'HEA',
        })

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = np.isin(split_vals, list(TRAIN_VALUES))
    test_mask = np.isin(split_vals, list(TEST_VALUES))

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
