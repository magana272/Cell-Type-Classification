"""mPancreas — mouse pancreas (GSE132188).

Train: day != 15.5
  Alpha	947
  Beta	646
  Delta	33
  Ductal	3769
  Epsilon	89
  Fev+ Alpha	339
  Fev+ Beta	902
  Fev+ Delta	59
  Fev+ Epsilon	66
  Fev+ Pyy	100
  Mat. Acinar	108
  Multipotent	1224
  Ngn3 High early	771
  Ngn3 High late	1090
  Ngn3 low EP	2354
  Prlf. Acinar	3194
  Prlf. Ductal	1890
  Prlf. Tip	2284
  Prlf. Trunk	1168
  Tip	3815
  Trunk	617
Test:  day = 15.5
  Alpha	481
  Beta	591
  Delta	70
  Ductal	499
  Epsilon	142
  Fev+ Alpha	6
  Fev+ Beta	457
  Fev+ Delta	51
  Fev+ Epsilon	46
  Fev+ Pyy	32
  Mat. Acinar	6959
  Multipotent	1
  Ngn3 High early	0
  Ngn3 High late	642
  Ngn3 low EP	262
  Prlf. Acinar	171
  Prlf. Ductal	417
  Prlf. Tip	0
  Prlf. Trunk	37
  Tip	6
  Trunk	16

       
GSM3852752	E12_5
GSM3852753	E13_5
GSM3852754	E14_5
GSM3852755	E15_5


"""
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

# Direct download URL (bypasses GEOparse)
_H5AD_URL = (
    'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE132188'
    '&format=file&file=GSE132188%5Fadata%2Eh5ad%2Eh5'
)


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save mPancreas dataset."""
    h5ad_path = os.path.join(data_dir, 'mPancreas.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up mPancreas ({GEO})[/bold]')
    adata = read_h5ad_or_download(h5ad_path, url=_H5AD_URL)

    # Build Celltype from fine + broad cluster annotations:
    # Use fine types (Fev+ subtypes, Ngn3 subtypes, etc.) but replace
    # Acinar/Ductal/Tip/Trunk with their broad subdivisions (Mat./Prlf.)
    if LABEL_COL not in adata.obs.columns:
        fine = adata.obs['clusters_fig6_fine_final'].astype(str)
        broad = adata.obs['clusters_fig6_broad_final'].astype(str)
        use_broad = fine.isin(['Acinar', 'Ductal', 'Tip', 'Trunk'])
        adata.obs[LABEL_COL] = fine.where(~use_broad, broad)
        # Drop doublets
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
    
