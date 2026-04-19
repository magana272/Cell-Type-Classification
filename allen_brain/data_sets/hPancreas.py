"""hPancreas — human pancreas (multiple studies).

Train: Baron (GSE84133) + Muraro (GSE85241)
PP	356
PSC	444
Delta	793
Ductal	1290
Beta	2966
Acinar	1144
Alpha	3136
Epsilon	21
Endothelial	273
Macrophage	52
Schwann	13
Mast	25
T_cell	7
Mesenchymal	80
MHC class II	0


Test:  Xin (GSE81608) + Segerstolpe (E-MTAB-5061) + Lawlor (GSE86473)
PP	282
PSC	73
Delta	188
Ductal	414
Beta	1006
Acinar	209
Alpha	2011
Epsilon	7
Endothelial	16
Macrophage	0
Schwann	0
Mast	7
T_cell	0
Mesenchymal	0
MHC class II	5


"""
from __future__ import annotations

import os

import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/hPancreas'
LABEL_COL = 'Celltype'
SPLIT_COL = 'study'

TRAIN_STUDIES = {'Baron', 'Muraro'}
TEST_STUDIES = {'Xin', 'Segerstolpe', 'Lawlor'}

# Figshare pancreas benchmark h5ad (all 5 studies with annotations)
_PANCREAS_URL = 'https://ndownloader.figshare.com/files/24539828'
_PANCREAS_H5AD = 'data/pancreas/pancreas.h5ad'

# tech → study mapping
TECH_TO_STUDY = {
    'inDrop1': 'Baron', 'inDrop2': 'Baron',
    'inDrop3': 'Baron', 'inDrop4': 'Baron',
    'celseq2': 'Muraro',
    'smartseq2': 'Segerstolpe',
    'smarter': 'Xin',
    'fluidigmc1': 'Lawlor',
    # 'celseq' (Grün) is not part of the TOSICA benchmark
}

# celltype → TOSICA Celltype mapping
CELLTYPE_MAP = {
    'alpha': 'Alpha', 'beta': 'Beta', 'gamma': 'PP',
    'delta': 'Delta', 'epsilon': 'Epsilon',
    'acinar': 'Acinar', 'ductal': 'Ductal',
    'endothelial': 'Endothelial',
    'activated_stellate': 'PSC', 'quiescent_stellate': 'PSC',
    'macrophage': 'Macrophage', 'schwann': 'Schwann',
    'mast': 'Mast', 't_cell': 'T_cell',
}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hPancreas dataset."""
    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print('[bold]Setting up hPancreas (multi-study)[/bold]')

    h5ad_path = os.path.join(data_dir, 'hPancreas.h5ad')
    if os.path.exists(h5ad_path):
        adata = read_h5ad_or_download(h5ad_path)
    else:
        # Download figshare pancreas benchmark h5ad
        if not os.path.exists(_PANCREAS_H5AD):
            from allen_brain.cell_data.cell_download import download_h5ad
            os.makedirs(os.path.dirname(_PANCREAS_H5AD), exist_ok=True)
            download_h5ad(_PANCREAS_URL, _PANCREAS_H5AD)

        import anndata as ad
        adata = ad.read_h5ad(_PANCREAS_H5AD)

        # Map tech → study, drop techs not in the benchmark
        adata.obs[SPLIT_COL] = adata.obs['tech'].map(TECH_TO_STUDY)
        adata = adata[adata.obs[SPLIT_COL].notna()].copy()

        # Map celltype → TOSICA Celltype
        adata.obs[LABEL_COL] = adata.obs['celltype'].map(CELLTYPE_MAP)
        adata = adata[adata.obs[LABEL_COL].notna()].copy()

        os.makedirs(data_dir, exist_ok=True)
        adata.write_h5ad(h5ad_path)
        console.print(f'[green]Saved[/green] {h5ad_path} '
                      f'({adata.n_obs:,} cells, '
                      f'{adata.obs[LABEL_COL].nunique()} types)')

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = np.isin(split_vals, list(TRAIN_STUDIES))
    test_mask = np.isin(split_vals, list(TEST_STUDIES))

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
