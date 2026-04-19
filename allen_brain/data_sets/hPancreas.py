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
Baron (GSE84133)
GSM2230757	human pancreatic islets, sample 1
GSM2230757_human1_umifm_counts.csv.gz
GSM2230758	human pancreatic islets, sample 2
GSM2230759	human pancreatic islets, sample 3
GSM2230760	human pancreatic islets, sample 4
GSM2230761	mouse pancreatic islets, sample 1
GSM2230762	mouse pancreatic islets, sample 2
Muraro (GSE85241)
GSM2262792	Donor D28, live sorted cells, library 1
GSM2262793	Donor D28, live sorted cells, library 2
GSM2262794	Donor D28, live sorted cells, library 3
GSM2262795	Donor D28, live sorted cells, library 4
GSM2262796	Donor D28, live sorted cells, library 5
GSM2262797	Donor D28, live sorted cells, library 6
GSM2262798	Donor D28, live sorted cells, library 7
GSM2262799	Donor D28, live sorted cells, library 8
GSM2262800	Donor D29, live sorted cells, library 1
GSM2262801	Donor D29, live sorted cells, library 2
GSM2262802	Donor D29, live sorted cells, library 3
GSM2262803	Donor D29, live sorted cells, library 4
GSM2262804	Donor D29, live sorted cells, library 5
GSM2262805	Donor D29, live sorted cells, library 6
GSM2262806	Donor D29, live sorted cells, library 7
GSM2262807	Donor D29, live sorted cells, library 8
GSM2262808	Donor D30, live sorted cells, library 1
GSM2262809	Donor D30, live sorted cells, library 2
GSM2262810	Donor D30, live sorted cells, library 3
GSM2262811	Donor D30, live sorted cells, library 4
GSM2262812	Donor D30, live sorted cells, library 5
GSM2262813	Donor D30, live sorted cells, library 6
GSM2262814	Donor D30, live sorted cells, library 7
GSM2262815	Donor D30, live sorted cells, library 8
GSM2262816	Donor D31, live sorted cells, library 1
GSM2262817	Donor D31, live sorted cells, library 2
GSM2262818	Donor D31, live sorted cells, library 3
GSM2262819	Donor D31, live sorted cells, library 4
GSM2262820	Donor D31, live sorted cells, library 5
GSM2262821	Donor D31, live sorted cells, library 6
GSM2262822	Donor D31, live sorted cells, library 7
GSM2262823	Donor D31, live sorted cells, library 8


Xin (GSE81608)
Supplementary file	Size	Download	File type/resource
GSE81608_human_islets_rpkm.txt.gz	35.5 Mb	(ftp)(http)	TXT


Segerstolpe (E-MTAB-5061)
https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061



Lawlor (GSE86473):          
GSM2303122	baseline-03182015_S2
GSM2303123	baseline-03272015_S8
GSM2303124	baseline-05202015-Islet-15_S1
GSM2303125	baseline-05222015-Islet-16_S6
GSM2303126	baseline-09032015_S2
GSM2303127	baseline-09102015_S7
GSM2303128	baseline-10012015_S12
GSM2303129	baseline-10292015_S17
GSM2303130	Single-cell-bulk-03182015_S4
GSM2303131	Single-cell-bulk-03272015_S10
GSM2303132	Single-cell-bulk-05202015_S3
GSM2303133	Single-cell-bulk-05222015_S8
GSM2303134	Single-cell-bulk-c1-09032015_S5
GSM2303135	Single-cell-bulk-c1-09102015_S10
GSM2303136	Single-cell-bulk-c1-10012015_S15
GSM2303137	Single-cell-bulk-c1-10292015_S20
GSM2303138	undissociated-c1-ctrl-09032015_S3
GSM2303139	undissociated-c1-ctrl-09102015_S8
GSM2303140	undissociated-c1-ctrl-10012015_S13
GSM2303141	undissociated-c1-ctrl-10292015_S18
GSM2303142	Untreated-Ctrl-03182015_S3
GSM2303143	Untreated-Ctrl-03272015_S9
GSM2303144	Untreated-Ctrl-05202015_S2
GSM2303145	Untreated-Ctrl-05222015_S7


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

# The figshare pancreas benchmark h5ad contains all 5 studies.
TECH_TO_STUDY = {
    'inDrop1': 'Baron', 'inDrop2': 'Baron',
    'inDrop3': 'Baron', 'inDrop4': 'Baron',
    'celseq2': 'Muraro',
    'smartseq2': 'Segerstolpe',
    'smarter': 'Xin',
    'fluidigmc1': 'Lawlor',
}

CELLTYPE_MAP = {
    'alpha': 'Alpha', 'beta': 'Beta', 'gamma': 'PP',
    'delta': 'Delta', 'epsilon': 'Epsilon',
    'acinar': 'Acinar', 'ductal': 'Ductal',
    'endothelial': 'Endothelial',
    'activated_stellate': 'PSC', 'quiescent_stellate': 'PSC',
    'macrophage': 'Macrophage', 'schwann': 'Schwann',
    'mast': 'Mast', 't_cell': 'T_cell',
}

_PANCREAS_H5AD = 'data/pancreas/pancreas.h5ad'


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
        # Build from the shared figshare pancreas benchmark h5ad
        if not os.path.exists(_PANCREAS_H5AD):
            from allen_brain.cell_data.cell_download import H5AD_SOURCES, download_h5ad
            download_h5ad(H5AD_SOURCES['pancreas'], _PANCREAS_H5AD)

        import anndata as ad
        adata = ad.read_h5ad(_PANCREAS_H5AD)

        adata.obs[SPLIT_COL] = adata.obs['tech'].map(TECH_TO_STUDY)
        adata = adata[adata.obs[SPLIT_COL].notna()].copy()

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
