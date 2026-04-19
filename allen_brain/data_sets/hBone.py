"""hBone — human bone marrow (GSE152805).

Train: state = OLT

HomC	6930
preFC	497
RegC	3385
RepC	669
HTC	2474
preHTC 	201
FC	459
Test:  state = MT
HomC	397
preFC	4423
RegC	184
RepC	2211
HTC	50
preHTC 	1994
FC	2266
GSM4626766	Osteoarthritic cartilage-oLT-1
GSM4626767	Osteoarthritic cartilage-oLT-2
GSM4626768	Osteoarthritic cartilage-oLT-3
GSM4626769	Osteoarthritic cartilage-MT-1
GSM4626770	Osteoarthritic cartilage-MT-2
GSM4626771	Osteoarthritic cartilage-MT-3

GSM4626766_OA_oLT_113.barcodes.tsv.gz	24.6 Kb
GSM4626766_OA_oLT_113.genes.tsv.gz	251.3 Kb
GSM4626766_OA_oLT_113.matrix.mtx.gz	42.5 Mb
GSM4626767_OA_oLT_116.barcodes.tsv.gz	15.8 Kb
GSM4626767_OA_oLT_116.genes.tsv.gz	251.3 Kb
GSM4626767_OA_oLT_116.matrix.mtx.gz	30.0 Mb
GSM4626768_OA_oLT_118.barcodes.tsv.gz	18.9 Kb
GSM4626768_OA_oLT_118.genes.tsv.gz	251.3 Kb
GSM4626768_OA_oLT_118.matrix.mtx.gz	43.4 Mb
GSM4626769_OA_MT_113.barcodes.tsv.gz	13.4 Kb
GSM4626769_OA_MT_113.genes.tsv.gz	251.3 Kb
GSM4626769_OA_MT_113.matrix.mtx.gz	30.3 Mb
GSM4626770_OA_MT_116.barcodes.tsv.gz	14.8 Kb
GSM4626770_OA_MT_116.genes.tsv.gz	251.3 Kb
GSM4626770_OA_MT_116.matrix.mtx.gz	36.2 Mb
GSM4626771_OA_MT_118.barcodes.tsv.gz	19.8 Kb
GSM4626771_OA_MT_118.genes.tsv.gz	251.3 Kb
GSM4626771_OA_MT_118.matrix.mtx.gz


"""
from __future__ import annotations

import os

import numpy as np

from allen_brain.data_sets._utils import (
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/hBone'
GEO = 'GSE152805'
LABEL_COL = 'Celltype'
SPLIT_COL = 'state'
TRAIN_VALUES = {'OLT'}
TEST_VALUES = {'MT'}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hBone dataset."""
    h5ad_path = os.path.join(data_dir, 'hBone.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up hBone ({GEO})[/bold]')
    adata = read_h5ad_or_download(h5ad_path, accession=GEO)

    if SPLIT_COL not in adata.obs.columns:
        adata.obs[SPLIT_COL] = adata.obs['title'].apply(
            lambda t: 'OLT' if 'oLT' in t else 'MT'
        )

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = np.isin(split_vals, list(TRAIN_VALUES))
    test_mask = np.isin(split_vals, list(TEST_VALUES))

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
