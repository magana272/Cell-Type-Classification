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


"""
from __future__ import annotations

import os
import tarfile
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from allen_brain.data_sets._utils import (
    _cluster_adata,
    _download_geo_file,
    _read_mtx_files,
    condition_split_and_save,
    console,
)

DATA_DIR = 'data/hArtery'
GEO = 'GSE159677'
LABEL_COL = 'Celltype'
SPLIT_COL = 'state'
TRAIN_VALUES = {'HEA'}
TEST_VALUES = {'DIS'}

# Cellranger aggr output — count matrix + analysis/clustering
_AGGR_URL = (
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159677/suppl/'
    'GSE159677_AGGREGATEMAPPED-tisCAR6samples_featurebcmatrixfiltered.tar.gz'
)

# Barcode suffix → condition (from Aggregated.Sample.Meta.txt.gz)
_SUFFIX_TO_STATE = {
    '-1': 'HEA', '-2': 'DIS', '-3': 'HEA',
    '-4': 'DIS', '-5': 'HEA', '-6': 'DIS',
}


def _build_h5ad(h5ad_path: str, data_dir: str) -> None:
    """Download aggregated cellranger tar and build h5ad."""
    os.makedirs(data_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, 'aggr.tar.gz')
        _download_geo_file(_AGGR_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            try:
                tf.extractall(tmp, filter='data')
            except TypeError:
                tf.extractall(tmp)

        # Find count matrix and analysis directory
        mtx_path = bc_path = feat_path = None
        analysis_dir = None
        for root, dirs, files in os.walk(tmp):
            for f in files:
                fp = os.path.join(root, f)
                if f.endswith(('.mtx.gz', '.mtx')) and mtx_path is None:
                    mtx_path = fp
                elif 'barcodes' in f and f.endswith(('.tsv.gz', '.tsv')):
                    bc_path = fp
                elif ('features' in f or 'genes' in f) and f.endswith(('.tsv.gz', '.tsv')):
                    feat_path = fp
            if 'analysis' in dirs:
                analysis_dir = os.path.join(root, 'analysis')

        adata = _read_mtx_files(mtx_path, bc_path, feat_path)
        console.print(f'  Loaded {adata.n_obs:,} cells, {adata.n_vars:,} genes')

        # Map barcode suffix → condition
        suffixes = adata.obs_names.str.extract(r'(-\d+)$')[0]
        adata.obs[SPLIT_COL] = suffixes.map(_SUFFIX_TO_STATE).values

        # Read cellranger clustering from analysis directory
        if analysis_dir:
            for root, _dirs, files in os.walk(analysis_dir):
                for f in sorted(files):
                    if f == 'clusters.csv':
                        clust_df = pd.read_csv(os.path.join(root, f))
                        n = clust_df['Cluster'].nunique()
                        console.print(f'  Found clustering: {n} clusters')
                        adata.obs[LABEL_COL] = (
                            'Cluster_' + clust_df['Cluster'].astype(str).values
                        )

        if LABEL_COL not in adata.obs.columns:
            _cluster_adata(adata)
            adata.obs[LABEL_COL] = adata.obs['Celltype']

    adata.write_h5ad(h5ad_path)
    console.print(f'  Saved: {adata.obs[LABEL_COL].nunique()} types')


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hArtery dataset."""
    h5ad_path = os.path.join(data_dir, 'hArtery.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print(f'[bold]Setting up hArtery ({GEO})[/bold]')
    if not os.path.exists(h5ad_path):
        _build_h5ad(h5ad_path, data_dir)
    adata = ad.read_h5ad(h5ad_path)

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = np.isin(split_vals, list(TRAIN_VALUES))
    test_mask = np.isin(split_vals, list(TEST_VALUES))

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
