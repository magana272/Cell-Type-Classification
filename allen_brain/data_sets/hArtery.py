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

# Per-sample 10X tar.gz URLs — filename encodes condition (conDIS / conHEA)
_GEO_SAMPLES = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4837nnn'
_SAMPLE_URLS = {
    'GSM4837523': f'{_GEO_SAMPLES}/GSM4837523/suppl/GSM4837523_02dat20190515tisCARconDIS_featurebcmatrixfiltered.tar.gz',
    'GSM4837524': f'{_GEO_SAMPLES}/GSM4837524/suppl/GSM4837524_01dat20190515tisCARconHEA_featurebcmatrixfiltered.tar.gz',
    'GSM4837525': f'{_GEO_SAMPLES}/GSM4837525/suppl/GSM4837525_02dat20190620tisCARconDIS_featurebcmatrixfiltered.tar.gz',
    'GSM4837526': f'{_GEO_SAMPLES}/GSM4837526/suppl/GSM4837526_01dat20190620tisCARconHEA_featurebcmatrixfiltered.tar.gz',
    'GSM4837527': f'{_GEO_SAMPLES}/GSM4837527/suppl/GSM4837527_02dat20190717tisCARconDIS_featurebcmatrixfiltered.tar.gz',
    'GSM4837528': f'{_GEO_SAMPLES}/GSM4837528/suppl/GSM4837528_01dat20190717tisCARconHEA_featurebcmatrixfiltered.tar.gz',
}


def _build_h5ad(h5ad_path: str, data_dir: str) -> None:
    """Download per-sample 10X tars, parse condition from filename."""
    os.makedirs(data_dir, exist_ok=True)
    adatas = []

    for gsm, url in _SAMPLE_URLS.items():
        fname = os.path.basename(url)
        # Parse condition from filename: conDIS → DIS, conHEA → HEA
        state = 'DIS' if 'conDIS' in fname else 'HEA'

        with tempfile.TemporaryDirectory() as tmp:
            tar_path = os.path.join(tmp, 'sample.tar.gz')
            _download_geo_file(url, tar_path)
            with tarfile.open(tar_path) as tf:
                try:
                    tf.extractall(tmp, filter='data')
                except TypeError:
                    tf.extractall(tmp)

            # Find 10X matrix files
            mtx_path = bc_path = feat_path = None
            for root, _dirs, files in os.walk(tmp):
                for f in files:
                    fp = os.path.join(root, f)
                    if f.endswith(('.mtx.gz', '.mtx')) and mtx_path is None:
                        mtx_path = fp
                    elif 'barcodes' in f and f.endswith(('.tsv.gz', '.tsv')):
                        bc_path = fp
                    elif ('features' in f or 'genes' in f) and f.endswith(('.tsv.gz', '.tsv')):
                        feat_path = fp

            adata = _read_mtx_files(mtx_path, bc_path, feat_path)
            adata.obs[SPLIT_COL] = state
            adata.obs_names = gsm + '_' + adata.obs_names.astype(str)
            adatas.append(adata)
            console.print(f'  {gsm} ({state}): {adata.n_obs:,} cells')

    adata = ad.concat(adatas, join='inner')
    adata.var_names_make_unique()

    # Cluster for cell-type labels
    _cluster_adata(adata)

    adata.write_h5ad(h5ad_path)
    console.print(f'  Saved: {adata.n_obs:,} cells, {adata.obs[LABEL_COL].nunique()} types')


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
