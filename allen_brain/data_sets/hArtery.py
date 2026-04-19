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

# Per-sample molecule_info.h5 — filename encodes condition (conDIS / conHEA)
_GEO_SAMPLES = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4837nnn'
_SAMPLE_H5 = {
    'GSM4837523': f'{_GEO_SAMPLES}/GSM4837523/suppl/GSM4837523_02dat20190515tisCARconDIS_moleculeinfo.h5',
    'GSM4837524': f'{_GEO_SAMPLES}/GSM4837524/suppl/GSM4837524_01dat20190515tisCARconHEA_moleculeinfo.h5',
    'GSM4837525': f'{_GEO_SAMPLES}/GSM4837525/suppl/GSM4837525_02dat20190620tisCARconDIS_moleculeinfo.h5',
    'GSM4837526': f'{_GEO_SAMPLES}/GSM4837526/suppl/GSM4837526_01dat20190620tisCARconHEA_moleculeinfo.h5',
    'GSM4837527': f'{_GEO_SAMPLES}/GSM4837527/suppl/GSM4837527_02dat20190717tisCARconDIS_moleculeinfo.h5',
    'GSM4837528': f'{_GEO_SAMPLES}/GSM4837528/suppl/GSM4837528_01dat20190717tisCARconHEA_moleculeinfo.h5',
}


def _read_molecule_info_h5(h5_path: str):
    """Read a cellranger molecule_info.h5 into AnnData count matrix.

    Only keeps barcodes that pass the cell-calling filter.
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        console.print(f'    h5 keys: {list(f.keys())}')

        # Gene/feature names
        if 'features' in f:
            gene_names = [g.decode() for g in f['features/name'][:]]
        elif 'gene_ids' in f:
            gene_names = [g.decode() for g in f['gene_ids'][:]]
        else:
            gene_names = None

        all_barcodes = np.array([b.decode() for b in f['barcodes'][:]])
        n_barcodes = len(all_barcodes)
        n_features = len(gene_names) if gene_names else f['feature_idx'][:].max() + 1

        # pass_filter: (n_filtered_cells, n_genomes) — values are barcode indices
        pass_filter = f['barcode_info/pass_filter'][:]
        filtered_idx = pass_filter[:, 0].astype(np.int64)  # first genome column
        filtered_barcodes = all_barcodes[filtered_idx]
        console.print(f'    {len(filtered_barcodes):,} / {n_barcodes:,} barcodes pass filter')

        # Build count matrix only for filtered barcodes
        barcode_idx = f['barcode_idx'][:]
        feature_idx = f['feature_idx'][:]
        count = f['count'][:] if 'count' in f else np.ones(len(barcode_idx), dtype=np.int32)

        # Remap barcode indices: old_idx → new_idx (only for passing barcodes)
        remap = np.full(n_barcodes, -1, dtype=np.int64)
        remap[filtered_idx] = np.arange(len(filtered_idx))
        new_bc_idx = remap[barcode_idx]

        # Keep only molecules from filtered barcodes
        mask = new_bc_idx >= 0
        X = scipy.sparse.coo_matrix(
            (count[mask], (new_bc_idx[mask], feature_idx[mask])),
            shape=(len(filtered_idx), n_features),
            dtype=np.float32,
        ).tocsr()

    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index(filtered_barcodes)
    if gene_names:
        adata.var_names = pd.Index(gene_names)
    adata.var_names_make_unique()
    return adata


def _build_h5ad(h5ad_path: str, data_dir: str) -> None:
    """Download per-sample molecule_info.h5 files, parse condition from filename."""
    os.makedirs(data_dir, exist_ok=True)
    adatas = []

    for gsm, url in _SAMPLE_H5.items():
        fname = os.path.basename(url)
        state = 'DIS' if 'conDIS' in fname else 'HEA'

        h5_path = os.path.join(data_dir, f'{gsm}.h5')
        _download_geo_file(url, h5_path)

        adata = _read_molecule_info_h5(h5_path)
        adata.obs[SPLIT_COL] = state
        adata.obs_names = gsm + '_' + adata.obs_names.astype(str)
        adatas.append(adata)
        console.print(f'  {gsm} ({state}): {adata.n_obs:,} cells')
        os.remove(h5_path)

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
