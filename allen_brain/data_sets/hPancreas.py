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
import tarfile
import tempfile
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from allen_brain.data_sets._utils import (
    _download_geo_file,
    condition_split_and_save,
    console,
)

DATA_DIR = 'data/hPancreas'
LABEL_COL = 'Celltype'
SPLIT_COL = 'study'

TRAIN_STUDIES = {'Baron', 'Muraro'}
TEST_STUDIES = {'Xin', 'Segerstolpe', 'Lawlor'}

# ---------------------------------------------------------------------------
# Per-study download URLs
# ---------------------------------------------------------------------------

_BARON_URL = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE84133&format=file'

_MURARO_DATA_URL = (
    'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE85241&format=file'
    '&file=GSE85241%5Fcellsystems%5Fdataset%5F4donors%5Fupdated%2Ecsv%2Egz'
)
_MURARO_ANNO_URL = (
    'https://s3.amazonaws.com/scrnaseq-public-datasets/manual-data/muraro/'
    'cell_type_annotation_Cels2016.csv'
)

_XIN_DATA_URL = (
    'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE81608&format=file'
    '&file=GSE81608%5Fhuman%5Fislets%5Frpkm%2Etxt%2Egz'
)
_XIN_IDENTITY_URL = (
    'https://s3.amazonaws.com/scrnaseq-public-datasets/manual-data/xin/'
    'human_islet_cell_identity.txt'
)

_SEGER_DATA_URL = (
    'https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5061/'
    'E-MTAB-5061.processed.1.zip'
)
_SEGER_SDRF_URL = (
    'https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5061/E-MTAB-5061.sdrf.txt'
)


# ---------------------------------------------------------------------------
# Per-study loaders
# ---------------------------------------------------------------------------

def _download_baron(study_path: str, data_dir: str) -> None:
    """Baron (GSE84133): tar of per-donor CSV files with assigned_cluster."""
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, 'baron.tar')
        _download_geo_file(_BARON_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp)

        import gzip
        import glob
        adatas = []
        for gz in sorted(glob.glob(os.path.join(tmp, '*.csv.gz'))):
            # Decompress
            csv_path = gz[:-3]
            with gzip.open(gz, 'rb') as fi, open(csv_path, 'wb') as fo:
                fo.write(fi.read())
            # Only human donors
            if 'human' not in os.path.basename(gz).lower():
                continue
            df = pd.read_csv(csv_path, index_col=0)
            if 'assigned_cluster' not in df.columns:
                continue
            labels = df['assigned_cluster'].values
            # Drop non-expression columns
            meta_cols = [c for c in df.columns if not c[0].isupper() or c == 'assigned_cluster']
            expr = df.drop(columns=meta_cols, errors='ignore')
            adata = ad.AnnData(
                X=scipy.sparse.csr_matrix(expr.values, dtype=np.float32),
                obs=pd.DataFrame({LABEL_COL: labels}, index=expr.index.astype(str)),
            )
            adata.var_names = pd.Index(expr.columns.astype(str))
            adatas.append(adata)

        adata = ad.concat(adatas, join='inner')
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.write_h5ad(study_path)
        console.print(f'  Baron: {adata.n_obs:,} cells')


def _download_muraro(study_path: str, data_dir: str) -> None:
    """Muraro (GSE85241): CSV + S3 annotation."""
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, 'muraro.csv.gz')
        anno_path = os.path.join(tmp, 'muraro_anno.csv')
        _download_geo_file(_MURARO_DATA_URL, data_path)
        _download_geo_file(_MURARO_ANNO_URL, anno_path)

        df = pd.read_csv(data_path, index_col=0)  # genes × cells
        anno = pd.read_csv(anno_path)

        X = scipy.sparse.csr_matrix(df.values.T, dtype=np.float32)
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(df.columns.astype(str))
        adata.var_names = pd.Index(df.index.astype(str))

        # Map annotations
        if 'cell_type1' in anno.columns:
            anno_map = dict(zip(anno.iloc[:, 0].astype(str), anno['cell_type1']))
            adata.obs[LABEL_COL] = adata.obs_names.map(anno_map)
        else:
            adata.obs[LABEL_COL] = 'unknown'

        adata = adata[adata.obs[LABEL_COL].notna()].copy()
        adata.var_names_make_unique()
        adata.write_h5ad(study_path)
        console.print(f'  Muraro: {adata.n_obs:,} cells')


def _download_xin(study_path: str, data_dir: str) -> None:
    """Xin (GSE81608): RPKM txt + S3 cell identity."""
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, 'xin.txt.gz')
        id_path = os.path.join(tmp, 'xin_identity.txt')
        _download_geo_file(_XIN_DATA_URL, data_path)
        _download_geo_file(_XIN_IDENTITY_URL, id_path)

        df = pd.read_csv(data_path, sep='\t', index_col=0)  # genes × cells
        identity = pd.read_csv(id_path, sep='\t')

        X = scipy.sparse.csr_matrix(df.values.T, dtype=np.float32)
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(df.columns.astype(str))
        adata.var_names = pd.Index(df.index.astype(str))

        # Map cell identity to cell type
        if 'cell_type1' in identity.columns:
            id_map = dict(zip(identity.iloc[:, 0].astype(str), identity['cell_type1']))
            adata.obs[LABEL_COL] = adata.obs_names.map(id_map)
        elif len(identity.columns) >= 2:
            id_map = dict(zip(identity.iloc[:, 0].astype(str), identity.iloc[:, 1]))
            adata.obs[LABEL_COL] = adata.obs_names.map(id_map)

        adata = adata[adata.obs[LABEL_COL].notna()].copy()
        adata.var_names_make_unique()
        adata.write_h5ad(study_path)
        console.print(f'  Xin: {adata.n_obs:,} cells')


def _download_segerstolpe(study_path: str, data_dir: str) -> None:
    """Segerstolpe (E-MTAB-5061): EBI zip + sdrf metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, 'seger.zip')
        sdrf_path = os.path.join(tmp, 'seger_sdrf.txt')
        _download_geo_file(_SEGER_DATA_URL, zip_path)
        _download_geo_file(_SEGER_SDRF_URL, sdrf_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Find the counts/rpkm file
        import glob
        txt_files = glob.glob(os.path.join(tmp, '*.txt'))
        expr_file = [f for f in txt_files if 'rpkm' in f.lower() or 'counts' in f.lower()]
        if not expr_file:
            expr_file = [f for f in txt_files if f != sdrf_path]
        df = pd.read_csv(expr_file[0], sep='\t', index_col=0)

        # First row might be sample labels — check if numeric
        try:
            df.iloc[0].astype(float)
        except (ValueError, TypeError):
            # First row is metadata, skip it
            df = df.iloc[1:]

        X = scipy.sparse.csr_matrix(df.values.astype(np.float32).T)
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(df.columns.astype(str))
        adata.var_names = pd.Index(df.index.astype(str))

        # Read sdrf for cell-type annotations
        sdrf = pd.read_csv(sdrf_path, sep='\t')
        ct_col = [c for c in sdrf.columns if 'cell type' in c.lower()]
        name_col = [c for c in sdrf.columns if 'source name' in c.lower() or 'assay name' in c.lower()]
        if ct_col and name_col:
            sdrf_map = dict(zip(sdrf[name_col[0]].astype(str), sdrf[ct_col[0]]))
            adata.obs[LABEL_COL] = adata.obs_names.map(sdrf_map)
            adata = adata[adata.obs[LABEL_COL].notna()].copy()

        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.write_h5ad(study_path)
        console.print(f'  Segerstolpe: {adata.n_obs:,} cells')


# ---------------------------------------------------------------------------

_STUDY_DOWNLOADERS = {
    'Baron': _download_baron,
    'Muraro': _download_muraro,
    'Xin': _download_xin,
    'Segerstolpe': _download_segerstolpe,
    # Lawlor: no script in tos_dataset — will be skipped if not present
}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save hPancreas dataset."""
    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print('[bold]Setting up hPancreas (multi-study)[/bold]')
    os.makedirs(data_dir, exist_ok=True)

    h5ad_path = os.path.join(data_dir, 'hPancreas.h5ad')
    if os.path.exists(h5ad_path):
        adata = ad.read_h5ad(h5ad_path)
    else:
        parts = []
        for study_name, downloader in _STUDY_DOWNLOADERS.items():
            study_path = os.path.join(data_dir, f'{study_name}.h5ad')
            if not os.path.exists(study_path):
                console.print(f'[bold]Downloading {study_name}...[/bold]')
                downloader(study_path, data_dir)
            part = ad.read_h5ad(study_path)
            part.obs[SPLIT_COL] = study_name
            parts.append(part)
            console.print(f'  {study_name}: {part.n_obs:,} cells loaded')

        adata = ad.concat(parts, join='inner')
        adata.var_names_make_unique()
        adata.write_h5ad(h5ad_path)
        console.print(f'[green]Merged[/green] {len(parts)} studies → {h5ad_path}')

    split_vals = adata.obs[SPLIT_COL].astype(str).values
    train_mask = np.isin(split_vals, list(TRAIN_STUDIES))
    test_mask = np.isin(split_vals, list(TEST_STUDIES))

    return condition_split_and_save(
        adata, data_dir, label_col=LABEL_COL,
        train_mask=train_mask, test_mask=test_mask, seed=seed,
    )
