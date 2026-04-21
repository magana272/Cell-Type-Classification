"""Shared helpers for TOSICA benchmark dataset setup."""
from __future__ import annotations

import os
import pickle
import tarfile
import tempfile

import anndata as ad
import numpy as np
import scipy.io
import scipy.sparse
from rich.console import Console
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

console = Console()

VAL_FRAC = 0.10  # fraction of *train* set held out for validation


# ---------------------------------------------------------------------------
# GEO 10X download helpers
# ---------------------------------------------------------------------------

def _ftp_to_https(url: str) -> str:
    """Convert FTP URLs to HTTPS (NCBI supports both)."""
    if url.startswith('ftp://'):
        return 'https://' + url[6:]
    return url


def _download_geo_file(url: str, dest: str) -> None:
    """Download a GEO supplementary file (handles FTP→HTTPS)."""
    from allen_brain.cell_data.cell_download import download_url
    download_url(_ftp_to_https(url), dest)


def _read_10x_from_geo_sample(supp_urls: list[str], tmp_dir: str):
    """Read a 10X count matrix from GEO sample supplementary files.

    Handles two layouts:
      1. tar.gz archive containing a cellranger output directory
      2. Individual barcodes / genes / matrix files
    Returns AnnData or None.
    """
    import scanpy as sc

    # Strategy 1: tar.gz with 10X matrix directory
    for url in supp_urls:
        fname = os.path.basename(url).lower()
        if fname.endswith('.tar.gz') and (
                'featurebcmatrix' in fname or 'filtered_feature' in fname):
            tar_path = os.path.join(tmp_dir, 'matrix.tar.gz')
            _download_geo_file(url, tar_path)
            with tarfile.open(tar_path) as tf:
                try:
                    tf.extractall(tmp_dir, filter='data')
                except TypeError:          # Python < 3.12
                    tf.extractall(tmp_dir)  # noqa: S202
            for root, _dirs, files in os.walk(tmp_dir):
                if any(f.endswith(('.mtx.gz', '.mtx')) for f in files):
                    return sc.read_10x_mtx(root)

    # Strategy 2: individual barcodes/genes/matrix files
    mtx_url = bc_url = gene_url = None
    for url in supp_urls:
        fname = os.path.basename(url).lower()
        if '.matrix.mtx' in fname or fname == 'matrix.mtx.gz':
            mtx_url = url
        elif '.barcodes.tsv' in fname or fname == 'barcodes.tsv.gz':
            bc_url = url
        elif '.genes.tsv' in fname or '.features.tsv' in fname:
            gene_url = url

    if mtx_url and bc_url:
        mtx_dir = os.path.join(tmp_dir, 'mtx')
        os.makedirs(mtx_dir, exist_ok=True)
        mtx_path = os.path.join(mtx_dir, 'matrix.mtx.gz')
        bc_path = os.path.join(mtx_dir, 'barcodes.tsv.gz')
        _download_geo_file(mtx_url, mtx_path)
        _download_geo_file(bc_url, bc_path)
        gene_path = None
        if gene_url:
            gene_path = os.path.join(mtx_dir, 'genes.tsv.gz')
            _download_geo_file(gene_url, gene_path)
        return _read_mtx_files(mtx_path, bc_path, gene_path)

    return None


def _read_mtx_files(mtx_path: str, bc_path: str, gene_path: str | None):
    """Read 10X-style matrix/barcodes/genes files into AnnData."""
    import pandas as pd

    X = scipy.sparse.csr_matrix(scipy.io.mmread(mtx_path).T)
    barcodes = pd.read_csv(bc_path, header=None, sep='\t')[0].values
    if gene_path and os.path.exists(gene_path):
        genes_df = pd.read_csv(gene_path, header=None, sep='\t')
        gene_names = genes_df[1].values if genes_df.shape[1] >= 2 else genes_df[0].values
    else:
        gene_names = np.array([f'Gene_{i}' for i in range(X.shape[1])])
    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = pd.Index(gene_names)
    adata.var_names_make_unique()
    return adata


def _cluster_adata(adata) -> None:
    """Normalize, HVG, PCA, neighbors, Leiden → adds ``Celltype`` column."""
    import scanpy as sc
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
    sc.pp.pca(adata, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.8)
    adata.obs['Celltype'] = 'Cluster_' + adata.obs['leiden'].astype(str)


def _build_h5ad_from_geo_10x(gse, dest: str) -> None:
    """Build an h5ad from per-sample 10X matrices in *gse*."""
    adatas = []
    for gsm_name, gsm in sorted(gse.gsms.items()):
        # Collect every supplementary-file URL for this sample
        supp_urls: list[str] = []
        for key, vals in gsm.metadata.items():
            if key.startswith('supplementary_file'):
                supp_urls.extend(vals if isinstance(vals, list) else [vals])
        if not supp_urls:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            sample_ad = _read_10x_from_geo_sample(supp_urls, tmp)
            if sample_ad is None:
                console.print(f'[yellow]  {gsm_name}: no 10X data, skipping[/yellow]')
                continue

            # Attach GEO sample metadata as obs columns
            sample_ad.obs['geo_accession'] = gsm_name
            for meta_key in ('title', 'source_name_ch1'):
                vals = gsm.metadata.get(meta_key, [''])
                sample_ad.obs[meta_key] = vals[0] if vals else ''
            for ch in gsm.metadata.get('characteristics_ch1', []):
                if ':' in ch:
                    k, v = ch.split(':', 1)
                    sample_ad.obs[k.strip()] = v.strip()

            sample_ad.var_names_make_unique()
            sample_ad.obs_names = gsm_name + '_' + sample_ad.obs_names.astype(str)
            adatas.append(sample_ad)
            console.print(f'  {gsm_name}: {sample_ad.n_obs:,} cells')

    if not adatas:
        raise FileNotFoundError('No 10X data found in GEO supplementary files')

    adata = ad.concat(adatas, join='inner')
    adata.var_names_make_unique()

    console.print('Preprocessing and clustering …')
    _cluster_adata(adata)

    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    adata.write_h5ad(dest)
    console.print(
        f'[green]Saved {os.path.basename(dest)}[/green]: '
        f'{adata.n_obs:,} cells, {adata.n_vars:,} genes, '
        f'{adata.obs["Celltype"].nunique()} clusters')


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def download_geo_h5ad(accession: str, dest: str) -> None:
    """Download a GEO supplementary h5ad, or build one from 10X matrices."""
    if os.path.exists(dest):
        console.print(f'[skip] {os.path.basename(dest)} already exists')
        return

    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

    import GEOparse
    gse = GEOparse.get_GEO(geo=accession, destdir=os.path.dirname(dest),
                           silent=True)

    # Try series-level h5ad supplementary files first
    for url in gse.metadata.get('supplementary_file', []):
        if url.endswith('.h5ad') or url.endswith('.h5ad.gz') or url.endswith('.h5'):
            from allen_brain.cell_data.cell_download import download_h5ad
            download_h5ad(_ftp_to_https(url), dest)
            return

    # Fallback: build from per-sample 10X matrices
    console.print(f'[yellow]No h5ad in {accession} — building from 10X matrices[/yellow]')
    _build_h5ad_from_geo_10x(gse, dest)


def read_h5ad_or_download(path: str, accession: str | None = None,
                          url: str | None = None) -> ad.AnnData:
    """Read an h5ad file, downloading it first if needed."""
    if not os.path.exists(path):
        if url:
            from allen_brain.cell_data.cell_download import download_h5ad
            download_h5ad(url, path)
        elif accession:
            download_geo_h5ad(accession, path)
        else:
            raise FileNotFoundError(f'{path} not found and no download source given')
    return ad.read_h5ad(path)


def condition_split_and_save(
    adata: ad.AnnData,
    data_dir: str,
    label_col: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    min_cells: int = 0,
    seed: int = 42,
) -> str:
    """Split by pre-defined train/test masks, carve val from train, save arrays.

    Parameters
    ----------
    adata : AnnData with .X and .obs
    data_dir : output directory for .npy / .npz files
    label_col : obs column containing cell-type labels
    train_mask, test_mask : boolean arrays selecting train / test cells
    min_cells : drop classes with fewer cells in the *train* set
    seed : random seed for train/val split

    Returns
    -------
    data_dir
    """
    # Check if already processed
    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    labels_all = adata.obs[label_col].astype(str).values
    is_sparse = scipy.sparse.issparse(adata.X)

    # --- Extract train / test ------------------------------------------------
    labels_train = labels_all[train_mask]
    labels_test = labels_all[test_mask]

    if is_sparse:
        X_train_full = adata.X[train_mask]
        X_test_full = adata.X[test_mask]
        if not scipy.sparse.isspmatrix_csr(X_train_full):
            X_train_full = scipy.sparse.csr_matrix(X_train_full)
        if not scipy.sparse.isspmatrix_csr(X_test_full):
            X_test_full = scipy.sparse.csr_matrix(X_test_full)
    else:
        X_train_full = np.asarray(adata.X[train_mask], dtype=np.float32)
        X_test_full = np.asarray(adata.X[test_mask], dtype=np.float32)

    # --- Filter rare classes (based on train counts) -------------------------
    if min_cells > 0:
        import pandas as pd
        counts = pd.Series(labels_train).value_counts()
        keep = set(counts[counts >= min_cells].index)

        train_keep = np.array([l in keep for l in labels_train])
        test_keep = np.array([l in keep for l in labels_test])

        labels_train = labels_train[train_keep]
        labels_test = labels_test[test_keep]
        if is_sparse:
            X_train_full = X_train_full[train_keep]
            X_test_full = X_test_full[test_keep]
        else:
            X_train_full = X_train_full[train_keep]
            X_test_full = X_test_full[test_keep]

    # --- Encode labels -------------------------------------------------------
    le = LabelEncoder()
    le.fit(np.concatenate([labels_train, labels_test]))
    y_train_full = le.transform(labels_train)
    y_test = le.transform(labels_test)

    # --- Split train → train + val -------------------------------------------
    idx = np.arange(len(y_train_full))
    try:
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx, y_train_full, test_size=VAL_FRAC,
            stratify=y_train_full, random_state=seed,
        )
    except ValueError:
        # Some classes too small for stratified split — fall back to shuffle
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx, y_train_full, test_size=VAL_FRAC, random_state=seed,
        )

    # --- Save ----------------------------------------------------------------
    os.makedirs(data_dir, exist_ok=True)
    gene_names = np.array(adata.var_names)

    for name, idxs, y_split in [('train', idx_train, y_train),
                                 ('val', idx_val, y_val)]:
        if is_sparse:
            scipy.sparse.save_npz(
                os.path.join(data_dir, f'X_{name}.npz'),
                X_train_full[idxs].tocsr(),
            )
        else:
            np.save(os.path.join(data_dir, f'X_{name}.npy'),
                    X_train_full[idxs].astype(np.float32))
        np.save(os.path.join(data_dir, f'y_{name}.npy'), y_split)

    # Test set
    if is_sparse:
        scipy.sparse.save_npz(
            os.path.join(data_dir, 'X_test.npz'),
            X_test_full.tocsr(),
        )
    else:
        np.save(os.path.join(data_dir, 'X_test.npy'),
                X_test_full.astype(np.float32))
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    np.save(os.path.join(data_dir, 'gene_names.npy'), gene_names)
    np.save(os.path.join(data_dir, 'class_names.npy'), le.classes_)
    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    n_classes = len(le.classes_)
    console.print(
        f'[green]Saved[/green] {data_dir}: '
        f'train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}  '
        f'classes={n_classes}  genes={len(gene_names):,}'
        f'{" [sparse]" if is_sparse else ""}'
    )
    return data_dir
