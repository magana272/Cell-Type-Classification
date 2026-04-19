"""mBrain — mouse brain (multiple studies).

Train: Saunders (GSE116470) + Tabula muris (GSE109774) + Rosenberg (GSE110823)
astrocyte	3843
brain pericyte	767
endothelial cell	2987
ependymal cell	119
macrophage	170
microglial cell	4669
neuron	28241
olfactory ensheathing cell	16
oligodendrocyte	7178
oligodendrocyte precursor cell	811
Test:  Zeisel (GSE60361)
astrocyte	1059
brain pericyte	281
endothelial cell	392
ependymal cell	79
macrophage	96
microglial cell	275
neuron	3584
olfactory ensheathing cell	107
oligodendrocyte	1485
oligodendrocyte precursor cell	36


Saunders (GSE116470):
GSE116470_F_GRCm38.81.P60Cerebellum_ALT.raw.dge.txt.gz	80.2 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.txt.gz	586.6 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60Cortex_noRep5_POSTERIORonly.raw.dge.txt.gz	346.1 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60EntoPeduncular.raw.dge.txt.gz	54.1 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60GlobusPallidus.raw.dge.txt.gz	190.4 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60Hippocampus.raw.dge.txt.gz	424.1 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60Striatum.raw.dge.txt.gz	269.7 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60SubstantiaNigra.raw.dge.txt.gz	126.7 Mb	(ftp)(http)	TXT
GSE116470_F_GRCm38.81.P60Thalamus.raw.dge.txt.gz	259.8 Mb	(ftp)(http)	TXT
GSE116470_metacells.BrainCellAtlas_Saunders_version_2018.04.01.csv.gz	8.6 Mb	(ftp)(http)	CSV

Tabula muris (GSE109774):
SE109774_Brain_Microglia.tar.gz	315.3 Mb	(ftp)(http)	TAR
GSE109774_Brain_Neurons.tar.gz	382.0 Mb	(ftp)(http)	TAR


Rosenberg (GSE110823):
GSM3017260_100_CNS_nuclei.mat.gz	539.5 Kb
GSM3017261_150000_CNS_nuclei.mat.gz	217.1 Mb
GSM3017262_same_day_cells_nuclei_3000_UBCs.mat.gz	25.4 Mb
GSM3017263_same_day_cells_nuclei_300_UBCs.mat.gz	3.1 Mb
GSM3017264_frozen_preserved_cells_nuclei_1000_UBCs.mat.gz	8.8 Mb
GSM3017265_frozen_preserved_cells_nuclei_200_UBCs.mat.gz	2.3 Mb


"""
from __future__ import annotations

import os
import tarfile
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from allen_brain.data_sets._utils import (
    _cluster_adata,
    _download_geo_file,
    _read_mtx_files,
    condition_split_and_save,
    console,
    read_h5ad_or_download,
)

DATA_DIR = 'data/mBrain'
LABEL_COL = 'cell_ontology_class'
SPLIT_COL = 'study'

TRAIN_STUDIES = {'Saunders', 'Tabula muris', 'Rosenberg'}
TEST_STUDIES = {'Zeisel'}

# ---- GEO series-level supplementary URLs ----------------------------------
_GEO_BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/series'

_TM_BRAIN_URLS = [
    f'{_GEO_BASE}/GSE109nnn/GSE109774/suppl/GSE109774_Brain_Microglia.tar.gz',
    f'{_GEO_BASE}/GSE109nnn/GSE109774/suppl/GSE109774_Brain_Neurons.tar.gz',
]

# DropViz "DGE By Class" files — cells pre-sorted by cell type
_DROPVIZ_BASE = 'https://storage.googleapis.com/dropviz-downloads/static/classes'
_SAUNDERS_CLASS_FILES = {
    'astrocyte': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Astrocytes_9-13-17.raw.dge.txt.gz',
    'endothelial cell': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Endothelial_5-3-17.raw.dge.txt.gz',
    'microglial cell': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Microglia_Macrophage_5-3-17.raw.dge.txt.gz',
    'brain pericyte': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Mural_5-3-17.raw.dge.txt.gz',
    'oligodendrocyte': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Oligodendrocytes_5-3-17.raw.dge.txt.gz',
    'oligodendrocyte precursor cell': f'{_DROPVIZ_BASE}/H_1stRound_CrossTissue_Polydendrocytes_5-3-17.raw.dge.txt.gz',
}

_ROSENBERG_URL = (
    'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3017nnn/GSM3017261/suppl/'
    'GSM3017261_150000_CNS_nuclei.mat.gz'
)

# GEO version (S3 is in Glacier)
_ZEISEL_URL = (
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60361/suppl/'
    'GSE60361_C1-3005-Expression.txt.gz'
)

# Tabula muris FACS metadata with cell_ontology_class annotations
_TM_METADATA_URL = 'https://raw.githubusercontent.com/czbiohub-sf/tabula-muris-vignettes/refs/heads/master/data/TM_facs_metadata.csv'


# ---- Per-study download helpers -------------------------------------------

def _download_tabula_muris_brain(study_path: str, data_dir: str) -> None:
    """Download Tabula muris brain FACS data (per-cell CSVs) from GEO."""
    adatas = []
    for url in _TM_BRAIN_URLS:
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = os.path.join(tmp, 'brain.tar.gz')
            _download_geo_file(url, tar_path)
            with tarfile.open(tar_path) as tf:
                try:
                    tf.extractall(tmp, filter='data')
                except TypeError:
                    tf.extractall(tmp)

            # FACS format: each CSV is one cell (gene_name, count)
            import glob
            csvs = sorted(glob.glob(os.path.join(tmp, '**', '*.csv'), recursive=True))
            if not csvs:
                console.print(f'[yellow]  No CSVs in {os.path.basename(url)}[/yellow]')
                continue

            # Read first CSV to get gene names
            first = pd.read_csv(csvs[0], header=None)
            gene_names = first[0].values

            # Read all cells
            expr = np.zeros((len(csvs), len(gene_names)), dtype=np.float32)
            cell_names = []
            for i, csv_path in enumerate(csvs):
                cell_names.append(os.path.basename(csv_path).replace('.csv', ''))
                df = pd.read_csv(csv_path, header=None)
                expr[i] = df[1].values

            adata = ad.AnnData(X=scipy.sparse.csr_matrix(expr))
            adata.obs_names = pd.Index(cell_names)
            adata.var_names = pd.Index(gene_names.astype(str))
            adatas.append(adata)
            console.print(f'  Tabula muris {os.path.basename(url)}: {adata.n_obs:,} cells')

    if not adatas:
        raise FileNotFoundError('No data found in Tabula muris brain tar files')
    adata = ad.concat(adatas, join='inner')
    adata.var_names_make_unique()

    # Download FACS metadata CSV for cell_ontology_class annotations
    meta_path = os.path.join(data_dir, 'TM_facs_metadata.csv')
    _download_geo_file(_TM_METADATA_URL, meta_path)
    meta = pd.read_csv(meta_path, index_col=0)
    os.remove(meta_path)

    # Match cells by name (CSV filenames map to metadata index)
    common = adata.obs_names.intersection(meta.index)
    if len(common) > 0 and 'cell_ontology_class' in meta.columns:
        adata = adata[common].copy()
        adata.obs[LABEL_COL] = meta.loc[common, 'cell_ontology_class'].values
    else:
        _cluster_adata(adata)
        adata.obs[LABEL_COL] = adata.obs['Celltype']

    adata.write_h5ad(study_path)


def _read_dge_sparse(path: str):
    """Read a Drop-seq DGE file into sparse AnnData.

    Supports two formats:
      1. MatrixMarket with %%GENES / %%CELLS headers (DropViz .raw.dge.txt.gz)
      2. Dense TSV/CSV (genes × cells)
    """
    import gzip

    opener = gzip.open if path.endswith('.gz') else open

    with opener(path, 'rt') as f:
        first_line = f.readline().strip()

        # --- Format 1: MatrixMarket with %%GENES header ---
        if first_line.startswith('%%MatrixMarket'):
            gene_names = []
            cell_names = []
            for line in f:
                line = line.strip()
                if line.startswith('%%GENES'):
                    gene_names = line.split('\t')[1:]
                elif line.startswith('%%CELLS'):
                    cell_names = line.split('\t')[1:]
                elif line.startswith('%'):
                    continue
                else:
                    # Size line: n_rows n_cols n_entries
                    parts = line.split()
                    n_genes, n_cells, n_entries = int(parts[0]), int(parts[1]), int(parts[2])
                    if not cell_names:
                        cell_names = [str(i) for i in range(n_cells)]
                    if not gene_names:
                        gene_names = [str(i) for i in range(n_genes)]
                    break

            # Read triplets: gene_idx cell_idx value (1-indexed)
            rows, cols, vals = [], [], []
            for line in f:
                parts = line.strip().split()
                g, c, v = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
                rows.append(c)   # cell = row in AnnData
                cols.append(g)   # gene = column
                vals.append(v)

            X = scipy.sparse.coo_matrix(
                (vals, (rows, cols)),
                shape=(n_cells, n_genes),
                dtype=np.float32,
            ).tocsr()

        # --- Format 2: Dense TSV/CSV ---
        else:
            sep = ',' if first_line.count(',') > first_line.count('\t') else '\t'
            header = first_line.split(sep)
            cell_names = [c.strip('"') for c in header[1:]]
            gene_names = []
            rows, cols, vals = [], [], []
            gene_idx = 0
            for line in f:
                parts = line.strip().split(sep)
                try:
                    parsed = [int(float(v.strip('"'))) for v in parts[1:]]
                except ValueError:
                    continue
                gene_names.append(parts[0].strip('"'))
                for cell_idx, v in enumerate(parsed):
                    if v > 0:
                        rows.append(cell_idx)
                        cols.append(gene_idx)
                        vals.append(v)
                gene_idx += 1

            X = scipy.sparse.coo_matrix(
                (vals, (rows, cols)),
                shape=(len(cell_names), len(gene_names)),
                dtype=np.float32,
            ).tocsr()

    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index([str(c) for c in cell_names])
    adata.var_names = pd.Index([str(g) for g in gene_names])
    return adata


def _download_saunders(study_path: str, data_dir: str) -> None:
    """Download Saunders brain atlas DGE By Class from DropViz."""
    adatas = []
    for cell_class, url in _SAUNDERS_CLASS_FILES.items():
        fname = os.path.basename(url)
        dge_path = os.path.join(data_dir, fname)
        _download_geo_file(url, dge_path)
        adata = _read_dge_sparse(dge_path)
        adata.obs[LABEL_COL] = cell_class
        adatas.append(adata)
        console.print(f'  Saunders {cell_class}: {adata.n_obs:,} cells')
        os.remove(dge_path)

    adata = ad.concat(adatas, join='inner')
    adata.var_names_make_unique()
    adata.write_h5ad(study_path)


def _download_rosenberg(study_path: str, data_dir: str) -> None:
    """Download Rosenberg SPLiT-seq CNS nuclei .mat file."""
    import gzip
    import shutil

    mat_gz_path = os.path.join(data_dir, 'rosenberg.mat.gz')
    mat_path = os.path.join(data_dir, 'rosenberg.mat')
    _download_geo_file(_ROSENBERG_URL, mat_gz_path)

    # Decompress
    with gzip.open(mat_gz_path, 'rb') as f_in, open(mat_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(mat_gz_path)

    mat = scipy.io.loadmat(mat_path)
    os.remove(mat_path)

    # Follow rosen.py format exactly
    X = scipy.sparse.csr_matrix(mat['DGE'].T)
    n_cells, n_genes = X.shape

    # MATLAB cell arrays of strings → extract properly
    def _extract_strings(arr):
        """Unpack nested MATLAB string arrays into a flat list."""
        flat = arr.flatten()
        return [str(x[0]).strip() if hasattr(x, '__len__') else str(x).strip()
                for x in flat]

    raw_genes = _extract_strings(mat['genes'])
    # If gene list doesn't match DGE columns, use generic names
    if len(raw_genes) != n_genes:
        console.print(f'    [yellow]genes field has {len(raw_genes)} entries '
                      f'but DGE has {n_genes} columns — using generic names[/yellow]')
        raw_genes = [f'Gene_{i}' for i in range(n_genes)]

    cluster_assignment = _extract_strings(mat['cluster_assignment'])

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(raw_genes)
    adata.obs[LABEL_COL] = cluster_assignment[:n_cells]

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.write_h5ad(study_path)
    console.print(f'  Rosenberg: {adata.n_obs:,} cells, '
                  f'{adata.obs[LABEL_COL].nunique()} types')


def _download_zeisel(study_path: str, data_dir: str) -> None:
    """Download Zeisel cortex expression data (GEO)."""
    txt_path = os.path.join(data_dir, 'zeisel_expression.txt.gz')
    _download_geo_file(_ZEISEL_URL, txt_path)

    # GEO format: row 0 = "cell_id" header + cell IDs, rows 1+ = gene expression
    adata = _read_dge_sparse(txt_path)
    os.remove(txt_path)
    adata.var_names_make_unique()

    _cluster_adata(adata)
    adata.obs[LABEL_COL] = adata.obs['Celltype']

    adata.write_h5ad(study_path)
    console.print(f'  Zeisel: {adata.n_obs:,} cells, '
                  f'{adata.obs[LABEL_COL].nunique()} types')



_STUDY_DOWNLOADERS = {
    'Tabula muris': _download_tabula_muris_brain,
    'Saunders': _download_saunders,
    'Rosenberg': _download_rosenberg,
    'Zeisel': _download_zeisel,
}


def setup(data_dir: str = DATA_DIR, seed: int = 1) -> str:
    """Download, split, and save mBrain dataset."""
    h5ad_path = os.path.join(data_dir, 'mBrain.h5ad')

    if (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {data_dir}')
        return data_dir

    console.print('[bold]Setting up mBrain (multi-study)[/bold]')
    os.makedirs(data_dir, exist_ok=True)

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
