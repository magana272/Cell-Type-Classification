import gc
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from .load import CellSplit


@dataclass
class PreprocessedSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    hvg_gene_idx: np.ndarray 
    scaler: StandardScaler


def gene_filter(
    X: np.ndarray,
    idx_val: np.ndarray,
    min_gene_frac: float,
) -> np.ndarray:
    """Keep genes expressed (>0) in at least max(3, frac*n_val) val cells."""
    n_val = len(idx_val)
    min_cells = max(3, int(min_gene_frac * n_val))
    gene_nonzero = (X[idx_val] > 0).sum(axis=0)
    filtered_gene_idx = np.flatnonzero(gene_nonzero >= min_cells)
    print(f'[Phase 1] Genes retained: {filtered_gene_idx.size:,} / {X.shape[1]:,} '
          f'(>= {min_cells} val cells)')
    return filtered_gene_idx


def select_hvg(
    X: np.ndarray,
    idx_val: np.ndarray,
    filtered_gene_idx: np.ndarray,
    n_hvg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-normalize val cells, return top-variance gene indices.

    Returns (hvg_gene_idx, hvg_local) — the first indexes into the original
    gene axis of X; the second indexes into ``filtered_gene_idx`` so callers
    can subset a pre-filtered (cells, n_filtered) block.
    """
    val_filt = X[idx_val][:, filtered_gene_idx].astype(np.float32, copy=False)
    lib = np.maximum(val_filt.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
    val_norm = np.log1p(val_filt / lib * 1e4).astype(np.float32)
    var = val_norm.var(axis=0)

    n_hvg_eff = min(n_hvg, var.size)
    top = np.argpartition(-var, n_hvg_eff - 1)[:n_hvg_eff]
    hvg_local = top[np.argsort(-var[top])]  # variance-descending
    hvg_gene_idx = filtered_gene_idx[hvg_local]
    print(f'[Phase 2] Top {n_hvg_eff} HVGs selected')
    return hvg_gene_idx, hvg_local


def normalize_rows(
    X: np.ndarray,
    rows: np.ndarray,
    filtered_gene_idx: np.ndarray,
    hvg_local: np.ndarray,
    row_chunk: int = 4096,
) -> np.ndarray:
    """Library-size normalize + log1p for ``rows``, returning only HVG columns.

    Library size is summed over *filtered* genes (matches the notebook's
    phase-3 pipeline), then HVG columns are divided by that library size.
    Runs in row chunks so the temporary (chunk, n_filtered) block stays small.
    """
    n_hvg = hvg_local.size
    out = np.empty((len(rows), n_hvg), dtype=np.float32)
    for start in range(0, len(rows), row_chunk):
        end = min(start + row_chunk, len(rows))
        block = X[rows[start:end]][:, filtered_gene_idx]
        lib = np.maximum(block.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
        hvg = block[:, hvg_local].astype(np.float32, copy=False)
        out[start:end] = np.log1p(hvg / lib.astype(np.float32) * 1e4)
    return out


def fit_scale(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, transform all three splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, scaler


def preprocess_hvg(
    split: CellSplit,
    n_hvg: int = 2000,
    min_gene_frac: float = 0.01,
    row_chunk: int = 4096,
) -> PreprocessedSplit:
    """Gene filter + HVG selection + log-norm + StandardScaler, in-memory.

    Mirrors the notebook's 3-phase polars pipeline on an already-loaded
    CellSplit. Gene filter and HVG selection use val cells only.
    """
    X = split.X

    filtered_gene_idx = gene_filter(X, split.idx_val, min_gene_frac)
    hvg_gene_idx, hvg_local = select_hvg(X, split.idx_val, filtered_gene_idx, n_hvg)

    X_train_hvg = normalize_rows(X, split.idx_train, filtered_gene_idx, hvg_local, row_chunk)
    X_val_hvg = normalize_rows(X, split.idx_val, filtered_gene_idx, hvg_local, row_chunk)
    X_test_hvg = normalize_rows(X, split.idx_test, filtered_gene_idx, hvg_local, row_chunk)
    print(f'[Phase 3] train {X_train_hvg.shape}, val {X_val_hvg.shape}, '
          f'test {X_test_hvg.shape}')

    X_train_s, X_val_s, X_test_s, scaler = fit_scale(X_train_hvg, X_val_hvg, X_test_hvg)
    del X_train_hvg, X_val_hvg, X_test_hvg
    gc.collect()

    return PreprocessedSplit(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        hvg_gene_idx=hvg_gene_idx,
        scaler=scaler,
    )
