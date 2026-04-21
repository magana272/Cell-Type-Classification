from __future__ import annotations

import os
import sys

from allen_brain.TOSICA.train import set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TOSICA'))

import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
import allen_brain.TOSICA as TOSICA

from allen_brain.cell_data.cell_dataset import make_dataset, GeneExpressionDataset
from allen_brain.cell_data.cell_preprocess import select_hvg
from allen_brain.models.train import _compute_metrics, append_results_csv

DATA_DIR: str = 'data/mPancreas'
_ROOT: str = os.path.dirname(os.path.abspath(__file__))
GMT_PATH: str = os.path.join(_ROOT, 'allen_brain', 'TOSICA', 'resources', 'm_reactome.gmt')
PROJECT: str = 'tosica_run'
LABEL_COL: str = 'cell_type'
N_HVG: int = 10_000
EPOCHS: int = 20
BATCH_SIZE: int = 64
SEED: int = 1


def main() -> None:
    set_seed(SEED)
    ds_train: GeneExpressionDataset = make_dataset(DATA_DIR, split='train')
    ds_test: GeneExpressionDataset = make_dataset(DATA_DIR, split='test')

    if N_HVG and 0 < N_HVG < len(ds_train.gene_names):
        X_train_dense = ds_train.X.toarray() if scipy.sparse.issparse(ds_train.X) else np.asarray(ds_train.X)
        hvg_idx: np.ndarray = np.sort(select_hvg(X_train_dense, N_HVG))
        for ds in (ds_train, ds_test):
            X_sub = ds.X[:, hvg_idx]
            ds.X = X_sub.toarray() if scipy.sparse.issparse(X_sub) else np.asarray(X_sub)
            ds._sparse = False
            ds.gene_names = ds.gene_names[hvg_idx]

    gene_names: list[str] = [str(g) for g in ds_train.gene_names]
    all_class_names: list[str] = list(ds_train.class_names)

    X_train: np.ndarray = np.asarray(ds_train.X).astype(np.float32)
    y_str: list[str] = [all_class_names[int(yi)] for yi in ds_train.y]

    train_adata: ad.AnnData = ad.AnnData(X=X_train, var=pd.DataFrame(index=gene_names))
    train_adata.obs[LABEL_COL] = y_str

    TOSICA.train(
        train_adata,
        gmt_path=GMT_PATH,
        project=PROJECT,
        label_name=LABEL_COL,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_gs=300,
        max_g=300,
    )

    X_test: np.ndarray = np.asarray(ds_test.X).astype(np.float32)
    y_test: np.ndarray = np.asarray(ds_test.y)

    test_adata: ad.AnnData = ad.AnnData(X=X_test, var=pd.DataFrame(index=gene_names))

    bs: int = BATCH_SIZE
    while len(ds_test) > 1 and len(ds_test) % bs == 1:
        bs += 1

    result: ad.AnnData = TOSICA.pre(
        test_adata,
        model_weight_path=f'./{PROJECT}/model-{EPOCHS - 1}.pth',
        project=PROJECT,
        batch_size=bs,
        laten=True,
    )

    predictions: np.ndarray = result.obs['Prediction'].values.astype(str)
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(all_class_names)}
    y_pred_int: np.ndarray = np.array([name_to_idx.get(p, -1) for p in predictions])

    metrics = _compute_metrics(y_test, y_pred_int, all_class_names, save_dir=PROJECT)
    append_results_csv('TOSICA', metrics)


if __name__ == '__main__':
    main()
