"""Shared training/evaluation workflow for all 5 models on a single dataset.

Called by 5_hPancreas.py, 5_mPancreas.py, 5_mAtlas.py with dataset-specific config.
"""
from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from rich.console import Console
from rich.panel import Panel

from allen_brain.TOSICA.train import set_seed
from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig
from allen_brain.models.gnn_train import GraphTrainer, train_graph
from allen_brain.models.CellTypeGNN import GraphBuilder
from allen_brain.models.CellTypeAttention import PathwayMaskBuilder
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_preprocess import select_hvg

console = Console()

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Shared defaults
SEED = 1
N_HVG = 10_000
EPOCHS = 20
LRF = 0.01
NORMALIZE = 'None'

# GMT paths
MOUSE_GMT = os.path.join(_ROOT, 'allen_brain', 'TOSICA', 'resources', 'm_reactome.gmt')
HUMAN_GMT = os.path.join(_ROOT, 'allen_brain', 'TOSICA', 'resources', 'reactome.gmt')
MOUSE_GMT_URL = ('https://data.broadinstitute.org/gsea-msigdb/msigdb/'
                 'release/2023.2.Mm/m2.cp.reactome.v2023.2.Mm.symbols.gmt')
HUMAN_GMT_URL = ('https://data.broadinstitute.org/gsea-msigdb/msigdb/'
                 'release/2023.2.Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt')


def _make_scheduler(optimizer, epochs):
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - LRF) + LRF
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def train_mlp(data_dir: str, tag: str, csv_path: str) -> None:
    console.print(Panel(f'[bold]MLP — {tag}[/bold]', border_style='cyan'))
    cfg = ExperimentConfig(model='CellTypeMLP', seed=SEED, batch_size=8192,
                           epochs=EPOCHS, normalize=NORMALIZE, lr=0.01)
    trainer = T.Trainer(cfg)
    tl, vl, hvg_idx, scaler = trainer.make_dataloaders(data_dir, n_hvg=N_HVG,
                                                         normalize=NORMALIZE)
    ds = tl.dataset
    model = T.build_model('CellTypeMLP', len(ds.gene_names), ds.n_classes)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.01, momentum=0.9, weight_decay=5e-5)
    scheduler = _make_scheduler(optimizer, EPOCHS)
    writer, ckpt = T.make_writer_and_ckpt(cfg, len(ds.gene_names), data_tag=tag)
    ckpt_dir = os.path.dirname(ckpt)
    T._save_model_kwargs(ckpt_dir, {})
    if hvg_idx is not None:
        np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
    if scaler is not None:
        with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
        f.write(NORMALIZE)
    T.print_header()
    T.train(model, (tl, vl), torch.nn.CrossEntropyLoss(), optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=True)
    metrics = trainer.evaluate(data_dir, ckpt, squeeze_channel=True)
    T.append_results_csv(f'MLP_{tag}', metrics, csv_path=csv_path)


def train_cnn(data_dir: str, tag: str, csv_path: str) -> None:
    console.print(Panel(f'[bold]CNN — {tag}[/bold]', border_style='cyan'))
    cfg = ExperimentConfig(model='CellTypeCNN', seed=SEED, batch_size=16384,
                           epochs=EPOCHS, normalize=NORMALIZE, n_hvg=N_HVG)
    trainer = T.Trainer(cfg)
    tl, vl, hvg_idx, scaler = trainer.make_dataloaders(data_dir, n_hvg=N_HVG,
                                                         normalize=NORMALIZE)
    ds = tl.dataset
    model = T.build_model('CellTypeCNN', len(ds.gene_names), ds.n_classes)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=5e-5)
    scheduler = _make_scheduler(optimizer, EPOCHS)
    writer, ckpt = T.make_writer_and_ckpt(cfg, len(ds.gene_names), data_tag=tag)
    ckpt_dir = os.path.dirname(ckpt)
    T._save_model_kwargs(ckpt_dir, {})
    if hvg_idx is not None:
        np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
    if scaler is not None:
        with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
        f.write(NORMALIZE)
    T.print_header()
    T.train(model, (tl, vl), torch.nn.CrossEntropyLoss(), optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=False)
    metrics = trainer.evaluate(data_dir, ckpt, squeeze_channel=False)
    T.append_results_csv(f'CNN_{tag}', metrics, csv_path=csv_path)


def train_gnn(data_dir: str, tag: str, csv_path: str) -> None:
    console.print(Panel(f'[bold]GNN — {tag}[/bold]', border_style='cyan'))
    K_NEIGHBORS = 10
    cfg = ExperimentConfig(model='CellTypeGNN', seed=SEED, batch_size=256,
                           epochs=EPOCHS, k_neighbors=K_NEIGHBORS,
                           normalize=NORMALIZE)
    set_seed(SEED)
    gb = GraphBuilder(k_neighbors=K_NEIGHBORS, normalize=NORMALIZE)
    data = gb.build_graph_data(data_dir, n_hvg=N_HVG).to(T.DEVICE)
    n_classes = int(data.y.max().item()) + 1
    class_names = list(np.load(f'{data_dir}/class_names.npy', allow_pickle=True))
    n_features = data.x.shape[1]
    model = T.build_model('CellTypeGNN', n_features, n_classes)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=5e-5)
    scheduler = _make_scheduler(optimizer, EPOCHS)
    writer, ckpt = T.make_writer_and_ckpt(cfg, n_features, data_tag=tag)
    T._save_model_kwargs(os.path.dirname(ckpt), {})
    T.print_header()
    train_graph(model, data, torch.nn.CrossEntropyLoss(), optimizer, scheduler,
                EPOCHS, writer, ckpt)
    trainer = GraphTrainer(cfg)
    metrics = trainer.evaluate(data, ckpt, n_features, n_classes,
                               class_names=class_names)
    T.append_results_csv(f'GNN_{tag}', metrics, csv_path=csv_path)


def train_transformer(data_dir: str, tag: str, gmt_path: str, gmt_url: str,
                      csv_path: str) -> None:
    console.print(Panel(f'[bold]Transformer — {tag}[/bold]', border_style='cyan'))
    cfg = ExperimentConfig(model='CellTypeTOSICA', seed=SEED, batch_size=64,
                           epochs=EPOCHS, n_hvg=N_HVG)
    set_seed(SEED)
    trainer = T.Trainer(cfg)
    tl, vl, hvg_idx, _ = trainer.make_dataloaders(data_dir, n_hvg=N_HVG)
    ds = tl.dataset
    gene_names = [str(g) for g in ds.gene_names]
    mask, n_pathways = PathwayMaskBuilder(
        gmt_path=gmt_path, gmt_url=gmt_url,
        min_overlap=5, max_pathways=300, max_gene_set_size=300,
    ).build_mask(gene_names)
    extra_kw = dict(mask=mask, n_pathways=n_pathways, n_layers=2)
    model = T.build_model('CellTypeTOSICA', len(ds.gene_names), ds.n_classes,
                          **extra_kw)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=5e-5)
    scheduler = _make_scheduler(optimizer, EPOCHS)
    writer, ckpt = T.make_writer_and_ckpt(cfg, len(ds.gene_names), data_tag=tag)
    ckpt_dir = os.path.dirname(ckpt)
    if hvg_idx is not None:
        np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
    T._save_model_kwargs(ckpt_dir, extra_kw)
    T.print_header()
    T.train(model, (tl, vl), torch.nn.CrossEntropyLoss(), optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=True)
    metrics = trainer.evaluate(data_dir, ckpt, squeeze_channel=True,
                               extra_model_kwargs=extra_kw)
    T.append_results_csv(f'Transformer_{tag}', metrics, csv_path=csv_path)


def train_tosica(data_dir: str, tag: str, gmt_path: str, csv_path: str) -> None:
    console.print(Panel(f'[bold]TOSICA — {tag}[/bold]', border_style='cyan'))
    import anndata as ad
    import scanpy as sc
    sys.path.insert(0, os.path.join(_ROOT, 'TOSICA'))
    import allen_brain.TOSICA as TOSICA
    from allen_brain.models.train import _compute_metrics

    set_seed(SEED)
    BATCH_SIZE = 64
    PROJECT = f'tosica_{tag}'

    ds_train = make_dataset(data_dir, split='train')
    ds_test = make_dataset(data_dir, split='test')

    if N_HVG and 0 < N_HVG < len(ds_train.gene_names):
        X_dense = ds_train.X.toarray() if scipy.sparse.issparse(ds_train.X) else np.asarray(ds_train.X)
        hvg_idx = np.sort(select_hvg(X_dense, N_HVG))
        for ds in (ds_train, ds_test):
            X_sub = ds.X[:, hvg_idx]
            ds.X = X_sub.toarray() if scipy.sparse.issparse(X_sub) else np.asarray(X_sub)
            ds._sparse = False
            ds.gene_names = ds.gene_names[hvg_idx]

    gene_names = [str(g) for g in ds_train.gene_names]
    all_class_names = list(ds_train.class_names)

    X_train = np.asarray(ds_train.X).astype(np.float32)
    y_str = [all_class_names[int(yi)] for yi in ds_train.y]
    train_adata = ad.AnnData(X=X_train, var=pd.DataFrame(index=gene_names))
    train_adata.obs['cell_type'] = y_str

    TOSICA.train(train_adata, gmt_path=gmt_path, project=PROJECT,
                 label_name='cell_type', batch_size=BATCH_SIZE,
                 epochs=EPOCHS, max_gs=300, max_g=300)

    X_test = np.asarray(ds_test.X).astype(np.float32)
    y_test = np.asarray(ds_test.y)
    test_adata = ad.AnnData(X=X_test, var=pd.DataFrame(index=gene_names))

    bs = BATCH_SIZE
    while len(ds_test) > 1 and len(ds_test) % bs == 1:
        bs += 1

    result = TOSICA.pre(test_adata,
                        model_weight_path=f'./{PROJECT}/model-{EPOCHS - 1}.pth',
                        project=PROJECT, batch_size=bs, laten=True)

    predictions = result.obs['Prediction'].values.astype(str)
    name_to_idx = {n: i for i, n in enumerate(all_class_names)}
    y_pred_int = np.array([name_to_idx.get(p, -1) for p in predictions])

    metrics = _compute_metrics(y_test, y_pred_int, all_class_names,
                               save_dir=PROJECT)
    T.append_results_csv(f'TOSICA_{tag}', metrics, csv_path=csv_path)


def compare_to_baselines(tag: str, csv_path: str) -> None:
    """Print a ranked table comparing our models to published TOSICA baselines."""
    from rich.table import Table
    from allen_brain.cell_data.tosica_baselines import PUBLISHED_ACCURACY

    published = PUBLISHED_ACCURACY.get(tag, {})
    if not published:
        console.print(f'[yellow]No published baselines for {tag}[/yellow]')
        return

    # Load our results
    ours: dict[str, float] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            ours[row['model']] = row['accuracy']

    # Merge: published + ours
    combined: dict[str, float | None] = dict(published)
    for model_name, acc in ours.items():
        combined[f'Ours-{model_name}'] = acc

    # Sort by accuracy descending
    ranked = sorted(combined.items(), key=lambda x: x[1] if x[1] is not None else -1,
                    reverse=True)

    table = Table(title=f'{tag}: All Methods Ranked by Accuracy', show_lines=True)
    table.add_column('Rank', justify='right', style='dim')
    table.add_column('Method', style='bold')
    table.add_column('Accuracy', justify='right')

    for rank, (method, acc) in enumerate(ranked, 1):
        acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
        if method.startswith('Ours-'):
            table.add_row(str(rank), f'[bold blue]{method}[/bold blue]', acc_str)
        else:
            table.add_row(str(rank), method, acc_str)

    console.print(table)


def run_all(data_dir: str, tag: str, gmt_path: str, gmt_url: str,
            csv_path: str = 'results.csv') -> None:
    """Train and evaluate all 5 models on one dataset, then compare to baselines."""
    console.print(Panel(f'[bold green]Dataset: {tag}[/bold green]  ·  {data_dir}',
                        border_style='green'))
    set_seed(SEED)
    train_mlp(data_dir, tag, csv_path)
    train_cnn(data_dir, tag, csv_path)
    train_gnn(data_dir, tag, csv_path)
    train_transformer(data_dir, tag, gmt_path, gmt_url, csv_path)
    train_tosica(data_dir, tag, gmt_path, csv_path)
    compare_to_baselines(tag, csv_path)
    console.print(Panel(f'[bold green]Done: {tag}[/bold green]', border_style='green'))
