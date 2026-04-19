"""Compare our 4 models against 19 published annotators on TOSICA benchmarks.

Trains MLP, CNN, Transformer (TOSICA), and GNN on the 6 TOSICA benchmark
datasets, then merges results with published accuracy from Chen et al. (2023)
and generates comparison figures.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from allen_brain.cell_data.cell_load import ALL_DATASETS
from allen_brain.cell_data.cell_vis import ModelComparisonVisualizer
from allen_brain.cell_data.tosica_baselines import (
    PUBLISHED_ACCURACY,
    TOSICA_DATASET_NAMES,
)
from allen_brain.data_sets import TOSICA_DATASETS
from allen_brain.models import train as T
from allen_brain.models.config import EvalMetrics

# Reuse training infrastructure from 9_multiple_dataset
from importlib import import_module
_multi = import_module('9_multiple_dataset')
train_and_eval_model = _multi.train_and_eval_model
MODELS = _multi.MODELS

console: Console = Console()

SAVE_DIR: str = 'figures/tosica_benchmark'
OUR_MODEL_NAMES: list[str] = ['Ours-MLP', 'Ours-CNN', 'Ours-Transformer', 'Ours-GNN']


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 1. Ensure datasets are set up --------------------------------------
    console.print(Panel('[bold]TOSICA Benchmark: Setup Datasets[/bold]',
                        border_style='cyan'))
    for ds_name, mod in TOSICA_DATASETS.items():
        data_dir = ALL_DATASETS[ds_name]['dir']
        if not (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
                or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
            try:
                mod.setup()
            except Exception as e:
                console.print(f'[red]Failed to set up {ds_name}: {e}[/red]')

    # --- 2. Train & evaluate our 4 models on each dataset --------------------
    console.print(Panel('[bold]TOSICA Benchmark: Train & Evaluate[/bold]',
                        border_style='cyan'))

    our_results: dict[str, dict[str, EvalMetrics | None]] = {}
    model_keys = list(MODELS.keys())  # MLP, CNN, Transformer, GNN

    for ds_name in TOSICA_DATASET_NAMES:
        if ds_name not in ALL_DATASETS:
            continue
        data_dir = ALL_DATASETS[ds_name]['dir']
        if not (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
                or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
            console.print(f'[yellow]Skipping {ds_name}: no data[/yellow]')
            continue

        console.print(Panel(f'[bold]Dataset: {ds_name}[/bold]',
                            border_style='green'))
        our_results[ds_name] = {}

        for model_label, model_cfg in MODELS.items():
            console.print(f'  Training {model_label} on {ds_name}...')
            try:
                metrics = train_and_eval_model(
                    model_label, model_cfg, data_dir, ds_name)
                our_results[ds_name][model_label] = metrics
                if metrics:
                    T.append_results_csv(
                        f'{model_label}_{ds_name}', metrics,
                        csv_path='results_tosica_benchmark.csv')
            except Exception as e:
                console.print(f'  [red]{model_label} failed: {e}[/red]')
                our_results[ds_name][model_label] = None

    # --- 3. Merge our results with published baselines -----------------------
    console.print(Panel('[bold]TOSICA Benchmark: Comparison[/bold]',
                        border_style='cyan'))

    combined: dict[str, dict[str, float | None]] = {}
    for ds_name in TOSICA_DATASET_NAMES:
        published = dict(PUBLISHED_ACCURACY.get(ds_name, {}))
        ours = our_results.get(ds_name, {})

        for model_key in model_keys:
            our_name = f'Ours-{model_key}'
            m = ours.get(model_key)
            if m is not None:
                published[our_name] = m.accuracy
        combined[ds_name] = published

    # --- 4. Generate comparison figures --------------------------------------
    ModelComparisonVisualizer.plot_annotator_comparison_heatmap(
        combined, OUR_MODEL_NAMES, SAVE_DIR)

    ModelComparisonVisualizer.plot_mean_accuracy_bar(
        combined, OUR_MODEL_NAMES, SAVE_DIR)

    # Our-models-only heatmap
    if our_results:
        ModelComparisonVisualizer.plot_metric_heatmap(
            our_results, 'accuracy', model_keys, SAVE_DIR,
            title='Our Models: Accuracy on TOSICA Benchmarks',
            filename='our_models_accuracy_heatmap.png')
        ModelComparisonVisualizer.plot_metric_heatmap(
            our_results, 'f1_macro', model_keys, SAVE_DIR,
            title='Our Models: F1-Macro on TOSICA Benchmarks',
            cmap='YlOrRd', filename='our_models_f1_heatmap.png')

    # --- 5. Print summary table ----------------------------------------------
    _print_comparison_table(combined)


def _print_comparison_table(
    combined: dict[str, dict[str, float | None]],
) -> None:
    """Print Rich table comparing all methods across datasets."""
    datasets = list(combined.keys())
    all_methods: set[str] = set()
    for d in combined.values():
        all_methods.update(d.keys())

    # Compute mean accuracy
    method_means: list[tuple[str, float]] = []
    for m in all_methods:
        vals = [combined[ds].get(m) for ds in datasets]
        valid = [v for v in vals if v is not None]
        method_means.append((m, np.mean(valid) if valid else 0.0))
    method_means.sort(key=lambda x: x[1], reverse=True)

    table = Table(title='TOSICA Benchmark: All Methods Ranked by Mean Accuracy',
                  show_lines=True)
    table.add_column('Rank', justify='right', style='dim')
    table.add_column('Method', style='bold')
    for ds in datasets:
        table.add_column(ds, justify='right')
    table.add_column('Mean', justify='right', style='bold')

    for rank, (method, mean_acc) in enumerate(method_means, 1):
        row = [str(rank)]
        if method.startswith('Ours-'):
            row.append(f'[bold blue]{method}[/bold blue]')
        else:
            row.append(method)
        for ds in datasets:
            v = combined[ds].get(method)
            row.append(f'{v:.4f}' if v is not None else 'N/A')
        row.append(f'{mean_acc:.4f}')
        table.add_row(*row)

    console.print(table)


if __name__ == '__main__':
    main()
