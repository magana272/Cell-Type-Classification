from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_vis import DatasetVisualizer
from allen_brain.cell_data.tosica_baselines import PUBLISHED_ACCURACY

console = Console()

SAVE_DIR = 'figures'
DPI = 300
SEED = 42

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUR_MODELS = ['MLP', 'CNN', 'GNN', 'Transformer', 'TOSICA']
OUR_COLORS = {'MLP': '#1f77b4', 'CNN': '#ff7f0e', 'GNN': '#2ca02c',
              'Transformer': '#d62728', 'TOSICA': '#9467bd'}


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    console.print(f'[green]Saved[/green] {path}')


def fig_dataset_overview() -> None:
    console.print('[bold]Fig 1: Dataset overview[/bold]')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (data_dir, title) in zip(axes, [
        ('data/hPancreas', 'hPancreas (Human, 14 classes)'),
        ('data/mPancreas', 'mPancreas (Mouse, 21 classes)'),
        ('data/mAtlas', 'mAtlas (Mouse, 120 classes)'),
    ]):
        ds = make_dataset(data_dir, split='train')
        classes = ds.class_names
        counts = np.bincount(np.asarray(ds.y), minlength=ds.n_classes)
        order = np.argsort(counts)[::-1]
        ax.barh(range(len(classes)), counts[order], color='steelblue', edgecolor='none')
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([classes[i] for i in order], fontsize=7)
        ax.set_xlabel('Number of cells (train split)')
        ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    _save(fig, 'fig_dataset_overview.png')


def fig_umap(data_dir: str, tag: str) -> None:
    console.print(f'[bold]Fig: UMAP {tag}[/bold]')
    ds = make_dataset(data_dir, split='train')
    vis = DatasetVisualizer(ds, fig_dir=SAVE_DIR, seed=SEED)
    pca, X_pca = vis.plot_pca(n_components=30,
                              save_path=SAVE_DIR,
                              file_name=f'fig_{tag}_pca.png')
    vis.plot_umap(X_pca, max_cells=6000,
                  save_path=os.path.join(SAVE_DIR, f'fig_{tag}_umap.png'))


def fig_results_bar(csv_path: str, dataset: str, tag: str) -> None:
    console.print(f'[bold]Fig: Results bar {tag}[/bold]')
    if not os.path.exists(csv_path):
        console.print(f'[yellow]  {csv_path} not found, skipping[/yellow]')
        return

    df = pd.read_csv(csv_path)
    our_acc: dict[str, float] = {}
    for _, row in df.iterrows():
        name = row['model'].replace(f'_{dataset}', '')
        our_acc[name] = row['accuracy']

    published = PUBLISHED_ACCURACY.get(dataset, {})
    top_pub = sorted(published.items(), key=lambda x: x[1] or 0, reverse=True)[:5]

    methods: list[str] = []
    accs: list[float] = []
    colors: list[str] = []
    for name, acc in top_pub:
        if acc is not None:
            methods.append(name)
            accs.append(acc)
            colors.append('#aaaaaa')
    for name in OUR_MODELS:
        if name in our_acc:
            methods.append(f'Ours-{name}')
            accs.append(our_acc[name])
            colors.append(OUR_COLORS[name])

    order = np.argsort(accs)[::-1]
    methods = [methods[i] for i in order]
    accs = [accs[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.35)))
    bars = ax.barh(range(len(methods)), accs, color=colors, edgecolor='none')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'{dataset}: Model Comparison')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    _save(fig, f'fig_{tag}_results.png')


def fig_leaderboard() -> None:
    console.print('[bold]Fig: Leaderboard heatmap[/bold]')

    datasets = ['hPancreas', 'mPancreas', 'mAtlas']
    csv_map = {'hPancreas': 'results_hPancreas.csv',
               'mPancreas': 'results_mPancreas.csv',
               'mAtlas': 'results_mAtlas.csv'}

    all_methods: dict[str, dict[str, float | None]] = {}

    for ds_name in datasets:
        published = PUBLISHED_ACCURACY.get(ds_name, {})
        for method, acc in published.items():
            all_methods.setdefault(method, {})[ds_name] = acc

        csv_path = csv_map.get(ds_name, '')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                name = f"Ours-{row['model'].replace(f'_{ds_name}', '')}"
                all_methods.setdefault(name, {})[ds_name] = row['accuracy']

    methods_sorted = sorted(all_methods.keys(),
                            key=lambda m: np.nanmean([
                                all_methods[m].get(d) or 0 for d in datasets
                            ]), reverse=True)
    ours = [m for m in methods_sorted if m.startswith('Ours-')]
    others = [m for m in methods_sorted if not m.startswith('Ours-')][:12]
    methods_show = others + ours

    matrix = np.full((len(methods_show), len(datasets)), np.nan)
    for i, m in enumerate(methods_show):
        for j, d in enumerate(datasets):
            v = all_methods.get(m, {}).get(d)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(7, max(6, len(methods_show) * 0.3)))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGn',
                xticklabels=datasets, yticklabels=methods_show,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Accuracy: All Methods vs. Datasets')
    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text().startswith('Ours-'):
            label.set_fontweight('bold')
            label.set_color('#1f77b4')
    plt.tight_layout()
    _save(fig, 'fig_leaderboard.png')


def fig_model_comparison() -> None:
    console.print('[bold]Fig: Model comparison[/bold]')

    datasets = {'hPancreas': 'results_hPancreas.csv',
                'mPancreas': 'results_mPancreas.csv',
                'mAtlas': 'results_mAtlas.csv'}

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    metric_labels = ['Accuracy', 'F1 (macro)', 'F1 (weighted)']
    bar_width = 0.25

    for ax, (ds_name, csv_path) in zip(axes, datasets.items()):
        if not os.path.exists(csv_path):
            ax.set_title(f'{ds_name} (no results)')
            continue
        df = pd.read_csv(csv_path)
        models: list[str] = []
        values: dict[str, list[float]] = {m: [] for m in metrics}
        for _, row in df.iterrows():
            name = row['model'].replace(f'_{ds_name}', '')
            models.append(name)
            for m in metrics:
                values[m].append(row[m])

        x = np.arange(len(models))
        for i, (m, label) in enumerate(zip(metrics, metric_labels)):
            offset = (i - 1) * bar_width
            ax.bar(x + offset, values[m], bar_width, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Score')
        ax.set_title(ds_name)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=7)

    plt.suptitle('Model Performance Across Datasets', fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, 'fig_model_comparison.png')


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig_dataset_overview()

    for data_dir, tag in [('data/hPancreas', 'hpan'), ('data/mPancreas', 'mpan'),
                           ('data/mAtlas', 'matlas')]:
        if os.path.exists(os.path.join(data_dir, 'X_train.npy')) or \
           os.path.exists(os.path.join(data_dir, 'X_train.npz')):
            fig_umap(data_dir, tag)

    fig_results_bar('results_hPancreas.csv', 'hPancreas', 'hpan')
    fig_results_bar('results_mPancreas.csv', 'mPancreas', 'mpan')
    fig_results_bar('results_mAtlas.csv', 'mAtlas', 'matlas')

    fig_leaderboard()

    fig_model_comparison()

    console.print('\n[bold green]All figures generated.[/bold green]')


if __name__ == '__main__':
    main()
