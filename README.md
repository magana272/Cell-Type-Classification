# Deep Learning for Single-Cell RNA-seq Cell Type Classification

Benchmarking five deep learning architectures for automated cell type annotation from scRNA-seq data, evaluated on the [TOSICA benchmark](https://doi.org/10.1038/s41467-023-35923-4) (Chen et al., 2023).

## Key Results

| Dataset | Best Model | Accuracy | Benchmark Rank |
|---------|-----------|----------|----------------|
| hPancreas (14 classes) | GNN | 97.3% | 3rd of 24 methods |
| mPancreas (21 classes) | Transformer | 78.9% | 4th of 19 methods |
| mAtlas (120 classes) | MLP | 84.1% | 1st of 21 methods |

## Architectures

| Model | Key Mechanism | Params |
|-------|--------------|--------|
| **MLP** | Squeeze-and-excitation channel attention | 2.6M |
| **1D CNN** | Residual convolutions + SE + dual-pool head | 58K |
| **GraphSAGE GNN** | Cosine k-NN graph, transductive learning | 2.9M |
| **Transformer** | Reactome pathway-masked attention with [CLS] token | 144M |
| **TOSICA** | Published pathway-masked transformer (Chen et al.) | ~144M |

All models use cross-entropy loss, SGD with cosine annealing, and 10K highly variable genes.

## Project Structure

```
.
├── 1_download.py                # Download and preprocess datasets
├── 2_visualize.py               # Dataset visualization (UMAP, PCA, etc.)
├── 3_MLP.py                     # Train MLP
├── 3_CNN.py                     # Train 1D CNN
├── 3_GNN.py                     # Train GraphSAGE GNN
├── 3_Transformer.py             # Train pathway-masked Transformer
├── 3_TOSICA.py                  # Train using original TOSICA library
├── 5_hPancreas.py               # Evaluate all models on hPancreas
├── 5_mPancreas.py               # Evaluate all models on mPancreas
├── 5_mAtlas.py                  # Evaluate all models on mAtlas
├── 6_figures.py                 # Generate paper figures
├── allen_brain/
│   ├── models/                  # Model definitions and training loop
│   │   ├── CellTypeMLP.py
│   │   ├── CellTypeCNN.py
│   │   ├── CellTypeGNN.py
│   │   ├── CellTypeAttention.py # Pathway-masked Transformer
│   │   ├── train.py             # Shared training infrastructure
│   │   ├── gnn_train.py         # GNN-specific training (transductive)
│   │   ├── losses.py            # Cross-entropy and focal loss
│   │   ├── blocks.py            # SE block
│   │   └── config.py            # Hyperparameter configs
│   ├── TOSICA/                  # Original TOSICA implementation
│   ├── cell_data/               # Data loading and preprocessing
│   └── data_sets/               # Per-dataset download scripts
├── hyperparametertuning/        # Optuna tuning scripts
├── tests/                       # Unit tests
├── figures/                     # Generated figures
└── data/                        # Downloaded datasets (not tracked)
```

## Setup

```bash
pip install torch torchvision torch-geometric
pip install anndata scanpy optuna rich scikit-learn scipy
```

## Usage

```bash
# 1. Download datasets
python 1_download.py

# 2. Train models (example: MLP on mPancreas)
python 3_MLP.py

# 3. Evaluate on all datasets
python 5_hPancreas.py
python 5_mPancreas.py
python 5_mAtlas.py

# 4. Generate figures
python 6_figures.py
```

## Datasets

Three datasets from the TOSICA benchmark with condition-based train/test splits:

| Dataset | Species | Train | Test | Classes | Split Criterion |
|---------|---------|-------|------|---------|----------------|
| hPancreas | Human | 9,540 | 4,218 | 14 | Study of origin |
| mPancreas | Mouse | 22,918 | 10,886 | 21 | Dev. day != 15.5 |
| mAtlas | Mouse | 30,624 | 76,797 | 120 | Biological condition |

## Code Availability

The source code for this project is available at [github.com/magana272/CellTypeClassification](https://github.com/magana272/CellTypeClassification).

## References

Chen, J., Xu, H., Tao, W. et al. Transformer for one stop interpretable cell type annotation. *Nature Communications* 14, 223 (2023). https://doi.org/10.1038/s41467-023-35923-4
