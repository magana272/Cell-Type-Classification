import torch

from .config import TrainConfig
from .CellTypeMLP import MLP_Model, TRAIN_CONFIG as _mlp_tc
from .CellTypeCNN import CellTypeCNN, ResBlock, TRAIN_CONFIG as _cnn_tc
from .CellTypeAttention import TOSICA, MaskedEmbedding, TRAIN_CONFIG as _tosica_tc
from .CellTypeGNN import CellTypeGNN, TRAIN_CONFIG as _gnn_tc

AVAILABLE_MODELS = ["CellTypeCNN", "CellTypeTOSICA", "CellTypeMLP", "CellTypeGNN"]

_CHANNEL_DIM_MODELS = {"CellTypeCNN"}


def needs_channel_dim(model_name: str) -> bool:
    return model_name in _CHANNEL_DIM_MODELS


def _identity_mask(n_genes: int) -> torch.Tensor:
    return torch.eye(n_genes)


def get_model(
    name: str,
    n_genes: int,
    n_classes: int,
    *,
    mask: torch.Tensor | None = None,
    n_pathways: int | None = None,
    embed_dim: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    n_stages: int = 3,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    use_checkpointing: bool = False,
) -> torch.nn.Module:
    if name == "CellTypeCNN":
        return CellTypeCNN(seq_len=n_genes, n_classes=n_classes,
                           dropout=dropout, n_stages=n_stages,
                           use_checkpointing=use_checkpointing)

    if name == "CellTypeMLP":
        return MLP_Model(input_dim=n_genes, n_classes=n_classes,
                         dropout=dropout, n_layers=n_layers,
                         hidden_dim=hidden_dim)

    if name == "CellTypeGNN":
        return CellTypeGNN(in_dim=n_genes, hidden_dim=hidden_dim,
                           n_classes=n_classes, dropout=dropout,
                           n_layers=n_layers)

    if name == "CellTypeTOSICA":
        if mask is None:
            n_pw = n_pathways or n_genes
            mask = _identity_mask(n_genes) if n_pw == n_genes else torch.ones(n_genes, n_pw)
        else:
            n_pw = mask.shape[1]
        return TOSICA(
            n_genes=n_genes,
            n_pathways=n_pw,
            n_classes=n_classes,
            mask=mask,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown model '{name}'. Choose from: {AVAILABLE_MODELS}"
    )


TRAIN_CONFIGS: dict[str, TrainConfig] = {
    'CellTypeMLP': _mlp_tc,
    'CellTypeCNN': _cnn_tc,
    'CellTypeTOSICA': _tosica_tc,
    'CellTypeGNN': _gnn_tc,
}


def get_train_config(name: str) -> TrainConfig | None:
    return TRAIN_CONFIGS.get(name)
