"""
models — Model registry and factory for cell-type classifiers.

Supported model names:
    CellTypeCNN         5-block 1D CNN  (seq_len, n_classes)
    CellTypeCNN_3Layer  3-block 1D CNN  (seq_len, n_classes)
    TOSICA              Transformer     (n_genes, n_pathways, n_classes, mask, ...)

Usage:
    from models import get_model, AVAILABLE_MODELS, needs_channel_dim
"""

import torch
import numpy as np

from .CellTypeCNN import CellTypeCNN, CellTypeCNN_3Layer, GeneExpressionDataset
from .CellTypeAttention import TOSICA, MaskedEmbedding

AVAILABLE_MODELS = ["CellTypeCNN", "CellTypeCNN_3Layer", "TOSICA"]

_CHANNEL_DIM_MODELS = {"CellTypeCNN", "CellTypeCNN_3Layer"}


def needs_channel_dim(model_name: str) -> bool:
    """Return True if the model expects a leading channel dimension."""
    return model_name in _CHANNEL_DIM_MODELS


def _identity_mask(n_genes: int) -> torch.Tensor:
    """Create a default identity-like pathway mask (each gene = its own pathway)."""
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
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name       : one of AVAILABLE_MODELS
    n_genes    : number of input genes (= seq_len for CNNs, = n_genes for TOSICA)
    n_classes  : number of cell-type classes
    mask       : (n_genes, n_pathways) binary tensor for TOSICA  (optional)
    n_pathways : override pathway count; defaults to n_genes when mask is None
    """
    if name in ("CellTypeCNN", "CellTypeCNN_3Layer"):
        cls = CellTypeCNN if name == "CellTypeCNN" else CellTypeCNN_3Layer
        return cls(seq_len=n_genes, n_classes=n_classes)

    if name == "TOSICA":
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
