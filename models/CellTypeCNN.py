import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> Dropout -> GELU -> optional MaxPool."""

    def __init__(self, in_ch, out_ch, kernel, pool=None, padding="same", dropout=0.2):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool1d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CellTypeCNN(nn.Module):
    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding="same", bias=False),
            ConvBlock(64, 64, kernel=3, pool=4, dropout=dropout),
            ConvBlock(64, 64, kernel=3, pool=4, dropout=dropout),
            ConvBlock(64, 64, kernel=3, pool=4, dropout=dropout),
            ConvBlock(64, 64, kernel=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 200),
            nn.ReLU(),
            nn.LayerNorm(200),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))