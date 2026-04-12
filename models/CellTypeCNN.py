import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GeneExpressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, add_channel: bool = True):
        if add_channel:
            self.X = torch.from_numpy(X[:, np.newaxis, :])  # (n, 1, g)
        else:
            self.X = torch.from_numpy(X)                     # (n, g)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> Dropout -> GELU -> optional MaxPool."""

    def __init__(self, in_ch, out_ch, kernel, pool=None, padding="same"):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(0.1),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool1d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CellTypeCNN_3Layer(nn.Module):
    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding="same", bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=9, padding="same", bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=7, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CellTypeCNN(nn.Module):
    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding="same", bias=False),
            ConvBlock(64, 128, kernel=15, pool=4),
            ConvBlock(128, 256, kernel=9, pool=4),
            ConvBlock(256, 512, kernel=7, pool=4),
            ConvBlock(512, 512, kernel=5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def batch_correction_combat(X: np.ndarray, batch=None) -> np.ndarray:
    if batch is None:
        print("No batch labels supplied; skipping ComBat correction.")
        return X

    batch = np.asarray(batch)
    if len(np.unique(batch)) < 2:
        print("Only one batch present; skipping ComBat (needs >= 2 batches).")
        return X

    try:
        from inmoose.pycombat import pycombat_norm

        corrected = pycombat_norm(X.T, batch)
        print("ComBat correction applied (inmoose).")
        return corrected.T.astype(np.float32)
    except ImportError:
        pass
    try:
        from combat.pycombat import pycombat
        import pandas as _pd

        corrected = pycombat(_pd.DataFrame(X.T), batch)
        print("ComBat correction applied (pyComBat).")
        return corrected.values.T.astype(np.float32)
    except ImportError:
        pass

    print("Warning: neither inmoose nor pyComBat found; skipping batch correction.")
    return X
