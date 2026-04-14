
from torch import nn

class MLP_SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w
class MLPBlock(nn.Module):
    """Simple MLP block: Linear -> ReLU -> Dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)
    

class MLP_Model(nn.Module):
    """Simple MLP with two hidden layers and SE attention."""
    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = MLPBlock(input_dim, 512, dropout=dropout)
        self.se = MLP_SEBlock(512)
        self.fc2 = MLPBlock(512, 256, dropout=dropout)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)        
        x = self.fc1(x)
        x = self.se(x.unsqueeze(-1)).squeeze(-1)
        x = self.fc2(x)
        return self.classifier(x)