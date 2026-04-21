"""TOSICA benchmark dataset registry.

Each module exposes a ``setup(data_dir=...)`` function that downloads,
splits (by the paper's biological condition), and saves NumPy arrays
ready for ``GeneExpressionDataset``.
"""
from __future__ import annotations

from allen_brain.data_sets import (
    hPancreas,
    mAtlas,
    mPancreas,
)

TOSICA_DATASETS: dict[str, object] = {
    'hPancreas': hPancreas,
    'mPancreas': mPancreas,
    'mAtlas': mAtlas,
}
