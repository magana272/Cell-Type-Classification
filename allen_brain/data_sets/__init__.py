"""TOSICA benchmark dataset registry.

Each module exposes a ``setup(data_dir=...)`` function that downloads,
splits (by the paper's biological condition), and saves NumPy arrays
ready for ``GeneExpressionDataset``.
"""
from __future__ import annotations

from allen_brain.data_sets import (
    hArtery,
    hBone,
    hPancreas,
    mAtlas,
    mBrain,
    mPancreas,
)

TOSICA_DATASETS: dict[str, object] = {
    'hArtery': hArtery,
    'hBone': hBone,
    'hPancreas': hPancreas,
    'mBrain': mBrain,
    'mPancreas': mPancreas,
    'mAtlas': mAtlas,
}
