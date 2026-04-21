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
