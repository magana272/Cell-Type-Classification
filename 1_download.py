"""Download all datasets and process into train/val/test .npy splits."""

from allen_brain.TOSICA.train import set_seed
from allen_brain.cell_data.cell_download import download_data
from allen_brain.cell_data.cell_load import (
    load_10x, load_smartseq, load_pbmc, load_pancreas,
    load_tabula_muris, load_lung,
)
from allen_brain.data_sets import TOSICA_DATASETS


def main():
    # # Download raw files (CSVs + h5ad)
    download_data()

    load_10x()
    load_smartseq()
    # load_pbmc()
    # load_pancreas()
    # #load_tabula_muris()
    # load_lung()

    # TOSICA benchmark datasets
    set_seed(1)
    for name, mod in TOSICA_DATASETS.items():
        try:
            mod.setup()
        except Exception as e:
            print(f'[WARN] {name}: {e}')


if __name__ == '__main__':
    main()
