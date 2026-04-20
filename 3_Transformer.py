import math
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from allen_brain.TOSICA.train import set_seed
from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig
from allen_brain.models.CellTypeAttention import PathwayMaskBuilder
from allen_brain.cell_data.cell_dataset import make_dataset

DATA_DIR = 'data/mPancreas'
_ROOT = os.path.dirname(os.path.abspath(__file__))
GMT_PATH = os.path.join(_ROOT, 'allen_brain', 'TOSICA', 'resources', 'm_reactome.gmt')
GMT_URL = ('https://data.broadinstitute.org/gsea-msigdb/msigdb/'
           'release/2023.2.Mm/m2.cp.reactome.v2023.2.Mm.symbols.gmt')
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300
N_HVG = 10000
SEED = 1
BATCH_SIZE = 8192
EPOCHS = 20
LR = 0.001
LRF = 0.01

cfg = ExperimentConfig(
    model='CellTypeTOSICA',
    seed=SEED,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    n_hvg=N_HVG,
)


def _build_pathway_kwargs(gene_names: list[str]) -> dict:
    """Build extra_model_kwargs for TOSICA from (HVG-filtered) gene names."""
    mask, n_pathways = PathwayMaskBuilder(
        gmt_path=GMT_PATH, gmt_url=GMT_URL,
        min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE
    ).build_mask(gene_names)
    return dict(mask=mask, n_pathways=n_pathways)


def main() -> None:
    set_seed(SEED)
    trainer = T.Trainer(cfg)
    train_loader, val_loader, hvg_idx, _ = trainer.make_dataloaders(DATA_DIR, n_hvg=N_HVG)
    ds = train_loader.dataset

    extra_kw = _build_pathway_kwargs([str(g) for g in ds.gene_names])
    extra_kw.update(n_layers=2)
    model = T.build_model('CellTypeTOSICA', len(ds.gene_names), ds.n_classes, **extra_kw)

    criterion = torch.nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=LR, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / EPOCHS)) / 2) * (1 - LRF) + LRF
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    writer, ckpt = T.make_writer_and_ckpt(cfg, len(ds.gene_names))
    ckpt_dir = os.path.dirname(ckpt)
    if hvg_idx is not None:
        import numpy as np
        np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
    T._save_model_kwargs(ckpt_dir, extra_kw)

    T.print_header()
    T.train(model, (train_loader, val_loader), criterion, optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=True)

    metrics = trainer.evaluate(DATA_DIR, ckpt, squeeze_channel=True,
                               extra_model_kwargs=extra_kw)
    T.append_results_csv('Transformer', metrics)


if __name__ == '__main__':
    main()
