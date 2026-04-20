import math
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from allen_brain.TOSICA.train import set_seed
from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig

DATA_DIR = 'data/mPancreas'
SEED = 1
BATCH_SIZE = 16384
N_HVG = 10000
EPOCHS = 20
LR = 0.001
LRF = 0.01
NORMALIZE = 'None'

cfg = ExperimentConfig(
    model='CellTypeCNN',
    seed=SEED,
    batch_size=BATCH_SIZE,
    n_hvg=N_HVG,
    epochs=EPOCHS,
    normalize=NORMALIZE,
)


def main() -> None:
    set_seed(SEED)
    trainer = T.Trainer(cfg)
    train_loader, val_loader, hvg_idx, scaler = trainer.make_dataloaders(
        DATA_DIR, n_hvg=N_HVG, normalize=NORMALIZE)
    ds = train_loader.dataset

    model = T.build_model('CellTypeCNN', len(ds.gene_names), ds.n_classes)

    criterion = torch.nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=LR, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / EPOCHS)) / 2) * (1 - LRF) + LRF
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    writer, ckpt = T.make_writer_and_ckpt(cfg, len(ds.gene_names))
    ckpt_dir = os.path.dirname(ckpt)
    T._save_model_kwargs(ckpt_dir, {})
    if hvg_idx is not None:
        np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
    if scaler is not None:
        with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
        f.write(NORMALIZE)

    T.print_header()
    T.train(model, (train_loader, val_loader), criterion, optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=False)

    metrics = trainer.evaluate(DATA_DIR, ckpt, squeeze_channel=False)
    T.append_results_csv('CNN', metrics)


if __name__ == '__main__':
    main()
