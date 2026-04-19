import math
import os

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from allen_brain.TOSICA.train import set_seed
from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig
from allen_brain.models.gnn_train import GraphTrainer, train_graph
from allen_brain.models.CellTypeGNN import GraphBuilder

DATA_DIR = 'data/mPancreas'
SEED = 42
BATCH_SIZE = 256
K_NEIGHBORS = 10
EPOCHS = 200
LR = 0.001
LRF = 0.01
HVG  = 2000
NORMALIZE = 'None'

cfg = ExperimentConfig(
    model='CellTypeGNN',
    seed=SEED,
    batch_size=BATCH_SIZE,
    n_hvg=0,
    epochs=EPOCHS,
    k_neighbors=K_NEIGHBORS,
    normalize=NORMALIZE,
)


def main() -> None:
    set_seed(SEED)
    gb = GraphBuilder(k_neighbors=K_NEIGHBORS, normalize=NORMALIZE)
    data = gb.build_graph_data(DATA_DIR).to(T.DEVICE)
    n_classes: int = int(data.y.max().item()) + 1
    class_names: list[str] = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))
    n_features: int = data.x.shape[1]

    model = T.build_model('CellTypeGNN', n_features, n_classes)

    criterion = torch.nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=LR, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / EPOCHS)) / 2) * (1 - LRF) + LRF
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    writer, ckpt = T.make_writer_and_ckpt(cfg, n_features)

    T.print_header()
    train_graph(model, data, criterion, optimizer, scheduler,
                EPOCHS, writer, ckpt)

    trainer = GraphTrainer(cfg)
    metrics = trainer.evaluate(data, ckpt, n_features, n_classes,
                               class_names=class_names)
    T.append_results_csv('GNN', metrics)


if __name__ == '__main__':
    main()
