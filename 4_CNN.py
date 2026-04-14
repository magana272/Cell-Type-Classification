import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from allen_brain.models import get_model
from allen_brain.models import train as T

torch.set_float32_matmul_precision('high')
SEED = 42
BATCH_SIZE = 4096
N_HVG = 2000
DATA_DIR = 'data/10x'

COFIG = {
    'model': 'CellTypeCNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': optim.AdamW,
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': 20,
    'loss': nn.CrossEntropyLoss,
}


def _build_model(ds):
    model = get_model(COFIG['model'], len(ds.gene_names), ds.n_classes).to(T.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    return model


def _make_writer_and_ckpt(ds):
    run_name = T.make_run_name(COFIG['model'], len(ds.gene_names), BATCH_SIZE,
                               COFIG['epochs'], COFIG['lr'], COFIG['weight_decay'])
    writer = SummaryWriter(log_dir=f'runs/{COFIG["model"]}/{run_name}')
    return writer, f'best_model_{run_name}.pt'


def main():
    ds, _, train_loader, val_loader = T.make_dataloaders(DATA_DIR, COFIG['batch_size'])
    model = _build_model(ds)
    criterion = COFIG['loss'](weight=T.class_weights(ds), label_smoothing=0.1)
    optimizer, scheduler = T.build_optimizer(
        model, COFIG['lr'], COFIG['weight_decay'], COFIG['epochs'], opt_cls=COFIG['optimizer'])
    writer, ckpt = _make_writer_and_ckpt(ds)
    print(f'Training {COFIG["epochs"]} epochs on {T.DEVICE}...')
    T.print_header()
    best = T.train(model, (train_loader, val_loader), criterion,
                   optimizer, scheduler, COFIG['epochs'], writer, ckpt,
                   squeeze_channel=False)
    print(f'\nBest validation accuracy: {best:.4f}')


if __name__ == '__main__':
    main()
