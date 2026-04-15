import torch
from torch import nn, optim

from allen_brain.models import train as T

SEED = 42
BATCH_SIZE = 2048
N_HVG = 2000
DATA_DIR = 'data/10x'
N_TRIALS = 3
TUNE_EPOCHS = 20
TUNE_BATCH_SIZE = 2048

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


def main():
    cfg = COFIG
    squeeze_channel = False

    # 1. Dataloaders — full matrix preloaded to pinned host RAM, async H2D per batch.
    train_loader, val_loader = T.make_dataloaders(DATA_DIR, cfg['batch_size'])
    loaders = (train_loader, val_loader)
    ds = train_loader.dataset
    n_features = len(ds.gene_names)

    def builder():
        return T.build_model(cfg['model'], n_features, ds.n_classes)

    # 2. Optuna hparam search over lr / weight_decay with Hyperband pruner.
    best_params = T.run_hparam_search(
        cfg, builder, ds, loaders, squeeze_channel,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS,
    )
    if best_params is not None:
        cfg['lr'] = best_params['lr']
        cfg['weight_decay'] = best_params['weight_decay']

    # 3. Final training run with the tuned hparams.
    model = builder()
    criterion = cfg['loss'](
        weight=T.class_weights(ds), label_smoothing=0.1,
    )
    optimizer, scheduler = T.build_optimizer(
        model, cfg['lr'], cfg['weight_decay'], cfg['epochs'],
        opt_cls=cfg['optimizer'],
    )
    writer, ckpt = T.make_writer_and_ckpt(cfg, n_features)
    print(f'Training {cfg["epochs"]} epochs with best params on {T.DEVICE}...')
    T.print_header()
    best = T.train(
        model, loaders, criterion, optimizer, scheduler,
        cfg['epochs'], writer, ckpt, squeeze_channel=squeeze_channel,
    )
    print(f'\nBest validation accuracy: {best:.4f}')
    return best


if __name__ == '__main__':
    main()
