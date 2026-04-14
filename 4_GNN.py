import gc

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.models import get_model
from allen_brain.models import train as T


SEED = 42
BATCH_SIZE = 128
N_HVG = 2000
DATA_DIR = 'data/smartseq'
K_NEIGHBORS = 15

COFIG = {
    'model': 'CellTypeGNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': optim.AdamW,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 200,
    'loss': nn.CrossEntropyLoss,
}


def _to_numpy_2d(ds_X):
    arr = ds_X.numpy() if torch.is_tensor(ds_X) else np.asarray(ds_X)
    if arr.ndim == 3:
        arr = arr.squeeze(1)
    return arr.astype(np.float32, copy=False)


def _to_numpy_1d(ds_y):
    arr = ds_y.numpy() if torch.is_tensor(ds_y) else np.asarray(ds_y)
    return arr.astype(np.int64, copy=False)


def _stack_splits(ds_train, ds_val, ds_test):
    X_all = np.vstack([_to_numpy_2d(ds_train.X), _to_numpy_2d(ds_val.X), _to_numpy_2d(ds_test.X)])
    y_all = np.concatenate([_to_numpy_1d(ds_train.y), _to_numpy_1d(ds_val.y), _to_numpy_1d(ds_test.y)])
    return X_all, y_all


def _build_masks(n_train, n_val, n_test):
    n_total = n_train + n_val + n_test
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask   = torch.zeros(n_total, dtype=torch.bool)
    test_mask  = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    return train_mask, val_mask, test_mask


def _build_knn_edges(X_all, k):
    n_total = X_all.shape[0]
    print(f'Building k={k} cosine-NN graph on {n_total:,} cells...')
    nn_finder = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nn_finder.fit(X_all)
    _, indices = nn_finder.kneighbors(X_all)
    src = np.repeat(np.arange(n_total), k)
    dst = indices[:, 1:].reshape(-1)
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    edge_index = torch.tensor(np.stack([src_sym, dst_sym], axis=0), dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def data_set_up():
    ds_train = make_dataset(DATA_DIR, split='train')
    ds_val   = make_dataset(DATA_DIR, split='val')
    ds_test  = make_dataset(DATA_DIR, split='test')
    X_all, y_all = _stack_splits(ds_train, ds_val, ds_test)
    tr_m, vl_m, te_m = _build_masks(len(ds_train.y), len(ds_val.y), len(ds_test.y))
    edge_index = _build_knn_edges(X_all, K_NEIGHBORS)
    n_total = X_all.shape[0]
    print(f'Graph: {n_total:,} nodes, {edge_index.shape[1]:,} edges, avg deg {edge_index.shape[1] / n_total:.1f}')
    data = Data(x=torch.from_numpy(X_all), edge_index=edge_index,
                y=torch.from_numpy(y_all).long(),
                train_mask=tr_m, val_mask=vl_m, test_mask=te_m)
    del ds_train, ds_val, ds_test
    gc.collect()
    return data


def _masked_class_weights(y, mask, n_classes, device=T.DEVICE):
    counts = np.bincount(y[mask].cpu().numpy(), minlength=n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * n_classes


def _train_step(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    acc = (logits[data.train_mask].argmax(1) == data.y[data.train_mask]).float().mean().item()
    return loss.item(), acc


def _eval_step(model, data, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.val_mask], data.y[data.val_mask]).item()
        acc  = (logits[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
    return loss, acc


def _build_model(data, n_classes):
    model = get_model(COFIG['model'], data.x.shape[1], n_classes).to(T.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    return model


def _make_writer_and_ckpt(data):
    run_name = T.make_run_name(COFIG['model'], data.x.shape[1], BATCH_SIZE,
                               COFIG['epochs'], COFIG['lr'], COFIG['weight_decay'])
    writer = SummaryWriter(log_dir=f'runs/{COFIG["model"]}/{run_name}')
    return writer, f'best_gnn_{run_name}.pt'


def _train_loop(model, data, criterion, optimizer, scheduler, writer, ckpt, epochs, patience=20):
    best_loss, best_acc, no_improve = float('inf'), 0.0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_step(model, data, criterion, optimizer)
        vl_loss, vl_acc = _eval_step(model, data, criterion)
        scheduler.step()
        improved = vl_loss < best_loss - 1e-4
        if improved:
            best_loss, best_acc, no_improve = vl_loss, vl_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        T.print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc,
                    scheduler.get_last_lr()[0], '<' if improved else '')
        T.log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc)
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    return best_acc


def main():
    data = data_set_up().to(T.DEVICE)
    n_classes = int(data.y.max().item()) + 1
    model = _build_model(data, n_classes)
    criterion = COFIG['loss'](
        weight=_masked_class_weights(data.y, data.train_mask, n_classes),
        label_smoothing=0.1,
    )
    optimizer, scheduler = T.build_optimizer(
        model, COFIG['lr'], COFIG['weight_decay'], COFIG['epochs'], opt_cls=COFIG['optimizer'])
    writer, ckpt = _make_writer_and_ckpt()
    print(f'\nData: {data}')
    print(f'Training {COFIG["epochs"]} epochs on {T.DEVICE}...')
    T.print_header()
    best = _train_loop(model, data, criterion, optimizer, scheduler,
                       writer, ckpt, COFIG['epochs'])
    print(f'\nBest validation accuracy: {best:.4f}')


if __name__ == '__main__':
    main()
