"""Shared training utilities: dataloaders, class weights, epoch loop, checkpointing."""

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from allen_brain.cell_data.cell_dataset import make_dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(data_dir, batch_size, drop_last_train=True, device=DEVICE):
    ds = make_dataset(data_dir, split='train').to(device)
    ds_val = make_dataset(data_dir, split='val').to(device)
    on_gpu = device.type == 'cuda'
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last_train, num_workers=0, pin_memory=not on_gpu)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=not on_gpu)
    print(f'train: {len(ds)} cells, {ds.n_classes} classes, {len(ds.gene_names)} genes (on {device})')
    print(f'val:   {len(ds_val)} cells')
    return ds, ds_val, train_loader, val_loader


def class_weights(ds, device=DEVICE):
    counts = np.bincount(ds.y.cpu().numpy(), minlength=ds.n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * ds.n_classes


def build_optimizer(model, lr, weight_decay, epochs, opt_cls=optim.AdamW):
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def prep_batch(xb, yb, device=DEVICE, squeeze_channel=False):
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    if squeeze_channel and xb.dim() == 3:
        xb = xb.squeeze(1)
    return xb, yb


def train_batch(model, xb, yb, criterion, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return loss, logits


def run_epoch(model, loader, criterion, optimizer, scaler,
              device=DEVICE, train=True, squeeze_channel=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = prep_batch(xb, yb, device, squeeze_channel)
            if train:
                loss, logits = train_batch(model, xb, yb, criterion, optimizer, scaler)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
    return total_loss / total, correct / total


def log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc):
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('Loss/val', vl_loss, epoch)
    writer.add_scalar('Accuracy/train', tr_acc, epoch)
    writer.add_scalar('Accuracy/val', vl_acc, epoch)


def print_header():
    print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | {"Val Loss":>9} | {"Val Acc":>8} | LR')
    print('-' * 72)


def print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr, flag):
    print(f'{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {lr:.2e}{flag}')


def make_run_name(model_name, n_hvg, batch_size, epochs, lr, wd):
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f'{model_name}_hvg{n_hvg}_bs{batch_size}_epochs{epochs}_lr{lr}_wd{wd}_{ts}'


def _step_epoch(model, loaders, criterion, optimizer, scheduler, scaler, device, squeeze_channel):
    train_loader, val_loader = loaders
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                device=device, train=True, squeeze_channel=squeeze_channel)
    vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer, scaler,
                                device=device, train=False, squeeze_channel=squeeze_channel)
    scheduler.step()
    return tr_loss, tr_acc, vl_loss, vl_acc


def train(model, loaders, criterion, optimizer, scheduler, epochs, writer, ckpt,
          device=DEVICE, squeeze_channel=False, patience=15, compile_model=True):
    if compile_model and device.type == 'cuda':
        model = torch.compile(model)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_loss, best_acc, no_improve = float('inf'), 0.0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, vl_loss, vl_acc = _step_epoch(
            model, loaders, criterion, optimizer, scheduler, scaler, device, squeeze_channel)
        improved = vl_loss < best_loss - 1e-4
        if improved:
            best_loss, best_acc, no_improve = vl_loss, vl_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc,
                  scheduler.get_last_lr()[0], '<' if improved else '')
        log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc)
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    return best_acc
