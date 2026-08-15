"""Train one ResNet-18 per diversity level.

    python src/train.py --config configs/default.yaml
    python src/train.py --levels 1000
    python src/train.py --levels 0 1000 --scratch    # pretraining ablation

Early stopping runs against a stratified per-class validation split, and the
weights restored at the end are the best-by-validation-loss ones, not the
final epoch's.

ABLATION CAVEAT: --scratch reuses the fine-tuning hyperparameters (lr 1e-4,
same patience) so that diversity stays the only variable across conditions.
Training from random initialization would normally want a higher learning rate
and a longer schedule, so the from-scratch numbers are a lower bound on what
random init could reach, not a ceiling.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import base_parser, load_config  # noqa: E402
from data import (SyntheticDataset, build_transforms, get_device,  # noqa: E402
                  stratified_split)


def build_model(cfg, pretrained: bool, device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    return model.to(device)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train_level(cfg, level: int, pretrained: bool, device, verbose=True):
    suffix = "" if pretrained else "_scratch"
    name = cfg.level_name(level)
    cfg.checkpoints.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoints / f"model_{name}{suffix}.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            h = ckpt["history"]
            print(f"  [{name}{suffix}] checkpoint exists "
                  f"(stopped epoch {h.get('stopped_epoch', '?')}, "
                  f"best val acc {h.get('best_val_acc', float('nan')):.1f}%) - skipping")
        return ckpt["history"]

    train_tf, _ = build_transforms(cfg.image_size)
    full_ds = SyntheticDataset(cfg, level, transform=train_tf)
    if len(full_ds) == 0:
        print(f"  [{name}] no images found - run generate_dataset.py first")
        return None

    train_ds, val_ds = stratified_split(full_ds, cfg.val_split, cfg.split_seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    if verbose:
        print(f"\n{'-' * 62}")
        print(f"  {name}{suffix}  |  {len(train_ds):,} train / {len(val_ds):,} val")
        print(f"{'-' * 62}")

    model = build_model(cfg, pretrained, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_state, best_epoch = float("inf"), None, 0
    no_improve, stopped = 0, cfg.max_epochs

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history["loss"].append(round(tr_loss, 5))
        history["acc"].append(round(tr_acc, 2))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 2))
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch <= 3 or epoch % 5 == 0 or no_improve >= cfg.patience):
            print(f"  epoch {epoch:03d}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:5.1f}%  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:5.1f}%  "
                  f"(no improve: {no_improve})")

        if no_improve >= cfg.patience:
            stopped = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Report the epoch whose weights were actually kept, not the last one run.
    history["stopped_epoch"] = stopped
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val_loss, 5)
    history["best_val_acc"] = history["val_acc"][best_epoch - 1]
    history["train_acc_at_best"] = history["acc"][best_epoch - 1]

    torch.save({
        "texture_level": level,
        "level_name": name,
        "pretrained": pretrained,
        "model_state": model.state_dict(),
        "history": history,
        "objects": cfg.objects,
        "num_classes": cfg.num_classes,
    }, ckpt_path)

    if verbose:
        print(f"  stopped epoch {stopped}, kept epoch {best_epoch} "
              f"(val acc {history['best_val_acc']:.1f}%)  ->  {ckpt_path.name}")
    return history


def main():
    ap = base_parser("Train ResNet-18 classifiers per diversity level")
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--scratch", action="store_true",
                    help="random initialization instead of ImageNet weights")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = args.levels if args.levels else cfg.levels
    pretrained = cfg.pretrained and not args.scratch
    device = get_device()

    print(f"Device: {device}  |  pretrained: {pretrained}")
    if not pretrained:
        print("Ablation run - hyperparameters unchanged from the pretrained "
              "setting, so these are a lower bound (see module docstring).")

    t0 = time.time()
    histories = {}
    for level in levels:
        h = train_level(cfg, level, pretrained, device)
        if h:
            histories[cfg.level_name(level)] = h

    print(f"\nTotal {(time.time() - t0) / 60:.1f} min")

    cfg.results.mkdir(parents=True, exist_ok=True)
    log_path = cfg.results / ("training_history_scratch.json" if not pretrained
                              else "training_history.json")
    with open(log_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"History -> {log_path}")

    print(f"\n  {'Level':<26} {'Stopped':>8} {'Kept':>6} "
          f"{'Train acc':>10} {'Val acc':>8}")
    for level in levels:
        name = cfg.level_name(level)
        if name in histories:
            h = histories[name]
            print(f"  {name:<26} {h['stopped_epoch']:>8} {h['best_epoch']:>6} "
                  f"{h['train_acc_at_best']:>9.1f}% {h['best_val_acc']:>7.1f}%")


if __name__ == "__main__":
    main()