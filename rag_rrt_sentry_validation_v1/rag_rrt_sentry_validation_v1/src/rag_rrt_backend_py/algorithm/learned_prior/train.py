"""
Train lightweight CNN for learned prior heatmap prediction.

Example:
python learned_prior/train.py \
  --dataset_dir data/learned_prior_dataset \
  --output_dir checkpoints/learned_prior_cnn \
  --epochs 20 \
  --batch_size 8 \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from learned_prior.dataset import (
    LearnedPriorNPZDataset,
    compute_pos_weight,
    learned_prior_collate_fn,
)
from learned_prior.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/learned_prior_dataset")
    parser.add_argument("--output_dir", default="checkpoints/learned_prior_cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """
    Resolve device safely. If CUDA is requested but unavailable, raise a clear error.
    """
    device_name = str(device_name).lower().strip()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check that the CUDA-enabled PyTorch build is installed."
        )
    return torch.device(device_name)


def path_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    probs = torch.sigmoid(logits)
    pred = probs >= float(threshold)
    tgt = target >= 0.5

    intersection = torch.logical_and(pred, tgt).flatten(1).sum(dim=1).float()
    union = torch.logical_or(pred, tgt).flatten(1).sum(dim=1).float()
    iou = intersection / (union + 1e-8)

    return float(iou.mean().detach().cpu())


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ious = []

    for batch in loader:
        x = batch["input"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ious.append(path_iou_from_logits(logits, y, threshold=threshold))

    return float(np.mean(losses)), float(np.mean(ious))


def write_log(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_set = LearnedPriorNPZDataset(args.dataset_dir, split="train")
    val_set = LearnedPriorNPZDataset(args.dataset_dir, split="val")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=learned_prior_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=learned_prior_collate_fn,
    )

    if args.pos_weight is None:
        pos_weight_value = compute_pos_weight(train_set)
    else:
        pos_weight_value = float(args.pos_weight)

    model_kwargs = {
        "in_channels": 3,
        "base_channels": int(args.base_channels),
    }
    model = build_model("lightweight_cnn", **model_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )
    )

    rows = [[
        "epoch",
        "train_loss",
        "val_loss",
        "train_path_iou",
        "val_path_iou",
        "epoch_time_sec",
        "pos_weight",
        "device",
        "batch_size",
        "base_channels",
    ]]

    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_iou = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            threshold=args.threshold,
        )

        with torch.no_grad():
            val_loss, val_iou = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                threshold=args.threshold,
            )

        epoch_time = time.time() - t0

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} "
            f"time={epoch_time:.1f}s"
        )

        rows.append([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{train_iou:.6f}",
            f"{val_iou:.6f}",
            f"{epoch_time:.2f}",
            f"{pos_weight_value:.6f}",
            str(device),
            int(args.batch_size),
            int(args.base_channels),
        ])
        write_log(log_dir / "train_record.csv", rows)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_name": "lightweight_cnn",
            "model_kwargs": model_kwargs,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_path_iou": val_iou,
            "pos_weight": pos_weight_value,
            "threshold": float(args.threshold),
        }

        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(ckpt, ckpt_dir / "best.pt")

    print(f"Saved best checkpoint to: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
