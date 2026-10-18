"""
Dataset utilities for learned prior heatmap training.

Expected .npz sample fields:
    map
    start
    goal
    input_tensor
    target_heatmap
    astar_path
    map_name
    seed

Important:
    astar_path has variable length across samples. The default PyTorch
    DataLoader collate function cannot stack variable-length arrays.
    Use learned_prior_collate_fn in training/evaluation loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_scalar_str(value: Any) -> str:
    """Convert numpy scalar/string object to a normal Python string."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def _to_scalar_int(value: Any) -> int:
    """Convert numpy scalar/integer object to a normal Python int."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return int(value.item())
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


class LearnedPriorNPZDataset(Dataset):
    """
    Dataset for learned prior heatmap prediction.

    The training target is target_heatmap. The A* path is kept only as
    metadata/debug information and is not stacked as a tensor in the default
    training loop.
    """

    def __init__(self, dataset_dir: str | Path, split: str = "train"):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.split_dir = self.dataset_dir / self.split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.split_dir}"
            )

        self.files = sorted(self.split_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz samples found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_path = self.files[idx]
        data = np.load(sample_path, allow_pickle=True)

        x = data["input_tensor"].astype(np.float32)  # (3,H,W)

        y = data["target_heatmap"].astype(np.float32)
        if y.ndim == 2:
            y = y[None, :, :]  # (1,H,W)

        astar_path = data["astar_path"].astype(np.float32)
        if astar_path.size == 0:
            astar_path = np.empty((0, 2), dtype=np.float32)
        else:
            astar_path = np.asarray(astar_path, dtype=np.float32).reshape(-1, 2)

        sample = {
            "input": torch.from_numpy(x),
            "target": torch.from_numpy(y),
            "map": torch.from_numpy(data["map"].astype(np.float32)),
            "start": torch.from_numpy(data["start"].astype(np.float32).reshape(2)),
            "goal": torch.from_numpy(data["goal"].astype(np.float32).reshape(2)),
            # Variable-length metadata; do not let the default collate stack it.
            "astar_path": astar_path,
            "map_name": _to_scalar_str(data["map_name"]),
            "seed": _to_scalar_int(data["seed"]),
            "file": str(sample_path),
        }
        return sample


def learned_prior_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that stacks fixed-size tensors and keeps variable-length
    fields such as astar_path as lists.

    This fixes errors like:
        RuntimeError: stack expects each tensor to be equal size,
        but got [38, 2] and [92, 2]
    """
    return {
        "input": torch.stack([item["input"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "map": torch.stack([item["map"] for item in batch], dim=0),
        "start": torch.stack([item["start"] for item in batch], dim=0),
        "goal": torch.stack([item["goal"] for item in batch], dim=0),
        "astar_path": [item["astar_path"] for item in batch],
        "map_name": [item["map_name"] for item in batch],
        "seed": torch.tensor([item["seed"] for item in batch], dtype=torch.long),
        "file": [item["file"] for item in batch],
    }


def compute_pos_weight(dataset: Dataset, max_samples: int | None = None) -> float:
    """
    Compute negative/positive pixel ratio for BCEWithLogitsLoss(pos_weight=...).
    """
    pos = 0.0
    total = 0.0
    n = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))

    for i in range(n):
        y = dataset[i]["target"].numpy()
        pos += float((y > 0.5).sum())
        total += float(y.size)

    neg = max(total - pos, 1.0)
    pos = max(pos, 1.0)
    return float(neg / pos)
