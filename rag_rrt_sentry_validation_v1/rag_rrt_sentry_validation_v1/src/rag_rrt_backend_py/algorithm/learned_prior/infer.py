"""
Inference utilities for learned heatmap prior prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .model import build_model


def build_input_tensor(
    binary_mask: np.ndarray,
    start,
    goal,
    start_goal_radius: int = 1,
) -> np.ndarray:
    """
    Build a 3-channel input tensor for heatmap prediction.

    Channels:
        0: free-space map, free=1, obstacle=0
        1: start indicator map
        2: goal indicator map
    """
    binary_mask = (np.asarray(binary_mask) > 0).astype(np.float32)
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array.")

    h, w = binary_mask.shape
    start_map = np.zeros((h, w), dtype=np.float32)
    goal_map = np.zeros((h, w), dtype=np.float32)

    def paint(marker: np.ndarray, point):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        r = max(0, int(start_goal_radius))
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 < x1 and y0 < y1:
            marker[y0:y1, x0:x1] = 1.0

    paint(start_map, start)
    paint(goal_map, goal)

    return np.stack([binary_mask, start_map, goal_map], axis=0).astype(np.float32)


def _load_checkpoint(checkpoint_path: str | Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Learned prior checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "lightweight_cnn")
        model_kwargs = checkpoint.get("model_kwargs", {})
    else:
        state_dict = checkpoint
        model_name = "lightweight_cnn"
        model_kwargs = {}

    model = build_model(model_name=model_name, **model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "model_kwargs": model_kwargs,
    }


def predict_heatmap(
    binary_mask: np.ndarray,
    start,
    goal,
    checkpoint_path: str | Path,
    device: str = "cpu",
    start_goal_radius: int = 1,
) -> Tuple[np.ndarray, dict]:
    """
    Predict a path-probability heatmap.

    Returns:
        heatmap: 2D numpy array in [0, 1]
        metadata: model/checkpoint metadata
    """
    device = str(device)
    input_np = build_input_tensor(
        binary_mask=binary_mask,
        start=start,
        goal=goal,
        start_goal_radius=start_goal_radius,
    )

    model, metadata = _load_checkpoint(checkpoint_path, device=device)

    x = torch.from_numpy(input_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    heatmap = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    metadata.update({
        "input_shape": list(input_np.shape),
        "heatmap_shape": list(heatmap.shape),
        "device": device,
    })
    return heatmap, metadata
