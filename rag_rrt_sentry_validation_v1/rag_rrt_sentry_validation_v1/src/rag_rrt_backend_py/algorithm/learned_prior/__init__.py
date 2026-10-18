"""
Minimal learned prior generator package.

This package implements the first runnable loop for learned external prior
generation:
    map + start + goal -> CNN heatmap -> heatmap-derived path -> PriorResult

It does not implement reliability control, alpha update, BPC, ConnInit, or TopK.
"""

from .model import LightweightHeatmapCNN
from .infer import predict_heatmap, build_input_tensor
from .heatmap_to_path import extract_path_from_heatmap

__all__ = [
    "LightweightHeatmapCNN",
    "predict_heatmap",
    "build_input_tensor",
    "extract_path_from_heatmap",
]
