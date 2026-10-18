"""
Lightweight CNN heatmap predictor for learned external prior generation.

The model is intentionally small. Its role in the current project is only to
produce a path-probability heatmap that can be converted into an external prior
path. It does not control alpha and does not replace ConnInit / BPC / TopK.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightHeatmapCNN(nn.Module):
    """
    Small encoder-decoder CNN for binary / soft path heatmap prediction.

    Input:
        tensor with shape (B, 3, H, W)
        channel 0: free-space map, free=1, obstacle=0
        channel 1: start map
        channel 2: goal map

    Output:
        logits with shape (B, 1, H, W)
        Apply sigmoid(logits) to obtain heatmap probabilities.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c = int(base_channels)

        self.enc1 = nn.Sequential(
            ConvRelu(in_channels, c),
            ConvRelu(c, c),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            ConvRelu(c, 2 * c),
            ConvRelu(2 * c, 2 * c),
        )
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ConvRelu(2 * c, 4 * c),
            ConvRelu(4 * c, 4 * c),
        )

        self.dec2 = nn.Sequential(
            ConvRelu(4 * c + 2 * c, 2 * c),
            ConvRelu(2 * c, 2 * c),
        )
        self.dec1 = nn.Sequential(
            ConvRelu(2 * c + c, c),
            ConvRelu(c, c),
        )

        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        x = self.bottleneck(x)

        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        logits = self.out(x)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


def build_model(model_name: str = "lightweight_cnn", **kwargs) -> nn.Module:
    name = str(model_name).lower().strip()
    if name in ["lightweight_cnn", "lightweight_heatmap_cnn", "cnn"]:
        return LightweightHeatmapCNN(**kwargs)
    raise ValueError(f"Unknown learned prior model_name: {model_name}")
