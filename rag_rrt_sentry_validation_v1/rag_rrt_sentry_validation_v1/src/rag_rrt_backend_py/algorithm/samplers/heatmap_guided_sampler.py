"""
Heatmap-guided sampler for an NRRT*-style learning-guided baseline.

This sampler consumes a predicted heatmap as a non-uniform sampling
probability distribution. It does not consume path priors and does not perform
ConnInit, BPC, alpha adaptation, or Top-K edge checking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class HeatmapGuidedSampler:
    """
    Sample valid 2D points from a CNN-predicted heatmap with uniform fallback.

    Sampling rule:
        with probability p_neural:
            sample from the heatmap distribution;
        otherwise:
            sample uniformly from free space.

    If heatmap-guided attempts repeatedly fail or the heatmap is unusable, the
    sampler falls back to grid_map.sample_free().

    Coordinate convention:
        points are [x, y], while heatmap / binary_mask are indexed as [y, x].
    """

    def __init__(
        self,
        grid_map,
        heatmap: Optional[np.ndarray],
        p_neural: float = 0.5,
        max_heatmap_attempts: int = 50,
        heatmap_threshold: float = 0.5,
        jitter: bool = True,
        eps: float = 1e-12,
    ):
        self.map = grid_map
        self.heatmap = None if heatmap is None else np.asarray(heatmap, dtype=float)

        self.p_neural = float(np.clip(p_neural, 0.0, 1.0))
        self.max_heatmap_attempts = int(max(1, max_heatmap_attempts))
        self.heatmap_threshold = float(heatmap_threshold)
        self.jitter = bool(jitter)
        self.eps = float(eps)

        self.valid_heatmap = False
        self.heatmap_shape = None
        self.map_shape = self._get_map_shape()
        self.num_heatmap_candidate_cells = 0
        self.heatmap_weight_sum = 0.0
        self.used_threshold_distribution = False
        self.invalid_reason = ""

        self._candidate_x = np.empty((0,), dtype=float)
        self._candidate_y = np.empty((0,), dtype=float)
        self._prob = np.empty((0,), dtype=float)

        self.num_neural_attempts = 0
        self.num_neural_successes = 0
        self.num_neural_fallbacks = 0
        self.num_uniform_samples = 0

        self._prepare_distribution()

    def _get_map_shape(self) -> Optional[Tuple[int, int]]:
        if hasattr(self.map, "binary_mask"):
            mask = np.asarray(self.map.binary_mask)
            if mask.ndim == 2:
                return int(mask.shape[0]), int(mask.shape[1])

        height = getattr(self.map, "height", None)
        width = getattr(self.map, "width", None)
        if height is not None and width is not None:
            return int(height), int(width)

        return None

    def _align_heatmap_shape(self, heatmap: np.ndarray) -> Optional[np.ndarray]:
        if heatmap.ndim != 2:
            self.invalid_reason = "heatmap is not a 2D array"
            return None

        self.heatmap_shape = tuple(int(v) for v in heatmap.shape)

        if self.map_shape is None:
            return heatmap

        map_h, map_w = self.map_shape
        if heatmap.shape == (map_h, map_w):
            return heatmap

        # Defensive guard: if a heatmap is accidentally provided as [x, y],
        # transpose it only when this exactly matches the map shape.
        if heatmap.shape == (map_w, map_h):
            return heatmap.T

        self.invalid_reason = (
            f"heatmap shape {tuple(heatmap.shape)} does not match map shape "
            f"{(map_h, map_w)}"
        )
        return None

    def _build_free_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape

        if hasattr(self.map, "binary_mask"):
            binary_mask = (np.asarray(self.map.binary_mask) > 0)
            if binary_mask.shape == shape:
                free_mask = binary_mask.copy()
            else:
                free_mask = np.ones(shape, dtype=bool)
        else:
            free_mask = np.ones(shape, dtype=bool)

        # Use point_valid to incorporate clearance and any map-specific validity
        # checks. Maps in this project are small enough that this preprocessing
        # cost is negligible relative to planning experiments.
        if hasattr(self.map, "point_valid"):
            for y in range(h):
                for x in range(w):
                    if free_mask[y, x]:
                        free_mask[y, x] = bool(
                            self.map.point_valid(np.array([x, y], dtype=float))
                        )

        return free_mask

    def _prepare_distribution(self) -> None:
        if self.heatmap is None:
            self.invalid_reason = "heatmap is None"
            return

        heatmap = self._align_heatmap_shape(self.heatmap)
        if heatmap is None:
            return

        heatmap = np.asarray(heatmap, dtype=float)
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        heatmap = np.maximum(heatmap, 0.0)

        free_mask = self._build_free_mask(heatmap.shape)
        if not np.any(free_mask):
            self.invalid_reason = "no valid free cells in map"
            return

        raw_weights = heatmap * free_mask.astype(float)

        threshold_weights = raw_weights.copy()
        if np.isfinite(self.heatmap_threshold):
            threshold_weights[threshold_weights < self.heatmap_threshold] = 0.0

        if float(threshold_weights.sum()) > self.eps:
            weights = threshold_weights
            self.used_threshold_distribution = True
        else:
            weights = raw_weights
            self.used_threshold_distribution = False

        if float(weights.sum()) <= self.eps:
            self.invalid_reason = "heatmap has no positive mass on valid cells"
            return

        ys, xs = np.nonzero(weights > self.eps)
        flat_weights = weights[ys, xs].astype(float)
        weight_sum = float(flat_weights.sum())

        if weight_sum <= self.eps or len(xs) == 0:
            self.invalid_reason = "empty heatmap sampling distribution"
            return

        self._candidate_x = xs.astype(float)
        self._candidate_y = ys.astype(float)
        self._prob = flat_weights / weight_sum

        self.num_heatmap_candidate_cells = int(len(xs))
        self.heatmap_weight_sum = weight_sum
        self.valid_heatmap = True
        self.invalid_reason = ""

    def _clip_point_to_map(self, point: np.ndarray) -> np.ndarray:
        x = float(point[0])
        y = float(point[1])

        clearance = float(getattr(self.map, "clearance", 0.0))
        width = float(getattr(self.map, "width", self.map_shape[1] if self.map_shape else 0))
        height = float(getattr(self.map, "height", self.map_shape[0] if self.map_shape else 0))

        if width > 0:
            x = float(np.clip(x, clearance, max(clearance, width - 1.0 - clearance)))
        if height > 0:
            y = float(np.clip(y, clearance, max(clearance, height - 1.0 - clearance)))

        return np.array([x, y], dtype=float)

    def sample_heatmap(self) -> Optional[np.ndarray]:
        if not self.valid_heatmap or len(self._prob) == 0:
            return None

        idx = int(np.random.choice(len(self._prob), p=self._prob))
        x = self._candidate_x[idx]
        y = self._candidate_y[idx]

        if self.jitter:
            x += float(np.random.uniform(-0.5, 0.5))
            y += float(np.random.uniform(-0.5, 0.5))

        point = self._clip_point_to_map(np.array([x, y], dtype=float))

        if hasattr(self.map, "point_valid") and not self.map.point_valid(point):
            return None

        return point

    def sample_uniform(self) -> np.ndarray:
        self.num_uniform_samples += 1
        return self.map.sample_free()

    def sample(self) -> np.ndarray:
        if np.random.random() < self.p_neural and self.valid_heatmap:
            self.num_neural_attempts += 1

            for _ in range(self.max_heatmap_attempts):
                point = self.sample_heatmap()
                if point is not None:
                    self.num_neural_successes += 1
                    return point

            self.num_neural_fallbacks += 1
            return self.sample_uniform()

        return self.sample_uniform()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "neural_guided_valid_heatmap": int(self.valid_heatmap),
            "neural_guided_invalid_reason": self.invalid_reason,
            "neural_guided_p_neural": self.p_neural,
            "neural_guided_max_heatmap_attempts": self.max_heatmap_attempts,
            "neural_guided_heatmap_threshold": self.heatmap_threshold,
            "neural_guided_used_threshold_distribution": int(
                self.used_threshold_distribution
            ),
            "neural_guided_heatmap_shape": (
                "" if self.heatmap_shape is None else str(tuple(self.heatmap_shape))
            ),
            "neural_guided_map_shape": (
                "" if self.map_shape is None else str(tuple(self.map_shape))
            ),
            "neural_guided_num_candidate_cells": self.num_heatmap_candidate_cells,
            "neural_guided_heatmap_weight_sum": self.heatmap_weight_sum,
            "neural_guided_num_neural_attempts": self.num_neural_attempts,
            "neural_guided_num_neural_successes": self.num_neural_successes,
            "neural_guided_num_neural_fallbacks": self.num_neural_fallbacks,
            "neural_guided_num_uniform_samples": self.num_uniform_samples,
        }
