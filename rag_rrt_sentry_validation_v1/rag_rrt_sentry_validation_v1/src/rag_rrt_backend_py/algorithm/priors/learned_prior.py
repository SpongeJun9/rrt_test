"""
Learned external prior generator.

This module implements the minimal learned-prior loop:
    map + start + goal -> lightweight CNN heatmap -> heatmap-derived path -> PriorResult

Important boundary:
    LearnedPriorGenerator only produces an external PriorResult. It does not
    perform trust regulation, alpha update, BPC computation, ConnInit, or TopK
    edge checking.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from priors.base import PriorGenerator
from priors.prior_result import PriorResult


class LearnedPriorGenerator(PriorGenerator):
    """
    Learned heatmap prior generator.

    Expected config keys:
        checkpoint_path: str
        device: "cpu" or "cuda"
        heatmap_threshold: float, recorded for analysis
        heatmap_cost_weight: float
        start_goal_radius: int
        allow_fallback: bool
        fallback_to_astar: bool, backward-compatible alias of allow_fallback
        heuristic_type: str
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)
        self.checkpoint_path = self.config.get("checkpoint_path", "")
        self.device = self.config.get("device", "cpu")
        self.heatmap_threshold = float(self.config.get("heatmap_threshold", 0.5))
        self.heatmap_cost_weight = float(self.config.get("heatmap_cost_weight", 5.0))
        self.start_goal_radius = int(self.config.get("start_goal_radius", 1))
        self.allow_fallback = bool(
            self.config.get("allow_fallback", self.config.get("fallback_to_astar", False))
        )
        self.heuristic_type = self.config.get("heuristic_type", "euclidean")

    def _base_metadata(self, **kwargs) -> Dict[str, Any]:
        metadata = {
            "generator": "LearnedPriorGenerator",
            "model": "LightweightHeatmapCNN",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "heatmap_threshold": self.heatmap_threshold,
            "heatmap_cost_weight": self.heatmap_cost_weight,
            "start_goal_radius": self.start_goal_radius,
            "used_fallback": False,
            "fallback_type": "",
            "path_extraction_method": "astar_on_heatmap_cost",
        }
        for key in ["map_name", "task_id", "seed"]:
            if key in kwargs:
                metadata[key] = kwargs[key]
        return metadata

    def _fallback_astar(self, env, start, goal, prior_mode: str, message: str, metadata: Dict[str, Any], **kwargs) -> PriorResult:
        from priors.astar_prior import AStarPriorGenerator

        fallback = AStarPriorGenerator(config=self.config)
        prior = fallback.generate(
            env=env,
            start=start,
            goal=goal,
            prior_mode="coarse_valid",
            **kwargs,
        )

        metadata = dict(metadata)
        metadata.update(getattr(prior, "metadata", {}) or {})
        metadata["generator"] = "LearnedPriorGenerator"
        metadata["used_fallback"] = True
        metadata["fallback_type"] = "astar"
        metadata["fallback_reason"] = message
        metadata["path_extraction_method"] = "fallback_astar"

        return PriorResult(
            prior_type="learned",
            prior_mode="heatmap_cnn",
            path=prior.path,
            heatmap=None,
            start=prior.start,
            goal=prior.goal,
            success=bool(prior.success),
            message=(
                "LearnedPriorGenerator used fallback A*; "
                f"reason: {message}"
            ),
            metadata=metadata,
        )

    def generate(
        self,
        env,
        start,
        goal,
        prior_mode: str = "learned_heatmap",
        **kwargs,
    ) -> PriorResult:
        start_arr = np.asarray(start, dtype=float).reshape(2)
        goal_arr = np.asarray(goal, dtype=float).reshape(2)
        metadata = self._base_metadata(requested_prior_mode=prior_mode, **kwargs)

        if not hasattr(env, "binary_mask"):
            message = "Environment does not provide binary_mask."
            if self.allow_fallback:
                return self._fallback_astar(env, start, goal, prior_mode, message, metadata, **kwargs)
            return PriorResult(
                prior_type="learned",
                prior_mode="heatmap_cnn",
                path=[],
                heatmap=None,
                start=(float(start_arr[0]), float(start_arr[1])),
                goal=(float(goal_arr[0]), float(goal_arr[1])),
                success=False,
                message=message,
                metadata=metadata,
            )

        if self.checkpoint_path is None or str(self.checkpoint_path).strip() == "":
            message = "No learned prior checkpoint_path is provided."
            if self.allow_fallback:
                return self._fallback_astar(env, start, goal, prior_mode, message, metadata, **kwargs)
            return PriorResult(
                prior_type="learned",
                prior_mode="heatmap_cnn",
                path=[],
                heatmap=None,
                start=(float(start_arr[0]), float(start_arr[1])),
                goal=(float(goal_arr[0]), float(goal_arr[1])),
                success=False,
                message=message,
                metadata=metadata,
            )

        try:
            from learned_prior.infer import predict_heatmap
            from learned_prior.heatmap_to_path import extract_path_from_heatmap

            heatmap, infer_metadata = predict_heatmap(
                binary_mask=env.binary_mask,
                start=start_arr,
                goal=goal_arr,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                start_goal_radius=self.start_goal_radius,
            )
            metadata.update(infer_metadata)

            clearance = int(getattr(env, "clearance", self.config.get("clearance", 0)))
            path, extraction_metadata = extract_path_from_heatmap(
                binary_mask=env.binary_mask,
                heatmap=heatmap,
                start=start_arr,
                goal=goal_arr,
                clearance=clearance,
                heatmap_cost_weight=self.heatmap_cost_weight,
                heuristic_type=self.heuristic_type,
            )
            metadata.update(extraction_metadata)
            metadata["clearance"] = clearance

            success = path is not None and len(path) >= 2
            if not success:
                message = "Learned heatmap path extraction failed."
                if self.allow_fallback:
                    return self._fallback_astar(env, start, goal, prior_mode, message, metadata, **kwargs)
                prior_path = []
            else:
                message = ""
                prior_path = [(float(p[0]), float(p[1])) for p in path]

            return PriorResult(
                prior_type="learned",
                prior_mode="heatmap_cnn",
                path=prior_path,
                heatmap=heatmap,
                start=(float(start_arr[0]), float(start_arr[1])),
                goal=(float(goal_arr[0]), float(goal_arr[1])),
                success=success,
                message=message,
                metadata=metadata,
            )

        except Exception as exc:
            message = f"Learned prior inference failed: {repr(exc)}"
            metadata["exception"] = repr(exc)
            if self.allow_fallback:
                return self._fallback_astar(env, start, goal, prior_mode, message, metadata, **kwargs)

            return PriorResult(
                prior_type="learned",
                prior_mode="heatmap_cnn",
                path=[],
                heatmap=None,
                start=(float(start_arr[0]), float(start_arr[1])),
                goal=(float(goal_arr[0]), float(goal_arr[1])),
                success=False,
                message=message,
                metadata=metadata,
            )
