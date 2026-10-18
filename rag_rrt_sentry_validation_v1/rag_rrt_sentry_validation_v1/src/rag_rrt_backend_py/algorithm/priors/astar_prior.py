# priors/astar_prior.py
"""
A* external prior generator.

This module wraps the existing AStar2D implementation and exposes it through
the unified PriorGenerator interface.

Design principle:
    AStarPriorGenerator only generates an external prior path.
    It does not perform guided sampling, trust regulation, BPC, ConnInit,
    or planner-side collision-check accounting.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from priors.base import PriorGenerator
from priors.prior_result import PriorResult
from utils.astar import AStar2D, compute_path_cost


class AStarPriorGenerator(PriorGenerator):
    """
    External prior generator based on A* search.

    Supported prior_mode:
        coarse_valid

    For corrupted priors such as broken_middle, endpoint_mismatch, or
    strong_shift, use SyntheticFailurePriorGenerator instead.
    """

    SUPPORTED_PRIOR_MODES = {"coarse_valid"}

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)

        self.heuristic_type = self.config.get("heuristic_type", "euclidean")

    def generate(
        self,
        env,
        start,
        goal,
        prior_mode: str = "coarse_valid",
        **kwargs,
    ) -> PriorResult:
        """
        Generate a coarse A* prior path.

        Args:
            env:
                Usually GridMap2D. Must provide:
                    binary_mask
                    clearance

            start:
                Start point.

            goal:
                Goal point.

            prior_mode:
                Only "coarse_valid" is supported in this generator.

            **kwargs:
                Optional metadata, e.g. map_name, task_id.

        Returns:
            PriorResult containing an A* path if search succeeds.
        """
        if prior_mode not in self.SUPPORTED_PRIOR_MODES:
            raise ValueError(
                "AStarPriorGenerator only supports prior_mode='coarse_valid'. "
                "Use SyntheticFailurePriorGenerator for corrupted priors such as "
                "'broken_middle', 'endpoint_mismatch', or 'strong_shift'."
            )

        start_arr = np.asarray(start, dtype=float).reshape(2)
        goal_arr = np.asarray(goal, dtype=float).reshape(2)

        if not hasattr(env, "binary_mask"):
            return PriorResult(
                prior_type="astar",
                prior_mode=prior_mode,
                path=[],
                heatmap=None,
                start=(float(start_arr[0]), float(start_arr[1])),
                goal=(float(goal_arr[0]), float(goal_arr[1])),
                success=False,
                message="Environment does not provide binary_mask.",
                metadata={
                    "generator": "AStarPriorGenerator",
                },
            )

        clearance = int(getattr(env, "clearance", self.config.get("clearance", 0)))

        astar = AStar2D(
            binary_mask=env.binary_mask,
            clearance=clearance,
            heuristic_type=self.heuristic_type,
        )

        path, visited = astar.search(start_arr, goal_arr)

        success = path is not None and len(path) > 0

        if success:
            prior_path = [
                (float(p[0]), float(p[1]))
                for p in path
            ]
            message = ""
            path_cost = compute_path_cost(prior_path)
        else:
            prior_path = []
            message = "A* failed to find a prior path."
            path_cost = float("inf")

        metadata = {
            "generator": "AStarPriorGenerator",
            "heuristic_type": self.heuristic_type,
            "clearance": clearance,
            "num_waypoints": len(prior_path),
            "num_visited": 0 if visited is None else len(visited),
            "path_cost": path_cost,
        }

        for key in ["map_name", "task_id", "seed"]:
            if key in kwargs:
                metadata[key] = kwargs[key]

        return PriorResult(
            prior_type="astar",
            prior_mode=prior_mode,
            path=prior_path,
            heatmap=None,
            start=(float(start_arr[0]), float(start_arr[1])),
            goal=(float(goal_arr[0]), float(goal_arr[1])),
            success=success,
            message=message,
            metadata=metadata,
        )