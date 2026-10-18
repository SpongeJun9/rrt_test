"""
NRRT*-style neural-guided RRT* baseline.

This baseline directly uses a learned heatmap as a non-uniform sampling
probability distribution. It intentionally does not use path priors,
AStarCorridorSampler, ConnInit, BPC, online alpha adaptation, or Top-K edge
checking.
"""

from __future__ import annotations

import numpy as np

from planners.rrt_star_2d import RRTStar2D
from samplers.heatmap_guided_sampler import HeatmapGuidedSampler


class NeuralGuidedRRTStar2D(RRTStar2D):
    """
    RRT* with CNN heatmap-biased sampling.

    Boundary relative to the proposed RAG method:
        - consumes PriorResult.heatmap only;
        - ignores PriorResult.path;
        - does not instantiate AStarCorridorSampler;
        - does not perform reliability-aware trust regulation.
    """

    def __init__(
        self,
        grid_map,
        start,
        goal,
        step_len: float = 10.0,
        search_radius: float = 50.0,
        iter_max: int = 50000,
        goal_radius: float | None = None,
        prior_result=None,
        heatmap=None,
        p_neural: float = 0.5,
        max_heatmap_attempts: int = 50,
        heatmap_threshold: float = 0.5,
    ):
        super().__init__(
            grid_map=grid_map,
            start=start,
            goal=goal,
            step_len=step_len,
            search_radius=search_radius,
            iter_max=iter_max,
            goal_radius=goal_radius,
        )

        self.external_prior_result = prior_result
        self.external_prior_path_ignored = getattr(prior_result, "path", None)

        if heatmap is None and prior_result is not None:
            heatmap = getattr(prior_result, "heatmap", None)

        self.heatmap_sampler = HeatmapGuidedSampler(
            grid_map=grid_map,
            heatmap=heatmap,
            p_neural=p_neural,
            max_heatmap_attempts=max_heatmap_attempts,
            heatmap_threshold=heatmap_threshold,
        )

        self.p_neural = float(np.clip(p_neural, 0.0, 1.0))
        self.max_heatmap_attempts = int(max_heatmap_attempts)
        self.heatmap_threshold = float(heatmap_threshold)

    def sample(self):
        return self.heatmap_sampler.sample()

    def solve(
        self,
        iter_after_initial: int | None = None,
        stop_at_first_solution: bool = False,
    ):
        result = super().solve(
            iter_after_initial=iter_after_initial,
            stop_at_first_solution=stop_at_first_solution,
        )

        result.metadata.update(self.heatmap_sampler.diagnostics())
        result.metadata["guided_prior_source"] = "learned_heatmap_sampling"
        result.metadata["uses_prior_path"] = 0
        result.metadata["uses_conninit"] = 0
        result.metadata["uses_bpc"] = 0
        result.metadata["use_topk_edge_checking"] = 0
        result.metadata["parent_check_top_k"] = np.nan
        result.metadata["rewire_check_top_k"] = np.nan

        if self.external_prior_result is not None:
            result.metadata["external_prior_type"] = getattr(
                self.external_prior_result, "prior_type", None
            )
            result.metadata["external_prior_mode"] = getattr(
                self.external_prior_result, "prior_mode", None
            )

        return result
