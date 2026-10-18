# priors/synthetic_failure_prior.py
"""
Synthetic corrupted external prior generator.

This module generates controlled imperfect priors based on an A* prior path.

Supported prior modes:
    - coarse_valid
    - broken_middle
    - endpoint_mismatch
    - strong_shift

Design principle:
    SyntheticFailurePriorGenerator only generates prior paths.
    It does not perform guided sampling, trust regulation, BPC, ConnInit,
    or planner-side collision-check accounting.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from priors.base import PriorGenerator
from priors.prior_result import PriorResult, Point2D
from priors.astar_prior import AStarPriorGenerator
from utils.astar import compute_path_cost


class SyntheticFailurePriorGenerator(PriorGenerator):
    """
    Controlled imperfect-prior generator.

    This generator first obtains a coarse A* prior path, then applies
    deterministic transformations to create specific prior failure modes.

    Supported modes:
        coarse_valid:
            Use the base A* prior path.

        broken_middle:
            Keep start-side and goal-side fragments while removing the middle.
            The two fragments are concatenated, intentionally creating a large
            invalid edge that can be detected by connectivity diagnostics.

        endpoint_mismatch:
            Shift the whole path away from the current start / goal.

        strong_shift:
            Shift the whole path toward a nearby but potentially wrong
            corridor / room / trap structure.
    """

    SUPPORTED_PRIOR_MODES = {
        "coarse_valid",
        "broken_middle",
        "endpoint_mismatch",
        "strong_shift",
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)
        self.base_generator = AStarPriorGenerator(config=config)

    def generate(
        self,
        env,
        start,
        goal,
        prior_mode: str = "coarse_valid",
        **kwargs,
    ) -> PriorResult:
        """
        Generate a controlled synthetic prior.

        Args:
            env:
                Usually GridMap2D.

            start:
                Start point.

            goal:
                Goal point.

            prior_mode:
                One of:
                    "coarse_valid"
                    "broken_middle"
                    "endpoint_mismatch"
                    "strong_shift"

            **kwargs:
                Optional metadata and corruption parameters, e.g.:
                    map_name
                    task_id
                    seed
                    endpoint_mismatch_shift
                    strong_shift_offset
                    keep_ratio_each_side

        Returns:
            PriorResult.
        """
        if prior_mode not in self.SUPPORTED_PRIOR_MODES:
            raise ValueError(
                f"Unknown prior_mode={prior_mode}. "
                f"Supported modes are {sorted(self.SUPPORTED_PRIOR_MODES)}."
            )

        base = self.base_generator.generate(
            env=env,
            start=start,
            goal=goal,
            prior_mode="coarse_valid",
            **kwargs,
        )

        if not base.success or not base.has_path():
            return PriorResult(
                prior_type="synthetic_failure",
                prior_mode=prior_mode,
                path=[],
                heatmap=None,
                start=base.start,
                goal=base.goal,
                success=False,
                message="Base A* prior generation failed.",
                metadata={
                    "generator": "SyntheticFailurePriorGenerator",
                    "base_prior_type": base.prior_type,
                    "base_prior_mode": base.prior_mode,
                    "base_prior_success": int(base.success),
                    "corruption": prior_mode,
                },
            )

        base_path = base.path_array()

        if prior_mode == "coarse_valid":
            transformed = base_path
            corruption_params = {}

        elif prior_mode == "broken_middle":
            keep_ratio_each_side = float(
                kwargs.get(
                    "keep_ratio_each_side",
                    self.config.get("keep_ratio_each_side", 0.30),
                )
            )
            transformed = self._make_broken_middle(
                base_path,
                keep_ratio_each_side=keep_ratio_each_side,
            )
            corruption_params = {
                "keep_ratio_each_side": keep_ratio_each_side,
            }

        elif prior_mode == "endpoint_mismatch":
            shift = self._resolve_endpoint_mismatch_shift(kwargs)

            endpoint_ratio = float(
                kwargs.get(
                    "endpoint_mismatch_ratio",
                    self.config.get("endpoint_mismatch_ratio", 0.20),
                )
            )

            # Use smooth tapered endpoint shifts instead of hard segment shifts.
            # This creates endpoint inconsistency without introducing abrupt
            # artificial breaks at the boundary between shifted and unshifted parts.
            transformed = self._make_endpoint_mismatch_tapered(
                base_path,
                start_shift=shift,
                goal_shift=shift,
                endpoint_ratio=endpoint_ratio,
            )

            corruption_params = {
                "start_shift": tuple(float(v) for v in shift),
                "goal_shift": tuple(float(v) for v in shift),
                "endpoint_ratio": endpoint_ratio,
                "tapered": True,
            }

        elif prior_mode == "strong_shift":
            shift = self._resolve_strong_shift(kwargs)
            transformed = self._shift_path(base_path, shift)
            corruption_params = {
                "shift": tuple(float(v) for v in shift),
            }

        else:
            # This should be unreachable due to earlier validation.
            raise ValueError(f"Unsupported prior_mode: {prior_mode}")

        prior_path = self._array_to_path(transformed)
        success = len(prior_path) > 0

        metadata = {
            "generator": "SyntheticFailurePriorGenerator",
            "base_prior_type": base.prior_type,
            "base_prior_mode": base.prior_mode,
            "base_prior_success": int(base.success),
            "base_num_waypoints": base.num_waypoints(),
            "base_path_cost": base.metadata.get("path_cost", np.nan),
            "corruption": prior_mode,
            "corruption_params": corruption_params,
            "num_waypoints": len(prior_path),
            "path_cost": compute_path_cost(prior_path),
        }

        for key in ["map_name", "task_id", "seed"]:
            if key in kwargs:
                metadata[key] = kwargs[key]

        return PriorResult(
            prior_type="synthetic_failure",
            prior_mode=prior_mode,
            path=prior_path,
            heatmap=None,
            start=base.start,
            goal=base.goal,
            success=success,
            message="" if success else "Synthetic prior transformation produced empty path.",
            metadata=metadata,
        )

    @staticmethod
    def _array_to_path(path_arr: np.ndarray) -> list[Point2D]:
        if path_arr is None or len(path_arr) == 0:
            return []

        path_arr = np.asarray(path_arr, dtype=float).reshape(-1, 2)
        return [
            (float(p[0]), float(p[1]))
            for p in path_arr
        ]

    @staticmethod
    def _make_broken_middle(
        path_arr: np.ndarray,
        keep_ratio_each_side: float = 0.30,
    ) -> np.ndarray:
        """
        Remove the middle part of the prior path.

        The returned path intentionally concatenates the start-side fragment
        and goal-side fragment. This preserves compatibility with the current
        connectivity diagnostic, which can detect the large invalid edge
        between the two fragments.
        """
        path_arr = np.asarray(path_arr, dtype=float).reshape(-1, 2)

        if len(path_arr) < 2:
            return path_arr.copy()

        keep_ratio_each_side = float(
            np.clip(keep_ratio_each_side, 0.1, 0.45)
        )

        n = len(path_arr)
        k = max(2, int(n * keep_ratio_each_side))

        start_part = path_arr[:k]
        goal_part = path_arr[-k:]

        return np.vstack([start_part, goal_part])
    

    @staticmethod
    def _make_endpoint_mismatch_tapered(
        path_arr: np.ndarray,
        start_shift,
        goal_shift,
        endpoint_ratio: float = 0.20,
    ) -> np.ndarray:
        """
        Create endpoint mismatch using smooth tapered shifts.

        Unlike a hard shift of the first/last path segments, this function
        gradually decays the displacement from the endpoint toward the middle
        of the path. This avoids artificial long jump edges at the boundary
        between shifted and unshifted parts.

        Semantic role:
            - The prior no longer starts near the true start.
            - The prior no longer ends near the true goal.
            - The middle part of the prior remains largely similar to the
            original A* path.

        This is intended to represent endpoint mismatch rather than a broken
        middle prior.
        """
        path_arr = np.asarray(path_arr, dtype=float).reshape(-1, 2)

        if len(path_arr) < 4:
            return path_arr.copy()

        endpoint_ratio = float(np.clip(endpoint_ratio, 0.05, 0.35))

        n = len(path_arr)
        k = max(2, int(n * endpoint_ratio))

        # Avoid overlap between start-side and goal-side taper regions.
        k = min(k, max(2, n // 3))

        start_shift = np.asarray(start_shift, dtype=float).reshape(2)
        goal_shift = np.asarray(goal_shift, dtype=float).reshape(2)

        transformed = path_arr.copy()

        # Start side:
        #   endpoint receives full shift, then smoothly decays to zero.
        start_weights = np.linspace(1.0, 0.0, k)
        transformed[:k] += start_weights[:, None] * start_shift

        # Goal side:
        #   smoothly increases from zero to full shift at the endpoint.
        goal_weights = np.linspace(0.0, 1.0, k)
        transformed[-k:] += goal_weights[:, None] * goal_shift

        return transformed
    

    @staticmethod
    def _shift_path(
        path_arr: np.ndarray,
        shift: Iterable[float],
    ) -> np.ndarray:
        path_arr = np.asarray(path_arr, dtype=float).reshape(-1, 2)
        shift = np.asarray(shift, dtype=float).reshape(2)
        return path_arr + shift
    
    @staticmethod
    def _make_endpoint_mismatch(
        path_arr: np.ndarray,
        shift,
        endpoint_ratio: float = 0.20,
    ) -> np.ndarray:
        """
        Perturb only the start-side and goal-side fragments of the prior path.

        This creates endpoint inconsistency while keeping the middle part of
        the prior path mostly unchanged.
        """
        path_arr = np.asarray(path_arr, dtype=float).reshape(-1, 2)

        if len(path_arr) < 4:
            return path_arr.copy()

        endpoint_ratio = float(np.clip(endpoint_ratio, 0.05, 0.35))
        n = len(path_arr)
        k = max(2, int(n * endpoint_ratio))

        shift = np.asarray(shift, dtype=float).reshape(2)

        transformed = path_arr.copy()
        transformed[:k] = transformed[:k] + shift
        transformed[-k:] = transformed[-k:] + shift

        return transformed

    def _resolve_endpoint_mismatch_shift(self, kwargs) -> Tuple[float, float]:
        """
        Resolve endpoint_mismatch shift.

        Priority:
            1. kwargs["endpoint_mismatch_shift"]
            2. config["endpoint_mismatch_shift"]
            3. map-specific config from existing complex prior utilities
            4. default fallback
        """
        if "endpoint_mismatch_shift" in kwargs:
            return tuple(kwargs["endpoint_mismatch_shift"])

        if "endpoint_mismatch_shift" in self.config:
            return tuple(self.config["endpoint_mismatch_shift"])

        map_name = kwargs.get("map_name", None)
        if map_name is not None:
            try:
                from utils.complex_prior_generation import get_endpoint_mismatch_shift
                return tuple(get_endpoint_mismatch_shift(map_name))
            except Exception:
                pass

        return (18.0, 18.0)

    def _resolve_strong_shift(self, kwargs) -> Tuple[float, float]:
        """
        Resolve strong_shift offset.

        Priority:
            1. kwargs["strong_shift_offset"]
            2. config["strong_shift_offset"]
            3. map-specific config from existing complex prior utilities
            4. default fallback
        """
        if "strong_shift_offset" in kwargs:
            return tuple(kwargs["strong_shift_offset"])

        if "strong_shift_offset" in self.config:
            return tuple(self.config["strong_shift_offset"])

        map_name = kwargs.get("map_name", None)
        if map_name is not None:
            try:
                from utils.complex_prior_generation import get_strong_shift
                return tuple(get_strong_shift(map_name))
            except Exception:
                pass

        return (18.0, 18.0)
    