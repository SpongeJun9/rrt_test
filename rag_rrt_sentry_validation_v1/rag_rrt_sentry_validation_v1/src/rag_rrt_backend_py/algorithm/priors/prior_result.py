# priors/prior_result.py
"""
Unified data structure for external prior generation.

This module defines PriorResult, the standard output format of all external
prior generators.

Important design principle:
    PriorResult only describes the generated prior itself.
    It should not contain planner-side states such as alpha, BPC,
    collision checks, success rate, or final path cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass
class PriorResult:
    """
    Standard output of an external prior generator.

    Attributes:
        prior_type:
            Type of prior generator, e.g.:
                "astar"
                "synthetic_failure"
                "learned"

        prior_mode:
            Specific prior mode, e.g.:
                "coarse_valid"
                "broken_middle"
                "endpoint_mismatch"
                "strong_shift"
                "learned_placeholder"

        path:
            Optional path-like prior represented as a list of 2D waypoints.

        heatmap:
            Optional heatmap-like prior represented as a 2D numpy array.
            This is reserved for future learned prior integration.

        start, goal:
            Planning query associated with this prior.

        success:
            Whether the prior generator successfully produced a usable prior.

        message:
            Optional human-readable diagnostic message.

        metadata:
            Prior-generation metadata only.
            Do not store planner-side states here.
    """

    prior_type: str
    prior_mode: str

    path: Optional[List[Point2D]] = None
    heatmap: Optional[np.ndarray] = None

    start: Optional[Point2D] = None
    goal: Optional[Point2D] = None

    success: bool = True
    message: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_path(self) -> bool:
        """Return True if a non-empty path prior is available."""
        return self.path is not None and len(self.path) > 0

    def has_heatmap(self) -> bool:
        """Return True if a heatmap prior is available."""
        return self.heatmap is not None

    def num_waypoints(self) -> int:
        """Return the number of waypoints in the path prior."""
        if self.path is None:
            return 0
        return int(len(self.path))

    def path_array(self) -> np.ndarray:
        """
        Return the path prior as an array with shape (N, 2).

        If no path exists, return an empty array with shape (0, 2).
        """
        if self.path is None or len(self.path) == 0:
            return np.empty((0, 2), dtype=float)

        return np.asarray(self.path, dtype=float).reshape(-1, 2)

    def summary_dict(self) -> Dict[str, Any]:
        """
        Return a compact summary suitable for logging or CSV records.
        """
        return {
            "prior_type": self.prior_type,
            "prior_mode": self.prior_mode,
            "prior_success": int(self.success),
            "prior_num_waypoints": self.num_waypoints(),
            "prior_has_path": int(self.has_path()),
            "prior_has_heatmap": int(self.has_heatmap()),
            "prior_message": self.message,
        }