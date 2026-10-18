# priors/base.py
"""
Base interface for external prior generators.

A PriorGenerator is responsible only for producing an external prior.
It should not perform trust regulation, alpha update, BPC computation,
or planner-side collision-check accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .prior_result import PriorResult


Point2D = Tuple[float, float]


class PriorGenerator(ABC):
    """
    Abstract base class for all external prior generators.

    Expected responsibility:
        map / start / goal / prior_mode
            -> PriorResult

    Not responsible for:
        - guided sampling probability alpha
        - connectivity-based alpha initialization
        - Best-Path Consistency
        - ReliabilityController update
        - RRT* parent selection or rewiring
        - planner collision-check statistics
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config: Dict[str, Any] = {} if config is None else dict(config)

    @abstractmethod
    def generate(
        self,
        env,
        start: Point2D,
        goal: Point2D,
        prior_mode: str = "coarse_valid",
        **kwargs,
    ) -> PriorResult:
        """
        Generate an external prior for a planning query.

        Args:
            env:
                Planning environment or grid map object.
                In the current project this is usually GridMap2D.

            start:
                Start point, shape-compatible with (x, y).

            goal:
                Goal point, shape-compatible with (x, y).

            prior_mode:
                Prior mode name, such as:
                    "coarse_valid"
                    "broken_middle"
                    "endpoint_mismatch"
                    "strong_shift"

            **kwargs:
                Optional future arguments, such as map_name, task_id,
                corruption strength, model checkpoint, etc.

        Returns:
            PriorResult.
        """
        raise NotImplementedError