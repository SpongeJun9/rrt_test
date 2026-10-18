# priors/__init__.py
"""
External prior generator package.

This package provides a unified interface for different external prior
sources, including:

    - A* prior
    - synthetic corrupted prior
    - future learned prior

The trust regulation mechanism remains in the planner/controller side.
"""

from .prior_result import PriorResult, Point2D
from .base import PriorGenerator
from .astar_prior import AStarPriorGenerator
from .synthetic_failure_prior import SyntheticFailurePriorGenerator
from .learned_prior import LearnedPriorGenerator
from .prior_factory import build_prior_generator

__all__ = [
    "PriorResult",
    "Point2D",
    "PriorGenerator",
    "AStarPriorGenerator",
    "SyntheticFailurePriorGenerator",
    "LearnedPriorGenerator",
    "build_prior_generator",
]