# priors/prior_factory.py
"""
Factory for external prior generators.

The experiment scripts should use this factory instead of directly importing
specific prior generator classes.
"""

from __future__ import annotations

from typing import Any, Dict

from priors.base import PriorGenerator
from priors.astar_prior import AStarPriorGenerator
from priors.synthetic_failure_prior import SyntheticFailurePriorGenerator
from priors.learned_prior import LearnedPriorGenerator


def build_prior_generator(
    generator_name: str,
    config: Dict[str, Any] | None = None,
) -> PriorGenerator:
    """
    Build an external prior generator by name.

    Args:
        generator_name:
            One of:
                "astar"
                "synthetic_failure"
                "learned"

        config:
            Optional generator configuration.

    Returns:
        PriorGenerator instance.
    """
    name = str(generator_name).lower().strip()

    if name == "astar":
        return AStarPriorGenerator(config=config)

    if name in ["synthetic_failure", "synthetic", "failure"]:
        return SyntheticFailurePriorGenerator(config=config)

    if name in ["learned", "learned_prior"]:
        return LearnedPriorGenerator(config=config)

    raise ValueError(
        f"Unknown prior generator: {generator_name}. "
        "Supported generators are: astar, synthetic_failure, learned."
    )