from __future__ import annotations
# utils/metrics.py
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlanningResult:
    success: bool = False
    path: Any = None

    final_cost: float = float("inf")
    first_solution_iter: int | None = None
    first_solution_time: float | None = None
    first_solution_cost: float = float("inf")
    planning_time: float = 0.0

    num_vertices: int = 0
    collision_checks: int = 0

    cost_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    reliability_history: list[float] = field(default_factory=list)

    guided_added_rate_history: list[float] = field(default_factory=list)
    guided_collision_rate_history: list[float] = field(default_factory=list)
    guided_progress_rate_history: list[float] = field(default_factory=list)
    guided_improvement_rate_history: list[float] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)


def path_cost(path) -> float:
    if path is None:
        return float("inf")

    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return float("inf")

    return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())


def first_finite_index(values):
    for i, value in enumerate(values):
        if np.isfinite(value):
            return i
    return None