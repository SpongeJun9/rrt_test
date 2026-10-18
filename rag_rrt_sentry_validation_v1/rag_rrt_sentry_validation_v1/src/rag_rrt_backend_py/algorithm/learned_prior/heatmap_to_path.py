"""
Heatmap-to-path extraction for learned external prior generation.

This module converts a learned path-probability heatmap into a path-like prior.
The current conservative method runs weighted A* on the original free-space map,
where high heatmap probability lowers traversal cost. Collision validity still
comes from the binary occupancy map; the heatmap never bypasses collision checks.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Tuple

import numpy as np

Point2D = Tuple[float, float]
GridState = Tuple[int, int]


MOTION_8 = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def _to_grid_state(point) -> GridState:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _heuristic(a: GridState, b: GridState, heuristic_type: str = "euclidean") -> float:
    if heuristic_type == "manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_state_valid(binary_mask: np.ndarray, state: GridState, clearance: int = 0) -> bool:
    h, w = binary_mask.shape
    x, y = state
    c = int(clearance)

    if x < c or x >= w - c:
        return False
    if y < c or y >= h - c:
        return False

    x0, x1 = max(0, x - c), min(w, x + c + 1)
    y0, y1 = max(0, y - c), min(h, y + c + 1)
    return bool(np.all(binary_mask[y0:y1, x0:x1] > 0))


def _is_edge_valid(binary_mask: np.ndarray, start: GridState, end: GridState, clearance: int = 0) -> bool:
    if not _is_state_valid(binary_mask, start, clearance=clearance):
        return False
    if not _is_state_valid(binary_mask, end, clearance=clearance):
        return False

    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy

    # Prevent diagonal corner cutting, matching utils/astar.py behavior.
    if abs(dx) == 1 and abs(dy) == 1:
        if not _is_state_valid(binary_mask, (sx + dx, sy), clearance=clearance):
            return False
        if not _is_state_valid(binary_mask, (sx, sy + dy), clearance=clearance):
            return False

    return True


def _extract_path(parent: Dict[GridState, GridState], start: GridState, goal: GridState) -> List[Point2D]:
    if goal not in parent:
        return []

    path = [goal]
    current = goal
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return [(float(x), float(y)) for x, y in path]


def extract_path_from_heatmap(
    binary_mask: np.ndarray,
    heatmap: np.ndarray,
    start,
    goal,
    clearance: int = 0,
    heatmap_cost_weight: float = 5.0,
    heuristic_type: str = "euclidean",
) -> tuple[List[Point2D], Dict[str, object]]:
    """
    Run weighted A* using heatmap-derived traversal costs.

    Args:
        binary_mask:
            2D map, free=1, obstacle=0.

        heatmap:
            2D probability map. Higher value means the learned model considers
            the cell more path-like.

        start, goal:
            Continuous or grid-compatible 2D coordinates in (x, y).

        clearance:
            Obstacle clearance in grid cells.

        heatmap_cost_weight:
            Strength of heatmap guidance. Larger values favor high-probability
            cells more strongly. This affects only the extracted prior path,
            not planner alpha.

    Returns:
        path, metadata
    """
    binary_mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array.")
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")
    if binary_mask.shape != heatmap.shape:
        raise ValueError(
            f"binary_mask and heatmap must have the same shape, got "
            f"{binary_mask.shape} and {heatmap.shape}."
        )

    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    start_state = _to_grid_state(start)
    goal_state = _to_grid_state(goal)

    metadata: Dict[str, object] = {
        "path_extraction_method": "astar_on_heatmap_cost",
        "heatmap_cost_weight": float(heatmap_cost_weight),
        "heuristic_type": heuristic_type,
        "clearance": int(clearance),
        "heatmap_min": float(np.min(heatmap)),
        "heatmap_max": float(np.max(heatmap)),
        "heatmap_mean": float(np.mean(heatmap)),
        "start_state": start_state,
        "goal_state": goal_state,
        "num_visited": 0,
        "success": False,
    }

    if not _is_state_valid(binary_mask, start_state, clearance=clearance):
        metadata["message"] = "Start state is invalid for heatmap path extraction."
        return [], metadata

    if not _is_state_valid(binary_mask, goal_state, clearance=clearance):
        metadata["message"] = "Goal state is invalid for heatmap path extraction."
        return [], metadata

    open_heap: list[tuple[float, GridState]] = []
    closed: set[GridState] = set()
    parent: Dict[GridState, GridState] = {start_state: start_state}
    g: Dict[GridState, float] = {start_state: 0.0}

    heapq.heappush(open_heap, (_heuristic(start_state, goal_state, heuristic_type), start_state))

    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal_state:
            found = True
            break

        cx, cy = current
        for dx, dy in MOTION_8:
            nxt = (cx + dx, cy + dy)
            if not _is_edge_valid(binary_mask, current, nxt, clearance=clearance):
                continue

            nx, ny = nxt
            move_cost = math.hypot(dx, dy)
            # Average endpoint heatmap probability. High heatmap -> lower cost.
            p = 0.5 * (float(heatmap[cy, cx]) + float(heatmap[ny, nx]))
            heatmap_penalty = 1.0 + float(heatmap_cost_weight) * (1.0 - p)
            edge_cost = move_cost * heatmap_penalty
            new_cost = g[current] + edge_cost

            if nxt not in g or new_cost < g[nxt]:
                g[nxt] = new_cost
                parent[nxt] = current
                f = new_cost + _heuristic(nxt, goal_state, heuristic_type)
                heapq.heappush(open_heap, (f, nxt))

    metadata["num_visited"] = int(len(closed))

    if not found:
        metadata["message"] = "Weighted A* failed on heatmap cost map."
        return [], metadata

    path = _extract_path(parent, start_state, goal_state)
    metadata["success"] = True
    metadata["num_waypoints"] = int(len(path))
    metadata["message"] = ""
    return path, metadata
