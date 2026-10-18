# utils/guidance_connectivity.py
import math
from collections import deque

import numpy as np


def point_valid_no_count(grid_map, point) -> bool:
    """
    Use grid_map.point_valid directly because it does not increase
    collision_checks in the current GridMap2D implementation.
    """
    return grid_map.point_valid(point)


def edge_valid_no_count(grid_map, start, end, resolution: float = 1.0) -> bool:
    """
    Collision check for diagnostic connectivity analysis.

    This intentionally does not call grid_map.edge_valid(), because
    grid_map.edge_valid() increments grid_map.collision_checks.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    dist = np.linalg.norm(end - start)
    if dist < 1e-12:
        return point_valid_no_count(grid_map, start)

    num_steps = max(2, int(math.ceil(dist / resolution)) + 1)

    for t in np.linspace(0.0, 1.0, num_steps):
        p = start + t * (end - start)
        if not point_valid_no_count(grid_map, p):
            return False

    return True


def _connected_components(num_nodes, adjacency):
    visited = np.zeros(num_nodes, dtype=bool)
    components = []

    for i in range(num_nodes):
        if visited[i]:
            continue

        queue = deque([i])
        visited[i] = True
        comp = []

        while queue:
            u = queue.popleft()
            comp.append(u)

            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        components.append(comp)

    return components


def _nearest_index(points, query):
    if len(points) == 0:
        return None

    points = np.asarray(points, dtype=float)
    query = np.asarray(query, dtype=float)
    dists = np.linalg.norm(points - query, axis=1)
    return int(np.argmin(dists))


def add_connectivity_score(features: dict) -> dict:
    endpoint_score = (
        0.5 * features.get("start_near_prior", 0.0)
        + 0.5 * features.get("goal_near_prior", 0.0)
    )

    score = (
        0.40 * features.get("goal_reachable_from_start", 0.0)
        + 0.25 * features.get("frontier_progress", 0.0)
        + 0.15 * features.get("largest_component_ratio", 0.0)
        + 0.10 * features.get("prior_edge_valid_ratio", 0.0)
        + 0.10 * endpoint_score
    )

    features["connectivity_score"] = float(np.clip(score, 0.0, 1.0))
    return features


def compute_prior_connectivity_features(
    grid_map,
    prior_path,
    start,
    goal,
    edge_resolution: float = 1.0,
    start_goal_near_radius: float | None = None,
) -> dict:
    """
    Compute diagnostic connectivity features for a prior path.

    This function is intended for logging and analysis. It should not
    modify planner behavior in the first version.

    Args:
        grid_map: GridMap2D.
        prior_path: array-like, shape (N, 2), usually guided_sampler.get_prior_path().
        start: start point.
        goal: goal point.
        edge_resolution: discretization resolution for collision checking.
        start_goal_near_radius: optional radius to judge whether prior has
            points near start/goal. If None, nearest prior points are used.

    Returns:
        dict of connectivity reliability features.
    """
    prior_path = np.asarray(prior_path, dtype=float)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if prior_path.ndim != 2 or prior_path.shape[1] < 2 or len(prior_path) == 0:
        return add_connectivity_score({
            "prior_num_points": 0,
            "prior_valid_ratio": 0.0,
            "prior_edge_valid_ratio": 0.0,
            "num_components": 0,
            "largest_component_ratio": 0.0,
            "start_reachable_ratio": 0.0,
            "goal_reachable_from_start": 0.0,
            "frontier_progress": 0.0,
            "start_near_prior": 0.0,
            "goal_near_prior": 0.0,
        })

    points = prior_path[:, :2]
    n = len(points)

    valid_mask = np.array(
        [point_valid_no_count(grid_map, p) for p in points],
        dtype=bool,
    )

    valid_indices = np.where(valid_mask)[0]
    num_valid = int(len(valid_indices))
    prior_valid_ratio = float(num_valid / n)

    if num_valid == 0:
        return add_connectivity_score({
            "prior_num_points": int(n),
            "prior_valid_ratio": prior_valid_ratio,
            "prior_edge_valid_ratio": 0.0,
            "num_components": 0,
            "largest_component_ratio": 0.0,
            "start_reachable_ratio": 0.0,
            "goal_reachable_from_start": 0.0,
            "frontier_progress": 0.0,
            "start_near_prior": 0.0,
            "goal_near_prior": 0.0,
        })

    # Map original path indices to compact valid-node indices.
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    valid_points = points[valid_indices]

    adjacency = [[] for _ in range(num_valid)]

    total_edges = max(0, n - 1)
    valid_edges = 0

    for i in range(n - 1):
        if not valid_mask[i] or not valid_mask[i + 1]:
            continue

        ok = edge_valid_no_count(
            grid_map,
            points[i],
            points[i + 1],
            resolution=edge_resolution,
        )

        if ok:
            valid_edges += 1
            u = old_to_new[i]
            v = old_to_new[i + 1]
            adjacency[u].append(v)
            adjacency[v].append(u)

    prior_edge_valid_ratio = (
        float(valid_edges / total_edges) if total_edges > 0 else 0.0
    )

    components = _connected_components(num_valid, adjacency)
    num_components = len(components)

    comp_id = np.full(num_valid, -1, dtype=int)
    comp_sizes = []

    for cid, comp in enumerate(components):
        comp_sizes.append(len(comp))
        for idx in comp:
            comp_id[idx] = cid

    largest_component_size = max(comp_sizes) if comp_sizes else 0
    largest_component_ratio = float(largest_component_size / num_valid)

    start_idx = _nearest_index(valid_points, start)
    goal_idx = _nearest_index(valid_points, goal)

    if start_idx is None or goal_idx is None:
        start_reachable_ratio = 0.0
        goal_reachable_from_start = 0.0
        frontier_progress = 0.0
    else:
        start_cid = comp_id[start_idx]
        goal_cid = comp_id[goal_idx]

        start_component_size = comp_sizes[start_cid]
        start_reachable_ratio = float(start_component_size / num_valid)
        goal_reachable_from_start = float(start_cid == goal_cid)

        start_component_indices = components[start_cid]
        start_component_points = valid_points[start_component_indices]

        sg = goal - start
        denom = float(np.dot(sg, sg))

        if denom < 1e-12:
            frontier_progress = 0.0
        else:
            proj = ((start_component_points - start) @ sg) / denom
            proj = np.clip(proj, 0.0, 1.0)
            frontier_progress = float(np.max(proj))

    if start_goal_near_radius is None:
        start_near_prior = 1.0
        goal_near_prior = 1.0
    else:
        start_dist = np.min(np.linalg.norm(valid_points - start, axis=1))
        goal_dist = np.min(np.linalg.norm(valid_points - goal, axis=1))
        start_near_prior = float(start_dist <= start_goal_near_radius)
        goal_near_prior = float(goal_dist <= start_goal_near_radius)


    
    return add_connectivity_score({
        "prior_num_points": int(n),
        "prior_valid_ratio": prior_valid_ratio,
        "prior_edge_valid_ratio": prior_edge_valid_ratio,
        "num_components": int(num_components),
        "largest_component_ratio": largest_component_ratio,
        "start_reachable_ratio": start_reachable_ratio,
        "goal_reachable_from_start": goal_reachable_from_start,
        "frontier_progress": frontier_progress,
        "start_near_prior": start_near_prior,
        "goal_near_prior": goal_near_prior,
    })
    
    
# added
def compute_connectivity_score(features: dict) -> float:
    endpoint_score = 0.5 * features.get("start_near_prior", 0.0) + 0.5 * features.get(
        "goal_near_prior", 0.0
    )

    score = (
        0.40 * features.get("goal_reachable_from_start", 0.0)
        + 0.25 * features.get("frontier_progress", 0.0)
        + 0.15 * features.get("largest_component_ratio", 0.0)
        + 0.10 * features.get("prior_edge_valid_ratio", 0.0)
        + 0.10 * endpoint_score
    )

    return float(np.clip(score, 0.0, 1.0))
