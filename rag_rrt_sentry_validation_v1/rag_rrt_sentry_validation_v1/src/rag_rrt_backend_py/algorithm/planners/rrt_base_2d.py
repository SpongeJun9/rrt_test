from __future__ import annotations
# planners/rrt_base_2d.py
import math
import time
from typing import Optional

import numpy as np

from utils.metrics import PlanningResult, path_cost


class RRTBase2D:
    """
    Base class for RRT-based planners.

    This class maintains:
        - vertices
        - parents
        - costs
        - solution node indices
        - common geometry utilities

    The map object should provide:
        - sample_free()
        - point_valid(point)
        - edge_valid(start, end)
        - path_cost(path)
        - collision_checks
    """

    def __init__(
        self,
        grid_map,
        start,
        goal,
        step_len: float = 10.0,
        search_radius: float = 50.0,
        iter_max: int = 50000,
        goal_radius: Optional[float] = None,
    ):
        self.map = grid_map

        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)

        self.step_len = float(step_len)
        self.search_radius = float(search_radius)
        self.iter_max = int(iter_max)

        if goal_radius is None:
            goal_radius = self.step_len
        self.goal_radius = float(goal_radius)

        self.vertices = np.zeros((self.iter_max + 1, 2), dtype=float)
        self.parents = np.full(self.iter_max + 1, -1, dtype=int)
        self.costs = np.full(self.iter_max + 1, np.inf, dtype=float)

        self.vertices[0] = self.start
        self.parents[0] = 0
        self.costs[0] = 0.0
        self.num_vertices = 1

        self.solution_indices: list[int] = []
        self.best_goal_index: Optional[int] = None
        self.best_cost: float = float("inf")

        self.cost_history: list[float] = []
        self.first_solution_iter: Optional[int] = None
        self.first_solution_time: Optional[float] = None
        self.first_solution_cost: float = float("inf")
        
        self.first_solution_collision_checks: Optional[int] = None

        self._reset_topk_edge_stats()

    def reset(self):
        self.vertices.fill(0.0)
        self.parents.fill(-1)
        self.costs.fill(np.inf)

        self.vertices[0] = self.start
        self.parents[0] = 0
        self.costs[0] = 0.0
        self.num_vertices = 1

        self.solution_indices = []
        self.best_goal_index = None
        self.best_cost = float("inf")

        self.cost_history = []
        self.first_solution_iter = None
        self.first_solution_time = None
        self.first_solution_cost = float("inf")
        
        self.first_solution_collision_checks = None

        self._reset_topk_edge_stats()

        if hasattr(self.map, "reset_statistics"):
            self.map.reset_statistics()

    def _reset_topk_edge_stats(self):
        """Reset diagnostic counters for Top-K edge checking.

        These counters are independent of the map-level collision_checks
        counter. The map counter remains the source of truth for total edge
        validity calls; these fields only decompose how many of those checks
        were spent in Top-K and fallback stages.
        """
        self.topk_edge_stats = {
            "parent_event_count": 0,
            "parent_eligible_candidate_count": 0,
            "parent_omitted_candidate_count": 0,
            "parent_topk_candidate_count": 0,
            "parent_topk_checked_count": 0,
            "parent_topk_success_count": 0,
            "parent_fallback_candidate_count": 0,
            "parent_fallback_checked_count": 0,
            "parent_fallback_success_count": 0,
            "parent_fallback_trigger_count": 0,
            "parent_fallback_recovery_count": 0,
            "rewire_event_count": 0,
            "rewire_eligible_candidate_count": 0,
            "rewire_omitted_candidate_count": 0,
            "rewire_topk_candidate_count": 0,
            "rewire_topk_checked_count": 0,
            "rewire_topk_success_count": 0,
            "rewire_fallback_candidate_count": 0,
            "rewire_fallback_checked_count": 0,
            "rewire_fallback_success_count": 0,
            "rewire_fallback_trigger_count": 0,
            "rewire_fallback_recovery_count": 0,
            "adaptive_rewire_event_count": 0,
            "adaptive_rewire_eligible_candidate_count": 0,
            "adaptive_rewire_primary_candidate_count": 0,
            "adaptive_rewire_primary_checked_count": 0,
            "adaptive_rewire_primary_success_count": 0,
            "adaptive_rewire_probe_candidate_count": 0,
            "adaptive_rewire_probe_checked_count": 0,
            "adaptive_rewire_probe_success_count": 0,
            "adaptive_rewire_omitted_candidate_count": 0,
        }

    def get_topk_edge_stats(self):
        """Return a copy of Top-K / fallback diagnostic counters."""
        if not hasattr(self, "topk_edge_stats"):
            self._reset_topk_edge_stats()
        return dict(self.topk_edge_stats)

    def _add_topk_edge_stat(self, key: str, value: int = 1):
        if not hasattr(self, "topk_edge_stats"):
            self._reset_topk_edge_stats()
        self.topk_edge_stats[key] = int(self.topk_edge_stats.get(key, 0)) + int(value)

    def sample(self):
        """
        Override this method in derived planners if needed.
        """
        return self.map.sample_free()

    def nearest(self, point):
        vertices = self.vertices[: self.num_vertices]
        dists = np.linalg.norm(vertices - point, axis=1)
        index = int(np.argmin(dists))
        return vertices[index], index

    def near(self, point):
        vertices = self.vertices[: self.num_vertices]
        dists = np.linalg.norm(vertices - point, axis=1)
        indices = np.where(dists <= self.search_radius)[0]
        return indices

    def steer(self, from_point, to_point):
        from_point = np.asarray(from_point, dtype=float)
        to_point = np.asarray(to_point, dtype=float)

        direction = to_point - from_point
        dist = np.linalg.norm(direction)

        if dist < 1e-12:
            return from_point.copy()

        if dist <= self.step_len:
            return to_point.copy()

        return from_point + self.step_len * direction / dist

    def add_vertex(self, point, parent_index):
        if self.num_vertices >= len(self.vertices):
            return None

        new_index = self.num_vertices
        self.vertices[new_index] = point
        self.parents[new_index] = parent_index
        self.costs[new_index] = (
            self.costs[parent_index]
            + np.linalg.norm(point - self.vertices[parent_index])
        )

        self.num_vertices += 1
        return new_index

    def choose_parent(self, point, neighbor_indices, default_parent_index):
        best_parent = default_parent_index
        best_cost = (
            self.costs[default_parent_index]
            + np.linalg.norm(point - self.vertices[default_parent_index])
        )

        for idx in neighbor_indices:
            candidate = self.vertices[idx]
            edge_cost = np.linalg.norm(point - candidate)
            candidate_cost = self.costs[idx] + edge_cost

            if candidate_cost < best_cost and self.map.edge_valid(candidate, point):
                best_parent = int(idx)
                best_cost = float(candidate_cost)

        return best_parent

    def rewire(self, new_index, neighbor_indices):
        """
        Return:
            rewire_count: number of rewired vertices
        """
        new_point = self.vertices[new_index]
        new_cost = self.costs[new_index]
        rewire_count = 0

        for idx in neighbor_indices:
            idx = int(idx)

            if idx == 0 or idx == new_index:
                continue

            neighbor = self.vertices[idx]
            edge_cost = np.linalg.norm(neighbor - new_point)
            candidate_cost = new_cost + edge_cost

            if candidate_cost + 1e-9 < self.costs[idx]:
                if self.map.edge_valid(new_point, neighbor):
                    self.parents[idx] = new_index
                    self.costs[idx] = candidate_cost
                    self.propagate_cost_to_children(idx)
                    rewire_count += 1

        if rewire_count > 0:
            self.refresh_best_solution()

        return rewire_count

    def propagate_cost_to_children(self, parent_index):
        """
        When a node is rewired, update its descendants' costs.
        Simple DFS over parent array.
        """
        stack = [parent_index]

        while stack:
            current = stack.pop()
            children = np.where(self.parents[: self.num_vertices] == current)[0]

            for child in children:
                if child == current:
                    continue

                self.costs[child] = (
                    self.costs[current]
                    + np.linalg.norm(self.vertices[child] - self.vertices[current])
                )
                stack.append(int(child))

    def in_goal_region(self, point):
        return (
            np.linalg.norm(point - self.goal) <= self.goal_radius
            and self.map.edge_valid(point, self.goal)
        )

    def update_solution(self, node_index):
        """
        If node_index can connect to goal, register it as a solution node.
        """
        if node_index is None:
            return False

        point = self.vertices[node_index]

        if not self.in_goal_region(point):
            return False

        if node_index not in self.solution_indices:
            self.solution_indices.append(node_index)

        solution_cost = self.costs[node_index] + np.linalg.norm(point - self.goal)

        improved = solution_cost + 1e-9 < self.best_cost

        if improved:
            self.best_cost = float(solution_cost)
            self.best_goal_index = int(node_index)

        return improved

    def refresh_best_solution(self):
        """Recompute the best registered goal connection after rewiring.

        Rewiring a solution node or one of its ancestors changes the true
        solution cost even though no new goal-region node was inserted.  The
        previous implementation left ``best_cost`` and ``best_goal_index``
        stale in that case, which also affected informed sampling and BPC.
        """
        if len(self.solution_indices) == 0:
            self.best_goal_index = None
            self.best_cost = float("inf")
            return False

        old_best_cost = float(self.best_cost)
        best_index = None
        best_cost = float("inf")

        for raw_index in self.solution_indices:
            index = int(raw_index)
            if index < 0 or index >= self.num_vertices:
                continue

            solution_cost = float(
                self.costs[index]
                + np.linalg.norm(self.vertices[index] - self.goal)
            )

            if solution_cost + 1e-12 < best_cost:
                best_cost = solution_cost
                best_index = index

        self.best_goal_index = best_index
        self.best_cost = best_cost
        return bool(best_cost + 1e-9 < old_best_cost)

    def get_best_path(self):
        if self.best_goal_index is None:
            return None

        path = [self.goal.copy()]
        idx = self.best_goal_index

        while idx != 0:
            path.append(self.vertices[idx].copy())
            idx = int(self.parents[idx])

        path.append(self.start.copy())
        path.reverse()

        return np.asarray(path, dtype=float)

    def record_cost(self):
        self.cost_history.append(self.best_cost)

    def make_result(self, planning_time: float):
        self.refresh_best_solution()
        path = self.get_best_path()
        success = path is not None and len(path) >= 2

        total_collision_checks = getattr(self.map, "collision_checks", 0)

        result = PlanningResult(
            success=success,
            path=path,
            final_cost=path_cost(path),
            first_solution_iter=self.first_solution_iter,
            first_solution_time=self.first_solution_time,
            first_solution_cost=self.first_solution_cost,
            planning_time=planning_time,
            num_vertices=self.num_vertices,
            collision_checks=total_collision_checks,
            cost_history=self.cost_history,
        )

        if self.first_solution_collision_checks is None:
            collision_checks_until_initial = np.nan
            collision_checks_after_initial = np.nan
        else:
            collision_checks_until_initial = self.first_solution_collision_checks
            collision_checks_after_initial = (
                total_collision_checks - self.first_solution_collision_checks
            )

        result.metadata["collision_checks_until_initial"] = (
            collision_checks_until_initial
        )
        result.metadata["collision_checks_after_initial"] = (
            collision_checks_after_initial
        )
        result.metadata["collision_check_samples"] = getattr(
            self.map,
            "collision_check_samples",
            np.nan,
        )
        result.metadata["collision_check_time"] = getattr(
            self.map,
            "collision_check_time",
            np.nan,
        )

        for key, value in self.get_topk_edge_stats().items():
            result.metadata[f"topk_{key}"] = value

        return result
    
    
    def choose_parent_topk(
        self,
        point,
        neighbor_indices,
        default_parent_index,
        parent_check_top_k: int | None = None,
        use_topk_fallback: bool = False,
        parent_fallback_check_k: int | None = None,
    ):
        """Choose a parent with Top-K edge validation and optional fallback.

        Strict Top-K checks only the lowest-cost K candidate parents. When
        use_topk_fallback=True, the planner checks a small number of additional
        candidates only if the Top-K stage fails to find a valid better parent.
        This keeps the edge-checking budget bounded while reducing the risk of
        missing a valid parent just outside the Top-K set.
        """
        best_parent = default_parent_index
        best_cost = (
            self.costs[default_parent_index]
            + np.linalg.norm(point - self.vertices[default_parent_index])
        )

        candidates = []

        for idx in neighbor_indices:
            idx = int(idx)
            candidate = self.vertices[idx]
            edge_cost = np.linalg.norm(point - candidate)
            candidate_cost = self.costs[idx] + edge_cost

            if candidate_cost < best_cost:
                candidates.append((candidate_cost, idx))

        candidates.sort(key=lambda x: x[0])

        self._add_topk_edge_stat("parent_event_count", 1)
        self._add_topk_edge_stat(
            "parent_eligible_candidate_count",
            len(candidates),
        )

        if parent_check_top_k is None:
            topk_candidates = candidates
            fallback_candidates = []
        else:
            k = max(0, int(parent_check_top_k))
            topk_candidates = candidates[:k]
            fallback_pool = candidates[k:]

            if use_topk_fallback:
                if parent_fallback_check_k is None:
                    fallback_candidates = fallback_pool
                else:
                    fallback_k = max(0, int(parent_fallback_check_k))
                    fallback_candidates = fallback_pool[:fallback_k]
            else:
                fallback_candidates = []

        omitted_count = max(
            0,
            len(candidates) - len(topk_candidates) - len(fallback_candidates),
        )
        self._add_topk_edge_stat("parent_omitted_candidate_count", omitted_count)

        self._add_topk_edge_stat(
            "parent_topk_candidate_count",
            len(topk_candidates),
        )
        self._add_topk_edge_stat(
            "parent_fallback_candidate_count",
            len(fallback_candidates),
        )

        topk_found = False

        for candidate_cost, idx in topk_candidates:
            candidate = self.vertices[idx]
            self._add_topk_edge_stat("parent_topk_checked_count", 1)

            if self.map.edge_valid(candidate, point):
                best_parent = int(idx)
                best_cost = float(candidate_cost)
                topk_found = True
                self._add_topk_edge_stat("parent_topk_success_count", 1)
                break

        if not topk_found and use_topk_fallback and len(fallback_candidates) > 0:
            self._add_topk_edge_stat("parent_fallback_trigger_count", 1)
            for candidate_cost, idx in fallback_candidates:
                if candidate_cost >= best_cost:
                    break

                candidate = self.vertices[idx]
                self._add_topk_edge_stat("parent_fallback_checked_count", 1)

                if self.map.edge_valid(candidate, point):
                    best_parent = int(idx)
                    best_cost = float(candidate_cost)
                    self._add_topk_edge_stat("parent_fallback_success_count", 1)
                    self._add_topk_edge_stat("parent_fallback_recovery_count", 1)
                    break

        return best_parent
    

    def _rank_rewire_candidates(self, new_index, neighbor_indices):
        """Return positive-improvement rewiring candidates in utility order."""
        new_point = self.vertices[new_index]
        new_cost = self.costs[new_index]
        candidates = []

        for idx in neighbor_indices:
            idx = int(idx)

            if idx == 0 or idx == new_index:
                continue

            neighbor = self.vertices[idx]
            edge_cost = np.linalg.norm(neighbor - new_point)
            candidate_cost = new_cost + edge_cost
            improvement = self.costs[idx] - candidate_cost

            if improvement > 1e-9:
                candidates.append((float(improvement), float(candidate_cost), idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _validate_and_apply_rewire_candidates(
        self,
        new_index,
        candidates,
        *,
        checked_stat_key: str,
        success_stat_key: str,
    ):
        """Validate ranked candidates and return (rewires, checks, gain)."""
        new_point = self.vertices[new_index]
        rewire_count = 0
        checked_count = 0
        realized_gain = 0.0

        for _, candidate_cost, idx in candidates:
            # Earlier rewires may already have reduced this node through an
            # ancestor. Do not spend an edge check or apply a stale non-improving
            # rewire in that case.
            current_gain = float(self.costs[idx] - candidate_cost)
            if current_gain <= 1e-9:
                continue

            neighbor = self.vertices[idx]
            self._add_topk_edge_stat(checked_stat_key, 1)
            checked_count += 1

            if self.map.edge_valid(new_point, neighbor):
                self.parents[idx] = new_index
                self.costs[idx] = candidate_cost
                self.propagate_cost_to_children(idx)
                rewire_count += 1
                realized_gain += current_gain
                self._add_topk_edge_stat(success_stat_key, 1)

        return rewire_count, checked_count, float(realized_gain)

    def rewire_topk(
        self,
        new_index,
        neighbor_indices,
        rewire_check_top_k: int | None = None,
        use_topk_fallback: bool = False,
        rewire_fallback_check_k: int | None = None,
    ):
        """Rewire neighbors with Top-K edge validation and optional fallback.

        Strict Top-K validates only the K candidates with the largest potential
        cost reduction. With fallback enabled, a bounded number of additional
        candidates are checked only when the Top-K stage produces no rewiring.
        """
        rewire_count = 0
        candidates = self._rank_rewire_candidates(new_index, neighbor_indices)

        self._add_topk_edge_stat("rewire_event_count", 1)
        self._add_topk_edge_stat(
            "rewire_eligible_candidate_count",
            len(candidates),
        )

        if rewire_check_top_k is None:
            topk_candidates = candidates
            fallback_candidates = []
        else:
            k = max(0, int(rewire_check_top_k))
            topk_candidates = candidates[:k]
            fallback_pool = candidates[k:]

            if use_topk_fallback:
                if rewire_fallback_check_k is None:
                    fallback_candidates = fallback_pool
                else:
                    fallback_k = max(0, int(rewire_fallback_check_k))
                    fallback_candidates = fallback_pool[:fallback_k]
            else:
                fallback_candidates = []

        omitted_count = max(
            0,
            len(candidates) - len(topk_candidates) - len(fallback_candidates),
        )
        self._add_topk_edge_stat("rewire_omitted_candidate_count", omitted_count)

        self._add_topk_edge_stat(
            "rewire_topk_candidate_count",
            len(topk_candidates),
        )
        self._add_topk_edge_stat(
            "rewire_fallback_candidate_count",
            len(fallback_candidates),
        )

        topk_rewire_count, _, _ = self._validate_and_apply_rewire_candidates(
            new_index,
            topk_candidates,
            checked_stat_key="rewire_topk_checked_count",
            success_stat_key="rewire_topk_success_count",
        )
        rewire_count += topk_rewire_count

        if (
            use_topk_fallback
            and topk_rewire_count == 0
            and len(fallback_candidates) > 0
        ):
            self._add_topk_edge_stat("rewire_fallback_trigger_count", 1)
            fallback_rewire_count, _, _ = self._validate_and_apply_rewire_candidates(
                new_index,
                fallback_candidates,
                checked_stat_key="rewire_fallback_checked_count",
                success_stat_key="rewire_fallback_success_count",
            )
            rewire_count += fallback_rewire_count
            if fallback_rewire_count > 0:
                self._add_topk_edge_stat("rewire_fallback_recovery_count", 1)

        if rewire_count > 0:
            self.refresh_best_solution()

        return rewire_count

    def rewire_adaptive(self, new_index, neighbor_indices, budget_controller):
        """Rewire with a solution-aware per-event budget and sparse probes."""
        candidates = self._rank_rewire_candidates(new_index, neighbor_indices)
        candidate_improvements = [item[0] for item in candidates]
        candidate_lower_bounds = [
            item[1] + np.linalg.norm(self.vertices[item[2]] - self.goal)
            for item in candidates
        ]
        decision = budget_controller.begin_event(
            candidate_improvements=candidate_improvements,
            candidate_lower_bounds=candidate_lower_bounds,
            best_cost=self.best_cost,
        )

        primary_candidates = candidates[: decision.budget]
        probe_start = decision.budget
        probe_end = probe_start + decision.probe_count
        probe_candidates = candidates[probe_start:probe_end]
        omitted_count = max(
            0,
            len(candidates) - len(primary_candidates) - len(probe_candidates),
        )

        self._add_topk_edge_stat("adaptive_rewire_event_count", 1)
        self._add_topk_edge_stat(
            "adaptive_rewire_eligible_candidate_count",
            len(candidates),
        )
        self._add_topk_edge_stat(
            "adaptive_rewire_primary_candidate_count",
            len(primary_candidates),
        )
        self._add_topk_edge_stat(
            "adaptive_rewire_probe_candidate_count",
            len(probe_candidates),
        )
        self._add_topk_edge_stat(
            "adaptive_rewire_omitted_candidate_count",
            omitted_count,
        )

        primary_rewires, primary_checked, primary_gain = (
            self._validate_and_apply_rewire_candidates(
                new_index,
                primary_candidates,
                checked_stat_key="adaptive_rewire_primary_checked_count",
                success_stat_key="adaptive_rewire_primary_success_count",
            )
        )
        probe_rewires, probe_checked, probe_gain = (
            self._validate_and_apply_rewire_candidates(
                new_index,
                probe_candidates,
                checked_stat_key="adaptive_rewire_probe_checked_count",
                success_stat_key="adaptive_rewire_probe_success_count",
            )
        )

        total_rewires = primary_rewires + probe_rewires
        if total_rewires > 0:
            self.refresh_best_solution()

        budget_controller.observe(
            primary_checked=primary_checked,
            primary_successes=primary_rewires,
            primary_gain=primary_gain,
            probe_checked=probe_checked,
            probe_successes=probe_rewires,
            probe_gain=probe_gain,
        )

        return total_rewires
