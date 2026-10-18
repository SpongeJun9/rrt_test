# planners/rrt_connect_2d.py
import time
import numpy as np

from planners.rrt_base_2d import RRTBase2D
from utils.metrics import PlanningResult, path_cost


class RRTConnect2D(RRTBase2D):
    """
    RRT-Connect baseline for fast feasible path finding.

    This planner is not an asymptotically optimal planner.
    It is included as a fast first-solution baseline.

    It uses two trees:
        - one rooted at start
        - one rooted at goal

    The planner stops when the two trees are connected.
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

        self.tree_start = None
        self.parent_start = None
        self.tree_goal = None
        self.parent_goal = None

        self.connected = False
        self.connect_start_index = None
        self.connect_goal_index = None

    def reset(self):
        super().reset()

        if hasattr(self.map, "reset_statistics"):
            self.map.reset_statistics()

        self.tree_start = [self.start.copy()]
        self.parent_start = [0]

        self.tree_goal = [self.goal.copy()]
        self.parent_goal = [0]

        self.connected = False
        self.connect_start_index = None
        self.connect_goal_index = None

        self.cost_history = []
        self.first_solution_iter = None
        self.first_solution_time = None
        self.first_solution_cost = float("inf")

    @staticmethod
    def _nearest_in_tree(tree, point):
        vertices = np.asarray(tree, dtype=float)
        dists = np.linalg.norm(vertices - point, axis=1)
        idx = int(np.argmin(dists))
        return tree[idx], idx

    def _extend(self, tree, parents, target):
        nearest, nearest_idx = self._nearest_in_tree(tree, target)
        new_point = self.steer(nearest, target)

        if np.linalg.norm(new_point - nearest) < 1e-12:
            return "trapped", None

        if not self.map.edge_valid(nearest, new_point):
            return "trapped", None

        tree.append(new_point)
        parents.append(nearest_idx)

        new_idx = len(tree) - 1

        if np.linalg.norm(new_point - target) <= self.step_len:
            if self.map.edge_valid(new_point, target):
                return "reached", new_idx

        return "advanced", new_idx

    def _connect(self, tree, parents, target):
        last_idx = None

        while True:
            status, new_idx = self._extend(tree, parents, target)

            if status == "trapped":
                return "trapped", last_idx

            last_idx = new_idx

            if status == "reached":
                return "reached", new_idx

    @staticmethod
    def _trace_path(tree, parents, idx):
        path = []

        while True:
            path.append(np.asarray(tree[idx], dtype=float).copy())
            parent = int(parents[idx])

            if parent == idx:
                break

            idx = parent

        path.reverse()
        return path

    def _build_connected_path(
        self,
        start_tree,
        start_parents,
        start_idx,
        goal_tree,
        goal_parents,
        goal_idx,
    ):
        path_from_start = self._trace_path(start_tree, start_parents, start_idx)

        path_from_goal = self._trace_path(goal_tree, goal_parents, goal_idx)
        path_from_goal = path_from_goal[::-1]

        if len(path_from_start) > 0 and len(path_from_goal) > 0:
            if np.linalg.norm(path_from_start[-1] - path_from_goal[0]) < 1e-9:
                full_path = path_from_start + path_from_goal[1:]
            else:
                full_path = path_from_start + path_from_goal
        else:
            full_path = path_from_start + path_from_goal

        return np.asarray(full_path, dtype=float)

    def solve(
        self,
        iter_after_initial: int | None = None,
        stop_at_first_solution: bool = True,
    ):
        """
        Run RRT-Connect.

        iter_after_initial is accepted for API compatibility but ignored by default,
        because RRT-Connect is used here as a fast feasible-solution baseline.
        """
        self.reset()
        start_time = time.time()

        best_path = None

        tree_a = self.tree_start
        parent_a = self.parent_start
        tree_b = self.tree_goal
        parent_b = self.parent_goal

        a_is_start_tree = True

        for iteration in range(self.iter_max):
            x_rand = self.sample()

            status_a, new_idx_a = self._extend(tree_a, parent_a, x_rand)

            if status_a != "trapped" and new_idx_a is not None:
                x_new = tree_a[new_idx_a]

                status_b, new_idx_b = self._connect(tree_b, parent_b, x_new)

                if status_b == "reached" and new_idx_b is not None:
                    self.connected = True
                    self.first_solution_iter = iteration
                    self.first_solution_time = time.time() - start_time

                    if a_is_start_tree:
                        best_path = self._build_connected_path(
                            start_tree=tree_a,
                            start_parents=parent_a,
                            start_idx=new_idx_a,
                            goal_tree=tree_b,
                            goal_parents=parent_b,
                            goal_idx=new_idx_b,
                        )
                    else:
                        best_path = self._build_connected_path(
                            start_tree=tree_b,
                            start_parents=parent_b,
                            start_idx=new_idx_b,
                            goal_tree=tree_a,
                            goal_parents=parent_a,
                            goal_idx=new_idx_a,
                        )

                    self.first_solution_cost = path_cost(best_path)
                    self.best_cost = self.first_solution_cost
                    self.first_solution_collision_checks = getattr(
                        self.map, "collision_checks", 0
                    )
                    self.cost_history.append(self.best_cost)
                    break

            self.cost_history.append(float("inf"))

            tree_a, tree_b = tree_b, tree_a
            parent_a, parent_b = parent_b, parent_a
            a_is_start_tree = not a_is_start_tree

        planning_time = time.time() - start_time

        success = best_path is not None and len(best_path) >= 2
        final_cost = path_cost(best_path) if success else float("inf")

        result = PlanningResult(
            success=success,
            path=best_path,
            final_cost=final_cost,
            first_solution_iter=self.first_solution_iter,
            first_solution_time=self.first_solution_time,
            first_solution_cost=self.first_solution_cost,
            planning_time=planning_time,
            num_vertices=len(self.tree_start) + len(self.tree_goal),
            collision_checks=getattr(self.map, "collision_checks", 0),
            cost_history=self.cost_history,
        )
        
        total_collision_checks = getattr(self.map, "collision_checks", 0)
        result.metadata["collision_checks_until_initial"] = (
            self.first_solution_collision_checks
            if self.first_solution_collision_checks is not None
            else np.nan
        )
        result.metadata["collision_checks_after_initial"] = (
            total_collision_checks - self.first_solution_collision_checks
            if self.first_solution_collision_checks is not None
            else np.nan
        )

        result.metadata["planner_category"] = "fast_feasible_solution"
        result.metadata["num_vertices_start_tree"] = len(self.tree_start)
        result.metadata["num_vertices_goal_tree"] = len(self.tree_goal)
        result.metadata["rrt_connect_stop_at_first_solution"] = True

        return result