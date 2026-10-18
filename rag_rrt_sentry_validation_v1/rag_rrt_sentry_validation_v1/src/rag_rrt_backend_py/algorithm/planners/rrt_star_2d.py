from __future__ import annotations
# planners/rrt_star_2d.py
import time
import numpy as np

from planners.rrt_base_2d import RRTBase2D


class RRTStar2D(RRTBase2D):
    """
    Standard RRT* with uniform sampling.
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

    def solve(
        self,
        iter_after_initial: int | None = None,
        stop_at_first_solution: bool = False,
    ):
        """
        Run RRT*.

        If iter_after_initial is not None:
            first find an initial solution, then continue for iter_after_initial iterations.
        """
        self.reset()

        start_time = time.time()
        initial_found = False
        after_initial_count = 0

        for iteration in range(self.iter_max):
            if initial_found and iter_after_initial is not None:
                if after_initial_count >= iter_after_initial:
                    break
                after_initial_count += 1

            x_rand = self.sample()

            x_nearest, nearest_index = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand)

            if np.linalg.norm(x_new - x_nearest) < 1e-12:
                self.record_cost()
                continue

            if not self.map.edge_valid(x_nearest, x_new):
                self.record_cost()
                continue

            neighbor_indices = self.near(x_new)
            parent_index = self.choose_parent(
                point=x_new,
                neighbor_indices=neighbor_indices,
                default_parent_index=nearest_index,
            )

            new_index = self.add_vertex(x_new, parent_index)

            if new_index is None:
                self.record_cost()
                break

            neighbor_indices = self.near(x_new)
            self.rewire(new_index, neighbor_indices)

            improved = self.update_solution(new_index)

            if improved and self.first_solution_iter is None:
                initial_found = True
                self.first_solution_iter = iteration
                self.first_solution_time = time.time() - start_time
                self.first_solution_cost = self.best_cost
                self.first_solution_collision_checks = getattr(
                    self.map, "collision_checks", 0
                )

                if stop_at_first_solution:
                    self.record_cost()
                    break

            if self.first_solution_iter is not None:
                initial_found = True

            self.record_cost()

        planning_time = time.time() - start_time
        return self.make_result(planning_time=planning_time)