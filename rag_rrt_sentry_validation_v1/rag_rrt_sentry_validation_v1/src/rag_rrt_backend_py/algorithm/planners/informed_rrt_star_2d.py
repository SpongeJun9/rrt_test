from __future__ import annotations
# planners/informed_rrt_star_2d.py
import math
import time

import numpy as np

from planners.rrt_star_2d import RRTStar2D


class InformedRRTStar2D(RRTStar2D):
    """
    Informed RRT*.

    Before an initial solution is found, it samples uniformly.
    After an initial solution is found, it samples inside the prolate ellipse
    defined by start, goal, and current best path cost.
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
        max_informed_sample_attempts: int = 100,
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

        self.max_informed_sample_attempts = int(max_informed_sample_attempts)

        self.c_min = float(np.linalg.norm(self.goal - self.start))
        self.x_center = (self.start + self.goal) / 2.0
        self.rotation = self._compute_rotation()

    def _compute_rotation(self):
        """
        Rotation matrix from unit x-axis to start-goal direction.
        """
        direction = self.goal - self.start
        norm = np.linalg.norm(direction)

        if norm < 1e-12:
            return np.eye(2)

        direction = direction / norm
        theta = math.atan2(direction[1], direction[0])

        return np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=float,
        )

    @staticmethod
    def sample_unit_ball():
        """
        Uniformly sample a point inside the 2D unit ball.
        """
        while True:
            p = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(p) <= 1.0:
                return p

    def sample_informed_subset(self):
        """
        Sample from the informed ellipse.

        If no solution exists yet, fallback to uniform sampling.
        """
        if not np.isfinite(self.best_cost):
            return self.map.sample_free()

        c_best = self.best_cost

        # Numerical guard.
        if c_best < self.c_min:
            c_best = self.c_min

        a = c_best / 2.0
        b_square = c_best**2 - self.c_min**2
        b = math.sqrt(max(b_square, 0.0)) / 2.0

        transform = self.rotation @ np.diag([a, b])

        for _ in range(self.max_informed_sample_attempts):
            x_ball = self.sample_unit_ball()
            x_rand = transform @ x_ball + self.x_center

            if self.map.point_valid(x_rand):
                return x_rand

        # If repeated informed samples are invalid, fallback to uniform.
        return self.map.sample_free()

    def sample(self):
        return self.sample_informed_subset()

    def solve(
        self,
        iter_after_initial: int | None = None,
        stop_at_first_solution: bool = False,
    ):
        """
        Same loop as RRT*, but sample() is now informed after initial solution.
        Kept explicit here for readability and later extension.
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