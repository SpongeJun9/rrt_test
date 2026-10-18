# planners/fixed_guided_rrt_star_2d.py
import time
import numpy as np

from planners.informed_rrt_star_2d import InformedRRTStar2D
from samplers.astar_corridor_sampler import AStarCorridorSampler


class FixedGuidedRRTStar2D(InformedRRTStar2D):
    """
    Fixed-ratio guided RRT*.

    Sampling rule:
        with probability alpha:
            sample from A* corridor guided sampler
        otherwise:
            use base sampler:
                - before initial solution: uniform sampling
                - after initial solution : informed sampling

    This is an important baseline for ReliabilityGuidedRRTStar2D.
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
        guide_sigma: float = 8.0,
        guide_max_attempts: int = 50,
        alpha: float = 0.5,
        guide_path_shift: tuple[float, float] | None = None,
        prior_failure_mode: str = "none",
        prior_failure_params: dict | None = None,
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

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

        self.alpha = float(alpha)
        
        self.prior_failure_mode = "none" if prior_failure_mode is None else str(prior_failure_mode)
        self.prior_failure_params = {} if prior_failure_params is None else dict(prior_failure_params)

        self.guided_sampler = AStarCorridorSampler(
            grid_map=grid_map,
            start=start,
            goal=goal,
            sigma=guide_sigma,
            max_attempts=guide_max_attempts,
            path_shift=guide_path_shift,
            prior_failure_mode=self.prior_failure_mode,
            prior_failure_params=self.prior_failure_params,
        )

        self.sample_type_history: list[str] = []

    def reset(self):
        super().reset()
        self.sample_type_history = []

    def sample_base(self):
        return self.sample_informed_subset()

    def sample_mixed(self):
        if np.random.random() < self.alpha:
            x_rand = self.guided_sampler.sample()

            if x_rand is not None:
                return x_rand, "guided"

            return self.sample_base(), "base"

        return self.sample_base(), "base"

    def solve(
        self,
        iter_after_initial: int | None = None,
        stop_at_first_solution: bool = False,
    ):
        self.reset()

        start_time = time.time()
        initial_found = False
        after_initial_count = 0

        for iteration in range(self.iter_max):
            if initial_found and iter_after_initial is not None:
                if after_initial_count >= iter_after_initial:
                    break
                after_initial_count += 1

            x_rand, sample_type = self.sample_mixed()
            self.sample_type_history.append(sample_type)

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
        result = self.make_result(planning_time=planning_time)

        result.alpha_history = [self.alpha] * len(self.cost_history)
        result.metadata["guided_prior_valid"] = self.guided_sampler.valid_prior
        result.metadata["guided_prior_cost"] = self.guided_sampler.path_cost
        result.metadata["num_guided_samples"] = self.sample_type_history.count("guided")
        result.metadata["num_base_samples"] = self.sample_type_history.count("base")
        result.metadata["fixed_alpha"] = self.alpha
        result.metadata["prior_failure_mode"] = self.prior_failure_mode
        result.metadata["prior_failure_params"] = str(self.prior_failure_params)

        return result