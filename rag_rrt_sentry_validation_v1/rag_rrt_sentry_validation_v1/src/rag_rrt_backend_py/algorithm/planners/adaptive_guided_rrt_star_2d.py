# planners/adaptive_guided_rrt_star_2d.py
import time
import numpy as np

from planners.fixed_guided_rrt_star_2d import FixedGuidedRRTStar2D


class AdaptiveGuidedRRTStar2D(FixedGuidedRRTStar2D):
    """
    Adaptive guided RRT* baseline without reliability diagnostics.

    This planner is intended as a representative baseline:
        - It uses the same guided prior sampler as FixedGuidedRRTStar2D.
        - It adaptively updates alpha based on local guided-sample performance.
        - It does NOT use connectivity diagnostics.
        - It does NOT use Best-Path Consistency.
        - It does NOT use current feasible best path corridor information.

    Purpose:
        Test whether simple performance-adaptive alpha regulation is sufficient,
        compared with RAG-BPC-ConnInit.

    External prior mode:
        prior_path / prior_result are passed through to FixedGuidedRRTStar2D,
        which passes them to AStarCorridorSampler.
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
        alpha_init: float = 0.5,
        alpha_min: float = 0.05,
        alpha_max: float = 0.8,
        update_interval: int = 50,
        beta: float = 0.25,
        w_valid: float = 0.7,
        w_improve: float = 0.3,
        guide_path_shift: tuple[float, float] | None = None,
        prior_failure_mode: str = "none",
        prior_failure_params: dict | None = None,
        prior_path=None,
        prior_result=None,
    ):
        super().__init__(
            grid_map=grid_map,
            start=start,
            goal=goal,
            step_len=step_len,
            search_radius=search_radius,
            iter_max=iter_max,
            goal_radius=goal_radius,
            guide_sigma=guide_sigma,
            guide_max_attempts=guide_max_attempts,
            alpha=alpha_init,
            guide_path_shift=guide_path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.update_interval = int(update_interval)
        self.beta = float(beta)

        self.w_valid = float(w_valid)
        self.w_improve = float(w_improve)

        self.alpha = float(np.clip(alpha_init, self.alpha_min, self.alpha_max))

        self.alpha_history: list[float] = []
        self.adaptive_window: list[dict] = []

    def reset(self):
        super().reset()
        self.alpha_history = []
        self.adaptive_window = []

    def _record_adaptive_event(
        self,
        sample_type: str,
        edge_valid: bool,
        added: bool,
        improved: bool,
    ):
        if sample_type != "guided":
            return

        self.adaptive_window.append(
            {
                "edge_valid": int(edge_valid),
                "added": int(added),
                "improved": int(improved),
            }
        )

        if len(self.adaptive_window) > self.update_interval:
            self.adaptive_window.pop(0)

    def _update_alpha(self):
        if len(self.adaptive_window) < max(5, self.update_interval // 4):
            return

        n = len(self.adaptive_window)

        valid_rate = sum(e["edge_valid"] for e in self.adaptive_window) / n
        improve_rate = sum(e["improved"] for e in self.adaptive_window) / n

        alpha_target = self.w_valid * valid_rate + self.w_improve * improve_rate
        alpha_target = float(np.clip(alpha_target, self.alpha_min, self.alpha_max))

        self.alpha = (1.0 - self.beta) * self.alpha + self.beta * alpha_target
        self.alpha = float(np.clip(self.alpha, self.alpha_min, self.alpha_max))

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

            c_best_before = self.best_cost

            x_rand, sample_type = self.sample_mixed()
            self.sample_type_history.append(sample_type)

            x_nearest, nearest_index = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand)

            edge_valid = False
            added = False
            improved = False

            if np.linalg.norm(x_new - x_nearest) < 1e-12:
                self._record_adaptive_event(
                    sample_type=sample_type,
                    edge_valid=False,
                    added=False,
                    improved=False,
                )

                if (iteration + 1) % self.update_interval == 0:
                    self._update_alpha()

                self.alpha_history.append(self.alpha)
                self.record_cost()
                continue

            if not self.map.edge_valid(x_nearest, x_new):
                self._record_adaptive_event(
                    sample_type=sample_type,
                    edge_valid=False,
                    added=False,
                    improved=False,
                )

                if (iteration + 1) % self.update_interval == 0:
                    self._update_alpha()

                self.alpha_history.append(self.alpha)
                self.record_cost()
                continue

            edge_valid = True

            neighbor_indices = self.near(x_new)
            parent_index = self.choose_parent(
                point=x_new,
                neighbor_indices=neighbor_indices,
                default_parent_index=nearest_index,
            )

            new_index = self.add_vertex(x_new, parent_index)

            if new_index is None:
                self._record_adaptive_event(
                    sample_type=sample_type,
                    edge_valid=edge_valid,
                    added=False,
                    improved=False,
                )

                if (iteration + 1) % self.update_interval == 0:
                    self._update_alpha()

                self.alpha_history.append(self.alpha)
                self.record_cost()
                break

            added = True

            neighbor_indices = self.near(x_new)
            self.rewire(new_index, neighbor_indices)

            improved = self.update_solution(new_index)

            if self.best_cost + 1e-9 < c_best_before:
                improved = True

            self._record_adaptive_event(
                sample_type=sample_type,
                edge_valid=edge_valid,
                added=added,
                improved=improved,
            )

            if (iteration + 1) % self.update_interval == 0:
                self._update_alpha()

            if improved and self.first_solution_iter is None:
                initial_found = True
                self.first_solution_iter = iteration
                self.first_solution_time = time.time() - start_time
                self.first_solution_cost = self.best_cost
                self.first_solution_collision_checks = getattr(
                    self.map, "collision_checks", 0
                )

                if stop_at_first_solution:
                    self.alpha_history.append(self.alpha)
                    self.record_cost()
                    break

            if self.first_solution_iter is not None:
                initial_found = True

            self.alpha_history.append(self.alpha)
            self.record_cost()

        planning_time = time.time() - start_time
        result = self.make_result(planning_time=planning_time)

        result.alpha_history = self.alpha_history
        result.metadata["guided_prior_valid"] = self.guided_sampler.valid_prior
        result.metadata["guided_prior_cost"] = self.guided_sampler.path_cost
        result.metadata["guided_prior_source"] = getattr(
            self.guided_sampler, "prior_source", "unknown"
        )
        result.metadata["num_guided_samples"] = self.sample_type_history.count("guided")
        result.metadata["num_base_samples"] = self.sample_type_history.count("base")
        result.metadata["adaptive_alpha_init"] = self.alpha_history[0] if self.alpha_history else self.alpha
        result.metadata["adaptive_alpha_final"] = self.alpha
        result.metadata["adaptive_alpha_min"] = self.alpha_min
        result.metadata["adaptive_alpha_max"] = self.alpha_max
        result.metadata["adaptive_update_interval"] = self.update_interval
        result.metadata["adaptive_beta"] = self.beta
        result.metadata["adaptive_w_valid"] = self.w_valid
        result.metadata["adaptive_w_improve"] = self.w_improve
        result.metadata["prior_failure_mode"] = getattr(self, "prior_failure_mode", "none")
        result.metadata["prior_failure_params"] = str(getattr(self, "prior_failure_params", {}))

        if self.external_prior_result is not None:
            result.metadata["external_prior_type"] = getattr(
                self.external_prior_result, "prior_type", None
            )
            result.metadata["external_prior_mode"] = getattr(
                self.external_prior_result, "prior_mode", None
            )

        return result
