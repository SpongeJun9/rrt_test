# planners/reliability_guided_rrt_star_2d.py
import time
import numpy as np

from planners.informed_rrt_star_2d import InformedRRTStar2D
from samplers.astar_corridor_sampler import AStarCorridorSampler
from controllers.edge_budget_controller import AdaptiveRewireBudgetController
from controllers.reliability_controller import ReliabilityController
from utils.guidance_connectivity import compute_prior_connectivity_features


class ReliabilityGuidedRRTStar2D(InformedRRTStar2D):
    """
    Reliability-aware guided-uniform / guided-informed RRT*.

    Sampling rule:
        with probability alpha_t:
            sample from A* corridor guided sampler
        otherwise:
            use base sampler. Two base sampler modes are supported:
                - "informed_after_initial" (default): uniform before the first
                  solution and informed sampling after the first solution;
                - "uniform": uniform sampling both before and after the first
                  solution, used for the RAG-uniform-base ablation.

    This version adds best-path corridor consistency:
        after an initial solution is found, guided samples are evaluated by
        their distance to the current best path. If guided samples are often
        far from the best-path corridor, alpha is penalized and capped.
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
        alpha_min: float = 0.1,
        alpha_max: float = 0.8,
        window_size: int = 50,
        update_interval: int = 20,
        eta: float = 0.05,
        margin: float = 0.05,
        min_samples_to_update: int = 10,
        w_add: float = 1.0,
        w_collision: float = 1.5,
        w_improve: float = 2.0,
        guide_path_shift: tuple[float, float] | None = None,
        prior_failure_mode: str = "none",           # added
        prior_failure_params: dict | None = None,   # added
        prior_path=None,
        prior_result=None,
        w_progress: float = 1.0,
        stagnation_check_interval: int = 100,
        stagnation_start_iter: int = 100,
        stagnation_alpha_decay: float = 0.03,
        # Master switch for guided/base relative online alpha regulation.
        # When disabled, alpha remains fixed at its initial value (which can
        # still be produced by ConnInit), while diagnostic signals are logged.
        enable_online_reliability_update: bool = True,
        # Best-path corridor consistency parameters.
        best_path_corridor_radius: float = 10.0,
        consistency_window_size: int | None = None,
        min_consistency_samples: int = 8,
        consistency_low_threshold: float = 0.35,
        consistency_mid_threshold: float = 0.55,
        consistency_high_threshold: float = 0.70,
        consistency_penalty: float = 0.20,
        consistency_bonus: float = 0.05,
        low_consistency_alpha_ceiling: float = 0.30,
        mid_consistency_alpha_ceiling: float = 0.50,
        high_consistency_recovery: bool = True,
        high_consistency_recovery_eta: float = 0.03,
        # Connectivity-based pre-solution trust regulation.
        use_connectivity_alpha_ceiling: bool = False,
        connectivity_alpha_ceiling_min: float | None = None,
        # Connectivity-based initial trust modulation.
        use_connectivity_alpha_init: bool = False,
        connectivity_alpha_init_min: float = 0.30,
        connectivity_alpha_init_max: float = 0.50,
        use_topk_edge_checking: bool = False,
        parent_check_top_k: int | None = None,
        rewire_check_top_k: int | None = None,
        use_topk_fallback: bool = False,
        parent_fallback_check_k: int | None = None,
        rewire_fallback_check_k: int | None = None,
        # Independent edge-validation strategies.  ``None`` preserves the
        # legacy behavior selected by use_topk_edge_checking.
        parent_edge_strategy: str | None = None,
        rewire_edge_strategy: str | None = None,
        adaptive_rewire_config: dict | None = None,
        # Base-branch sampling mode.
        #   "informed_after_initial": default RAG behavior.
        #   "uniform": RAG-uniform-base ablation.
        base_sampler_mode: str = "informed_after_initial",
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

        self.base_sampler_mode = self._normalize_base_sampler_mode(
            base_sampler_mode
        )

        # added
        self.prior_failure_mode = (
            "none" if prior_failure_mode is None else str(prior_failure_mode)
        )
        # added
        self.prior_failure_params = (
            {} if prior_failure_params is None else dict(prior_failure_params)
        )
        self.external_prior_result = prior_result
        self.external_prior_path = prior_path

        self.guided_sampler = AStarCorridorSampler(
            grid_map=grid_map,
            start=start,
            goal=goal,
            sigma=guide_sigma,
            max_attempts=guide_max_attempts,
            path_shift=guide_path_shift,
            prior_failure_mode=self.prior_failure_mode,
            prior_failure_params=self.prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )
        # added
        self.prior_connectivity_features = compute_prior_connectivity_features(
            grid_map=grid_map,
            prior_path=self.guided_sampler.get_prior_path(),
            start=start,
            goal=goal,
            edge_resolution=1.0,
            start_goal_near_radius=10.0,
        )
        
        self.use_connectivity_alpha_ceiling = bool(use_connectivity_alpha_ceiling)
        
        self.use_topk_edge_checking = bool(use_topk_edge_checking)
        self.parent_check_top_k = parent_check_top_k
        self.rewire_check_top_k = rewire_check_top_k
        self.use_topk_fallback = bool(use_topk_fallback)
        self.parent_fallback_check_k = parent_fallback_check_k
        self.rewire_fallback_check_k = rewire_fallback_check_k

        if parent_edge_strategy is None:
            parent_edge_strategy = (
                "topk" if self.use_topk_edge_checking else "legacy"
            )
        if rewire_edge_strategy is None:
            rewire_edge_strategy = (
                "topk" if self.use_topk_edge_checking else "full"
            )

        parent_aliases = {
            "legacy": "legacy",
            "legacy_full": "legacy",
            "full": "legacy",
            "delaycc": "delay_cc",
            "delay_cc": "delay_cc",
            "topk": "topk",
            "fixed_topk": "topk",
        }
        rewire_aliases = {
            "legacy": "full",
            "legacy_full": "full",
            "full": "full",
            "topk": "topk",
            "fixed_topk": "topk",
            "adaptive": "adaptive",
            "adaptive_budget": "adaptive",
        }

        parent_key = str(parent_edge_strategy).lower().strip()
        rewire_key = str(rewire_edge_strategy).lower().strip()
        if parent_key not in parent_aliases:
            raise ValueError(
                "parent_edge_strategy must be legacy, delay_cc, or topk"
            )
        if rewire_key not in rewire_aliases:
            raise ValueError(
                "rewire_edge_strategy must be full, topk, or adaptive"
            )

        self.parent_edge_strategy = parent_aliases[parent_key]
        self.rewire_edge_strategy = rewire_aliases[rewire_key]
        self.adaptive_rewire_config = (
            {} if adaptive_rewire_config is None else dict(adaptive_rewire_config)
        )
        self.adaptive_rewire_controller = (
            AdaptiveRewireBudgetController(**self.adaptive_rewire_config)
            if self.rewire_edge_strategy == "adaptive"
            else None
        )
        
        connectivity_score = float(
            self.prior_connectivity_features.get("connectivity_score", 1.0)
        )
        
        conn_floor = (
            alpha_min
            if connectivity_alpha_ceiling_min is None
            else float(connectivity_alpha_ceiling_min)
        )

        conn_floor = float(np.clip(conn_floor, alpha_min, alpha_max))

        raw_conn_ceiling = conn_floor + connectivity_score * (alpha_max - conn_floor)
        
        self.connectivity_alpha_ceiling = float(
            np.clip(raw_conn_ceiling, alpha_min, alpha_max)
        )
        
        pre_solution_alpha_ceiling = (
            self.connectivity_alpha_ceiling
            if self.use_connectivity_alpha_ceiling
            else None
        )
        # added
        self.use_connectivity_alpha_init = bool(use_connectivity_alpha_init)

        init_min = float(np.clip(connectivity_alpha_init_min, alpha_min, alpha_max))
        init_max = float(np.clip(connectivity_alpha_init_max, alpha_min, alpha_max))
        
        if init_min > init_max:
            raise ValueError(
                "connectivity_alpha_init_min must be <= connectivity_alpha_init_max"
            )
        
        self.connectivity_alpha_init = float(
            init_min + connectivity_score * (init_max - init_min)
        )
        
        effective_alpha_init = (
            self.connectivity_alpha_init
            if self.use_connectivity_alpha_init
            else alpha_init
        )
        

        self.best_path_corridor_radius = float(best_path_corridor_radius)

        self.controller = ReliabilityController(
            alpha_init=effective_alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            window_size=window_size,
            eta=eta,
            margin=margin,
            min_samples_to_update=min_samples_to_update,
            w_add=w_add,
            w_progress=w_progress,
            w_collision=w_collision,
            w_improve=w_improve,
            consistency_window_size=consistency_window_size,
            min_consistency_samples=min_consistency_samples,
            consistency_low_threshold=consistency_low_threshold,
            consistency_mid_threshold=consistency_mid_threshold,
            consistency_high_threshold=consistency_high_threshold,
            consistency_penalty=consistency_penalty,
            consistency_bonus=consistency_bonus,
            low_consistency_alpha_ceiling=low_consistency_alpha_ceiling,
            mid_consistency_alpha_ceiling=mid_consistency_alpha_ceiling,
            high_consistency_recovery=high_consistency_recovery,
            high_consistency_recovery_eta=high_consistency_recovery_eta,
            # added
            pre_solution_alpha_ceiling=pre_solution_alpha_ceiling,
        )

        self.update_interval = int(update_interval)

        self.sample_type_history: list[str] = []
        self.guided_distance_to_best_path_sample_history: list[float] = []
        self.guided_best_path_consistent_sample_history: list[float] = []
        
        # added
        self.iteration_history: list[dict] = []

        self.stagnation_check_interval = int(stagnation_check_interval)
        self.stagnation_start_iter = int(stagnation_start_iter)
        self.stagnation_alpha_decay = float(stagnation_alpha_decay)
        self.enable_online_reliability_update = bool(
            enable_online_reliability_update
        )
        

    def reset(self):
        super().reset()

        # The new controller has its own reset method.
        # Keep the fallback for compatibility with an older controller file.
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        else:
            self.controller.guided_window.clear()
            self.controller.base_window.clear()
            self.controller.alpha_history = [self.controller.alpha]
            self.controller.reliability_history = []
            self.controller.guided_score_history = []
            self.controller.base_score_history = []
            self.controller.added_rate_history = []
            self.controller.collision_rate_history = []
            self.controller.progress_rate_history = []
            self.controller.improvement_rate_history = []

        if self.adaptive_rewire_controller is not None:
            self.adaptive_rewire_controller.reset()

        self.sample_type_history = []
        self.guided_distance_to_best_path_sample_history = []
        self.guided_best_path_consistent_sample_history = []
        
        # added
        self.iteration_history = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_base_sampler_mode(mode: str) -> str:
        """
        Normalize the base-branch sampling mode.

        Supported canonical modes:
            - informed_after_initial:
                use Informed-RRT* behavior for the base branch. This means
                uniform sampling before the first solution and informed-subset
                sampling after the first solution.
            - uniform:
                use uniform free-space sampling for the base branch throughout
                the whole run. This is intended for the RAG-uniform-base
                ablation.
        """
        mode = str(mode).lower().strip()

        aliases = {
            "informed": "informed_after_initial",
            "informed_after_initial": "informed_after_initial",
            "informed-base": "informed_after_initial",
            "informed_base": "informed_after_initial",
            "default": "informed_after_initial",
            "uniform": "uniform",
            "uniform_base": "uniform",
            "uniform-base": "uniform",
            "rag_uniform_base": "uniform",
            "rag-uniform-base": "uniform",
        }

        if mode not in aliases:
            raise ValueError(
                "Unknown base_sampler_mode: "
                f"{mode!r}. Supported modes are 'informed_after_initial' "
                "and 'uniform'."
            )

        return aliases[mode]

    def sample_base(self):
        """
        Sample from the non-prior base branch.

        In the default mode, the base branch follows Informed-RRT* behavior:
        uniform sampling before the first solution and informed-subset sampling
        after the first solution. In the uniform ablation mode, the base branch
        always uses uniform free-space sampling, including after the first
        solution.
        """
        if self.base_sampler_mode == "uniform":
            return self.map.sample_free()

        return self.sample_informed_subset()

    def sample_mixed(self):
        """
        Return:
            sample: np.ndarray
            sample_type: "guided" or "base"
        """
        alpha = self.controller.get_alpha()

        if np.random.random() < alpha:
            x_rand = self.guided_sampler.sample()

            if x_rand is not None:
                return x_rand, "guided"

            # If guided sampler fails internally, fallback to base sampler.
            return self.sample_base(), "base"

        return self.sample_base(), "base"

    # ------------------------------------------------------------------
    # Best-path corridor consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _as_point_array(point):
        """
        Convert a point-like object to a 2D numpy array.

        This helper accepts common formats:
            - np.ndarray/list/tuple: [x, y]
            - object with .point
            - object with .x and .y
            - object with .pos
        """
        if point is None:
            return None

        if hasattr(point, "point"):
            point = point.point
        elif hasattr(point, "pos"):
            point = point.pos
        elif hasattr(point, "x") and hasattr(point, "y"):
            point = [point.x, point.y]

        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]

    def _point_to_segment_distance(self, p, a, b):
        p = self._as_point_array(p)
        a = self._as_point_array(a)
        b = self._as_point_array(b)

        if p is None or a is None or b is None:
            return None

        ab = b - a
        ap = p - a
        denom = float(np.dot(ab, ab))

        if denom < 1e-12:
            return float(np.linalg.norm(p - a))

        t = float(np.dot(ap, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _distance_to_path(self, point, path):
        if path is None or len(path) == 0:
            return None

        if len(path) == 1:
            p = self._as_point_array(point)
            q = self._as_point_array(path[0])
            if p is None or q is None:
                return None
            return float(np.linalg.norm(p - q))

        min_dist = float("inf")
        for i in range(len(path) - 1):
            dist = self._point_to_segment_distance(point, path[i], path[i + 1])
            if dist is not None:
                min_dist = min(min_dist, dist)

        if not np.isfinite(min_dist):
            return None
        return float(min_dist)

    def _extract_path_from_parent_arrays(self, goal_index):
        """
        Fallback path extraction for common RRT implementations.

        This is only used when the base class does not expose get_best_path()
        or get_path().
        """
        vertices = None
        for vertices_attr in ["vertices", "vertex", "nodes", "node_list", "V"]:
            if hasattr(self, vertices_attr):
                vertices = getattr(self, vertices_attr)
                break

        parents = None
        for parents_attr in ["parents", "parent", "parent_indices", "parent_index"]:
            if hasattr(self, parents_attr):
                parents = getattr(self, parents_attr)
                break

        if vertices is None or parents is None or goal_index is None:
            return None

        try:
            idx = int(goal_index)
        except Exception:
            return None

        path = []
        visited = set()

        while idx is not None and idx >= 0 and idx not in visited:
            visited.add(idx)

            try:
                vertex = vertices[idx]
            except Exception:
                return None

            point = self._as_point_array(vertex)
            if point is None:
                return None
            path.append(point)

            try:
                parent_idx = parents[idx]
            except Exception:
                break

            if parent_idx is None:
                break

            try:
                parent_idx = int(parent_idx)
            except Exception:
                break

            if parent_idx == idx:
                break

            idx = parent_idx

        if len(path) == 0:
            return None

        path.reverse()
        return path

    def _get_current_best_path(self):
        """
        Return the current best path after an initial solution is found.

        The exact path API may differ among your base classes, so this method
        tries several common interfaces. If none is available, consistency
        simply remains inactive instead of breaking the planner.
        """
        if self.first_solution_iter is None:
            return None

        for method_name in ["get_best_path", "get_path"]:
            method = getattr(self, method_name, None)
            if callable(method):
                try:
                    path = method()
                    if path is not None and len(path) > 0:
                        return path
                except TypeError:
                    pass
                except Exception:
                    pass

        for attr_name in ["best_path", "path", "solution_path"]:
            path = getattr(self, attr_name, None)
            if path is not None:
                try:
                    if len(path) > 0:
                        return path
                except Exception:
                    pass

        for index_attr in [
            "best_goal_index",
            "goal_index",
            "best_goal_node_index",
            "goal_node_index",
            "solution_index",
        ]:
            if hasattr(self, index_attr):
                path = self._extract_path_from_parent_arrays(getattr(self, index_attr))
                if path is not None and len(path) > 0:
                    return path

        return None

    def _compute_best_path_consistency(self, x_rand, sample_type):
        """
        Compute distance from guided sample to current best path.

        Only guided samples are evaluated. Base samples are not considered
        evidence for or against the reliability of the guided prior.
        """
        if sample_type != "guided":
            return None, None

        path = self._get_current_best_path()
        if path is None:
            return None, None

        distance = self._distance_to_path(x_rand, path)
        if distance is None:
            return None, None

        consistent = bool(distance <= self.best_path_corridor_radius)
        return distance, consistent

    def _record_sample(
        self,
        sample_type,
        x_rand,
        added,
        collision,
        improved,
        progress,
    ):
        distance_to_best_path, best_path_consistent = self._compute_best_path_consistency(
            x_rand=x_rand,
            sample_type=sample_type,
        )

        if sample_type == "guided":
            self.guided_distance_to_best_path_sample_history.append(
                np.nan if distance_to_best_path is None else float(distance_to_best_path)
            )
            self.guided_best_path_consistent_sample_history.append(
                np.nan if best_path_consistent is None else float(best_path_consistent)
            )

        self.controller.record(
            sample_type=sample_type,
            added=added,
            collision=collision,
            improved=improved,
            progress=progress,
            has_solution=self.first_solution_iter is not None,
            distance_to_best_path=distance_to_best_path,
            best_path_consistent=best_path_consistent,
        )

    # ------------------------------------------------------------------
    # Controller update
    # ------------------------------------------------------------------
    # modified
    def _post_iteration_update(
        self,
        iteration: int,
        sample_type: str,
        added: bool,
        collision: bool,
        improved: bool,
        progress: bool,
    ):
        allow_increase = self.first_solution_iter is not None

        self._maybe_update_controller(
            iteration=iteration,
            allow_increase=allow_increase,
        )

        stagnation_penalty_applied = False

        if (
            self.enable_online_reliability_update
            and
            self.first_solution_iter is None
            and iteration + 1 >= self.stagnation_start_iter
            and (iteration + 1) % self.stagnation_check_interval == 0
        ):
            self.controller.apply_stagnation_penalty(
                decay=self.stagnation_alpha_decay
            )
            stagnation_penalty_applied = True

        self._record_iteration()

        self._record_iteration_trajectory(
            iteration=iteration,
            sample_type=sample_type,
            added=added,
            collision=collision,
            improved=improved,
            progress=progress,
        )

        if len(self.iteration_history) > 0:
            self.iteration_history[-1]["stagnation_penalty_applied"] = int(
                stagnation_penalty_applied
            )

    def _maybe_update_controller(self, iteration: int, allow_increase: bool = True):
        if (iteration + 1) % self.update_interval == 0:
            if not self.enable_online_reliability_update:
                snapshot = getattr(self.controller, "record_snapshot", None)
                if callable(snapshot):
                    snapshot(allow_increase=allow_increase)
                return

            try:
                self.controller.update(allow_increase=allow_increase)
            except TypeError:
                # Fallback for the older controller implementation.
                old_alpha = self.controller.get_alpha()
                self.controller.update()
                if not allow_increase:
                    self.controller.alpha = min(self.controller.alpha, old_alpha)

    
    # added
    def _get_num_vertices_for_log(self):
        """
        Return the current number of valid vertices.

        In this project, vertices / parents / costs are numpy arrays preallocated
        with length iter_max + 1. Therefore len(vertices) is the capacity, not the
        current number of inserted vertices.

        The actual number of valid vertices is inferred from finite costs.
        """
        if hasattr(self, "costs"):
            try:
                costs = np.asarray(self.costs, dtype=float)
                return int(np.sum(np.isfinite(costs)))
            except Exception:
                pass

        if hasattr(self, "parents"):
            try:
                parents = np.asarray(self.parents)

                # Common convention:
                #   parent = -1 means unused
                #   start node may also have parent = -1
                # Therefore this fallback may undercount or overcount depending on
                # implementation, so costs-based counting is preferred.
                valid_parent_count = int(np.sum(parents >= 0))

                if valid_parent_count > 0:
                    return valid_parent_count + 1
            except Exception:
                pass

        for attr_name in ["nodes", "node_list", "V"]:
            if hasattr(self, attr_name):
                try:
                    return int(len(getattr(self, attr_name)))
                except Exception:
                    pass

        return np.nan
    
    
    # added
    def _record_iteration_trajectory(
        self,
        iteration: int,
        sample_type: str,
        added: bool,
        collision: bool,
        improved: bool,
        progress: bool,
    ):
        """
            Record one aligned per-iteration trajectory row.

            This is different from controller histories. Controller histories are
            update-level. This trajectory is iteration-level and is intended for
            plotting alpha / BPC / cost / sample-source curves.
        """ 
        allow_increase = self.first_solution_iter is not None

        if hasattr(self.controller, "get_current_state"):
            controller_state = self.controller.get_current_state(
                allow_increase=allow_increase
            )
        else:
            controller_state = {
                "alpha": self.controller.get_alpha(),
                "alpha_ceiling": np.nan,
                "relative_reliability": np.nan,
                "adjusted_reliability": np.nan,
                "guided_score": np.nan,
                "base_score": np.nan,
                "best_path_consistency": np.nan,
                "mean_distance_to_best_path": np.nan,
                "consistency_level": "unknown",
                "guided_window_size": np.nan,
                "base_window_size": np.nan,
                "consistency_window_size": np.nan,
            }

        row = {
            "iteration": int(iteration),

            "sample_type": sample_type,
            "is_guided": int(sample_type == "guided"),
            "is_base": int(sample_type == "base"),

            "added": int(added),
            "collision": int(collision),
            "improved": int(improved),
            "progress": int(progress),

            "has_solution": int(self.first_solution_iter is not None),
            "first_solution_iter": (
                np.nan
                if self.first_solution_iter is None
                else int(self.first_solution_iter)
            ),
            "first_solution_cost": (
                np.nan
                if self.first_solution_cost is None
                else float(self.first_solution_cost)
            ),
            "best_cost": (
                np.nan
                if self.best_cost is None or not np.isfinite(self.best_cost)
                else float(self.best_cost)
            ),

            "num_vertices": self._get_num_vertices_for_log(),

            "connectivity_score": float(
                self.prior_connectivity_features.get("connectivity_score", np.nan)
            ),
            "connectivity_alpha_init": float(
                getattr(self, "connectivity_alpha_init", np.nan)
            ),
            "connectivity_alpha_ceiling": float(
                getattr(self, "connectivity_alpha_ceiling", np.nan)
            ),
            "use_connectivity_alpha_init": int(
                getattr(self, "use_connectivity_alpha_init", False)
            ),
            "use_connectivity_alpha_ceiling": int(
                getattr(self, "use_connectivity_alpha_ceiling", False)
            ),
            "prior_failure_mode": self.prior_failure_mode,
            "base_sampler_mode": self.base_sampler_mode,
        }

        row.update(controller_state)

        self.iteration_history.append(row)
    
    
    def _record_iteration(self):
        self.record_cost()

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

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

            added = False
            collision = False
            improved = False
            progress = False

            if np.linalg.norm(x_new - x_nearest) < 1e-12:
                self._record_sample(
                    sample_type=sample_type,
                    x_rand=x_rand,
                    added=False,
                    collision=False,
                    improved=False,
                    progress=False,
                )
                # modified
                self._post_iteration_update(
                    iteration=iteration,
                    sample_type=sample_type,
                    added=False,
                    collision=False,
                    improved=False,
                    progress=False,
                )
                continue

            if not self.map.edge_valid(x_nearest, x_new):
                collision = True

                self._record_sample(
                    sample_type=sample_type,
                    x_rand=x_rand,
                    added=False,
                    collision=True,
                    improved=False,
                    progress=False,
                )
                self._post_iteration_update(
                    iteration=iteration,
                    sample_type=sample_type,
                    added=False,
                    collision=True,
                    improved=False,
                    progress=False,
                )
                continue

            neighbor_indices = self.near(x_new)
            if self.parent_edge_strategy == "legacy":
                parent_index = self.choose_parent(
                    point=x_new,
                    neighbor_indices=neighbor_indices,
                    default_parent_index=nearest_index,
                )
            elif self.parent_edge_strategy == "delay_cc":
                parent_index = self.choose_parent_topk(
                    point=x_new,
                    neighbor_indices=neighbor_indices,
                    default_parent_index=nearest_index,
                    parent_check_top_k=None,
                    use_topk_fallback=False,
                    parent_fallback_check_k=None,
                )
            else:
                parent_index = self.choose_parent_topk(
                    point=x_new,
                    neighbor_indices=neighbor_indices,
                    default_parent_index=nearest_index,
                    parent_check_top_k=self.parent_check_top_k,
                    use_topk_fallback=self.use_topk_fallback,
                    parent_fallback_check_k=self.parent_fallback_check_k,
                )

            new_index = self.add_vertex(x_new, parent_index)

            if new_index is None:
                self._record_sample(
                    sample_type=sample_type,
                    x_rand=x_rand,
                    added=False,
                    collision=False,
                    improved=False,
                    progress=False,
                )
                self._post_iteration_update(
                    iteration=iteration,
                    sample_type=sample_type,
                    added=False,
                    collision=False,
                    improved=False,
                    progress=False,
                )
                continue

            added = True

            nearest_goal_dist = np.linalg.norm(x_nearest - self.goal)
            new_goal_dist = np.linalg.norm(x_new - self.goal)
            progress = new_goal_dist < nearest_goal_dist - 1e-6

            neighbor_indices = self.near(x_new)
            if self.rewire_edge_strategy == "full":
                self.rewire(new_index, neighbor_indices)
            elif self.rewire_edge_strategy == "topk":
                self.rewire_topk(
                    new_index,
                    neighbor_indices,
                    rewire_check_top_k=self.rewire_check_top_k,
                    use_topk_fallback=self.use_topk_fallback,
                    rewire_fallback_check_k=self.rewire_fallback_check_k,
                )
            else:
                self.rewire_adaptive(
                    new_index,
                    neighbor_indices,
                    budget_controller=self.adaptive_rewire_controller,
                )

            improved = self.update_solution(new_index)

            # If rewiring changed the current best path cost indirectly,
            # check whether best cost improved compared with before this iteration.
            if self.best_cost + 1e-9 < c_best_before:
                improved = True

            if improved and self.first_solution_iter is None:
                initial_found = True
                self.first_solution_iter = iteration
                self.first_solution_time = time.time() - start_time
                self.first_solution_cost = self.best_cost
                self.first_solution_collision_checks = getattr(
                    self.map, "collision_checks", 0
                )

                self._record_sample(
                    sample_type=sample_type,
                    x_rand=x_rand,
                    added=added,
                    collision=collision,
                    improved=improved,
                    progress=progress,
                )
                self._post_iteration_update(
                    iteration=iteration,
                    sample_type=sample_type,
                    added=added,
                    collision=collision,
                    improved=improved,
                    progress=progress,
                )

                if stop_at_first_solution:
                    break

                continue

            if self.first_solution_iter is not None:
                initial_found = True

            self._record_sample(
                sample_type=sample_type,
                x_rand=x_rand,
                added=added,
                collision=collision,
                improved=improved,
                progress=progress,
            )

            self._post_iteration_update(
                iteration=iteration,
                sample_type=sample_type,
                added=added,
                collision=collision,
                improved=improved,
                progress=progress,
            )

        planning_time = time.time() - start_time
        result = self.make_result(planning_time=planning_time)

        histories = self.controller.get_histories()

        result.alpha_history = histories["alpha_history"]
        result.reliability_history = histories["reliability_history"]
        result.guided_added_rate_history = histories["guided_added_rate_history"]
        result.guided_collision_rate_history = histories["guided_collision_rate_history"]
        result.guided_progress_rate_history = histories["guided_progress_rate_history"]
        result.guided_improvement_rate_history = histories["guided_improvement_rate_history"]

        result.best_path_consistency_rate_history = histories.get(
            "best_path_consistency_rate_history", []
        )
        result.mean_distance_to_best_path_history = histories.get(
            "mean_distance_to_best_path_history", []
        )
        result.alpha_ceiling_history = histories.get(
            "alpha_ceiling_history", []
        )

        result.metadata["guided_score_history"] = histories["guided_score_history"]
        result.metadata["base_score_history"] = histories["base_score_history"]

        result.metadata["best_path_consistency_rate_history"] = histories.get(
            "best_path_consistency_rate_history", []
        )
        result.metadata["mean_distance_to_best_path_history"] = histories.get(
            "mean_distance_to_best_path_history", []
        )
        result.metadata["alpha_ceiling_history"] = histories.get(
            "alpha_ceiling_history", []
        )
        result.metadata["guided_distance_to_best_path_sample_history"] = (
            self.guided_distance_to_best_path_sample_history
        )
        result.metadata["guided_best_path_consistent_sample_history"] = (
            self.guided_best_path_consistent_sample_history
        )
        
        result.metadata["pre_solution_alpha_ceiling_history"] = histories.get(
            "pre_solution_alpha_ceiling_history", []
        )

        result.metadata["guided_prior_valid"] = self.guided_sampler.valid_prior
        result.metadata["guided_prior_cost"] = self.guided_sampler.path_cost
        result.metadata["guided_prior_source"] = getattr(
            self.guided_sampler, "prior_source", "unknown"
        )
        result.metadata["num_guided_samples"] = self.sample_type_history.count("guided")
        result.metadata["num_base_samples"] = self.sample_type_history.count("base")
        result.metadata["best_path_corridor_radius"] = self.best_path_corridor_radius
        
        # added
        for key, value in self.prior_connectivity_features.items():
            result.metadata[f"prior_conn_{key}"] = value
        
        # added
        result.metadata["adjusted_reliability_history"] = histories.get(
            "adjusted_reliability_history", []
        )
        result.metadata["base_added_rate_history"] = histories.get(
            "base_added_rate_history", []
        )
        result.metadata["base_collision_rate_history"] = histories.get(
            "base_collision_rate_history", []
        )
        result.metadata["base_progress_rate_history"] = histories.get(
            "base_progress_rate_history", []
        )
        result.metadata["base_improvement_rate_history"] = histories.get(
            "base_improvement_rate_history", []
        )
        result.metadata["consistency_level_history"] = histories.get(
            "consistency_level_history", []
        )
        
        result.metadata["use_connectivity_alpha_ceiling"] = (
            self.use_connectivity_alpha_ceiling
        )
        result.metadata["connectivity_alpha_ceiling"] = (
            self.connectivity_alpha_ceiling
        )
        
        result.metadata["use_connectivity_alpha_init"] = (
            self.use_connectivity_alpha_init
        )
        result.metadata["connectivity_alpha_init"] = self.connectivity_alpha_init
        
        # added
        result.metadata["prior_failure_mode"] = self.prior_failure_mode
        result.metadata["prior_failure_params"] = str(self.prior_failure_params)

        if self.external_prior_result is not None:
            result.metadata["external_prior_type"] = getattr(
                self.external_prior_result, "prior_type", None
            )
            result.metadata["external_prior_mode"] = getattr(
                self.external_prior_result, "prior_mode", None
            )
        
        # added
        result.metadata["iteration_history"] = self.iteration_history
        
        result.metadata["base_sampler_mode"] = self.base_sampler_mode
        result.metadata["uses_informed_base_after_initial"] = int(
            self.base_sampler_mode == "informed_after_initial"
        )
        result.metadata["is_rag_uniform_base_ablation"] = int(
            self.base_sampler_mode == "uniform"
        )
        result.metadata["online_reliability_update_enabled"] = int(
            self.enable_online_reliability_update
        )
        result.metadata["uses_prior_path"] = 1
        result.metadata["uses_conninit"] = int(
            self.use_connectivity_alpha_init
        )
        result.metadata["uses_bpc"] = 1

        result.metadata["use_topk_edge_checking"] = self.use_topk_edge_checking
        result.metadata["parent_edge_strategy"] = self.parent_edge_strategy
        result.metadata["rewire_edge_strategy"] = self.rewire_edge_strategy
        result.metadata["parent_check_top_k"] = self.parent_check_top_k
        result.metadata["rewire_check_top_k"] = self.rewire_check_top_k
        result.metadata["use_topk_fallback"] = self.use_topk_fallback
        result.metadata["parent_fallback_check_k"] = self.parent_fallback_check_k
        result.metadata["rewire_fallback_check_k"] = self.rewire_fallback_check_k

        for key, value in self.get_topk_edge_stats().items():
            result.metadata[f"topk_{key}"] = value

        if self.adaptive_rewire_controller is not None:
            result.metadata.update(
                self.adaptive_rewire_controller.get_summary()
            )
        
        return result
