# controllers/reliability_controller.py
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class SamplerStats:
    added_rate: float = 0.0
    collision_rate: float = 0.0
    progress_rate: float = 0.0
    improvement_rate: float = 0.0
    score: float = 0.0


class ReliabilityController:
    """
    Relative reliability controller with best-path corridor consistency.

    Base idea:
        alpha <- alpha + eta * (guided_score - base_score - margin)

    Additional mechanism:
        after an initial solution exists, guided samples are checked against
        the current best-path corridor. If guided samples are inconsistent
        with the best path, alpha is penalized and capped.

    High-consistency recovery:
        when guided samples are highly consistent with the current best path,
        alpha is allowed to recover gradually toward the active alpha ceiling.
    """

    def __init__(
        self,
        alpha_init: float = 0.5,
        alpha_min: float = 0.1,
        alpha_max: float = 0.8,
        window_size: int = 50,
        eta: float = 0.05,
        margin: float = 0.05,
        min_samples_to_update: int = 10,
        w_add: float = 0.2,
        w_progress: float = 1.0,
        w_collision: float = 1.5,
        w_improve: float = 2.0,
        # Best-path consistency parameters.
        consistency_window_size: int | None = None,
        min_consistency_samples: int = 8,
        consistency_low_threshold: float = 0.35,
        consistency_mid_threshold: float = 0.55,
        consistency_high_threshold: float = 0.70,
        consistency_penalty: float = 0.20,
        consistency_bonus: float = 0.05,
        low_consistency_alpha_ceiling: float = 0.30,
        mid_consistency_alpha_ceiling: float = 0.50,
        # High-consistency recovery parameters.
        high_consistency_recovery: bool = True,
        high_consistency_recovery_eta: float = 0.03,
        # Optional pre-solution alpha ceiling, used by connectivity reliability.
        pre_solution_alpha_ceiling: float | None = None,
    ):
        if not 0.0 <= alpha_min <= alpha_init <= alpha_max <= 1.0:
            raise ValueError("Require 0 <= alpha_min <= alpha_init <= alpha_max <= 1.")

        self.alpha = float(alpha_init)
        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        self.window_size = int(window_size)
        self.guided_window = deque(maxlen=self.window_size)
        self.base_window = deque(maxlen=self.window_size)

        self.eta = float(eta)
        self.margin = float(margin)
        self.min_samples_to_update = int(min_samples_to_update)

        self.w_add = float(w_add)
        self.w_progress = float(w_progress)
        self.w_collision = float(w_collision)
        self.w_improve = float(w_improve)

        if consistency_window_size is None:
            consistency_window_size = window_size
        self.consistency_window_size = int(consistency_window_size)
        self.consistency_window = deque(maxlen=self.consistency_window_size)
        self.distance_to_best_path_window = deque(maxlen=self.consistency_window_size)

        self.min_consistency_samples = int(min_consistency_samples)
        self.consistency_low_threshold = float(consistency_low_threshold)
        self.consistency_mid_threshold = float(consistency_mid_threshold)
        self.consistency_high_threshold = float(consistency_high_threshold)
        self.consistency_penalty = float(consistency_penalty)
        self.consistency_bonus = float(consistency_bonus)
        self.low_consistency_alpha_ceiling = float(low_consistency_alpha_ceiling)
        self.mid_consistency_alpha_ceiling = float(mid_consistency_alpha_ceiling)

        self.high_consistency_recovery = bool(high_consistency_recovery)
        self.high_consistency_recovery_eta = float(high_consistency_recovery_eta)

        # Whether BPC consistency is allowed to change alpha. Relative-only
        # configurations pass zero penalty/bonus, alpha_max ceilings, and
        # disable recovery. BPC is then retained for diagnostics only.
        self.consistency_adjustment_enabled = bool(
            abs(self.consistency_penalty) > 1e-12
            or abs(self.consistency_bonus) > 1e-12
            or self.low_consistency_alpha_ceiling < self.alpha_max - 1e-12
            or self.mid_consistency_alpha_ceiling < self.alpha_max - 1e-12
            or self.high_consistency_recovery
        )

        # added
        self.pre_solution_alpha_ceiling = (
            None
            if pre_solution_alpha_ceiling is None
            else float(
                np.clip(
                    pre_solution_alpha_ceiling,
                    self.alpha_min,
                    self.alpha_max,
                )
            )
        )
        
        if self.pre_solution_alpha_ceiling is not None:
            self.alpha = float(min(self.alpha, self.pre_solution_alpha_ceiling))

        self.reset_histories()

    def reset_histories(self) -> None:
        self.alpha_history: list[float] = [self.alpha]

        self.reliability_history: list[float] = []
        self.guided_score_history: list[float] = []
        self.base_score_history: list[float] = []

        # self.added_rate_history: list[float] = []
        # self.collision_rate_history: list[float] = []
        # self.progress_rate_history: list[float] = []
        # self.improvement_rate_history: list[float] = []
        # changed and added
        self.guided_added_rate_history: list[float] = []
        self.guided_collision_rate_history: list[float] = []
        self.guided_progress_rate_history: list[float] = []
        self.guided_improvement_rate_history: list[float] = []

        self.base_added_rate_history: list[float] = []
        self.base_collision_rate_history: list[float] = []
        self.base_progress_rate_history: list[float] = []
        self.base_improvement_rate_history: list[float] = []

        self.adjusted_reliability_history: list[float] = []
        self.consistency_level_history: list[str] = []
        # changed to there
        
        self.best_path_consistency_rate_history: list[float] = []
        self.mean_distance_to_best_path_history: list[float] = []
        self.alpha_ceiling_history: list[float] = []
        
        # added
        self.pre_solution_alpha_ceiling_history: list[float] = []

    def reset(self) -> None:
        self.alpha = float(self.alpha_init)
        
        if self.pre_solution_alpha_ceiling is not None:
            self.alpha = float(min(self.alpha, self.pre_solution_alpha_ceiling))

        self.guided_window.clear()
        self.base_window.clear()
        self.consistency_window.clear()
        self.distance_to_best_path_window.clear()

        self.reset_histories()

    def record(
        self,
        sample_type: str,
        added: bool,
        collision: bool,
        improved: bool,
        progress: bool = False,
        has_solution: bool = False,
        distance_to_best_path: float | None = None,
        best_path_consistent: bool | None = None,
    ) -> None:
        item = {
            "added": int(added),
            "collision": int(collision),
            "progress": int(progress),
            "improved": int(improved),
        }

        if sample_type == "guided":
            self.guided_window.append(item)

            # Only guided samples after an initial solution can provide
            # best-path consistency evidence.
            if has_solution and best_path_consistent is not None:
                self.consistency_window.append(
                    1.0 if best_path_consistent else 0.0
                )

                if distance_to_best_path is not None:
                    self.distance_to_best_path_window.append(
                        float(distance_to_best_path)
                    )

        elif sample_type == "base":
            self.base_window.append(item)
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

    def compute_stats(self, window) -> SamplerStats:
        if len(window) == 0:
            return SamplerStats()

        added_rate = float(np.mean([item["added"] for item in window]))
        collision_rate = float(np.mean([item["collision"] for item in window]))
        progress_rate = float(np.mean([item["progress"] for item in window]))
        improvement_rate = float(np.mean([item["improved"] for item in window]))

        score = (
            self.w_add * added_rate
            + self.w_progress * progress_rate
            - self.w_collision * collision_rate
            + self.w_improve * improvement_rate
        )

        return SamplerStats(
            added_rate=added_rate,
            collision_rate=collision_rate,
            progress_rate=progress_rate,
            improvement_rate=improvement_rate,
            score=float(score),
        )

    def get_consistency_rate(self) -> float | None:
        if len(self.consistency_window) == 0:
            return None
        return float(np.mean(self.consistency_window))

    def get_mean_distance_to_best_path(self) -> float | None:
        if len(self.distance_to_best_path_window) == 0:
            return None
        return float(np.mean(self.distance_to_best_path_window))

    def _get_alpha_ceiling_from_consistency(
        self,
        consistency_rate: float | None,
    ) -> float:
        if not self.consistency_adjustment_enabled:
            return self.alpha_max

        if (
            consistency_rate is None
            or len(self.consistency_window) < self.min_consistency_samples
        ):
            return self.alpha_max

        if consistency_rate < self.consistency_low_threshold:
            return min(self.alpha_max, self.low_consistency_alpha_ceiling)

        if consistency_rate < self.consistency_mid_threshold:
            return min(self.alpha_max, self.mid_consistency_alpha_ceiling)

        return self.alpha_max
    
    def _get_active_alpha_ceiling(
        self,
        consistency_rate: float | None,
        allow_increase: bool,
    ) -> float:
        """
        Return the active alpha ceiling.

        Before an initial solution is found, an optional connectivity-based
        ceiling can restrict alpha. After an initial solution is found, the
        best-path-consistency ceiling is used.
        """
        bpc_ceiling = self._get_alpha_ceiling_from_consistency(consistency_rate)

        if not allow_increase and self.pre_solution_alpha_ceiling is not None:
            return float(min(bpc_ceiling, self.pre_solution_alpha_ceiling))

        return float(bpc_ceiling)
    

    def update(self, allow_increase: bool = True) -> float:
        old_alpha = self.alpha

        guided_stats = self.compute_stats(self.guided_window)
        base_stats = self.compute_stats(self.base_window)
        consistency_rate = self.get_consistency_rate()

        # alpha_ceiling = self._get_alpha_ceiling_from_consistency(consistency_rate)
        # changed to there
        alpha_ceiling = self._get_active_alpha_ceiling(
            consistency_rate=consistency_rate,
            allow_increase=allow_increase,
        )

        if (
            len(self.guided_window) >= self.min_samples_to_update
            and len(self.base_window) >= self.min_samples_to_update
        ):
            relative_reliability = guided_stats.score - base_stats.score

            adjusted_reliability = relative_reliability

            if (
                self.consistency_adjustment_enabled
                and consistency_rate is not None
            ):
                if consistency_rate < self.consistency_low_threshold:
                    adjusted_reliability -= self.consistency_penalty
                elif consistency_rate >= self.consistency_high_threshold:
                    adjusted_reliability += self.consistency_bonus

            new_alpha = float(
                np.clip(
                    self.alpha + self.eta * (adjusted_reliability - self.margin),
                    self.alpha_min,
                    alpha_ceiling,
                )
            )
        else:
            relative_reliability = 0.0
            adjusted_reliability = 0.0
            new_alpha = self.alpha

        # Before the first solution, the planner passes allow_increase=False.
        # This prevents premature trust in the guided prior.
        if not allow_increase:
            new_alpha = min(new_alpha, old_alpha)

        self.alpha = float(
            np.clip(
                new_alpha,
                self.alpha_min,
                alpha_ceiling,
            )
        )

        # High-consistency recovery:
        # If the guided samples are highly consistent with the current best
        # path, alpha recovers toward the active ceiling.
        if (
            self.high_consistency_recovery
            and allow_increase
            and consistency_rate is not None
            and len(self.consistency_window) >= self.min_consistency_samples
            and consistency_rate >= self.consistency_high_threshold
        ):
            self.alpha = float(
                np.clip(
                    self.alpha
                    + self.high_consistency_recovery_eta
                    * (alpha_ceiling - self.alpha),
                    self.alpha_min,
                    alpha_ceiling,
                )
            )

        self.alpha_history.append(self.alpha)
        self.reliability_history.append(relative_reliability)
        self.guided_score_history.append(guided_stats.score)
        self.base_score_history.append(base_stats.score)

        
        # self.added_rate_history.append(guided_stats.added_rate)
        # self.collision_rate_history.append(guided_stats.collision_rate)
        # self.progress_rate_history.append(guided_stats.progress_rate)
        # self.improvement_rate_history.append(guided_stats.improvement_rate)
        # changed and added
        self.guided_added_rate_history.append(guided_stats.added_rate)
        self.guided_collision_rate_history.append(guided_stats.collision_rate)
        self.guided_progress_rate_history.append(guided_stats.progress_rate)
        self.guided_improvement_rate_history.append(guided_stats.improvement_rate)

        self.base_added_rate_history.append(base_stats.added_rate)
        self.base_collision_rate_history.append(base_stats.collision_rate)
        self.base_progress_rate_history.append(base_stats.progress_rate)
        self.base_improvement_rate_history.append(base_stats.improvement_rate)

        self.adjusted_reliability_history.append(adjusted_reliability)
        self.consistency_level_history.append(
            self.get_consistency_level(consistency_rate)
        )
        # to there

        self.best_path_consistency_rate_history.append(
            np.nan if consistency_rate is None else consistency_rate
        )

        mean_distance = self.get_mean_distance_to_best_path()
        self.mean_distance_to_best_path_history.append(
            np.nan if mean_distance is None else mean_distance
        )

        self.alpha_ceiling_history.append(alpha_ceiling)
        
        self.pre_solution_alpha_ceiling_history.append(
            np.nan if self.pre_solution_alpha_ceiling is None
            else float(self.pre_solution_alpha_ceiling)
        )

        return self.alpha

    def get_alpha(self) -> float:
        return self.alpha

    def record_snapshot(self, allow_increase: bool = True) -> dict:
        """Record controller diagnostics without changing ``alpha``.

        This is used by ConnInit-only and fixed-trust ablations.  Those
        configurations must keep the guided sampling probability frozen while
        still exposing guided/base performance and BPC diagnostic signals for
        analysis.  Calling :meth:`update` with ``eta=0`` is not equivalent,
        because other controller mechanisms (ceilings, recovery, or planner
        stagnation penalties) could still change ``alpha``.
        """
        state = self.get_current_state(allow_increase=allow_increase)

        relative_reliability = state["relative_reliability"]
        adjusted_reliability = state["adjusted_reliability"]

        self.alpha_history.append(float(self.alpha))
        self.reliability_history.append(
            0.0
            if not np.isfinite(relative_reliability)
            else float(relative_reliability)
        )
        self.adjusted_reliability_history.append(
            0.0
            if not np.isfinite(adjusted_reliability)
            else float(adjusted_reliability)
        )
        self.guided_score_history.append(float(state["guided_score"]))
        self.base_score_history.append(float(state["base_score"]))

        self.guided_added_rate_history.append(
            float(state["guided_added_rate"])
        )
        self.guided_collision_rate_history.append(
            float(state["guided_collision_rate"])
        )
        self.guided_progress_rate_history.append(
            float(state["guided_progress_rate"])
        )
        self.guided_improvement_rate_history.append(
            float(state["guided_improvement_rate"])
        )

        self.base_added_rate_history.append(float(state["base_added_rate"]))
        self.base_collision_rate_history.append(
            float(state["base_collision_rate"])
        )
        self.base_progress_rate_history.append(
            float(state["base_progress_rate"])
        )
        self.base_improvement_rate_history.append(
            float(state["base_improvement_rate"])
        )

        consistency_rate = state["best_path_consistency"]
        mean_distance = state["mean_distance_to_best_path"]
        self.best_path_consistency_rate_history.append(
            np.nan
            if not np.isfinite(consistency_rate)
            else float(consistency_rate)
        )
        self.mean_distance_to_best_path_history.append(
            np.nan
            if not np.isfinite(mean_distance)
            else float(mean_distance)
        )
        self.alpha_ceiling_history.append(float(state["alpha_ceiling"]))
        self.consistency_level_history.append(str(state["consistency_level"]))
        self.pre_solution_alpha_ceiling_history.append(
            np.nan
            if self.pre_solution_alpha_ceiling is None
            else float(self.pre_solution_alpha_ceiling)
        )

        return state
    
    # added
    def get_current_state(self, allow_increase: bool = True) -> dict:
        """
        Return the current controller state for per-iteration logging.

        This function does not update alpha. It only exposes the current
        reliability-related state so that the planner can build an aligned
        iteration-level trajectory log.
        """
        guided_stats = self.compute_stats(self.guided_window)
        base_stats = self.compute_stats(self.base_window)

        consistency_rate = self.get_consistency_rate()
        mean_distance = self.get_mean_distance_to_best_path()

        alpha_ceiling = self._get_active_alpha_ceiling(
            consistency_rate=consistency_rate,
            allow_increase=allow_increase,
        )

        if (
            len(self.guided_window) >= self.min_samples_to_update
            and len(self.base_window) >= self.min_samples_to_update
        ):
            relative_reliability = guided_stats.score - base_stats.score

            adjusted_reliability = relative_reliability
            if (
                self.consistency_adjustment_enabled
                and consistency_rate is not None
            ):
                if consistency_rate < self.consistency_low_threshold:
                    adjusted_reliability -= self.consistency_penalty
                elif consistency_rate >= self.consistency_high_threshold:
                    adjusted_reliability += self.consistency_bonus
        else:
            relative_reliability = np.nan
            adjusted_reliability = np.nan

        return {
            "alpha": float(self.alpha),
            "alpha_ceiling": float(alpha_ceiling),
            "consistency_adjustment_enabled": int(
                self.consistency_adjustment_enabled
            ),
            "controller_mode": (
                "bpc_adjusted"
                if self.consistency_adjustment_enabled
                else "relative_only"
            ),

            "relative_reliability": float(relative_reliability)
            if np.isfinite(relative_reliability)
            else np.nan,

            "adjusted_reliability": float(adjusted_reliability)
            if np.isfinite(adjusted_reliability)
            else np.nan,

            "guided_score": float(guided_stats.score),
            "base_score": float(base_stats.score),

            "guided_added_rate": float(guided_stats.added_rate),
            "guided_collision_rate": float(guided_stats.collision_rate),
            "guided_progress_rate": float(guided_stats.progress_rate),
            "guided_improvement_rate": float(guided_stats.improvement_rate),

            "base_added_rate": float(base_stats.added_rate),
            "base_collision_rate": float(base_stats.collision_rate),
            "base_progress_rate": float(base_stats.progress_rate),
            "base_improvement_rate": float(base_stats.improvement_rate),

            "best_path_consistency": (
                np.nan if consistency_rate is None else float(consistency_rate)
            ),
            "mean_distance_to_best_path": (
                np.nan if mean_distance is None else float(mean_distance)
            ),
            "consistency_level": self.get_consistency_level(consistency_rate),

            "guided_window_size": int(len(self.guided_window)),
            "base_window_size": int(len(self.base_window)),
            "consistency_window_size": int(len(self.consistency_window)),
        }
    

    def apply_stagnation_penalty(self, decay: float = 0.03) -> float:
        """
        Decrease alpha when the planner stagnates before finding an initial
        solution. This is intentionally only a downward adjustment.
        """
        self.alpha = float(
            np.clip(
                self.alpha - decay,
                self.alpha_min,
                self.alpha_max,
            )
        )
        # added
        if len(self.alpha_history) == 0 or self.alpha_history[-1] != self.alpha:
            self.alpha_history.append(self.alpha)
        
        return self.alpha

    def get_histories(self) -> dict:
        return {
            "controller_mode": (
                "bpc_adjusted"
                if self.consistency_adjustment_enabled
                else "relative_only"
            ),
            "consistency_adjustment_enabled": int(
                self.consistency_adjustment_enabled
            ),
            "alpha_history": self.alpha_history,
            "reliability_history": self.reliability_history,
            # added
            "adjusted_reliability_history": self.adjusted_reliability_history,
            "guided_score_history": self.guided_score_history,
            "base_score_history": self.base_score_history,
            
            "guided_added_rate_history": self.guided_added_rate_history,
            "guided_collision_rate_history": self.guided_collision_rate_history,
            "guided_progress_rate_history": self.guided_progress_rate_history,
            "guided_improvement_rate_history": self.guided_improvement_rate_history,
            # added to there
            "base_added_rate_history": self.base_added_rate_history,
            "base_collision_rate_history": self.base_collision_rate_history,
            "base_progress_rate_history": self.base_progress_rate_history,
            "base_improvement_rate_history": self.base_improvement_rate_history,
            
            "best_path_consistency_rate_history": (
                self.best_path_consistency_rate_history
            ),
            "mean_distance_to_best_path_history": (
                self.mean_distance_to_best_path_history
            ),
            "alpha_ceiling_history": self.alpha_ceiling_history,
            # added
            "consistency_level_history": self.consistency_level_history,
            # added
            "pre_solution_alpha_ceiling_history": (
                self.pre_solution_alpha_ceiling_history
            ),
        }
        
    # add
    def get_consistency_level(self, consistency_rate: float | None) -> str:
        if (
            consistency_rate is None
            or len(self.consistency_window) < self.min_consistency_samples
        ):
            return "unknown"

        if consistency_rate < self.consistency_low_threshold:
            return "low"

        if consistency_rate < self.consistency_mid_threshold:
            return "mid"

        if consistency_rate >= self.consistency_high_threshold:
            return "high"

        return "normal"
