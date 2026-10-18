from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RewireBudgetDecision:
    """Budget selected for one rewiring event."""

    budget: int
    probe_count: int
    level: int
    opportunity_coverage: float
    tail_opportunity_fraction: float
    solution_aware: bool


class AdaptiveRewireBudgetController:
    """Select a rewiring validation budget from per-event opportunities.

    Stage 1 adapted a persistent budget from realized local rewiring gains.
    Experiments showed that those local gains were only weakly related to the
    final solution: successful tail rewires often improved irrelevant tree
    branches.  Stage 2 therefore makes a deterministic decision for every
    rewiring event.

    Candidates remain in the same potential-cost-reduction order used by the
    fixed-K baseline.  After an initial solution exists, their local potential
    improvement is discounted when the optimistic route through that node is
    not competitive with the current best solution.  The smallest configured
    budget whose prefix covers ``opportunity_coverage_target`` of the resulting
    opportunity mass is selected.  Before an initial solution exists, the
    fixed ``initial_budget`` is retained so that the feasibility-search phase
    is directly comparable with fixed K.

    A sparse deterministic tail probe is retained only for diagnosis.  The
    controller never consumes random numbers, so it does not perturb the
    planner's sampling stream.
    """

    def __init__(
        self,
        budgets: Iterable[int] = (4, 8, 12, 20),
        initial_budget: int = 12,
        opportunity_coverage_target: float = 0.90,
        solution_slack_ratio: float = 0.10,
        relevance_floor: float = 0.05,
        probe_interval: int = 16,
        probe_k: int = 1,
    ):
        normalized = sorted({max(1, int(value)) for value in budgets})
        if not normalized:
            raise ValueError("budgets must contain at least one positive integer")
        if int(initial_budget) not in normalized:
            raise ValueError("initial_budget must be one of budgets")
        if not 0.0 < float(opportunity_coverage_target) <= 1.0:
            raise ValueError("opportunity_coverage_target must be in (0, 1]")
        if float(solution_slack_ratio) <= 0.0:
            raise ValueError("solution_slack_ratio must be positive")
        if not 0.0 <= float(relevance_floor) <= 1.0:
            raise ValueError("relevance_floor must be in [0, 1]")
        if int(probe_interval) <= 0 or int(probe_k) <= 0:
            raise ValueError("probe_interval and probe_k must be positive")

        self.budgets = tuple(normalized)
        self.initial_budget = int(initial_budget)
        self.opportunity_coverage_target = float(opportunity_coverage_target)
        self.solution_slack_ratio = float(solution_slack_ratio)
        self.relevance_floor = float(relevance_floor)
        self.probe_interval = int(probe_interval)
        self.probe_k = int(probe_k)
        self._initial_level = self.budgets.index(self.initial_budget)
        self.reset()

    def reset(self) -> None:
        self.level = int(self._initial_level)
        self.event_count = 0
        self.pre_solution_event_count = 0
        self.solution_aware_event_count = 0
        self.opportunity_event_count = 0
        self.truncated_event_count = 0
        self.increase_count = 0
        self.decrease_count = 0
        self.budget_event_counts = {budget: 0 for budget in self.budgets}
        self.total_primary_gain = 0.0
        self.total_probe_gain = 0.0
        self.total_primary_checked = 0
        self.total_primary_successes = 0
        self.total_probe_checked = 0
        self.total_probe_successes = 0
        self.total_selected_opportunity_coverage = 0.0
        self.total_tail_opportunity_fraction = 0.0
        self.total_competitive_candidate_count = 0
        self.total_candidate_count_in_opportunity_events = 0
        self.total_opportunity_score = 0.0
        self.total_selected_opportunity_score = 0.0
        self.budget_change_history: list[dict] = []

    @property
    def current_budget(self) -> int:
        return int(self.budgets[self.level])

    def _solution_relevance_weights(
        self,
        candidate_lower_bounds: Sequence[float],
        best_cost: float,
    ) -> tuple[list[float], bool]:
        solution_aware = math.isfinite(float(best_cost)) and float(best_cost) > 0.0
        if not solution_aware:
            return [1.0 for _ in candidate_lower_bounds], False

        best_cost = float(best_cost)
        slack = max(1e-12, self.solution_slack_ratio * best_cost)
        competitive_limit = best_cost + slack
        weights = []

        for lower_bound in candidate_lower_bounds:
            raw = (competitive_limit - float(lower_bound)) / slack
            normalized = min(1.0, max(0.0, raw))
            weight = self.relevance_floor + (
                1.0 - self.relevance_floor
            ) * normalized
            weights.append(float(weight))

        return weights, True

    def _select_budget(self, opportunity_scores: Sequence[float]) -> tuple[int, float]:
        candidate_count = len(opportunity_scores)
        if candidate_count == 0:
            return int(self.budgets[0]), 1.0

        total_score = float(sum(max(0.0, float(v)) for v in opportunity_scores))
        if total_score <= 1e-12:
            return int(self.budgets[0]), 1.0

        selected_budget = int(self.budgets[-1])
        selected_score = float(
            sum(opportunity_scores[: min(selected_budget, candidate_count)])
        )

        for budget in self.budgets:
            prefix_count = min(int(budget), candidate_count)
            prefix_score = float(sum(opportunity_scores[:prefix_count]))
            coverage = prefix_score / total_score
            selected_budget = int(budget)
            selected_score = prefix_score
            if coverage + 1e-12 >= self.opportunity_coverage_target:
                break

        coverage = min(1.0, max(0.0, selected_score / total_score))
        return selected_budget, float(coverage)

    def begin_event(
        self,
        *,
        candidate_improvements: Sequence[float],
        candidate_lower_bounds: Sequence[float],
        best_cost: float,
    ) -> RewireBudgetDecision:
        if len(candidate_improvements) != len(candidate_lower_bounds):
            raise ValueError(
                "candidate_improvements and candidate_lower_bounds must match"
            )

        improvements = [max(0.0, float(v)) for v in candidate_improvements]
        relevance_weights, solution_aware = self._solution_relevance_weights(
            candidate_lower_bounds,
            best_cost,
        )
        opportunity_scores = [
            improvement * relevance
            for improvement, relevance in zip(improvements, relevance_weights)
        ]

        self.event_count += 1
        if solution_aware:
            self.solution_aware_event_count += 1
            selected_budget, coverage = self._select_budget(opportunity_scores)
        else:
            self.pre_solution_event_count += 1
            selected_budget = int(self.initial_budget)
            total_score = float(sum(opportunity_scores))
            selected_score = float(sum(opportunity_scores[:selected_budget]))
            coverage = (
                selected_score / total_score if total_score > 1e-12 else 1.0
            )

        previous_budget = self.current_budget
        self.level = self.budgets.index(selected_budget)
        self.budget_event_counts[selected_budget] += 1

        if selected_budget > previous_budget:
            self.increase_count += 1
        elif selected_budget < previous_budget:
            self.decrease_count += 1

        if selected_budget != previous_budget:
            self.budget_change_history.append(
                {
                    "event_count": int(self.event_count),
                    "old_budget": int(previous_budget),
                    "new_budget": int(selected_budget),
                    "solution_aware": bool(solution_aware),
                }
            )

        candidate_count = len(opportunity_scores)
        total_score = float(sum(opportunity_scores))
        selected_count = min(selected_budget, candidate_count)
        selected_score = float(sum(opportunity_scores[:selected_count]))
        tail_fraction = (
            max(0.0, 1.0 - selected_score / total_score)
            if total_score > 1e-12
            else 0.0
        )

        if candidate_count > 0:
            self.opportunity_event_count += 1
            self.total_selected_opportunity_coverage += float(coverage)
            self.total_tail_opportunity_fraction += float(tail_fraction)
            self.total_competitive_candidate_count += sum(
                weight > self.relevance_floor + 1e-12
                for weight in relevance_weights
            )
            self.total_candidate_count_in_opportunity_events += candidate_count
            self.total_opportunity_score += total_score
            self.total_selected_opportunity_score += selected_score

        probe_count = 0
        if candidate_count > selected_budget:
            self.truncated_event_count += 1
            if self.truncated_event_count % self.probe_interval == 0:
                probe_count = min(
                    self.probe_k,
                    candidate_count - selected_budget,
                )

        return RewireBudgetDecision(
            budget=int(selected_budget),
            probe_count=int(probe_count),
            level=int(self.level),
            opportunity_coverage=float(coverage),
            tail_opportunity_fraction=float(tail_fraction),
            solution_aware=bool(solution_aware),
        )

    def observe(
        self,
        *,
        primary_checked: int,
        primary_successes: int,
        primary_gain: float,
        probe_checked: int,
        probe_successes: int,
        probe_gain: float,
    ) -> None:
        """Record realized outcomes without changing a future budget."""
        self.total_primary_checked += max(0, int(primary_checked))
        self.total_primary_successes += max(0, int(primary_successes))
        self.total_probe_checked += max(0, int(probe_checked))
        self.total_probe_successes += max(0, int(probe_successes))
        self.total_primary_gain += max(0.0, float(primary_gain))
        self.total_probe_gain += max(0.0, float(probe_gain))

    @staticmethod
    def _safe_mean(total: float, count: int) -> float:
        return float(total / count) if count > 0 else 0.0

    def get_summary(self) -> dict:
        used_budgets = [
            budget
            for budget, count in self.budget_event_counts.items()
            if count > 0
        ]
        budget_sum = sum(
            budget * count for budget, count in self.budget_event_counts.items()
        )
        summary = {
            "adaptive_rewire_controller_version": "stage2_solution_opportunity",
            "adaptive_rewire_initial_budget": int(self.initial_budget),
            "adaptive_rewire_final_budget": int(self.current_budget),
            "adaptive_rewire_min_budget": int(
                min(used_budgets, default=self.current_budget)
            ),
            "adaptive_rewire_max_budget": int(
                max(used_budgets, default=self.current_budget)
            ),
            "adaptive_rewire_mean_budget": self._safe_mean(
                budget_sum,
                self.event_count,
            ),
            "adaptive_rewire_event_count": int(self.event_count),
            "adaptive_rewire_pre_solution_event_count": int(
                self.pre_solution_event_count
            ),
            "adaptive_rewire_solution_aware_event_count": int(
                self.solution_aware_event_count
            ),
            "adaptive_rewire_opportunity_event_count": int(
                self.opportunity_event_count
            ),
            "adaptive_rewire_truncated_event_count": int(
                self.truncated_event_count
            ),
            "adaptive_rewire_budget_increase_count": int(self.increase_count),
            "adaptive_rewire_budget_decrease_count": int(self.decrease_count),
            "adaptive_rewire_primary_realized_gain": float(
                self.total_primary_gain
            ),
            "adaptive_rewire_probe_realized_gain": float(self.total_probe_gain),
            "adaptive_rewire_selected_opportunity_coverage_mean": (
                self._safe_mean(
                    self.total_selected_opportunity_coverage,
                    self.opportunity_event_count,
                )
            ),
            "adaptive_rewire_tail_opportunity_fraction_mean": self._safe_mean(
                self.total_tail_opportunity_fraction,
                self.opportunity_event_count,
            ),
            "adaptive_rewire_competitive_candidate_count_mean": self._safe_mean(
                self.total_competitive_candidate_count,
                self.opportunity_event_count,
            ),
            "adaptive_rewire_candidate_count_mean": self._safe_mean(
                self.total_candidate_count_in_opportunity_events,
                self.opportunity_event_count,
            ),
            "adaptive_rewire_selected_opportunity_fraction_global": (
                self.total_selected_opportunity_score / self.total_opportunity_score
                if self.total_opportunity_score > 1e-12
                else 1.0
            ),
            "adaptive_rewire_opportunity_coverage_target": float(
                self.opportunity_coverage_target
            ),
            "adaptive_rewire_solution_slack_ratio": float(
                self.solution_slack_ratio
            ),
            "adaptive_rewire_relevance_floor": float(self.relevance_floor),
            "adaptive_rewire_probe_interval": int(self.probe_interval),
            "adaptive_rewire_probe_k": int(self.probe_k),
            "adaptive_rewire_budget_event_counts": dict(
                self.budget_event_counts
            ),
            "adaptive_rewire_budget_change_history": list(
                self.budget_change_history
            ),
        }

        for budget, count in self.budget_event_counts.items():
            summary[f"adaptive_rewire_budget_{budget}_event_count"] = int(count)

        return summary
