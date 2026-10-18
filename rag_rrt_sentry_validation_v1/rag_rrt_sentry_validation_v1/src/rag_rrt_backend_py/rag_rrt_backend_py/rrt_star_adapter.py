#!/usr/bin/env python3
"""Adapters from ROS OccupancyGrid/world coordinates to existing planners.

The adapter keeps all search logic inside the separately validated Python
RAG-RRT project.  It only performs:

* OccupancyGrid -> legacy ``GridMap2D`` conversion;
* world <-> continuous planner-grid coordinate conversion;
* deterministic planner construction and invocation;
* external A* prior generation for the final RAG-RRT*-Rel variant;
* final path, tree, and prior conversion back to map/world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional, Tuple

import numpy as np

from rag_rrt_backend_py.occupancy_grid_map import OccupancyGridMap
from rag_rrt_backend_py.final_path_validation import validate_polyline
from rag_rrt_backend_py.execution_grid_map import execution_map_class


RAG_REL_PLANNER = "rag_bpc_conn_init_rel_only"
RAG_REL_TOPK_FALLBACK_PLANNER = (
    "rag_bpc_conn_init_rel_only_topk_fallback"
)
RAG_PLANNERS = (RAG_REL_PLANNER, RAG_REL_TOPK_FALLBACK_PLANNER)
SUPPORTED_PLANNERS = (
    "rrt_star",
    "irrt_star",
    RAG_REL_PLANNER,
    RAG_REL_TOPK_FALLBACK_PLANNER,
)

PLANNER_DISPLAY_NAMES = {
    "rrt_star": "RRT*",
    "irrt_star": "Informed RRT*",
    RAG_REL_PLANNER: "RAG-RRT*-Rel",
    RAG_REL_TOPK_FALLBACK_PLANNER: "RAG-RRT*-Rel-TopK-Fallback",
}


@dataclass(frozen=True)
class RRTStarAdapterConfig:
    """Configuration expressed in the legacy planners' grid-cell units."""

    project_root: str = ""
    step_len_cells: float = 5.0
    search_radius_cells: float = 15.0
    iter_max: int = 3000
    iter_after_initial: int = 300
    clearance_cells: int = 2
    informed_max_sample_attempts: int = 100

    # Final RAG-RRT*-Rel configuration, frozen from the 2-D experiments.
    rag_prior_mode: str = "coarse_valid"
    rag_prior_heuristic_type: str = "euclidean"
    rag_guide_sigma_cells: float = 5.0
    rag_guide_max_attempts: int = 50
    rag_alpha_init: float = 0.5
    rag_alpha_min: float = 0.1
    rag_alpha_max: float = 0.8
    rag_window_size: int = 30
    rag_update_interval: int = 20
    rag_eta: float = 0.08
    rag_margin: float = 0.05
    rag_min_samples_to_update: int = 8
    rag_w_add: float = 0.2
    rag_w_progress: float = 1.0
    rag_w_collision: float = 1.5
    rag_w_improve: float = 2.0
    rag_stagnation_check_interval: int = 100
    rag_stagnation_start_iter: int = 100
    rag_stagnation_alpha_decay: float = 0.05
    rag_best_path_corridor_radius_cells: float = 10.0
    rag_min_consistency_samples: int = 8
    rag_consistency_low_threshold: float = 0.35
    rag_consistency_mid_threshold: float = 0.55
    rag_consistency_high_threshold: float = 0.70
    rag_connectivity_alpha_init_min: float = 0.30
    rag_connectivity_alpha_init_max: float = 0.50

    # Frozen TopK-Fallback efficiency configuration from the 2-D experiments.
    rag_parent_check_top_k: int = 8
    rag_rewire_check_top_k: int = 12
    rag_parent_fallback_check_k: int = 4
    rag_rewire_fallback_check_k: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.step_len_cells) or self.step_len_cells <= 0.0:
            raise ValueError("step_len_cells must be finite and positive")
        if (
            not math.isfinite(self.search_radius_cells)
            or self.search_radius_cells <= 0.0
        ):
            raise ValueError("search_radius_cells must be finite and positive")
        if self.iter_max <= 0:
            raise ValueError("iter_max must be positive")
        if self.iter_after_initial < 0:
            raise ValueError("iter_after_initial must be non-negative")
        if self.clearance_cells < 0:
            raise ValueError("clearance_cells must be non-negative")
        if self.informed_max_sample_attempts <= 0:
            raise ValueError("informed_max_sample_attempts must be positive")

        if self.rag_prior_mode != "coarse_valid":
            raise ValueError(
                "This ROS integration stage supports only "
                "rag_prior_mode='coarse_valid'"
            )
        if not math.isfinite(self.rag_guide_sigma_cells) or self.rag_guide_sigma_cells <= 0:
            raise ValueError("rag_guide_sigma_cells must be finite and positive")
        if self.rag_guide_max_attempts <= 0:
            raise ValueError("rag_guide_max_attempts must be positive")
        if not 0.0 <= self.rag_alpha_min <= self.rag_alpha_init <= self.rag_alpha_max <= 1.0:
            raise ValueError(
                "Require 0 <= rag_alpha_min <= rag_alpha_init <= "
                "rag_alpha_max <= 1"
            )
        if self.rag_window_size <= 0 or self.rag_update_interval <= 0:
            raise ValueError("RAG window and update intervals must be positive")
        if self.rag_min_samples_to_update <= 0:
            raise ValueError("rag_min_samples_to_update must be positive")
        if self.rag_stagnation_check_interval <= 0:
            raise ValueError("rag_stagnation_check_interval must be positive")
        if self.rag_stagnation_start_iter < 0:
            raise ValueError("rag_stagnation_start_iter must be non-negative")
        if (
            not math.isfinite(self.rag_best_path_corridor_radius_cells)
            or self.rag_best_path_corridor_radius_cells <= 0.0
        ):
            raise ValueError(
                "rag_best_path_corridor_radius_cells must be finite and positive"
            )
        if self.rag_min_consistency_samples <= 0:
            raise ValueError("rag_min_consistency_samples must be positive")
        if not (
            0.0 <= self.rag_consistency_low_threshold
            <= self.rag_consistency_mid_threshold
            <= self.rag_consistency_high_threshold
            <= 1.0
        ):
            raise ValueError("RAG consistency thresholds must be ordered in [0, 1]")
        if not (
            self.rag_alpha_min
            <= self.rag_connectivity_alpha_init_min
            <= self.rag_connectivity_alpha_init_max
            <= self.rag_alpha_max
        ):
            raise ValueError(
                "Connectivity alpha-init bounds must lie inside RAG alpha bounds"
            )
        for name, value in (
            ("rag_parent_check_top_k", self.rag_parent_check_top_k),
            ("rag_rewire_check_top_k", self.rag_rewire_check_top_k),
            ("rag_parent_fallback_check_k", self.rag_parent_fallback_check_k),
            ("rag_rewire_fallback_check_k", self.rag_rewire_fallback_check_k),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class RRTStarAdapterResult:
    """Planner output after conversion back to map/world coordinates."""

    success: bool
    world_path: Optional[np.ndarray]
    planner_result: Any
    prior_generation_time: float
    planner_solve_time: float
    path_conversion_time: float
    message: str
    planner_name: str
    tree_world_vertices: Optional[np.ndarray] = None
    tree_parents: Optional[np.ndarray] = None
    prior_world_path: Optional[np.ndarray] = None


@dataclass
class _SimplePriorResult:
    """Fallback PriorResult-compatible object for external A* generation."""

    prior_type: str = "astar"
    prior_mode: str = "coarse_valid"
    path: list[tuple[float, float]] = field(default_factory=list)
    heatmap: Any = None
    start: Optional[tuple[float, float]] = None
    goal: Optional[tuple[float, float]] = None
    success: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def path_array(self) -> np.ndarray:
        if not self.path:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.path, dtype=float).reshape(-1, 2)


class RRTStarAdapter:
    """Lazy-loading bridge to the existing sampling-planner implementations."""

    def __init__(self, config: RRTStarAdapterConfig) -> None:
        self.config = config
        self._project_root: Optional[Path] = None
        self._grid_map_class = None
        self._planner_classes: dict[str, type] = {}
        self._astar_prior_class = None
        self._astar_class = None
        self._compute_path_cost = None

    @property
    def configured_project_root(self) -> str:
        return self.config.project_root.strip() or os.environ.get(
            "RAG_RRT_PROJECT_ROOT",
            "",
        ).strip()

    def _resolve_project_root(self) -> Path:
        root_text = self.configured_project_root
        if not root_text:
            raise RuntimeError(
                "RAG-RRT project root is not configured. Set the ROS parameter "
                "algorithm_project_root or export RAG_RRT_PROJECT_ROOT."
            )

        root = Path(root_text).expanduser().resolve()
        required = (
            root / "envs" / "grid_map_2d.py",
            root / "planners" / "rrt_base_2d.py",
            root / "planners" / "rrt_star_2d.py",
            root / "planners" / "informed_rrt_star_2d.py",
            root / "planners" / "reliability_guided_rrt_star_2d.py",
            root / "controllers" / "reliability_controller.py",
            root / "samplers" / "astar_corridor_sampler.py",
            root / "utils" / "astar.py",
            root / "utils" / "guidance_connectivity.py",
            root / "utils" / "metrics.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Configured RAG-RRT project root is missing required files: "
                + ", ".join(missing)
            )
        return root

    def ensure_loaded(self) -> Path:
        """Import planner and prior classes only when a request arrives."""

        root = self._resolve_project_root()
        if (
            self._project_root == root
            and self._grid_map_class is not None
            and all(name in self._planner_classes for name in SUPPORTED_PLANNERS)
            and self._astar_class is not None
        ):
            return root

        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            grid_module = importlib.import_module("envs.grid_map_2d")
            rrt_module = importlib.import_module("planners.rrt_star_2d")
            irrt_module = importlib.import_module("planners.informed_rrt_star_2d")
            rag_module = importlib.import_module(
                "planners.reliability_guided_rrt_star_2d"
            )
            astar_module = importlib.import_module("utils.astar")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import the existing RAG-RRT project from {root}: {exc}"
            ) from exc

        # Prefer the project's official external-prior wrapper.  If importing
        # it pulls in optional learned-prior dependencies that are unavailable,
        # fall back to a local PriorResult-compatible wrapper around AStar2D.
        astar_prior_class = None
        try:
            astar_prior_module = importlib.import_module("priors.astar_prior")
            astar_prior_class = getattr(
                astar_prior_module,
                "AStarPriorGenerator",
                None,
            )
        except Exception:
            astar_prior_class = None

        grid_map_class = getattr(grid_module, "GridMap2D", None)
        rrt_star_class = getattr(rrt_module, "RRTStar2D", None)
        irrt_star_class = getattr(irrt_module, "InformedRRTStar2D", None)
        rag_rel_class = getattr(rag_module, "ReliabilityGuidedRRTStar2D", None)
        astar_class = getattr(astar_module, "AStar2D", None)
        compute_path_cost = getattr(astar_module, "compute_path_cost", None)

        if any(
            value is None
            for value in (
                grid_map_class,
                rrt_star_class,
                irrt_star_class,
                rag_rel_class,
                astar_class,
                compute_path_cost,
            )
        ):
            raise RuntimeError(
                "Project imports succeeded, but one or more required planner, "
                "map, or A* symbols were not found."
            )

        self._grid_map_class = execution_map_class(grid_map_class)
        self._planner_classes = {
            "rrt_star": rrt_star_class,
            "irrt_star": irrt_star_class,
            RAG_REL_PLANNER: rag_rel_class,
            RAG_REL_TOPK_FALLBACK_PLANNER: rag_rel_class,
        }
        self._astar_prior_class = astar_prior_class
        self._astar_class = astar_class
        self._compute_path_cost = compute_path_cost
        self._project_root = root
        return root

    def _generate_external_astar_prior(
        self,
        legacy_map,
        start_array: np.ndarray,
        goal_array: np.ndarray,
        seed: int,
    ) -> tuple[Any, float]:
        """Generate a coarse-valid external A* prior and measure its time."""

        assert self._astar_class is not None
        assert self._compute_path_cost is not None

        prior_start = time.perf_counter()
        if self._astar_prior_class is not None:
            generator = self._astar_prior_class(
                config={
                    "heuristic_type": self.config.rag_prior_heuristic_type,
                }
            )
            prior_result = generator.generate(
                env=legacy_map,
                start=start_array,
                goal=goal_array,
                prior_mode=self.config.rag_prior_mode,
                seed=int(seed),
            )
        else:
            astar = self._astar_class(
                binary_mask=legacy_map.binary_mask,
                clearance=int(legacy_map.clearance),
                heuristic_type=self.config.rag_prior_heuristic_type,
            )
            path, visited = astar.search(start_array, goal_array)
            success = path is not None and len(path) >= 2
            prior_path = (
                []
                if not success
                else [(float(p[0]), float(p[1])) for p in path]
            )
            prior_result = _SimplePriorResult(
                path=prior_path,
                start=(float(start_array[0]), float(start_array[1])),
                goal=(float(goal_array[0]), float(goal_array[1])),
                success=success,
                message="" if success else "A* failed to find a prior path.",
                metadata={
                    "generator": "RRTStarAdapterFallbackAStar",
                    "heuristic_type": self.config.rag_prior_heuristic_type,
                    "clearance": int(legacy_map.clearance),
                    "num_waypoints": len(prior_path),
                    "num_visited": 0 if visited is None else len(visited),
                    "path_cost": (
                        float(self._compute_path_cost(prior_path))
                        if success
                        else math.inf
                    ),
                },
            )

        prior_elapsed = time.perf_counter() - prior_start
        if not bool(getattr(prior_result, "success", False)):
            message = str(
                getattr(prior_result, "message", "A* prior generation failed")
            )
            raise RuntimeError(message or "A* prior generation failed")

        prior_path = self._prior_path_array(prior_result)
        if prior_path is None or len(prior_path) < 2:
            raise RuntimeError("A* prior generator returned fewer than two waypoints")

        return prior_result, prior_elapsed

    @staticmethod
    def _prior_path_array(prior_result) -> Optional[np.ndarray]:
        if prior_result is None:
            return None
        if hasattr(prior_result, "path_array"):
            path = prior_result.path_array()
        else:
            path = getattr(prior_result, "path", None)
        if path is None:
            return None
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
            return None
        if not np.all(np.isfinite(path)):
            raise ValueError("Prior path contains non-finite coordinates")
        return path

    def plan(
        self,
        grid_map: OccupancyGridMap,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        seed: int,
        *,
        planner_name: str = "rrt_star",
        collect_debug_tree: bool = False,
    ) -> RRTStarAdapterResult:
        """Run the selected planner in continuous planner-grid coordinates."""

        normalized_name = planner_name.strip().lower()
        if normalized_name not in SUPPORTED_PLANNERS:
            raise ValueError(
                f"Unsupported adapter planner '{planner_name}'. "
                f"Supported values: {', '.join(SUPPORTED_PLANNERS)}"
            )

        self.ensure_loaded()
        assert self._grid_map_class is not None
        planner_class = self._planner_classes[normalized_name]

        geometry = grid_map.geometry
        start_grid = geometry.world_to_planner(*start_world)
        goal_grid = geometry.world_to_planner(*goal_world)
        if start_grid is None:
            raise ValueError(
                f"Start world point {start_world} is outside the OccupancyGrid"
            )
        if goal_grid is None:
            raise ValueError(
                f"Goal world point {goal_world} is outside the OccupancyGrid"
            )

        binary_mask = grid_map.to_binary_mask()
        legacy_map = self._grid_map_class(
            binary_mask,
            clearance=self.config.clearance_cells,
        )

        start_array = np.asarray(start_grid, dtype=float)
        goal_array = np.asarray(goal_grid, dtype=float)

        if not legacy_map.point_valid(start_array):
            cell = geometry.world_to_grid(*start_world)
            raise ValueError(
                "Start is not valid after applying planner clearance "
                f"({self.config.clearance_cells} cells); containing cell={cell}"
            )
        if not legacy_map.point_valid(goal_array):
            cell = geometry.world_to_grid(*goal_world)
            raise ValueError(
                "Goal is not valid after applying planner clearance "
                f"({self.config.clearance_cells} cells); containing cell={cell}"
            )

        prior_result = None
        prior_generation_time = 0.0
        prior_world_path = None
        if normalized_name in RAG_PLANNERS:
            prior_result, prior_generation_time = self._generate_external_astar_prior(
                legacy_map,
                start_array,
                goal_array,
                int(seed),
            )
            prior_world_path = self._path_to_world(
                self._prior_path_array(prior_result),
                grid_map,
            )

        # Decouple planner random state from any prior-generator implementation.
        np.random.seed(int(seed))
        random.seed(int(seed))

        planner_kwargs = dict(
            grid_map=legacy_map,
            start=start_array,
            goal=goal_array,
            step_len=self.config.step_len_cells,
            search_radius=self.config.search_radius_cells,
            iter_max=self.config.iter_max,
        )
        if normalized_name == "irrt_star":
            planner_kwargs["max_informed_sample_attempts"] = (
                self.config.informed_max_sample_attempts
            )
        elif normalized_name in RAG_PLANNERS:
            planner_kwargs.update(
                guide_sigma=self.config.rag_guide_sigma_cells,
                guide_max_attempts=self.config.rag_guide_max_attempts,
                alpha_init=self.config.rag_alpha_init,
                alpha_min=self.config.rag_alpha_min,
                alpha_max=self.config.rag_alpha_max,
                window_size=self.config.rag_window_size,
                update_interval=self.config.rag_update_interval,
                eta=self.config.rag_eta,
                margin=self.config.rag_margin,
                min_samples_to_update=self.config.rag_min_samples_to_update,
                w_add=self.config.rag_w_add,
                w_progress=self.config.rag_w_progress,
                w_collision=self.config.rag_w_collision,
                w_improve=self.config.rag_w_improve,
                stagnation_check_interval=(
                    self.config.rag_stagnation_check_interval
                ),
                stagnation_start_iter=self.config.rag_stagnation_start_iter,
                stagnation_alpha_decay=self.config.rag_stagnation_alpha_decay,
                prior_result=prior_result,
                best_path_corridor_radius=(
                    self.config.rag_best_path_corridor_radius_cells
                ),
                min_consistency_samples=(
                    self.config.rag_min_consistency_samples
                ),
                consistency_low_threshold=(
                    self.config.rag_consistency_low_threshold
                ),
                consistency_mid_threshold=(
                    self.config.rag_consistency_mid_threshold
                ),
                consistency_high_threshold=(
                    self.config.rag_consistency_high_threshold
                ),
                # Relative-only final controller: BPC remains diagnostic.
                consistency_penalty=0.0,
                consistency_bonus=0.0,
                low_consistency_alpha_ceiling=self.config.rag_alpha_max,
                mid_consistency_alpha_ceiling=self.config.rag_alpha_max,
                high_consistency_recovery=False,
                use_connectivity_alpha_ceiling=False,
                connectivity_alpha_ceiling_min=(
                    self.config.rag_connectivity_alpha_init_min
                ),
                use_connectivity_alpha_init=True,
                connectivity_alpha_init_min=(
                    self.config.rag_connectivity_alpha_init_min
                ),
                connectivity_alpha_init_max=(
                    self.config.rag_connectivity_alpha_init_max
                ),
                base_sampler_mode="informed_after_initial",
                use_topk_edge_checking=(
                    normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                ),
                parent_check_top_k=(
                    self.config.rag_parent_check_top_k
                    if normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                    else None
                ),
                rewire_check_top_k=(
                    self.config.rag_rewire_check_top_k
                    if normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                    else None
                ),
                use_topk_fallback=(
                    normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                ),
                parent_fallback_check_k=(
                    self.config.rag_parent_fallback_check_k
                    if normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                    else None
                ),
                rewire_fallback_check_k=(
                    self.config.rag_rewire_fallback_check_k
                    if normalized_name == RAG_REL_TOPK_FALLBACK_PLANNER
                    else None
                ),
            )

        planner = planner_class(**planner_kwargs)

        solve_wall_start = time.perf_counter()
        result = planner.solve(
            iter_after_initial=self.config.iter_after_initial,
        )
        solve_wall_elapsed = time.perf_counter() - solve_wall_start

        planner_solve_time = float(
            getattr(result, "planning_time", solve_wall_elapsed)
        )

        conversion_start = time.perf_counter()
        world_path = self._path_to_world(
            getattr(result, "path", None),
            grid_map,
        )
        conversion_elapsed = time.perf_counter() - conversion_start

        tree_world_vertices = None
        tree_parents = None
        if collect_debug_tree:
            tree_world_vertices, tree_parents = self._tree_to_world(
                planner,
                grid_map,
            )

        success = bool(
            getattr(result, "success", False)
            and world_path is not None
            and len(world_path) >= 2
        )
        validation_start = time.perf_counter()
        validation_ok, validation_checks, validation_reason = (False, 0, "no candidate path")
        if success:
            validation_ok, validation_checks, validation_reason = validate_polyline(
                binary_mask, getattr(result, "path", None), self.config.clearance_cells
            )
        validation_time = time.perf_counter() - validation_start
        if success and not validation_ok:
            success = False
            world_path = None
        result.metadata.update({
            "final_path_validation_success": validation_ok,
            "final_path_validation_checks": validation_checks,
            "final_path_validation_time": validation_time,
            "final_path_validation_reason": validation_reason,
        })
        display_name = PLANNER_DISPLAY_NAMES[normalized_name]
        message = (
            f"{display_name} found a path"
            if success
            else (
                f"{display_name} did not find a path within the configured "
                "iteration budget"
            )
        )

        if getattr(result, "success", False) and not validation_ok:
            message = f"{display_name} candidate rejected by final path validation: {validation_reason}"
        message += (
            f"; final_path_validation_success={validation_ok}"
            f" checks={validation_checks} time_sec={validation_time:.9f}"
        )

        return RRTStarAdapterResult(
            success=success,
            world_path=world_path,
            planner_result=result,
            prior_generation_time=prior_generation_time,
            planner_solve_time=planner_solve_time,
            path_conversion_time=conversion_elapsed,
            message=message,
            planner_name=normalized_name,
            tree_world_vertices=tree_world_vertices,
            tree_parents=tree_parents,
            prior_world_path=prior_world_path,
        )

    @staticmethod
    def _tree_to_world(
        planner,
        grid_map: OccupancyGridMap,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Copy the final rewired tree and convert vertices to world coordinates."""

        num_vertices = int(getattr(planner, "num_vertices", 0))
        vertices = np.asarray(
            getattr(planner, "vertices", np.empty((0, 2))),
            dtype=float,
        )
        parents = np.asarray(
            getattr(planner, "parents", np.empty((0,), dtype=int)),
            dtype=int,
        )

        if num_vertices <= 0:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                "Planner vertices must have shape (N, 2), got "
                f"{vertices.shape}"
            )
        if len(vertices) < num_vertices or len(parents) < num_vertices:
            raise ValueError(
                "Planner tree arrays are shorter than num_vertices: "
                f"vertices={len(vertices)}, parents={len(parents)}, "
                f"num_vertices={num_vertices}"
            )

        vertices = vertices[:num_vertices].copy()
        parents = parents[:num_vertices].copy()
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Planner tree contains non-finite vertices")

        world_vertices = [
            grid_map.geometry.planner_to_world(
                float(point[0]),
                float(point[1]),
            )
            for point in vertices
        ]
        return np.asarray(world_vertices, dtype=float), parents

    @staticmethod
    def _path_to_world(
        planner_path,
        grid_map: OccupancyGridMap,
    ) -> Optional[np.ndarray]:
        if planner_path is None:
            return None

        planner_path = np.asarray(planner_path, dtype=float)
        if planner_path.ndim != 2 or planner_path.shape[1] != 2:
            raise ValueError(
                "Planner path must have shape (N, 2), got "
                f"{planner_path.shape}"
            )
        if len(planner_path) < 2:
            return None
        if not np.all(np.isfinite(planner_path)):
            raise ValueError("Planner path contains non-finite coordinates")

        world_points = [
            grid_map.geometry.planner_to_world(float(point[0]), float(point[1]))
            for point in planner_path
        ]
        return np.asarray(world_points, dtype=float)
