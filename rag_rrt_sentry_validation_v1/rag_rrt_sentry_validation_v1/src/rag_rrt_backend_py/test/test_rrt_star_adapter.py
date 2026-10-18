"""Optional integration tests against the user's separate RAG-RRT project."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rag_rrt_backend_py.occupancy_grid_map import MapGeometry, OccupancyGridMap
from rag_rrt_backend_py.rrt_star_adapter import (
    RAG_REL_PLANNER,
    RAG_REL_TOPK_FALLBACK_PLANNER,
    RRTStarAdapter,
    RRTStarAdapterConfig,
)


PLANNERS = [
    "rrt_star",
    "irrt_star",
    RAG_REL_PLANNER,
    RAG_REL_TOPK_FALLBACK_PLANNER,
]


def _project_root_or_skip() -> Path:
    root_text = os.environ.get("RAG_RRT_PROJECT_ROOT", "").strip()
    if not root_text:
        pytest.skip("RAG_RRT_PROJECT_ROOT is not set")

    root = Path(root_text).expanduser().resolve()
    required = (
        root / "planners" / "rrt_star_2d.py",
        root / "planners" / "informed_rrt_star_2d.py",
        root / "planners" / "reliability_guided_rrt_star_2d.py",
        root / "controllers" / "reliability_controller.py",
        root / "samplers" / "astar_corridor_sampler.py",
        root / "utils" / "astar.py",
        root / "utils" / "guidance_connectivity.py",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("RAG_RRT_PROJECT_ROOT is missing planner dependencies")
    return root


@pytest.mark.parametrize("planner_name", PLANNERS)
def test_existing_sampling_planner_adapter_finds_path(planner_name: str) -> None:
    root = _project_root_or_skip()

    height = width = 80
    values = np.zeros((height, width), dtype=np.int16)
    values[30:50, 25:55] = 100

    geometry = MapGeometry(
        width=width,
        height=height,
        resolution=0.05,
        origin_x=-2.0,
        origin_y=-2.0,
    )
    grid_map = OccupancyGridMap(
        geometry=geometry,
        data=values.ravel().tolist(),
    )

    start_world = geometry.planner_to_world(10.0, 40.0)
    goal_world = geometry.planner_to_world(70.0, 40.0)

    adapter = RRTStarAdapter(
        RRTStarAdapterConfig(
            project_root=str(root),
            step_len_cells=5.0,
            search_radius_cells=15.0,
            iter_max=5000,
            iter_after_initial=300,
            clearance_cells=2,
        )
    )
    result = adapter.plan(
        grid_map,
        start_world,
        goal_world,
        seed=0,
        planner_name=planner_name,
        collect_debug_tree=True,
    )

    assert result.success
    assert result.planner_name == planner_name
    assert result.world_path is not None
    assert len(result.world_path) >= 2
    assert result.planner_result.collision_checks > 0

    if planner_name in (RAG_REL_PLANNER, RAG_REL_TOPK_FALLBACK_PLANNER):
        metadata = result.planner_result.metadata
        assert result.prior_generation_time >= 0.0
        assert result.prior_world_path is not None
        assert len(result.prior_world_path) >= 2
        assert metadata["guided_prior_valid"]
        assert metadata["guided_prior_source"] == "external"
        assert metadata["use_connectivity_alpha_init"]
        assert metadata["base_sampler_mode"] == "informed_after_initial"
        assert len(result.planner_result.alpha_history) >= 1

        if planner_name == RAG_REL_TOPK_FALLBACK_PLANNER:
            assert metadata["use_topk_edge_checking"]
            assert metadata["parent_check_top_k"] == 8
            assert metadata["rewire_check_top_k"] == 12
            assert metadata["use_topk_fallback"]
            assert metadata["parent_fallback_check_k"] == 4
            assert metadata["rewire_fallback_check_k"] == 6
            for key in (
                "topk_parent_fallback_checked_count",
                "topk_parent_fallback_success_count",
                "topk_rewire_fallback_checked_count",
                "topk_rewire_fallback_success_count",
            ):
                assert key in metadata
                assert int(metadata[key]) >= 0


@pytest.mark.parametrize("planner_name", PLANNERS)
def test_debug_tree_matches_final_planner_state(planner_name: str) -> None:
    root = _project_root_or_skip()

    width = height = 50
    values = np.zeros((height, width), dtype=np.int16)
    geometry = MapGeometry(
        width=width,
        height=height,
        resolution=0.05,
        origin_x=-1.25,
        origin_y=-1.25,
    )
    grid_map = OccupancyGridMap(
        geometry=geometry,
        data=values.ravel().tolist(),
    )

    adapter = RRTStarAdapter(
        RRTStarAdapterConfig(
            project_root=str(root),
            iter_max=300,
            iter_after_initial=30,
            clearance_cells=1,
        )
    )
    result = adapter.plan(
        grid_map,
        geometry.planner_to_world(5.0, 5.0),
        geometry.planner_to_world(15.0, 5.0),
        seed=0,
        planner_name=planner_name,
        collect_debug_tree=True,
    )

    assert result.tree_world_vertices is not None
    assert result.tree_parents is not None
    assert len(result.tree_world_vertices) == result.planner_result.num_vertices
    assert len(result.tree_parents) == result.planner_result.num_vertices
    assert np.all(np.isfinite(result.tree_world_vertices))
    assert int(result.tree_parents[0]) == 0

    if planner_name in (RAG_REL_PLANNER, RAG_REL_TOPK_FALLBACK_PLANNER):
        assert result.prior_world_path is not None
        assert np.all(np.isfinite(result.prior_world_path))
