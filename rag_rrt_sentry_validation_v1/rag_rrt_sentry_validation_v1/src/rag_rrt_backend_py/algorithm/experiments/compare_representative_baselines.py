# experiments/compare_representative_baselines.py

import argparse
import csv
import sys
import os
import random
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from envs.complex_maps_2d import make_complex_problem
import envs.complex_maps_2d as complex_maps_2d


BENCHMARK_V4_MAPS = [
    "bench_simple_maze",
    "bench_complex_maze",
    "bench_concave_trap",
    "bench_multi_gap",
    "bench_cluttered",
]

BENCHMARK_STRUCTURAL_V5_MAPS = [
    "bench_zigzag_corridor",
    "bench_hourglass_narrow",
    "bench_classic_maze",
]

BENCHMARK_SCENE_MAPS = BENCHMARK_V4_MAPS + BENCHMARK_STRUCTURAL_V5_MAPS

BENCHMARK_SCENE_BUILDERS = {
    # Benchmark scenes v4
    "bench_simple_maze": getattr(
        complex_maps_2d,
        "make_bench_simple_maze_problem_v4",
        None,
    ),
    "bench_complex_maze": getattr(
        complex_maps_2d,
        "make_bench_complex_maze_problem_v4",
        None,
    ),
    "bench_concave_trap": getattr(
        complex_maps_2d,
        "make_bench_concave_trap_problem_v4",
        None,
    ),
    "bench_multi_gap": getattr(
        complex_maps_2d,
        "make_bench_multi_gap_problem_v4",
        None,
    ),
    "bench_cluttered": getattr(
        complex_maps_2d,
        "make_bench_cluttered_problem_v4",
        None,
    ),

    # Structural challenge scenes v5
    "bench_zigzag_corridor": getattr(
        complex_maps_2d,
        "make_bench_zigzag_corridor_problem_v5",
        None,
    ),
    "bench_hourglass_narrow": getattr(
        complex_maps_2d,
        "make_bench_hourglass_narrow_problem_v5",
        None,
    ),
    "bench_classic_maze": getattr(
        complex_maps_2d,
        "make_bench_classic_maze_problem_v5",
        None,
    ),
}
BENCHMARK_SCENE_BUILDERS = {
    name: builder
    for name, builder in BENCHMARK_SCENE_BUILDERS.items()
    if builder is not None
}
from utils.complex_prior_generation import get_complex_prior_config
from envs.grid_map_2d import GridMap2D
from planners.rrt_star_2d import RRTStar2D
from planners.informed_rrt_star_2d import InformedRRTStar2D
from planners.fixed_guided_rrt_star_2d import FixedGuidedRRTStar2D
from planners.reliability_guided_rrt_star_2d import ReliabilityGuidedRRTStar2D
from planners.rrt_connect_2d import RRTConnect2D
from planners.adaptive_guided_rrt_star_2d import AdaptiveGuidedRRTStar2D
from planners.neural_guided_rrt_star_2d import NeuralGuidedRRTStar2D
from priors.prior_factory import build_prior_generator


# ------------------------------------------------------------------
# Final controller / efficiency variant identities
# ------------------------------------------------------------------
RELIABILITY_FACTORIAL_PLANNERS = {
    "rag_rel_fixed_05",
    "rag_rel_conninit_only",
    "rag_rel_online_only",
    "rag_bpc_conn_init_rel_only",
}

ONLINE_DISABLED_PLANNERS = {
    "rag_rel_fixed_05",
    "rag_rel_conninit_only",
}

NO_CONNINIT_PLANNERS = {
    "rag_bpc_no_conn_init",
    "rag_rel_fixed_05",
    "rag_rel_online_only",
}

RELATIVE_ONLY_PLANNERS = {
    "rag_bpc_conn_init_rel_only",
    "rag_bpc_conn_init_rel_only_topk_fallback",
    "rag_rel_fixed_05",
    "rag_rel_conninit_only",
    "rag_rel_online_only",
    "rag_bpc_conn_init_rel_only_delaycc_full",
    "rag_bpc_conn_init_rel_only_delaycc_adaptive_rewire_stage2",
}

TOPK_FALLBACK_PLANNERS = {
    "rag_bpc_conn_init_topk_fallback",
    "rag_bpc_conn_init_topk_fallback_4_8",
    "rag_bpc_conn_init_topk_fallback_12_20",
    "rag_bpc_conn_init_rel_only_topk_fallback",
    "rag_bpc_conn_init_delaycc_rewire_topk_fallback_12_6",
}

EDGE_STRATEGY_PLANNERS = {
    "rag_bpc_conn_init_delaycc_full",
    "rag_bpc_conn_init_yang_k6_parent",
    "rag_bpc_conn_init_delaycc_rewire_topk_12",
    "rag_bpc_conn_init_delaycc_rewire_topk_fallback_12_6",
    "rag_bpc_conn_init_delaycc_adaptive_rewire",
    "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2",
    "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2_rho085",
    "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2_rho095",
    "rag_bpc_conn_init_rel_only_delaycc_full",
    "rag_bpc_conn_init_rel_only_delaycc_adaptive_rewire_stage2",
}

NO_BPC_ALPHA_CEILING_PLANNERS = (
    RELATIVE_ONLY_PLANNERS
    | {"rag_bpc_conn_init_no_ceiling"}
)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--maps",
        nargs="+",
        default=[
            "multi_room",
            "office_corridor",
            "warehouse_clutter",
            "urban_blocks",
            "hybrid_trap",
        ],
        choices=[
            "multi_room",
            "office_corridor",
            "warehouse_clutter",
            "urban_blocks",
            "hybrid_trap",
            "multi_room_hard",
            "office_corridor_hard",
            "warehouse_clutter_hard",
            "urban_blocks_hard",
            "hybrid_trap_hard",
            "multi_room_hard_v2",
            "office_corridor_hard_v2",
            "warehouse_clutter_hard_v2",
            "urban_blocks_hard_v2",
            "hybrid_trap_hard_v2",
            "multi_room_hard_v3",
            "office_corridor_hard_v3",
            "warehouse_clutter_hard_v3",
            "urban_blocks_hard_v3",
            "hybrid_trap_hard_v3",

            # benchmark scenes v4 + structural challenge scenes v5
            *BENCHMARK_SCENE_MAPS,
        ],
    )
    
    parser.add_argument(
        "--save_map_previews",
        action="store_true",
        help="Save obstacle map previews with start and goal markers.",
    )

    parser.add_argument(
        "--priors",
        nargs="+",
        default=[
            "coarse_valid",
            "broken_middle",
            "endpoint_mismatch",
            "strong_shift",
        ],
        choices=[
            "coarse_valid",
            "broken_middle",
            "endpoint_mismatch",
            "strong_shift",
            "learned_heatmap",
        ],
    )
    
    parser.add_argument(
        "--prior_generator",
        default="legacy_internal",
        choices=[
            "legacy_internal",
            "astar",
            "synthetic_failure",
            "learned",
        ],
        help=(
            "External prior generator interface. "
            "legacy_internal keeps the old sampler-internal A* / failure-mode logic. "
            "Use synthetic_failure, astar, or learned to generate an external PriorResult "
            "and pass it into guided planners."
        ),
    )

    parser.add_argument(
        "--learned_prior_checkpoint",
        default="",
        help="Checkpoint path for LearnedPriorGenerator when --prior_generator learned.",
    )

    parser.add_argument(
        "--learned_prior_device",
        default="cpu",
        help="Device for learned prior inference, e.g. cpu or cuda.",
    )

    parser.add_argument(
        "--learned_prior_heatmap_threshold",
        type=float,
        default=0.5,
        help="Heatmap threshold recorded for learned prior diagnostics.",
    )

    parser.add_argument(
        "--learned_prior_heatmap_cost_weight",
        type=float,
        default=5.0,
        help="Cost weight used when extracting a path from the learned heatmap.",
    )

    parser.add_argument(
        "--learned_prior_allow_fallback",
        action="store_true",
        help="Allow LearnedPriorGenerator to fallback to A* if checkpoint/inference/extraction fails.",
    )

    parser.add_argument(
        "--planners",
        nargs="+",
        default=[
            "rrt_star",
            "rrt_connect",
            "irrt_star",
            "fixed_guided_05",
            "fixed_guided_07",
            "adaptive_guided",
            "rag_bpc",
            "rag_bpc_conn_init",
        ],
        choices=[
            "rrt_star",
            "rrt_connect",
            "irrt_star",
            "neural_guided_rrt_star",
            "fixed_guided_03",
            "fixed_guided_05",
            "fixed_guided_07",
            "adaptive_guided",
            "rag_bpc",
            "rag_bpc_conn",
            "rag_bpc_conn_init",
            "rag_bpc_conn_init_uniform_base",
            "rag_bpc_conn_init_rel_only",
            "rag_bpc_conn_init_rel_only_topk_fallback",
            "rag_rel_fixed_05",
            "rag_rel_conninit_only",
            "rag_rel_online_only",
            "rag_bpc_conn_init_no_ceiling",
            "rag_bpc_no_conn_init",
            "rag_bpc_conn_init_topk",
            "rag_bpc_conn_init_topk_fallback",
            "rag_bpc_conn_init_topk_fallback_4_8",
            "rag_bpc_conn_init_topk_fallback_12_20",
            "rag_bpc_conn_init_topk_4_8",
            "rag_bpc_conn_init_topk_8_12",
            "rag_bpc_conn_init_topk_12_20",
            *sorted(EDGE_STRATEGY_PLANNERS),
        ],
        help="Planner variants to run.",
    )

    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--iter_max", type=int, default=5000)
    parser.add_argument("--iter_after_initial", type=int, default=1000)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--output_dir", default="results/complex_sim_maps")
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes.",
    )

    return parser.parse_args()


# =========================
# Map definitions
# =========================

def make_gap_horizontal_problem():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    wall_x0, wall_x1 = 58, 62
    binary_mask[:, wall_x0:wall_x1] = 0

    gap_y0, gap_y1 = 30, 38
    binary_mask[gap_y0:gap_y1, wall_x0:wall_x1] = 1

    start = np.array([15, 90], dtype=float)
    goal = np.array([105, 90], dtype=float)

    return binary_mask, start, goal


def make_gap_diagonal_problem():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    wall_x0, wall_x1 = 58, 62
    binary_mask[:, wall_x0:wall_x1] = 0

    gap_y0, gap_y1 = 30, 38
    binary_mask[gap_y0:gap_y1, wall_x0:wall_x1] = 1

    start = np.array([15, 95], dtype=float)
    goal = np.array([105, 25], dtype=float)

    return binary_mask, start, goal


def make_gap_false_opening_problem():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    wall_x0, wall_x1 = 58, 62
    binary_mask[:, wall_x0:wall_x1] = 0

    # True opening
    true_gap_y0, true_gap_y1 = 28, 36
    binary_mask[true_gap_y0:true_gap_y1, wall_x0:wall_x1] = 1

    # False opening
    false_gap_y0, false_gap_y1 = 76, 84
    binary_mask[false_gap_y0:false_gap_y1, wall_x0:wall_x1] = 1

    # Blocking wall behind the false opening
    block_x0, block_x1 = 72, 78
    block_y0, block_y1 = 64, 100
    binary_mask[block_y0:block_y1, block_x0:block_x1] = 0

    start = np.array([15, 90], dtype=float)
    goal = np.array([105, 30], dtype=float)

    return binary_mask, start, goal



def make_maze_like_problem():
    """
    Maze-like environment with alternating vertical walls and gaps.

    This map is intended to test whether the planner can handle multiple
    turns and detour structures. A shifted prior may still be partially
    useful but can also be misleading around wall openings.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    wall_thickness = 4

    # Alternating vertical walls.
    walls = [
        # x0, x1, gap_y0, gap_y1
        (25, 25 + wall_thickness, 12, 28),
        (45, 45 + wall_thickness, 82, 102),
        (65, 65 + wall_thickness, 16, 34),
        (85, 85 + wall_thickness, 78, 98),
    ]

    for x0, x1, gap_y0, gap_y1 in walls:
        binary_mask[:, x0:x1] = 0
        binary_mask[gap_y0:gap_y1, x0:x1] = 1

    start = np.array([10, 105], dtype=float)
    goal = np.array([110, 15], dtype=float)

    return binary_mask, start, goal



def make_narrow_passage_problem():
    """
    Diagonal narrow-passage environment.

    The planner must move from the lower-left region to the upper-right
    region through a long but feasible narrow corridor. The passage is
    narrower than the original gap maps, but still feasible under
    clearance=2 and step_len=5.0.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    # A long central corridor.
    # raw width = 12; with clearance=2, the effective width remains feasible.
    corridor_y0, corridor_y1 = 54, 66
    block_x0, block_x1 = 25, 95

    # Upper and lower blocks create the long narrow passage.
    binary_mask[0:corridor_y0, block_x0:block_x1] = 0
    binary_mask[corridor_y1:height, block_x0:block_x1] = 0

    # Mild side clutter, kept away from the passage entrances.
    binary_mask[15:35, 8:18] = 0
    binary_mask[85:105, 8:18] = 0

    binary_mask[15:35, 102:112] = 0
    binary_mask[85:105, 102:112] = 0

    start = np.array([10, 108], dtype=float)
    goal = np.array([110, 12], dtype=float)

    return binary_mask, start, goal



def make_cluttered_problem():
    """
    Cluttered obstacle environment.

    This map contains multiple rectangular obstacles. The purpose is to test
    whether the reliability controller remains stable when a prior is only
    locally useful and the path structure is less dominated by a single gap.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    rectangles = [
        # y0, y1, x0, x1
        (12, 42, 25, 35),
        (55, 105, 25, 35),

        (15, 65, 50, 60),
        (78, 112, 50, 60),

        (8, 35, 75, 85),
        (48, 95, 75, 85),

        (35, 48, 92, 112),
        (70, 82, 5, 25),

        (95, 108, 88, 108),
        (8, 20, 5, 20),
    ]

    for y0, y1, x0, x1 in rectangles:
        binary_mask[y0:y1, x0:x1] = 0

    start = np.array([10, 108], dtype=float)
    goal = np.array([110, 12], dtype=float)

    return binary_mask, start, goal



def make_problem(map_name):
    """Return the binary map, start, and goal for an experiment map.

    Most existing maps are still handled by make_complex_problem(). Benchmark
    v4 and structural v5 scenes are dispatched explicitly when their builder
    functions exist in envs/complex_maps_2d.py. This keeps the script backward
    compatible with the previous complex-map experiments while allowing the
    benchmark scenes to be used from --maps.
    """
    if map_name in BENCHMARK_SCENE_MAPS:
        builder = BENCHMARK_SCENE_BUILDERS.get(map_name)
        if builder is not None:
            return builder()

    return make_complex_problem(map_name)



def save_map_preview(binary_mask, start, goal, map_name, output_dir):
    """
    Save a simple map preview showing obstacles, start, and goal.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    # binary_mask: 1 = free, 0 = obstacle.
    ax.imshow(
        binary_mask,
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    ax.scatter(start[0], start[1], marker="o", s=60, label="start")
    ax.scatter(goal[0], goal[1], marker="*", s=100, label="goal")

    ax.set_title(map_name)
    ax.set_xlim(0, binary_mask.shape[1])
    ax.set_ylim(binary_mask.shape[0], 0)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.grid(False)

    save_path = output_dir / f"{map_name}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path



def get_path_shift(prior, map_name):
    if prior in [
        "correct",
        "partial_start",
        "partial_goal",
        "broken_middle",
        "endpoint_mismatch",
    ]:
        return None

    if map_name == "gap_horizontal":
        if prior == "mild_shift":
            return (0.0, 20.0)
        if prior == "strong_shift":
            return (0.0, 35.0)

    if map_name == "gap_diagonal":
        if prior == "mild_shift":
            return (0.0, 18.0)
        if prior == "strong_shift":
            return (0.0, 32.0)

    if map_name == "gap_false_opening":
        if prior == "mild_shift":
            return (0.0, 28.0)
        if prior == "strong_shift":
            return (0.0, 45.0)

    if map_name == "maze_like":
        if prior == "mild_shift":
            return (0.0, 14.0)
        if prior == "strong_shift":
            return (0.0, 28.0)

    if map_name == "narrow_passage":
        if prior == "mild_shift":
            return (0.0, 10.0)
        if prior == "strong_shift":
            return (0.0, 22.0)

    if map_name == "cluttered":
        if prior == "mild_shift":
            return (8.0, 8.0)
        if prior == "strong_shift":
            return (18.0, 18.0)

    raise ValueError(f"Unknown prior/map setting: prior={prior}, map={map_name}")



def get_prior_failure_mode(prior, map_name):
    """
        Return deterministic prior failure mode and parameters.

        Shift-based priors are handled by get_path_shift().
        Structural failure priors are handled here.
    """
    if prior in ["correct", "mild_shift", "strong_shift"]:
        return "none", {}

    """
        Boundary diagnostic mode:
            this prior is locally valid but globally incomplete.
            The current connectivity score may be overly permissive for this case.
    """
    if prior == "partial_start":
        return "partial_start", {
            "keep_ratio": 0.25,
        }
    """
        Boundary diagnostic mode:
            kept for future analysis, not recommended as a main failure mode yet.
    """
    if prior == "partial_goal":
        return "partial_goal", {
            "keep_ratio": 0.25,
        }

    if prior == "broken_middle":
        return "broken_middle", {
            "keep_ratio_each_side": 0.30,
        }

    if prior == "endpoint_mismatch":
        # Endpoint mismatch shifts the transformed prior away from start / goal.
        # For narrow_passage, a vertical shift is used because the main passage is
        # approximately horizontal; this makes the prior leave the valid corridor.
        if map_name == "narrow_passage":
            return "endpoint_mismatch", {
                "shift": (0.0, 32.0),
            }
        # Generic diagonal shift for the remaining maps.
        return "endpoint_mismatch", {
            "shift": (18.0, 18.0),
        }

    raise ValueError(
        f"Unknown prior failure setting: prior={prior}, map={map_name}"
    )



# =========================
# External prior interface
# =========================

def is_path_guided_planner(planner_name):
    """
    Return whether the planner consumes a path-like prior.

    These planners use AStarCorridorSampler / prior paths. They are distinct
    from the NRRT*-style heatmap-sampling baseline.
    """
    return planner_name in [
        "fixed_guided_03",
        "fixed_guided_05",
        "fixed_guided_07",
        "adaptive_guided",
        "rag_bpc",
        "rag_bpc_conn",
        "rag_bpc_conn_init",
        "rag_bpc_conn_init_uniform_base",
        "rag_bpc_conn_init_rel_only",
        "rag_bpc_conn_init_rel_only_topk_fallback",
        "rag_rel_fixed_05",
        "rag_rel_conninit_only",
        "rag_rel_online_only",
        "rag_bpc_conn_init_no_ceiling",
        "rag_bpc_no_conn_init",
        "rag_bpc_conn_init_topk",
        "rag_bpc_conn_init_topk_fallback",
        "rag_bpc_conn_init_topk_fallback_4_8",
        "rag_bpc_conn_init_topk_fallback_12_20",
        "rag_bpc_conn_init_topk_4_8",
        "rag_bpc_conn_init_topk_8_12",
        "rag_bpc_conn_init_topk_12_20",
        *sorted(EDGE_STRATEGY_PLANNERS),
    ]


def is_neural_guided_planner(planner_name):
    """Return whether the planner consumes a heatmap prior directly."""
    return planner_name == "neural_guided_rrt_star"


def is_guided_planner(planner_name):
    """Return whether a planner consumes any external prior result."""
    return is_path_guided_planner(planner_name) or is_neural_guided_planner(planner_name)


def get_prior_path_array_safe(prior_result):
    if prior_result is None:
        return None

    if hasattr(prior_result, "path_array"):
        try:
            return prior_result.path_array()
        except Exception:
            return None

    path = getattr(prior_result, "path", None)
    if path is None:
        return None

    try:
        return np.asarray(path, dtype=float)
    except Exception:
        return None


def prior_result_has_usable_path(prior_result):
    path = get_prior_path_array_safe(prior_result)
    return path is not None and len(path) >= 2


def get_prior_heatmap_array_safe(prior_result):
    if prior_result is None:
        return None

    heatmap = getattr(prior_result, "heatmap", None)
    if heatmap is None:
        return None

    try:
        heatmap = np.asarray(heatmap, dtype=float)
    except Exception:
        return None

    if heatmap.ndim != 2:
        return None

    if not np.any(np.isfinite(heatmap)):
        return None

    return heatmap


def prior_result_has_usable_heatmap(prior_result):
    heatmap = get_prior_heatmap_array_safe(prior_result)
    if heatmap is None:
        return False
    return bool(np.nanmax(heatmap) > 0.0)


def make_external_prior_result(
    prior_generator_name,
    grid_map,
    start,
    goal,
    prior,
    map_name,
    prior_generator_config=None,
):
    """
    Generate an external PriorResult through priors/prior_factory.py.

    legacy_internal:
        Return None and keep the old AStarCorridorSampler behavior.

    synthetic_failure / astar / learned:
        Build a generator and call generate(...). The returned PriorResult is
        later passed into guided planners only. Non-guided planners keep their
        original behavior.
    """
    if prior_generator_name in [None, "legacy_internal"]:
        return None

    generator = build_prior_generator(
        prior_generator_name,
        config={} if prior_generator_config is None else dict(prior_generator_config),
    )

    prior_result = generator.generate(
        env=grid_map,
        start=start,
        goal=goal,
        prior_mode=prior,
        map_name=map_name,
    )

    return prior_result

def summarize_external_prior_result(prior_result):
    """
    Convert a PriorResult into scalar CSV-friendly fields.

    These fields describe the external prior generator output, not planner
    performance. If legacy_internal is used, the values remain NaN / empty.
    Learned-prior-specific fields are read from PriorResult.metadata.
    """
    if prior_result is None:
        return {
            "external_prior_type": np.nan,
            "external_prior_mode": np.nan,
            "external_prior_success": np.nan,
            "external_prior_message": "",
            "external_prior_num_points": np.nan,
            "external_prior_has_path": np.nan,
            "external_prior_has_heatmap": np.nan,
            "external_prior_used_fallback": np.nan,
            "external_prior_fallback_type": "",
            "external_prior_path_extraction_method": "",
            "external_prior_heatmap_threshold": np.nan,
            "external_prior_heatmap_cost_weight": np.nan,
            "external_prior_checkpoint": "",
        }

    path = get_prior_path_array_safe(prior_result)
    metadata = getattr(prior_result, "metadata", {}) or {}

    if hasattr(prior_result, "has_heatmap"):
        try:
            has_heatmap = int(prior_result.has_heatmap())
        except Exception:
            has_heatmap = int(getattr(prior_result, "heatmap", None) is not None)
    else:
        has_heatmap = int(getattr(prior_result, "heatmap", None) is not None)

    return {
        "external_prior_type": getattr(prior_result, "prior_type", np.nan),
        "external_prior_mode": getattr(prior_result, "prior_mode", np.nan),
        "external_prior_success": int(bool(getattr(prior_result, "success", False))),
        "external_prior_message": str(getattr(prior_result, "message", "")),
        "external_prior_num_points": (
            np.nan if path is None else int(len(path))
        ),
        "external_prior_has_path": int(path is not None and len(path) >= 2),
        "external_prior_has_heatmap": has_heatmap,
        "external_prior_used_fallback": metadata.get("used_fallback", np.nan),
        "external_prior_fallback_type": metadata.get("fallback_type", ""),
        "external_prior_path_extraction_method": metadata.get(
            "path_extraction_method", ""
        ),
        "external_prior_heatmap_threshold": metadata.get(
            "heatmap_threshold", np.nan
        ),
        "external_prior_heatmap_cost_weight": metadata.get(
            "heatmap_cost_weight", np.nan
        ),
        "external_prior_checkpoint": metadata.get(
            "checkpoint_path", metadata.get("checkpoint", "")
        ),
    }

def is_missing_csv_value(value):
    """
    Return True for values that should be treated as missing in CSV fields.

    This handles None, empty strings, and numeric NaN. It intentionally keeps
    ordinary strings such as "none" because they are meaningful legacy labels.
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value == ""

    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def first_non_missing(*values, default=np.nan):
    """Return the first value that is not None / empty / NaN."""
    for value in values:
        if not is_missing_csv_value(value):
            return value
    return default


# =========================
# Planner construction
# =========================

def build_planner(
    planner_name,
    grid_map,
    start,
    goal,
    iter_max,
    path_shift,
    prior_failure_mode="none",
    prior_failure_params=None,
    prior_path=None,
    prior_result=None,
):
    common_kwargs = dict(
        grid_map=grid_map,
        start=start,
        goal=goal,
        step_len=5.0,
        search_radius=15.0,
        iter_max=iter_max,
    )

    if planner_name == "rrt_star":
        return RRTStar2D(**common_kwargs)
    
    if planner_name == "rrt_connect":
        return RRTConnect2D(**common_kwargs)

    if planner_name == "irrt_star":
        return InformedRRTStar2D(**common_kwargs)

    if planner_name == "neural_guided_rrt_star":
        return NeuralGuidedRRTStar2D(
            **common_kwargs,
            prior_result=prior_result,
            p_neural=0.5,
            max_heatmap_attempts=50,
            heatmap_threshold=0.5,
        )

    if planner_name == "fixed_guided_03":
        return FixedGuidedRRTStar2D(
            **common_kwargs,
            guide_sigma=5.0,
            alpha=0.3,
            guide_path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )

    if planner_name == "fixed_guided_05":
        return FixedGuidedRRTStar2D(
            **common_kwargs,
            guide_sigma=5.0,
            alpha=0.5,
            guide_path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )

    if planner_name == "fixed_guided_07":
        return FixedGuidedRRTStar2D(
            **common_kwargs,
            guide_sigma=5.0,
            alpha=0.7,
            guide_path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )
    
    if planner_name == "adaptive_guided":
        return AdaptiveGuidedRRTStar2D(
            **common_kwargs,
            guide_sigma=5.0,
            alpha_init=0.5,
            alpha_min=0.05,
            alpha_max=0.8,
            update_interval=50,
            beta=0.25,
            w_valid=0.7,
            w_improve=0.3,
            guide_path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,
        )

    topk_config = {
        "rag_bpc_conn_init_topk": (8, 12),
        "rag_bpc_conn_init_topk_fallback": (8, 12),
        "rag_bpc_conn_init_rel_only_topk_fallback": (8, 12),
        "rag_bpc_conn_init_topk_fallback_4_8": (4, 8),
        "rag_bpc_conn_init_topk_fallback_12_20": (12, 20),
        "rag_bpc_conn_init_topk_4_8": (4, 8),
        "rag_bpc_conn_init_topk_8_12": (8, 12),
        "rag_bpc_conn_init_topk_12_20": (12, 20),
    }

    edge_strategy_config = {
        # OMPL-style delayed collision checking in parent selection, with
        # otherwise full rewiring. This isolates the known sorting/early-stop
        # benefit from candidate truncation.
        "rag_bpc_conn_init_delaycc_full": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "full",
        },
        # Yang-style mechanism control: fixed K=6 parent candidates and full
        # rewiring on the same RAG backbone. It is not a full reproduction of
        # every module in the cited planner.
        "rag_bpc_conn_init_yang_k6_parent": {
            "parent_edge_strategy": "topk",
            "rewire_edge_strategy": "full",
            "parent_check_top_k": 6,
        },
        "rag_bpc_conn_init_delaycc_rewire_topk_12": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "topk",
            "rewire_check_top_k": 12,
        },
        "rag_bpc_conn_init_delaycc_rewire_topk_fallback_12_6": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "topk",
            "rewire_check_top_k": 12,
            "use_topk_fallback": True,
            "rewire_fallback_check_k": 6,
        },
        "rag_bpc_conn_init_delaycc_adaptive_rewire": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "adaptive",
            "adaptive_rewire_config": {
                "budgets": (4, 8, 12, 20),
                "initial_budget": 12,
                "opportunity_coverage_target": 0.90,
                "solution_slack_ratio": 0.10,
                "relevance_floor": 0.05,
                "probe_interval": 16,
                "probe_k": 1,
            },
        },
        "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "adaptive",
            "adaptive_rewire_config": {
                "budgets": (4, 8, 12, 20),
                "initial_budget": 12,
                "opportunity_coverage_target": 0.90,
                "solution_slack_ratio": 0.10,
                "relevance_floor": 0.05,
                "probe_interval": 16,
                "probe_k": 1,
            },
        },
        "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2_rho085": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "adaptive",
            "adaptive_rewire_config": {
                "budgets": (4, 8, 12, 20),
                "initial_budget": 12,
                "opportunity_coverage_target": 0.85,
                "solution_slack_ratio": 0.10,
                "relevance_floor": 0.05,
                "probe_interval": 16,
                "probe_k": 1,
            },
        },
        "rag_bpc_conn_init_delaycc_adaptive_rewire_stage2_rho095": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "adaptive",
            "adaptive_rewire_config": {
                "budgets": (4, 8, 12, 20),
                "initial_budget": 12,
                "opportunity_coverage_target": 0.95,
                "solution_slack_ratio": 0.10,
                "relevance_floor": 0.05,
                "probe_interval": 16,
                "probe_k": 1,
            },
        },
        # Relative-only integration controls for the final Stage-2 efficiency
        # mechanism.  These keep BPC diagnostic-only and vary only the edge
        # validation strategy.
        "rag_bpc_conn_init_rel_only_delaycc_full": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "full",
        },
        "rag_bpc_conn_init_rel_only_delaycc_adaptive_rewire_stage2": {
            "parent_edge_strategy": "delay_cc",
            "rewire_edge_strategy": "adaptive",
            "adaptive_rewire_config": {
                "budgets": (4, 8, 12, 20),
                "initial_budget": 12,
                "opportunity_coverage_target": 0.90,
                "solution_slack_ratio": 0.10,
                "relevance_floor": 0.05,
                "probe_interval": 16,
                "probe_k": 1,
            },
        },
    }
    
    rag_variants = [
        "rag_bpc",
        "rag_bpc_conn",
        "rag_bpc_conn_init",
        "rag_bpc_conn_init_uniform_base",
        "rag_bpc_conn_init_rel_only",
        "rag_bpc_conn_init_rel_only_topk_fallback",
        "rag_bpc_conn_init_no_ceiling",
        "rag_bpc_no_conn_init",
        "rag_rel_fixed_05",
        "rag_rel_conninit_only",
        "rag_rel_online_only",
        *topk_config.keys(),
        *edge_strategy_config.keys(),
    ]
    
    if planner_name in rag_variants:
        use_conn_ceiling = planner_name == "rag_bpc_conn"

        use_conn_init = planner_name in [
            "rag_bpc_conn_init",
            "rag_bpc_conn_init_uniform_base",
            "rag_bpc_conn_init_rel_only",
            "rag_bpc_conn_init_rel_only_topk_fallback",
            "rag_bpc_conn_init_no_ceiling",
            "rag_rel_conninit_only",
            *topk_config.keys(),
            *edge_strategy_config.keys(),
        ]

        use_topk_edge_checking = planner_name in topk_config
        use_topk_fallback = planner_name in TOPK_FALLBACK_PLANNERS

        # No-ConnInit configurations keep alpha_init=0.5. Their online and
        # BPC behavior is selected independently below.
        use_no_conninit_ablation = planner_name in NO_CONNINIT_PLANNERS

        # Relative-only controller:
        # keep ConnInit and guided/base relative scoring, while BPC remains a
        # diagnostic signal only. The selected final efficiency variant
        # "rag_bpc_conn_init_rel_only_topk_fallback" combines this controller with TopK-Fallback.
        # Consistency penalty/bonus, BPC ceilings, and high-consistency recovery
        # are disabled for every planner in RELATIVE_ONLY_PLANNERS.
        use_bpc_adjustment = planner_name not in RELATIVE_ONLY_PLANNERS

        enable_online_reliability_update = (
            planner_name not in ONLINE_DISABLED_PLANNERS
        )

        # No-ceiling ablation:
        # keep BPC penalty/bonus and high-consistency recovery, but disable
        # BPC alpha ceilings so that low/mid consistency cannot clamp alpha.
        use_bpc_alpha_ceiling = (
            planner_name not in NO_BPC_ALPHA_CEILING_PLANNERS
        )

        if use_topk_edge_checking:
            parent_check_top_k, rewire_check_top_k = topk_config[planner_name]
        else:
            parent_check_top_k, rewire_check_top_k = None, None

        strategy = edge_strategy_config.get(planner_name, {})
        parent_edge_strategy = strategy.get("parent_edge_strategy")
        rewire_edge_strategy = strategy.get("rewire_edge_strategy")
        adaptive_rewire_config = strategy.get("adaptive_rewire_config")
        parent_check_top_k = strategy.get(
            "parent_check_top_k",
            parent_check_top_k,
        )
        rewire_check_top_k = strategy.get(
            "rewire_check_top_k",
            rewire_check_top_k,
        )
        use_topk_fallback = bool(
            strategy.get("use_topk_fallback", use_topk_fallback)
        )
        parent_fallback_check_k = strategy.get(
            "parent_fallback_check_k",
            4 if use_topk_fallback and parent_edge_strategy != "delay_cc" else None,
        )
        rewire_fallback_check_k = strategy.get(
            "rewire_fallback_check_k",
            6 if use_topk_fallback else None,
        )

        return ReliabilityGuidedRRTStar2D(
            **common_kwargs,
            guide_sigma=5.0,
            alpha_init=0.5,
            alpha_min=0.1,
            alpha_max=0.8,
            window_size=30,
            update_interval=20,
            eta=0.08,
            margin=0.05,
            min_samples_to_update=8,
            w_add=0.2,
            w_progress=1.0,
            w_collision=1.5,
            w_improve=2.0,
            stagnation_check_interval=100,
            stagnation_start_iter=100,
            stagnation_alpha_decay=0.05,
            enable_online_reliability_update=(
                enable_online_reliability_update
            ),
            guide_path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,
            prior_failure_params=prior_failure_params,
            prior_path=prior_path,
            prior_result=prior_result,

            # Best-path corridor consistency parameters.
            best_path_corridor_radius=10.0,
            min_consistency_samples=8,
            consistency_low_threshold=0.35,
            consistency_mid_threshold=0.55,
            consistency_high_threshold=0.70,
            consistency_penalty=(0.20 if use_bpc_adjustment else 0.0),
            consistency_bonus=(0.05 if use_bpc_adjustment else 0.0),
            low_consistency_alpha_ceiling=(
                0.30 if (use_bpc_adjustment and use_bpc_alpha_ceiling) else 0.8
            ),
            mid_consistency_alpha_ceiling=(
                0.50 if (use_bpc_adjustment and use_bpc_alpha_ceiling) else 0.8
            ),
            high_consistency_recovery=use_bpc_adjustment,
            high_consistency_recovery_eta=0.03,

            # Connectivity-based pre-solution trust regulation.
            use_connectivity_alpha_ceiling=use_conn_ceiling,
            connectivity_alpha_ceiling_min=0.30,
            use_connectivity_alpha_init=use_conn_init,
            connectivity_alpha_init_min=0.30,
            connectivity_alpha_init_max=0.50,

            # Base-branch ablation.
            # Default RAG uses uniform before initial solution and informed
            # sampling after initial solution. The uniform-base ablation keeps
            # the base branch uniform throughout planning.
            base_sampler_mode=(
                "uniform"
                if planner_name == "rag_bpc_conn_init_uniform_base"
                else "informed_after_initial"
            ),

            # Collision-check-aware Top-K edge evaluation.
            use_topk_edge_checking=use_topk_edge_checking,
            parent_check_top_k=parent_check_top_k,
            rewire_check_top_k=rewire_check_top_k,
            use_topk_fallback=use_topk_fallback,
            parent_fallback_check_k=parent_fallback_check_k,
            rewire_fallback_check_k=rewire_fallback_check_k,
            parent_edge_strategy=parent_edge_strategy,
            rewire_edge_strategy=rewire_edge_strategy,
            adaptive_rewire_config=adaptive_rewire_config,
        )

    raise ValueError(f"Unknown planner_name: {planner_name}")


# =========================
# Result utilities
# =========================

def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def safe_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.min(values))


def safe_max(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.max(values))


def safe_last(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(values[-1])


def extract_run_record(
    map_name,
    prior,
    planner_name,
    seed,
    result,
    prior_generator="legacy_internal",
    prior_result=None,
):
    alpha_history = getattr(result, "alpha_history", [])
    
    metadata = getattr(result, "metadata", {})
    external_prior_summary = summarize_external_prior_result(prior_result)
    
    collision_checks_until_initial = metadata.get(
        "collision_checks_until_initial", np.nan
    )
    collision_checks_after_initial = metadata.get(
        "collision_checks_after_initial", np.nan
    )

    if len(alpha_history) > 0:
        initial_alpha = float(alpha_history[0])
        final_alpha = float(alpha_history[-1])
        min_alpha = float(np.min(alpha_history))
        max_alpha = float(np.max(alpha_history))

        # A simple alpha recovery indicator:
        # alpha drops below initial alpha and later returns upward.
        alpha_recovery = int(
            min_alpha < alpha_history[0] - 1e-9
            and final_alpha > min_alpha + 1e-9
        )
    else:
        initial_alpha = np.nan
        final_alpha = np.nan
        min_alpha = np.nan
        max_alpha = np.nan
        alpha_recovery = 0

    # New fields for the best-path corridor consistency controller.
    #
    # These attributes only exist for ReliabilityGuidedRRTStar2D after you
    # replaced planners/reliability_guided_rrt_star_2d.py with the best-path
    # version. For other planners, getattr(...) returns [] and all these
    # metrics become NaN. This keeps the CSV schema uniform for all planners.
    consistency_history = getattr(
        result,
        "best_path_consistency_rate_history",
        [],
    )
    distance_history = getattr(
        result,
        "mean_distance_to_best_path_history",
        [],
    )
    alpha_ceiling_history = getattr(
        result,
        "alpha_ceiling_history",
        [],
    )

    mean_best_path_consistency = safe_mean(consistency_history)
    final_best_path_consistency = safe_last(consistency_history)
    min_best_path_consistency = safe_min(consistency_history)
    max_best_path_consistency = safe_max(consistency_history)

    mean_distance_to_best_path = safe_mean(distance_history)
    final_distance_to_best_path = safe_last(distance_history)

    mean_alpha_ceiling = safe_mean(alpha_ceiling_history)
    final_alpha_ceiling = safe_last(alpha_ceiling_history)
    min_alpha_ceiling = safe_min(alpha_ceiling_history)
    
    prior_conn_fields = [
        "prior_num_points",
        "prior_valid_ratio",
        "prior_edge_valid_ratio",
        "num_components",
        "largest_component_ratio",
        "start_reachable_ratio",
        "goal_reachable_from_start",
        "frontier_progress",
        "start_near_prior",
        "goal_near_prior",
        "connectivity_score",
    ]

    # ------------------------------------------------------------------
    # Prior identity fields
    # ------------------------------------------------------------------
    # legacy_prior_failure_* describes the old sampler-internal corruption
    # mechanism. It is meaningful mainly when prior_generator ==
    # "legacy_internal".
    #
    # external_prior_* describes the output of priors/PriorGenerator. It is
    # meaningful when prior_generator != "legacy_internal".
    #
    # effective_prior_mode gives a single analysis field: external prior mode
    # when available; otherwise the legacy prior failure mode.
    guided_prior_source = first_non_missing(
        metadata.get("guided_prior_source", np.nan),
        default=np.nan,
    )

    external_prior_type = first_non_missing(
        metadata.get("external_prior_type", np.nan),
        external_prior_summary["external_prior_type"],
        default=np.nan,
    )

    external_prior_mode = first_non_missing(
        metadata.get("external_prior_mode", np.nan),
        external_prior_summary["external_prior_mode"],
        default=np.nan,
    )

    legacy_prior_failure_mode = first_non_missing(
        metadata.get("prior_failure_mode", np.nan),
        default="none",
    )

    legacy_prior_failure_params = first_non_missing(
        metadata.get("prior_failure_params", np.nan),
        default="{}",
    )

    effective_prior_mode = (
        external_prior_mode
        if not is_missing_csv_value(external_prior_mode)
        else legacy_prior_failure_mode
    )

    return {
        "map": map_name,
        "prior": prior,
        "planner": planner_name,
        "seed": seed,
        "prior_generator": prior_generator,
        "guided_prior_source": guided_prior_source,
        "external_prior_type": external_prior_type,
        "external_prior_mode": external_prior_mode,
        "external_prior_success": external_prior_summary["external_prior_success"],
        "external_prior_message": external_prior_summary["external_prior_message"],
        "external_prior_num_points": external_prior_summary["external_prior_num_points"],
        "external_prior_has_path": external_prior_summary["external_prior_has_path"],
        "external_prior_has_heatmap": external_prior_summary["external_prior_has_heatmap"],
        "external_prior_used_fallback": external_prior_summary["external_prior_used_fallback"],
        "external_prior_fallback_type": external_prior_summary["external_prior_fallback_type"],
        "external_prior_path_extraction_method": external_prior_summary["external_prior_path_extraction_method"],
        "external_prior_heatmap_threshold": external_prior_summary["external_prior_heatmap_threshold"],
        "external_prior_heatmap_cost_weight": external_prior_summary["external_prior_heatmap_cost_weight"],
        "external_prior_checkpoint": external_prior_summary["external_prior_checkpoint"],
        "success": int(result.success),
        "first_solution_iter": (
            result.first_solution_iter
            if result.first_solution_iter is not None
            else np.inf
        ),
        "first_solution_cost": result.first_solution_cost,
        "final_cost": result.final_cost,
        "planning_time": result.planning_time,
        "num_vertices": result.num_vertices,
        "collision_checks": result.collision_checks,
        "collision_check_samples": metadata.get(
            "collision_check_samples", np.nan
        ),
        "collision_check_time": metadata.get(
            "collision_check_time", np.nan
        ),
        "initial_alpha": initial_alpha,
        "alpha_drop_from_default": (
            0.5 - initial_alpha if np.isfinite(initial_alpha) else np.nan
        ),
        "final_alpha": final_alpha,
        "min_alpha": min_alpha,
        "max_alpha": max_alpha,
        "alpha_recovery": alpha_recovery,
        # Backward-compatible legacy columns.
        "prior_failure_mode": legacy_prior_failure_mode,
        "prior_failure_params": legacy_prior_failure_params,

        # Explicitly named legacy columns for new analysis scripts.
        "legacy_prior_failure_mode": legacy_prior_failure_mode,
        "legacy_prior_failure_params": legacy_prior_failure_params,

        # Recommended unified prior-mode field for analysis.
        "effective_prior_mode": effective_prior_mode,
        
        "use_connectivity_alpha_ceiling": metadata.get(
            "use_connectivity_alpha_ceiling", np.nan
        ),
        "connectivity_alpha_ceiling": metadata.get(
            "connectivity_alpha_ceiling", np.nan
        ),
        "use_connectivity_alpha_init": metadata.get(
            "use_connectivity_alpha_init", np.nan
        ),
        "connectivity_alpha_init": metadata.get(
            "connectivity_alpha_init", np.nan
        ),
        "use_topk_edge_checking": metadata.get(
            "use_topk_edge_checking", np.nan
        ),
        "online_reliability_update_enabled": metadata.get(
            "online_reliability_update_enabled", np.nan
        ),
        "parent_edge_strategy": metadata.get(
            "parent_edge_strategy", ""
        ),
        "rewire_edge_strategy": metadata.get(
            "rewire_edge_strategy", ""
        ),
        "base_sampler_mode": metadata.get(
            "base_sampler_mode", np.nan
        ),
        "uses_informed_base_after_initial": metadata.get(
            "uses_informed_base_after_initial", np.nan
        ),
        "is_rag_uniform_base_ablation": metadata.get(
            "is_rag_uniform_base_ablation", np.nan
        ),
        "conninit_ablation_mode": (
            "no_conninit"
            if planner_name in NO_CONNINIT_PLANNERS
            else ("conninit" if str(planner_name).startswith("rag_") else "")
        ),
        "uses_connectivity_alpha_init_effective": (
            0 if planner_name in NO_CONNINIT_PLANNERS
            else metadata.get("use_connectivity_alpha_init", np.nan)
        ),
        "is_rag_no_conninit_ablation": (
            1 if planner_name in NO_CONNINIT_PLANNERS
            else (0 if str(planner_name).startswith("rag_") else np.nan)
        ),
        "bpc_adjustment_mode": (
            "relative_only"
            if planner_name in RELATIVE_ONLY_PLANNERS
            else (
                "no_ceiling"
                if planner_name == "rag_bpc_conn_init_no_ceiling"
                else ("full" if str(planner_name).startswith("rag_") else "")
            )
        ),
        "uses_bpc_diagnostics": (
            1 if str(planner_name).startswith("rag_") else np.nan
        ),
        "uses_bpc_adjustment": (
            0 if planner_name in RELATIVE_ONLY_PLANNERS
            else (1 if str(planner_name).startswith("rag_") else np.nan)
        ),
        "uses_bpc_alpha_ceiling": (
            0
            if planner_name in NO_BPC_ALPHA_CEILING_PLANNERS
            else (1 if str(planner_name).startswith("rag_") else np.nan)
        ),
        "uses_high_consistency_recovery": (
            0 if planner_name in RELATIVE_ONLY_PLANNERS
            else (1 if str(planner_name).startswith("rag_") else np.nan)
        ),
        "parent_check_top_k": metadata.get(
            "parent_check_top_k", np.nan
        ),
        "rewire_check_top_k": metadata.get(
            "rewire_check_top_k", np.nan
        ),
        "use_topk_fallback": metadata.get(
            "use_topk_fallback", np.nan
        ),
        "parent_fallback_check_k": metadata.get(
            "parent_fallback_check_k", np.nan
        ),
        "rewire_fallback_check_k": metadata.get(
            "rewire_fallback_check_k", np.nan
        ),
        "topk_parent_event_count": metadata.get(
            "topk_parent_event_count", np.nan
        ),
        "topk_parent_eligible_candidate_count": metadata.get(
            "topk_parent_eligible_candidate_count", np.nan
        ),
        "topk_parent_omitted_candidate_count": metadata.get(
            "topk_parent_omitted_candidate_count", np.nan
        ),
        "topk_parent_topk_checked_count": metadata.get(
            "topk_parent_topk_checked_count", np.nan
        ),
        "topk_parent_fallback_checked_count": metadata.get(
            "topk_parent_fallback_checked_count", np.nan
        ),
        "topk_parent_fallback_success_count": metadata.get(
            "topk_parent_fallback_success_count", np.nan
        ),
        "topk_parent_fallback_trigger_count": metadata.get(
            "topk_parent_fallback_trigger_count", np.nan
        ),
        "topk_parent_fallback_recovery_count": metadata.get(
            "topk_parent_fallback_recovery_count", np.nan
        ),
        "topk_rewire_event_count": metadata.get(
            "topk_rewire_event_count", np.nan
        ),
        "topk_rewire_eligible_candidate_count": metadata.get(
            "topk_rewire_eligible_candidate_count", np.nan
        ),
        "topk_rewire_omitted_candidate_count": metadata.get(
            "topk_rewire_omitted_candidate_count", np.nan
        ),
        "topk_rewire_topk_checked_count": metadata.get(
            "topk_rewire_topk_checked_count", np.nan
        ),
        "topk_rewire_fallback_checked_count": metadata.get(
            "topk_rewire_fallback_checked_count", np.nan
        ),
        "topk_rewire_fallback_success_count": metadata.get(
            "topk_rewire_fallback_success_count", np.nan
        ),
        "topk_rewire_fallback_trigger_count": metadata.get(
            "topk_rewire_fallback_trigger_count", np.nan
        ),
        "topk_rewire_fallback_recovery_count": metadata.get(
            "topk_rewire_fallback_recovery_count", np.nan
        ),
        "adaptive_rewire_initial_budget": metadata.get(
            "adaptive_rewire_initial_budget", np.nan
        ),
        "adaptive_rewire_final_budget": metadata.get(
            "adaptive_rewire_final_budget", np.nan
        ),
        "adaptive_rewire_min_budget": metadata.get(
            "adaptive_rewire_min_budget", np.nan
        ),
        "adaptive_rewire_max_budget": metadata.get(
            "adaptive_rewire_max_budget", np.nan
        ),
        "adaptive_rewire_mean_budget": metadata.get(
            "adaptive_rewire_mean_budget", np.nan
        ),
        "adaptive_rewire_truncated_event_count": metadata.get(
            "adaptive_rewire_truncated_event_count", np.nan
        ),
        "adaptive_rewire_budget_increase_count": metadata.get(
            "adaptive_rewire_budget_increase_count", np.nan
        ),
        "adaptive_rewire_budget_decrease_count": metadata.get(
            "adaptive_rewire_budget_decrease_count", np.nan
        ),
        "adaptive_rewire_primary_realized_gain": metadata.get(
            "adaptive_rewire_primary_realized_gain", np.nan
        ),
        "adaptive_rewire_probe_realized_gain": metadata.get(
            "adaptive_rewire_probe_realized_gain", np.nan
        ),
        "adaptive_rewire_controller_version": metadata.get(
            "adaptive_rewire_controller_version", ""
        ),
        "adaptive_rewire_pre_solution_event_count": metadata.get(
            "adaptive_rewire_pre_solution_event_count", np.nan
        ),
        "adaptive_rewire_solution_aware_event_count": metadata.get(
            "adaptive_rewire_solution_aware_event_count", np.nan
        ),
        "adaptive_rewire_opportunity_event_count": metadata.get(
            "adaptive_rewire_opportunity_event_count", np.nan
        ),
        "adaptive_rewire_selected_opportunity_coverage_mean": metadata.get(
            "adaptive_rewire_selected_opportunity_coverage_mean", np.nan
        ),
        "adaptive_rewire_tail_opportunity_fraction_mean": metadata.get(
            "adaptive_rewire_tail_opportunity_fraction_mean", np.nan
        ),
        "adaptive_rewire_competitive_candidate_count_mean": metadata.get(
            "adaptive_rewire_competitive_candidate_count_mean", np.nan
        ),
        "adaptive_rewire_candidate_count_mean": metadata.get(
            "adaptive_rewire_candidate_count_mean", np.nan
        ),
        "adaptive_rewire_selected_opportunity_fraction_global": metadata.get(
            "adaptive_rewire_selected_opportunity_fraction_global", np.nan
        ),
        "adaptive_rewire_opportunity_coverage_target": metadata.get(
            "adaptive_rewire_opportunity_coverage_target", np.nan
        ),
        "adaptive_rewire_solution_slack_ratio": metadata.get(
            "adaptive_rewire_solution_slack_ratio", np.nan
        ),
        "adaptive_rewire_relevance_floor": metadata.get(
            "adaptive_rewire_relevance_floor", np.nan
        ),
        "adaptive_rewire_probe_interval": metadata.get(
            "adaptive_rewire_probe_interval", np.nan
        ),
        "adaptive_rewire_probe_k": metadata.get(
            "adaptive_rewire_probe_k", np.nan
        ),
        "adaptive_rewire_budget_4_event_count": metadata.get(
            "adaptive_rewire_budget_4_event_count", np.nan
        ),
        "adaptive_rewire_budget_8_event_count": metadata.get(
            "adaptive_rewire_budget_8_event_count", np.nan
        ),
        "adaptive_rewire_budget_12_event_count": metadata.get(
            "adaptive_rewire_budget_12_event_count", np.nan
        ),
        "adaptive_rewire_budget_20_event_count": metadata.get(
            "adaptive_rewire_budget_20_event_count", np.nan
        ),
        "topk_adaptive_rewire_event_count": metadata.get(
            "topk_adaptive_rewire_event_count", np.nan
        ),
        "topk_adaptive_rewire_eligible_candidate_count": metadata.get(
            "topk_adaptive_rewire_eligible_candidate_count", np.nan
        ),
        "topk_adaptive_rewire_primary_checked_count": metadata.get(
            "topk_adaptive_rewire_primary_checked_count", np.nan
        ),
        "topk_adaptive_rewire_primary_success_count": metadata.get(
            "topk_adaptive_rewire_primary_success_count", np.nan
        ),
        "topk_adaptive_rewire_probe_checked_count": metadata.get(
            "topk_adaptive_rewire_probe_checked_count", np.nan
        ),
        "topk_adaptive_rewire_probe_success_count": metadata.get(
            "topk_adaptive_rewire_probe_success_count", np.nan
        ),
        "topk_adaptive_rewire_omitted_candidate_count": metadata.get(
            "topk_adaptive_rewire_omitted_candidate_count", np.nan
        ),
        "topk_total_fallback_checked_count": (
            metadata.get("topk_parent_fallback_checked_count", 0)
            + metadata.get("topk_rewire_fallback_checked_count", 0)
        ),
        "topk_total_fallback_success_count": (
            metadata.get("topk_parent_fallback_success_count", 0)
            + metadata.get("topk_rewire_fallback_success_count", 0)
        ),
        "neural_guided_valid_heatmap": metadata.get(
            "neural_guided_valid_heatmap", np.nan
        ),
        "neural_guided_invalid_reason": metadata.get(
            "neural_guided_invalid_reason", ""
        ),
        "neural_guided_p_neural": metadata.get(
            "neural_guided_p_neural", np.nan
        ),
        "neural_guided_max_heatmap_attempts": metadata.get(
            "neural_guided_max_heatmap_attempts", np.nan
        ),
        "neural_guided_heatmap_threshold": metadata.get(
            "neural_guided_heatmap_threshold", np.nan
        ),
        "neural_guided_used_threshold_distribution": metadata.get(
            "neural_guided_used_threshold_distribution", np.nan
        ),
        "neural_guided_num_candidate_cells": metadata.get(
            "neural_guided_num_candidate_cells", np.nan
        ),
        "neural_guided_num_neural_attempts": metadata.get(
            "neural_guided_num_neural_attempts", np.nan
        ),
        "neural_guided_num_neural_successes": metadata.get(
            "neural_guided_num_neural_successes", np.nan
        ),
        "neural_guided_num_neural_fallbacks": metadata.get(
            "neural_guided_num_neural_fallbacks", np.nan
        ),
        "neural_guided_num_uniform_samples": metadata.get(
            "neural_guided_num_uniform_samples", np.nan
        ),
        "uses_prior_path": metadata.get(
            "uses_prior_path", np.nan
        ),
        "uses_conninit": metadata.get(
            "uses_conninit", np.nan
        ),
        "uses_bpc": metadata.get(
            "uses_bpc", np.nan
        ),
        # Best-path consistency diagnostics.
        "mean_best_path_consistency": mean_best_path_consistency,
        "final_best_path_consistency": final_best_path_consistency,
        "min_best_path_consistency": min_best_path_consistency,
        "max_best_path_consistency": max_best_path_consistency,
        "mean_distance_to_best_path": mean_distance_to_best_path,
        "final_distance_to_best_path": final_distance_to_best_path,
        "mean_alpha_ceiling": mean_alpha_ceiling,
        "final_alpha_ceiling": final_alpha_ceiling,
        "min_alpha_ceiling": min_alpha_ceiling,
        
        **{
            f"prior_conn_{field}": metadata.get(f"prior_conn_{field}", np.nan)
            for field in prior_conn_fields
        },
        
        "collision_checks_per_vertex": (
            result.collision_checks / max(result.num_vertices, 1)
        ),
        
        "collision_checks_per_success": (
            result.collision_checks if result.success else np.inf
        ),
        
        "collision_checks_until_initial": collision_checks_until_initial,
        "collision_checks_after_initial": collision_checks_after_initial,
        "collision_checks_per_first_solution": (
            collision_checks_until_initial
            if result.success and np.isfinite(collision_checks_until_initial)
            else np.inf
        ),
        "collision_checks_after_initial_ratio": (
            collision_checks_after_initial / result.collision_checks
            if result.collision_checks > 0
            and np.isfinite(collision_checks_after_initial)
            else np.nan
        ),
    }


def summarize_records(records):
    collision_checks_until_initial = [
        r["collision_checks_until_initial"] for r in records
    ]
    collision_checks_after_initial = [
        r["collision_checks_after_initial"] for r in records
    ]
    collision_checks_per_first_solution = [
        r["collision_checks_per_first_solution"] for r in records
    ]
    collision_checks_after_initial_ratio = [
        r["collision_checks_after_initial_ratio"] for r in records
    ]
    
    use_topk_edge_checking = [
        r["use_topk_edge_checking"] for r in records
    ]
    online_reliability_update_enabled = [
        r["online_reliability_update_enabled"] for r in records
    ]
    uses_informed_base_after_initial = [
        r["uses_informed_base_after_initial"] for r in records
    ]
    is_rag_uniform_base_ablation = [
        r["is_rag_uniform_base_ablation"] for r in records
    ]
    uses_bpc_diagnostics = [
        r["uses_bpc_diagnostics"] for r in records
    ]
    uses_bpc_adjustment = [
        r["uses_bpc_adjustment"] for r in records
    ]
    uses_bpc_alpha_ceiling = [
        r["uses_bpc_alpha_ceiling"] for r in records
    ]
    uses_high_consistency_recovery = [
        r["uses_high_consistency_recovery"] for r in records
    ]
    parent_check_top_k = [
        r["parent_check_top_k"] for r in records
    ]
    rewire_check_top_k = [
        r["rewire_check_top_k"] for r in records
    ]
    use_topk_fallback = [
        r["use_topk_fallback"] for r in records
    ]
    parent_fallback_check_k = [
        r["parent_fallback_check_k"] for r in records
    ]
    rewire_fallback_check_k = [
        r["rewire_fallback_check_k"] for r in records
    ]
    topk_total_fallback_checked_count = [
        r["topk_total_fallback_checked_count"] for r in records
    ]
    topk_total_fallback_success_count = [
        r["topk_total_fallback_success_count"] for r in records
    ]
    topk_parent_fallback_checked_count = [
        r["topk_parent_fallback_checked_count"] for r in records
    ]
    topk_rewire_fallback_checked_count = [
        r["topk_rewire_fallback_checked_count"] for r in records
    ]

    neural_guided_valid_heatmap = [
        r["neural_guided_valid_heatmap"] for r in records
    ]
    neural_guided_p_neural = [
        r["neural_guided_p_neural"] for r in records
    ]
    neural_guided_num_candidate_cells = [
        r["neural_guided_num_candidate_cells"] for r in records
    ]
    neural_guided_num_neural_attempts = [
        r["neural_guided_num_neural_attempts"] for r in records
    ]
    neural_guided_num_neural_successes = [
        r["neural_guided_num_neural_successes"] for r in records
    ]
    neural_guided_num_neural_fallbacks = [
        r["neural_guided_num_neural_fallbacks"] for r in records
    ]
    neural_guided_num_uniform_samples = [
        r["neural_guided_num_uniform_samples"] for r in records
    ]
    uses_prior_path = [r["uses_prior_path"] for r in records]
    uses_conninit = [r["uses_conninit"] for r in records]
    uses_bpc = [r["uses_bpc"] for r in records]
    
    successes = [r["success"] for r in records]
    external_prior_success = [r["external_prior_success"] for r in records]
    external_prior_has_path = [r["external_prior_has_path"] for r in records]
    external_prior_has_heatmap = [r["external_prior_has_heatmap"] for r in records]
    external_prior_used_fallback = [r["external_prior_used_fallback"] for r in records]
    external_prior_num_points = [r["external_prior_num_points"] for r in records]
    first_iters = [r["first_solution_iter"] for r in records]
    first_costs = [r["first_solution_cost"] for r in records]
    final_costs = [r["final_cost"] for r in records]
    times = [r["planning_time"] for r in records]
    vertices = [r["num_vertices"] for r in records]
    collision_checks = [r["collision_checks"] for r in records]

    initial_alpha = [r["initial_alpha"] for r in records]
    alpha_drop_from_default = [r["alpha_drop_from_default"] for r in records]
    final_alpha = [r["final_alpha"] for r in records]
    min_alpha = [r["min_alpha"] for r in records]
    max_alpha = [r["max_alpha"] for r in records]
    alpha_recovery = [r["alpha_recovery"] for r in records]

    mean_best_path_consistency = [
        r["mean_best_path_consistency"] for r in records
    ]
    final_best_path_consistency = [
        r["final_best_path_consistency"] for r in records
    ]
    min_best_path_consistency = [
        r["min_best_path_consistency"] for r in records
    ]
    max_best_path_consistency = [
        r["max_best_path_consistency"] for r in records
    ]
    mean_distance_to_best_path = [
        r["mean_distance_to_best_path"] for r in records
    ]
    final_distance_to_best_path = [
        r["final_distance_to_best_path"] for r in records
    ]
    mean_alpha_ceiling = [
        r["mean_alpha_ceiling"] for r in records
    ]
    final_alpha_ceiling = [
        r["final_alpha_ceiling"] for r in records
    ]
    min_alpha_ceiling = [
        r["min_alpha_ceiling"] for r in records
    ]
    
    connectivity_alpha_ceiling = [
        r["connectivity_alpha_ceiling"] for r in records
    ]
    connectivity_alpha_init = [
        r["connectivity_alpha_init"] for r in records
    ]
    
    collision_checks_per_vertex = [
        r["collision_checks_per_vertex"] for r in records
    ]
    collision_checks_per_success = [
        r["collision_checks_per_success"] for r in records
    ]
    
    
    prior_conn_fields = [
        "prior_conn_prior_valid_ratio",
        "prior_conn_prior_edge_valid_ratio",
        "prior_conn_num_components",
        "prior_conn_largest_component_ratio",
        "prior_conn_start_reachable_ratio",
        "prior_conn_goal_reachable_from_start",
        "prior_conn_frontier_progress",
        "prior_conn_start_near_prior",
        "prior_conn_goal_near_prior",
        "prior_conn_connectivity_score",
    ]
    
    prior_conn_values = {
        field: [r[field] for r in records]
        for field in prior_conn_fields
    }

    return {
        "num_runs": len(records),
        "success_rate": float(np.mean(successes)),
        "mean_external_prior_success": safe_mean(external_prior_success),
        "mean_external_prior_has_path": safe_mean(external_prior_has_path),
        "mean_external_prior_has_heatmap": safe_mean(external_prior_has_heatmap),
        "mean_external_prior_used_fallback": safe_mean(external_prior_used_fallback),
        "mean_external_prior_num_points": safe_mean(external_prior_num_points),
        "mean_first_solution_iter": safe_mean(first_iters),
        "std_first_solution_iter": safe_std(first_iters),
        "mean_first_solution_cost": safe_mean(first_costs),
        "mean_final_cost": safe_mean(final_costs),
        "std_final_cost": safe_std(final_costs),
        "mean_planning_time": safe_mean(times),
        "mean_num_vertices": safe_mean(vertices),
        "mean_collision_checks": safe_mean(collision_checks),
        "mean_initial_alpha": safe_mean(initial_alpha),
        "mean_alpha_drop_from_default": safe_mean(alpha_drop_from_default),
        "mean_final_alpha": safe_mean(final_alpha),
        "mean_min_alpha": safe_mean(min_alpha),
        "mean_max_alpha": safe_mean(max_alpha),
        "alpha_recovery_rate": float(np.mean(alpha_recovery)),
        # Best-path consistency summary.
        "mean_best_path_consistency": safe_mean(mean_best_path_consistency),
        "mean_final_best_path_consistency": safe_mean(final_best_path_consistency),
        "mean_min_best_path_consistency": safe_mean(min_best_path_consistency),
        "mean_max_best_path_consistency": safe_mean(max_best_path_consistency),
        "mean_distance_to_best_path": safe_mean(mean_distance_to_best_path),
        "mean_final_distance_to_best_path": safe_mean(final_distance_to_best_path),
        "mean_alpha_ceiling": safe_mean(mean_alpha_ceiling),
        "mean_final_alpha_ceiling": safe_mean(final_alpha_ceiling),
        "mean_min_alpha_ceiling": safe_mean(min_alpha_ceiling),
        
        "mean_connectivity_alpha_ceiling": safe_mean(
            connectivity_alpha_ceiling
        ),
        "mean_connectivity_alpha_init": safe_mean(connectivity_alpha_init),
        "num_failures": int(len(records) - np.sum(successes)),
        
        "mean_collision_checks_per_vertex": safe_mean(collision_checks_per_vertex),
        "mean_collision_checks_per_success": safe_mean(collision_checks_per_success),
        
        "mean_use_topk_edge_checking": safe_mean(use_topk_edge_checking),
        "mean_online_reliability_update_enabled": safe_mean(
            online_reliability_update_enabled
        ),
        "mean_uses_informed_base_after_initial": safe_mean(
            uses_informed_base_after_initial
        ),
        "mean_is_rag_uniform_base_ablation": safe_mean(
            is_rag_uniform_base_ablation
        ),
        "mean_uses_bpc_diagnostics": safe_mean(uses_bpc_diagnostics),
        "mean_uses_bpc_adjustment": safe_mean(uses_bpc_adjustment),
        "mean_uses_bpc_alpha_ceiling": safe_mean(uses_bpc_alpha_ceiling),
        "mean_uses_high_consistency_recovery": safe_mean(uses_high_consistency_recovery),
        "mean_parent_check_top_k": safe_mean(parent_check_top_k),
        "mean_rewire_check_top_k": safe_mean(rewire_check_top_k),
        "mean_use_topk_fallback": safe_mean(use_topk_fallback),
        "mean_parent_fallback_check_k": safe_mean(parent_fallback_check_k),
        "mean_rewire_fallback_check_k": safe_mean(rewire_fallback_check_k),
        "mean_topk_total_fallback_checked_count": safe_mean(
            topk_total_fallback_checked_count
        ),
        "mean_topk_total_fallback_success_count": safe_mean(
            topk_total_fallback_success_count
        ),
        "mean_topk_parent_fallback_checked_count": safe_mean(
            topk_parent_fallback_checked_count
        ),
        "mean_topk_rewire_fallback_checked_count": safe_mean(
            topk_rewire_fallback_checked_count
        ),
        "mean_neural_guided_valid_heatmap": safe_mean(
            neural_guided_valid_heatmap
        ),
        "mean_neural_guided_p_neural": safe_mean(neural_guided_p_neural),
        "mean_neural_guided_num_candidate_cells": safe_mean(
            neural_guided_num_candidate_cells
        ),
        "mean_neural_guided_num_neural_attempts": safe_mean(
            neural_guided_num_neural_attempts
        ),
        "mean_neural_guided_num_neural_successes": safe_mean(
            neural_guided_num_neural_successes
        ),
        "mean_neural_guided_num_neural_fallbacks": safe_mean(
            neural_guided_num_neural_fallbacks
        ),
        "mean_neural_guided_num_uniform_samples": safe_mean(
            neural_guided_num_uniform_samples
        ),
        "mean_uses_prior_path": safe_mean(uses_prior_path),
        "mean_uses_conninit": safe_mean(uses_conninit),
        "mean_uses_bpc": safe_mean(uses_bpc),
        
        **{
            f"mean_{field}": safe_mean(values)
            for field, values in prior_conn_values.items()
        },
        
        "mean_collision_checks_until_initial": safe_mean(
            collision_checks_until_initial
        ),
        "mean_collision_checks_after_initial": safe_mean(
            collision_checks_after_initial
        ),
        "mean_collision_checks_per_first_solution": safe_mean(
            collision_checks_per_first_solution
        ),
        "mean_collision_checks_after_initial_ratio": safe_mean(
            collision_checks_after_initial_ratio
        ),
    }


def write_csv(filepath, rows):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(filepath, result):
    histories = {
        "alpha": getattr(result, "alpha_history", []),
        "reliability": getattr(result, "reliability_history", []),
        "adjusted_reliability": result.metadata.get(
            "adjusted_reliability_history", []
        ) if hasattr(result, "metadata") else [],
        "guided_score": result.metadata.get(
            "guided_score_history", []
        ) if hasattr(result, "metadata") else [],
        "base_score": result.metadata.get(
            "base_score_history", []
        ) if hasattr(result, "metadata") else [],
        "best_path_consistency": getattr(
            result, "best_path_consistency_rate_history", []
        ),
        "mean_distance_to_best_path": getattr(
            result, "mean_distance_to_best_path_history", []
        ),
        "alpha_ceiling": getattr(result, "alpha_ceiling_history", []),
        
        "pre_solution_alpha_ceiling": result.metadata.get(
            "pre_solution_alpha_ceiling_history", []
        ) if hasattr(result, "metadata") else [],
    }

    max_len = max((len(v) for v in histories.values()), default=0)
    rows = []

    for i in range(max_len):
        row = {"update_index": i}
        for key, values in histories.items():
            row[key] = values[i] if i < len(values) else np.nan
        rows.append(row)

    write_csv(filepath, rows)



def write_trajectory_csv(filepath, result):
    if not hasattr(result, "metadata"):
        return

    trajectory = result.metadata.get("iteration_history", [])

    if trajectory is None or len(trajectory) == 0:
        return

    write_csv(filepath, trajectory)


def print_summary(row):
    print(f"success_rate: {row['success_rate']:.3f}")
    print(f"num_failures: {row['num_failures']}")
    print(f"mean_first_solution_iter: {row['mean_first_solution_iter']:.2f}")
    print(f"mean_final_cost: {row['mean_final_cost']:.2f}")
    print(f"mean_planning_time: {row['mean_planning_time']:.3f}")

    if row["planner"] in [
        "reliability_guided",
        "rag_bpc",
        "rag_bpc_conn",
        "rag_bpc_conn_init",
        "rag_bpc_conn_init_rel_only",
        "rag_bpc_conn_init_rel_only_topk_fallback",
        "rag_rel_fixed_05",
        "rag_rel_conninit_only",
        "rag_rel_online_only",
        "rag_bpc_conn_init_no_ceiling",
        "rag_bpc_no_conn_init",
        "rag_bpc_conn_init_topk",
        "rag_bpc_conn_init_topk_fallback",
        "rag_bpc_conn_init_topk_fallback_4_8",
        "rag_bpc_conn_init_topk_fallback_12_20",
        "rag_bpc_conn_init_topk_4_8",
        "rag_bpc_conn_init_topk_8_12",
        "rag_bpc_conn_init_topk_12_20",
    ] or row["planner"] in EDGE_STRATEGY_PLANNERS:
        print(f"mean_initial_alpha: {row['mean_initial_alpha']:.3f}")
        print(
            "mean_alpha_drop_from_default: "
            f"{row['mean_alpha_drop_from_default']:.3f}"
        )
        print(f"mean_final_alpha: {row['mean_final_alpha']:.3f}")
        print(f"mean_min_alpha: {row['mean_min_alpha']:.3f}")
        print(f"mean_max_alpha: {row['mean_max_alpha']:.3f}")
        print(f"alpha_recovery_rate: {row['alpha_recovery_rate']:.3f}")

        print(
            "mean_best_path_consistency: "
            f"{row['mean_best_path_consistency']:.3f}"
        )
        print(
            "mean_final_best_path_consistency: "
            f"{row['mean_final_best_path_consistency']:.3f}"
        )
        print(
            "mean_final_alpha_ceiling: "
            f"{row['mean_final_alpha_ceiling']:.3f}"
        )
        print(
            "mean_connectivity_alpha_ceiling: "
            f"{row['mean_connectivity_alpha_ceiling']:.3f}"
        )
        print(
            "mean_connectivity_alpha_init: "
            f"{row['mean_connectivity_alpha_init']:.3f}"
        )
        print(
            "mean_prior_conn_connectivity_score: "
            f"{row['mean_prior_conn_connectivity_score']:.3f}"
        )

    if row["planner"] == "neural_guided_rrt_star":
        print(
            "mean_neural_guided_valid_heatmap: "
            f"{row['mean_neural_guided_valid_heatmap']:.3f}"
        )
        print(
            "mean_neural_guided_p_neural: "
            f"{row['mean_neural_guided_p_neural']:.3f}"
        )
        print(
            "mean_neural_guided_num_candidate_cells: "
            f"{row['mean_neural_guided_num_candidate_cells']:.2f}"
        )


def get_prior_config_for_experiment(prior, map_name):
    """
    Return legacy sampler prior configuration.

    learned_heatmap is an external prior source label and should not trigger
    legacy path corruption. The learned prior itself is generated by
    LearnedPriorGenerator.
    """
    if prior == "learned_heatmap":
        return None, "none", {}
    return get_complex_prior_config(prior=prior, map_name=map_name)


def run_single_experiment(task):
    """
        Run one independent planner trial.

        This function is executed in a worker process. It must be top-level
        so that multiprocessing can pickle it.
    """
    (
        map_name,
        prior,
        planner_name,
        seed,
        iter_max,
        iter_after_initial,
        clearance,
        output_dir_str,
        prior_generator,
        prior_generator_config,
    ) = task

    try:
        # Make each worker run reproducible.
        np.random.seed(seed)
        random.seed(seed)

        output_dir = Path(output_dir_str)

        path_shift, prior_failure_mode, prior_failure_params = get_prior_config_for_experiment(
            prior=prior,
            map_name=map_name,
        )
        
        if prior in ["broken_middle", "endpoint_mismatch"]:
            if prior_failure_mode != prior:
                raise ValueError(
                    f"Prior config mismatch: "
                    f"prior={prior}, map={map_name}, "
                    f"prior_failure_mode={prior_failure_mode}, "
                    f"prior_failure_params={prior_failure_params}"
                )

        binary_mask, start, goal = make_problem(map_name)
        grid_map = GridMap2D(binary_mask, clearance=clearance)

        external_prior_result = make_external_prior_result(
            prior_generator_name=prior_generator,
            grid_map=grid_map,
            start=start,
            goal=goal,
            prior=prior,
            map_name=map_name,
            prior_generator_config=prior_generator_config,
        )

        # Path-guided planners consume PriorResult.path.
        # The NRRT*-style baseline consumes PriorResult.heatmap directly and
        # intentionally ignores PriorResult.path.
        if is_path_guided_planner(planner_name):
            planner_prior_result = (
                external_prior_result
                if prior_result_has_usable_path(external_prior_result)
                else None
            )
        elif is_neural_guided_planner(planner_name):
            planner_prior_result = (
                external_prior_result
                if prior_result_has_usable_heatmap(external_prior_result)
                else None
            )
        else:
            planner_prior_result = None

        planner = build_planner(
            planner_name=planner_name,
            grid_map=grid_map,
            start=start,
            goal=goal,
            iter_max=iter_max,
            path_shift=path_shift,
            prior_failure_mode=prior_failure_mode,      
            prior_failure_params=prior_failure_params,
            prior_result=planner_prior_result,
        )

        result = planner.solve(
            iter_after_initial=iter_after_initial
        )

        row = extract_run_record(
            map_name=map_name,
            prior=prior,
            planner_name=planner_name,
            seed=seed,
            result=result,
            prior_generator=prior_generator,
            prior_result=external_prior_result,
        )

        # Save per-run histories for reliability-guided variants.
        if planner_name in [
            "adaptive_guided",
            "reliability_guided",
            "rag_bpc",
            "rag_bpc_conn",
            "rag_bpc_conn_init",
            "rag_bpc_conn_init_uniform_base",
            "rag_bpc_conn_init_rel_only",
            "rag_bpc_conn_init_rel_only_topk_fallback",
            "rag_rel_fixed_05",
            "rag_rel_conninit_only",
            "rag_rel_online_only",
            "rag_bpc_conn_init_no_ceiling",
            "rag_bpc_no_conn_init",
            "rag_bpc_conn_init_topk",
            "rag_bpc_conn_init_topk_fallback",
            "rag_bpc_conn_init_topk_fallback_4_8",
            "rag_bpc_conn_init_topk_fallback_12_20",
            "rag_bpc_conn_init_topk_4_8",
            "rag_bpc_conn_init_topk_8_12",
            "rag_bpc_conn_init_topk_12_20",
        ] or planner_name in EDGE_STRATEGY_PLANNERS:
            history_dir = output_dir / "histories" / map_name / prior / planner_name
            history_path = history_dir / f"seed_{seed}.csv"
            write_history_csv(history_path, result)

            trajectory_dir = output_dir / "trajectories" / map_name / prior / planner_name
            trajectory_path = trajectory_dir / f"seed_{seed}.csv"
            write_trajectory_csv(trajectory_path, result)

        return {
            "ok": True,
            "row": row,
            "error": None,
            "traceback": None,
            "task": task,
        }

    except Exception as e:
        return {
            "ok": False,
            "row": None,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "task": task,
        }


# =========================
# Main experiment
# =========================
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_generator_config = {
        "checkpoint_path": args.learned_prior_checkpoint,
        "device": args.learned_prior_device,
        "heatmap_threshold": args.learned_prior_heatmap_threshold,
        "heatmap_cost_weight": args.learned_prior_heatmap_cost_weight,
        "allow_fallback": bool(args.learned_prior_allow_fallback),
    }
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "maps": args.maps,
        "priors": args.priors,
        "planners": args.planners,
        "prior_generator": args.prior_generator,
        "prior_generator_config": prior_generator_config,
        "num_runs": args.num_runs,
        "iter_max": args.iter_max,
        "iter_after_initial": args.iter_after_initial,
        "clearance": args.clearance,
        "num_workers": args.num_workers,
    }
    
    with open(output_dir / "config.txt", "w") as f:
        json.dump(config, f, indent=2)
    
    if args.save_map_previews:
        preview_dir = output_dir / "map_previews"

        for map_name in args.maps:
            binary_mask, start, goal = make_problem(map_name)
            save_path = save_map_preview(
                binary_mask=binary_mask,
                start=start,
                goal=goal,
                map_name=map_name,
                output_dir=preview_dir,
            )
            print(f"Saved map preview: {save_path}")

    planner_names = args.planners

    tasks = []

    for map_name in args.maps:
        for prior in args.priors:
            for planner_name in planner_names:
                for seed in range(args.num_runs):
                    tasks.append(
                        (
                            map_name,
                            prior,
                            planner_name,
                            seed,
                            args.iter_max,
                            args.iter_after_initial,
                            args.clearance,
                            str(output_dir),
                            args.prior_generator,
                            prior_generator_config,
                        )
                    )

    print(f"Total runs: {len(tasks)}")
    print(f"Using workers: {args.num_workers}")

    all_run_rows = []
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {
            executor.submit(run_single_experiment, task): task
            for task in tasks
        }

        completed = 0

        for future in as_completed(future_to_task):
            completed += 1
            result = future.result()

            if result["ok"]:
                row = result["row"]
                all_run_rows.append(row)

                print(
                    f"[{completed}/{len(tasks)}] "
                    f"{row['map']} | {row['prior']} | {row['planner']} | "
                    f"seed={row['seed']} | success={row['success']} | "
                    f"first_iter={row['first_solution_iter']} | "
                    f"final_cost={row['final_cost']}"
                )
            else:
                failed_tasks.append(result)
                task = result["task"]
                print(
                    f"[{completed}/{len(tasks)}] FAILED task={task}: "
                    f"{result['error']}"
                )

    if len(failed_tasks) > 0:
        error_path = output_dir / "failed_tasks.txt"
        with open(error_path, "w") as f:
            for item in failed_tasks:
                f.write("=" * 80 + "\n")
                f.write(f"task: {item['task']}\n")
                f.write(f"error: {item['error']}\n")
                f.write(item["traceback"])
                f.write("\n")

        print(f"\nSome tasks failed. See: {error_path}")

    # Sort rows for stable CSV output.
    all_run_rows.sort(
        key=lambda r: (
            r["map"],
            r["prior"],
            r["planner"],
            int(r["seed"]),
        )
    )

    # Build summary rows after all parallel tasks finish.
    summary_rows = []

    for map_name in args.maps:
        for prior in args.priors:
            print("\n" + "=" * 80)
            print(f"Map: {map_name} | Prior: {prior}")
            print("=" * 80)

            for planner_name in planner_names:
                records = [
                    r for r in all_run_rows
                    if r["map"] == map_name
                    and r["prior"] == prior
                    and r["planner"] == planner_name
                ]

                if len(records) == 0:
                    continue

                summary = summarize_records(records)
                _, legacy_prior_failure_mode, legacy_prior_failure_params = get_prior_config_for_experiment(
                    prior=prior,
                    map_name=map_name,
                )
                # These mode/source fields are copied from the first record in
                # the group because each summary group has one fixed
                # map-prior-planner setting.
                first_record = records[0]

                summary_row = {
                    "map": map_name,
                    "prior": prior,

                    # Requested command-line prior label.
                    "requested_prior_mode": prior,

                    # Source and effective identity of the prior actually used.
                    "prior_generator": first_record.get(
                        "prior_generator", args.prior_generator
                    ),
                    "guided_prior_source": first_record.get(
                        "guided_prior_source", np.nan
                    ),
                    "external_prior_type": first_record.get(
                        "external_prior_type", np.nan
                    ),
                    "external_prior_mode": first_record.get(
                        "external_prior_mode", np.nan
                    ),
                    "external_prior_has_heatmap": first_record.get(
                        "external_prior_has_heatmap", np.nan
                    ),
                    "external_prior_used_fallback": first_record.get(
                        "external_prior_used_fallback", np.nan
                    ),
                    "external_prior_path_extraction_method": first_record.get(
                        "external_prior_path_extraction_method", np.nan
                    ),
                    "neural_guided_valid_heatmap": first_record.get(
                        "neural_guided_valid_heatmap", np.nan
                    ),
                    "neural_guided_p_neural": first_record.get(
                        "neural_guided_p_neural", np.nan
                    ),
                    "neural_guided_heatmap_threshold": first_record.get(
                        "neural_guided_heatmap_threshold", np.nan
                    ),
                    "effective_prior_mode": first_record.get(
                        "effective_prior_mode", np.nan
                    ),
                    "base_sampler_mode": first_record.get(
                        "base_sampler_mode", np.nan
                    ),
                    "uses_informed_base_after_initial": first_record.get(
                        "uses_informed_base_after_initial", np.nan
                    ),
                    "is_rag_uniform_base_ablation": first_record.get(
                        "is_rag_uniform_base_ablation", np.nan
                    ),
                    "online_reliability_update_enabled": first_record.get(
                        "online_reliability_update_enabled", np.nan
                    ),
                    "parent_edge_strategy": first_record.get(
                        "parent_edge_strategy", ""
                    ),
                    "rewire_edge_strategy": first_record.get(
                        "rewire_edge_strategy", ""
                    ),
                    "conninit_ablation_mode": first_record.get(
                        "conninit_ablation_mode", ""
                    ),
                    "uses_connectivity_alpha_init_effective": first_record.get(
                        "uses_connectivity_alpha_init_effective", np.nan
                    ),
                    "is_rag_no_conninit_ablation": first_record.get(
                        "is_rag_no_conninit_ablation", np.nan
                    ),
                    "bpc_adjustment_mode": first_record.get(
                        "bpc_adjustment_mode", ""
                    ),
                    "uses_bpc_diagnostics": first_record.get(
                        "uses_bpc_diagnostics", np.nan
                    ),
                    "uses_bpc_adjustment": first_record.get(
                        "uses_bpc_adjustment", np.nan
                    ),
                    "uses_bpc_alpha_ceiling": first_record.get(
                        "uses_bpc_alpha_ceiling", np.nan
                    ),
                    "uses_high_consistency_recovery": first_record.get(
                        "uses_high_consistency_recovery", np.nan
                    ),

                    # Backward-compatible legacy columns.
                    "prior_failure_mode": legacy_prior_failure_mode,
                    "prior_failure_params": str(legacy_prior_failure_params),

                    # Explicitly named legacy fields for new analysis scripts.
                    "legacy_prior_failure_mode": legacy_prior_failure_mode,
                    "legacy_prior_failure_params": str(legacy_prior_failure_params),

                    "planner": planner_name,
                    **summary,
                }
                summary_rows.append(summary_row)

                print(f"\nPlanner: {planner_name}")
                print_summary(summary_row)

    write_csv(output_dir / "all_runs.csv", all_run_rows)
    write_csv(output_dir / "summary.csv", summary_rows)

    print("\nSaved:")
    print(output_dir / "all_runs.csv")
    print(output_dir / "summary.csv")


if __name__ == "__main__":
    main()
