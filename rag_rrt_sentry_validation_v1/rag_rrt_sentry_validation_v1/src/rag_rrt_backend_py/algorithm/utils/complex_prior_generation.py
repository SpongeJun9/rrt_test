# utils/complex_prior_generation.py
"""
Prior configuration utilities for Complex Simulation Environment Evaluation.

This module does not generate a new prior path directly. Instead, it returns
the existing parameters already supported by AStarCorridorSampler:

    path_shift
    prior_failure_mode
    prior_failure_params

This keeps the second-layer experiment compatible with the current
ReliabilityGuidedRRTStar2D implementation.
"""


def get_endpoint_mismatch_shift(map_name):
    """
    Shifts used for endpoint_mismatch.

    Semantic role:
        Move the prior away from start / goal proximity, making the prior
        structurally unreliable for the current start-goal task.

    These values should be checked with map previews and smoke tests.
    """
    if map_name == "multi_room":
        return (16.0, 16.0)

    if map_name == "office_corridor":
        return (0.0, 24.0)

    if map_name == "warehouse_clutter":
        return (18.0, 12.0)

    if map_name == "urban_blocks":
        return (20.0, 20.0)

    if map_name == "hybrid_trap":
        return (0.0, 28.0)

    raise ValueError(f"Unknown map_name for endpoint_mismatch: {map_name}")


def get_strong_shift(map_name):
    """
    Shifts used for strong_shift.

    Semantic role:
        Move the prior toward nearby but potentially wrong corridors, rooms,
        streets, or trap structures.

    This is intentionally different from endpoint_mismatch:
        - endpoint_mismatch focuses on start/goal mismatch.
        - strong_shift focuses on topological or corridor-level displacement.
    """
    if map_name == "multi_room":
        return (0.0, -22.0)

    if map_name == "office_corridor":
        return (18.0, 0.0)

    if map_name == "warehouse_clutter":
        return (0.0, 22.0)

    if map_name == "urban_blocks":
        return (-22.0, 0.0)

    if map_name == "hybrid_trap":
        return (20.0, -20.0)

    raise ValueError(f"Unknown map_name for strong_shift: {map_name}")


def get_complex_prior_config(prior, map_name):
    """
    Return:
        path_shift, prior_failure_mode, prior_failure_params

    Supported prior settings:
        coarse_valid
            Original A* corridor prior.

        broken_middle
            Remove the middle part of the transformed prior while keeping
            start-side and goal-side fragments.

        endpoint_mismatch
            Shift the transformed prior away from the current start-goal task.

        strong_shift
            Shift the whole A* prior toward nearby wrong structure or corridor.
    """
    if prior == "coarse_valid":
        return None, "none", {}

    if prior == "broken_middle":
        return None, "broken_middle", {
            "keep_ratio_each_side": 0.30,
        }

    if prior == "endpoint_mismatch":
        return None, "endpoint_mismatch", {
            "shift": get_endpoint_mismatch_shift(map_name),
        }

    if prior == "strong_shift":
        return get_strong_shift(map_name), "none", {}

    raise ValueError(
        f"Unknown complex prior setting: prior={prior}, map={map_name}"
    )