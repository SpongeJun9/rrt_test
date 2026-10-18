# envs/complex_maps_2d.py
"""
Complex 2D simulation maps for the second-stage evaluation.

These maps are designed for:
    Complex Simulation Environment Evaluation

Map list:
    multi_room
    office_corridor
    warehouse_clutter
    urban_blocks
    hybrid_trap

Conventions:
    binary_mask[y, x] = 1  -> free space
    binary_mask[y, x] = 0  -> obstacle

All maps use size 120 x 120 by default to remain compatible with the
existing planner parameters:
    step_len = 5.0
    search_radius = 15.0
    clearance = 2
"""

import numpy as np


def _add_rect(binary_mask, y0, y1, x0, x1):
    """
    Set a rectangular obstacle region.

    Args:
        binary_mask: 2D uint8 map.
        y0, y1, x0, x1: rectangle bounds.
    """
    h, w = binary_mask.shape
    y0 = max(0, int(y0))
    y1 = min(h, int(y1))
    x0 = max(0, int(x0))
    x1 = min(w, int(x1))

    if y0 < y1 and x0 < x1:
        binary_mask[y0:y1, x0:x1] = 0


def _open_rect(binary_mask, y0, y1, x0, x1):
    """
    Open a rectangular free-space region.
    """
    h, w = binary_mask.shape
    y0 = max(0, int(y0))
    y1 = min(h, int(y1))
    x0 = max(0, int(x0))
    x1 = min(w, int(x1))

    if y0 < y1 and x0 < x1:
        binary_mask[y0:y1, x0:x1] = 1


def _ensure_start_goal_free(binary_mask, start, goal, radius=4):
    """
    Make small free-space patches around start and goal.

    This is a safety operation for clearance-based validity.
    It does not create a global path by itself.
    """
    for p in [start, goal]:
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))

        _open_rect(
            binary_mask,
            y - radius,
            y + radius + 1,
            x - radius,
            x + radius + 1,
        )


def make_multi_room_problem():
    """
    Multi-room indoor-like environment.

    Structure:
        - Four large rooms.
        - Walls with door openings.
        - A few internal blocks create local detours.

    Intended stress:
        - Room-to-room transitions.
        - Doorway-level connectivity.
        - Prior degradation near room boundaries.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    wall = 4

    # Outer boundary.
    _add_rect(binary_mask, 0, wall, 0, width)
    _add_rect(binary_mask, height - wall, height, 0, width)
    _add_rect(binary_mask, 0, height, 0, wall)
    _add_rect(binary_mask, 0, height, width - wall, width)

    # Main room-separating walls.
    _add_rect(binary_mask, 10, 110, 58, 62)
    _add_rect(binary_mask, 58, 62, 10, 110)

    # Door openings in the central vertical wall.
    _open_rect(binary_mask, 22, 36, 58, 62)
    _open_rect(binary_mask, 82, 96, 58, 62)

    # Door openings in the central horizontal wall.
    _open_rect(binary_mask, 58, 62, 20, 34)
    _open_rect(binary_mask, 58, 62, 84, 98)

    # Local furniture / internal obstacles.
    obstacles = [
        (18, 34, 22, 34),
        (38, 50, 42, 52),
        (72, 92, 18, 30),
        (78, 90, 38, 50),
        (18, 35, 78, 90),
        (38, 50, 96, 106),
        (72, 86, 76, 90),
        (94, 106, 96, 108),
    ]

    for y0, y1, x0, x1 in obstacles:
        _add_rect(binary_mask, y0, y1, x0, x1)

    start = np.array([14, 104], dtype=float)
    goal = np.array([106, 14], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal)

    return binary_mask, start, goal


def make_office_corridor_problem():
    """
    Office corridor network.

    Structure:
        - Long horizontal and vertical corridors.
        - Multiple side rooms.
        - Branches and local dead-end-like regions.

    Intended stress:
        - Wrong branch guidance.
        - Shifted prior entering nearby corridor.
        - BPC correction after feasible solution.
    """
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Main corridors.
    _open_rect(binary_mask, 52, 68, 8, 112)     # horizontal corridor
    _open_rect(binary_mask, 8, 112, 52, 68)     # vertical corridor

    # Upper and lower side corridors.
    _open_rect(binary_mask, 18, 30, 18, 102)
    _open_rect(binary_mask, 90, 102, 18, 102)

    # Connector corridors.
    _open_rect(binary_mask, 18, 68, 20, 32)
    _open_rect(binary_mask, 18, 68, 88, 100)
    _open_rect(binary_mask, 52, 102, 20, 32)
    _open_rect(binary_mask, 52, 102, 88, 100)

    # Office rooms connected to corridors.
    rooms = [
        (8, 18, 12, 34),
        (8, 18, 42, 62),
        (8, 18, 76, 108),
        (30, 48, 12, 34),
        (30, 48, 76, 108),
        (72, 88, 12, 34),
        (72, 88, 76, 108),
        (102, 114, 12, 42),
        (102, 114, 78, 108),
    ]

    for y0, y1, x0, x1 in rooms:
        _open_rect(binary_mask, y0, y1, x0, x1)

    # Add a few internal blocks inside large rooms.
    obstacles = [
        (34, 44, 20, 28),
        (34, 44, 86, 94),
        (76, 84, 20, 28),
        (76, 84, 86, 94),
    ]
    for y0, y1, x0, x1 in obstacles:
        _add_rect(binary_mask, y0, y1, x0, x1)

    start = np.array([14, 60], dtype=float)
    goal = np.array([106, 24], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal)

    return binary_mask, start, goal


def make_warehouse_clutter_problem():
    """
    Warehouse-like clutter map.

    Structure:
        - Long shelf-like obstacles.
        - Multiple narrow but feasible aisles.
        - Some cross connections.

    Intended stress:
        - High obstacle density.
        - Shifted prior overlapping shelves.
        - Collision-heavy misleading guidance.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    # Outer boundary.
    _add_rect(binary_mask, 0, 4, 0, width)
    _add_rect(binary_mask, height - 4, height, 0, width)
    _add_rect(binary_mask, 0, height, 0, 4)
    _add_rect(binary_mask, 0, height, width - 4, width)

    # Shelf rows. Leave aisles between shelves.
    shelf_rects = [
        (14, 44, 18, 28),
        (56, 104, 18, 28),

        (14, 34, 40, 50),
        (46, 104, 40, 50),

        (14, 62, 62, 72),
        (76, 104, 62, 72),

        (14, 44, 84, 94),
        (56, 104, 84, 94),
    ]

    for y0, y1, x0, x1 in shelf_rects:
        _add_rect(binary_mask, y0, y1, x0, x1)

    # Local clutter blocks.
    clutter = [
        (20, 30, 104, 112),
        (42, 52, 6, 14),
        (70, 82, 104, 112),
        (96, 108, 6, 14),
        (8, 18, 52, 58),
        (102, 112, 52, 58),
    ]

    for y0, y1, x0, x1 in clutter:
        _add_rect(binary_mask, y0, y1, x0, x1)

    start = np.array([10, 108], dtype=float)
    goal = np.array([110, 12], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal)

    return binary_mask, start, goal


def make_urban_blocks_problem():
    """
    Urban-block / street-like map.

    Structure:
        - Block-shaped obstacles.
        - Grid-like roads.
        - Several intersections and alternative routes.

    Intended stress:
        - Shifted priors may remain collision-free but move to nearby streets.
        - Topological ambiguity.
        - Strong test for connectivity vs BPC complementarity.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    # Urban blocks.
    blocks = [
        (12, 34, 12, 34),
        (12, 34, 46, 70),
        (12, 34, 82, 108),

        (46, 70, 12, 34),
        (46, 70, 46, 70),
        (46, 70, 82, 108),

        (82, 108, 12, 34),
        (82, 108, 46, 70),
        (82, 108, 82, 108),
    ]

    for y0, y1, x0, x1 in blocks:
        _add_rect(binary_mask, y0, y1, x0, x1)

    # A few road closures / construction blocks.
    closures = [
        (36, 44, 70, 82),
        (70, 82, 36, 44),
        (36, 44, 4, 12),
    ]

    for y0, y1, x0, x1 in closures:
        _add_rect(binary_mask, y0, y1, x0, x1)

    start = np.array([8, 112], dtype=float)
    goal = np.array([112, 8], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal)

    return binary_mask, start, goal


def make_hybrid_trap_problem():
    """
    Hybrid trap map.

    Structure:
        - Combines a false-opening-like region, narrow passage, clutter,
          and a detour requirement.
        - Designed as a second-layer stress case.

    Intended stress:
        - Prior can be locally plausible but globally misleading.
        - Tests both pre-solution structural reliability and post-solution BPC.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    # Outer boundary.
    _add_rect(binary_mask, 0, 4, 0, width)
    _add_rect(binary_mask, height - 4, height, 0, width)
    _add_rect(binary_mask, 0, height, 0, 4)
    _add_rect(binary_mask, 0, height, width - 4, width)

    # Main vertical barrier with true and false openings.
    _add_rect(binary_mask, 8, 112, 56, 64)

    # True opening near upper-middle.
    _open_rect(binary_mask, 28, 42, 56, 64)

    # False opening near lower-middle.
    _open_rect(binary_mask, 76, 90, 56, 64)

    # Block behind false opening.
    _add_rect(binary_mask, 66, 104, 72, 80)

    # Narrow passage on the right side.
    _add_rect(binary_mask, 8, 28, 80, 104)
    _add_rect(binary_mask, 44, 72, 80, 104)
    _add_rect(binary_mask, 90, 112, 80, 104)

    # Small connector gaps on the right.
    _open_rect(binary_mask, 28, 44, 88, 96)
    _open_rect(binary_mask, 72, 90, 88, 96)

    # Left-side clutter.
    clutter = [
        (14, 30, 16, 28),
        (38, 54, 32, 44),
        (64, 78, 14, 28),
        (88, 104, 34, 48),
    ]

    for y0, y1, x0, x1 in clutter:
        _add_rect(binary_mask, y0, y1, x0, x1)

    start = np.array([12, 104], dtype=float)
    goal = np.array([108, 18], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal)

    return binary_mask, start, goal


def make_complex_problem(map_name):

    if map_name == "multi_room_hard_v3":
        return make_multi_room_hard_v3_problem()

    if map_name == "office_corridor_hard_v3":
        return make_office_corridor_hard_v3_problem()

    if map_name == "warehouse_clutter_hard_v3":
        return make_warehouse_clutter_hard_v3_problem()

    if map_name == "urban_blocks_hard_v3":
        return make_urban_blocks_hard_v3_problem()

    if map_name == "hybrid_trap_hard_v3":
        return make_hybrid_trap_hard_v3_problem()
    """
    Factory for complex simulation maps.
    """
    if map_name == "multi_room":
        return make_multi_room_problem()

    if map_name == "office_corridor":
        return make_office_corridor_problem()

    if map_name == "warehouse_clutter":
        return make_warehouse_clutter_problem()

    if map_name == "urban_blocks":
        return make_urban_blocks_problem()

    if map_name == "hybrid_trap":
        return make_hybrid_trap_problem()

    raise ValueError(f"Unknown complex map_name: {map_name}")

# === HARD_QUERY_BENCHMARK_EXTENSION_BEGIN ===
# The following functions define harder start-goal variants of the structured
# complex maps. They intentionally reuse the original obstacle layouts and only
# change the planning query. This creates a second benchmark layer without
# invalidating previous results obtained on the original maps.

HARD_QUERY_MAPS = [
    "multi_room_hard",
    "office_corridor_hard",
    "warehouse_clutter_hard",
    "urban_blocks_hard",
    "hybrid_trap_hard",
]


def _replace_start_goal(binary_mask, start_xy, goal_xy, radius=4):
    """
    Return a hard-query start/goal pair and clear small local patches.

    The clearing radius is the same style as _ensure_start_goal_free(). It is
    only used to avoid clearance-related invalidity around start/goal and does
    not create a global shortcut.
    """
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    _ensure_start_goal_free(binary_mask, start, goal, radius=radius)
    return binary_mask, start, goal


def make_multi_room_hard_problem():
    """
    Hard query for multi_room.

    The original obstacle layout is reused. The start and goal are placed in
    opposite deep rooms, forcing multiple room transitions through door openings.
    """
    binary_mask, _, _ = make_multi_room_problem()
    return _replace_start_goal(
        binary_mask=binary_mask,
        start_xy=(108, 108),
        goal_xy=(14, 14),
        radius=4,
    )


def make_office_corridor_hard_problem():
    """
    Hard query for office_corridor.

    The start and goal are placed inside different side-room regions rather than
    directly on the main corridor, increasing branch and turn requirements.
    """
    binary_mask, _, _ = make_office_corridor_problem()
    return _replace_start_goal(
        binary_mask=binary_mask,
        start_xy=(20, 12),
        goal_xy=(100, 108),
        radius=4,
    )


def make_warehouse_clutter_hard_problem():
    """
    Hard query for warehouse_clutter.

    The query is placed inside different aisle systems, making the route less
    dominated by a simple lower-left to upper-right diagonal trend.
    """
    binary_mask, _, _ = make_warehouse_clutter_problem()
    return _replace_start_goal(
        binary_mask=binary_mask,
        start_xy=(34, 100),
        goal_xy=(78, 20),
        radius=4,
    )


def make_urban_blocks_hard_problem():
    """
    Hard query for urban_blocks.

    The start and goal are placed in interior street corridors rather than on
    the easy outer boundary road, increasing topological ambiguity.
    """
    binary_mask, _, _ = make_urban_blocks_problem()
    return _replace_start_goal(
        binary_mask=binary_mask,
        start_xy=(40, 100),
        goal_xy=(76, 20),
        radius=4,
    )


def make_hybrid_trap_hard_problem():
    """
    Hard query for hybrid_trap.

    The goal is moved near the lower-right trap-side region. A naive direct
    tendency can be pulled toward the false-opening structure, while the feasible
    route still requires correct topological selection around the blocking wall.
    """
    binary_mask, _, _ = make_hybrid_trap_problem()
    return _replace_start_goal(
        binary_mask=binary_mask,
        start_xy=(14, 104),
        goal_xy=(108, 82),
        radius=4,
    )


# Preserve the original factory and extend it by redefining make_complex_problem.
# This avoids rewriting the original map definitions and keeps backward
# compatibility with all existing map names.
_ORIGINAL_MAKE_COMPLEX_PROBLEM = make_complex_problem


def make_complex_problem(map_name):
    """
    Factory for original and hard-query complex simulation maps.
    """
    if map_name == "multi_room_hard":
        return make_multi_room_hard_problem()

    if map_name == "office_corridor_hard":
        return make_office_corridor_hard_problem()

    if map_name == "warehouse_clutter_hard":
        return make_warehouse_clutter_hard_problem()

    if map_name == "urban_blocks_hard":
        return make_urban_blocks_hard_problem()

    if map_name == "hybrid_trap_hard":
        return make_hybrid_trap_hard_problem()

    return _ORIGINAL_MAKE_COMPLEX_PROBLEM(map_name)

# === HARD_QUERY_BENCHMARK_EXTENSION_END ===

# === HARD_V2_BENCHMARK_EXTENSION_BEGIN ===
# Hard-v2 structured benchmark.
#
# These maps are intentionally harder than the first hard-query benchmark:
#   - they modify obstacle topology, not only start/goal positions;
#   - they include narrower feasible corridors while remaining compatible with
#     clearance=2 and step_len=5.0;
#   - they introduce dead ends, blocked intersections, and longer detours.
#
# Convention:
#   binary_mask[y, x] = 1: free space
#   binary_mask[y, x] = 0: obstacle

HARD_V2_MAPS = [
    "multi_room_hard_v2",
    "office_corridor_hard_v2",
    "warehouse_clutter_hard_v2",
    "urban_blocks_hard_v2",
    "hybrid_trap_hard_v2",
]


def _hv2_empty_map(height=120, width=120, border=4):
    mask = np.ones((height, width), dtype=np.uint8)
    if border > 0:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
    return mask


def _hv2_obstacle(mask, x0, x1, y0, y1):
    h, w = mask.shape
    x0 = max(0, int(x0)); x1 = min(w, int(x1))
    y0 = max(0, int(y0)); y1 = min(h, int(y1))
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = 0


def _hv2_free(mask, x0, x1, y0, y1):
    h, w = mask.shape
    x0 = max(0, int(x0)); x1 = min(w, int(x1))
    y0 = max(0, int(y0)); y1 = min(h, int(y1))
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = 1


def _hv2_clear_point(mask, point, radius=5):
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    r = int(radius)
    _hv2_free(mask, x - r, x + r + 1, y - r, y + r + 1)


def _hv2_finalize(mask, start_xy, goal_xy, radius=5):
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    _hv2_clear_point(mask, start, radius=radius)
    _hv2_clear_point(mask, goal, radius=radius)
    return mask, start, goal


def make_multi_room_hard_v2_problem():
    """
    Multi-room hard-v2.

    Compared with the original multi_room map, this layout has a denser room
    partition, staggered doors, and a start-goal pair placed in opposite deep
    rooms. The route requires multiple room transitions rather than a simple
    diagonal expansion.
    """
    mask = _hv2_empty_map()

    # Main room partitions.
    _hv2_obstacle(mask, 56, 64, 8, 112)      # vertical wall
    _hv2_obstacle(mask, 8, 112, 56, 64)      # horizontal wall

    # Staggered door openings.
    _hv2_free(mask, 56, 64, 16, 28)
    _hv2_free(mask, 56, 64, 88, 100)
    _hv2_free(mask, 18, 32, 56, 64)
    _hv2_free(mask, 86, 100, 56, 64)

    # Secondary partitions and room blocks.
    _hv2_obstacle(mask, 22, 36, 16, 34)
    _hv2_obstacle(mask, 78, 94, 18, 36)
    _hv2_obstacle(mask, 18, 34, 76, 96)
    _hv2_obstacle(mask, 82, 100, 78, 98)
    _hv2_obstacle(mask, 40, 50, 36, 52)
    _hv2_obstacle(mask, 70, 82, 36, 52)
    _hv2_obstacle(mask, 38, 52, 82, 96)
    _hv2_obstacle(mask, 70, 80, 82, 106)

    # Narrow local blockers that create dead-end-like side regions.
    _hv2_obstacle(mask, 12, 34, 42, 48)
    _hv2_obstacle(mask, 86, 108, 70, 76)
    _hv2_obstacle(mask, 42, 50, 12, 24)
    _hv2_obstacle(mask, 96, 106, 102, 112)

    return _hv2_finalize(mask, start_xy=(108, 108), goal_xy=(14, 14), radius=5)


def make_office_corridor_hard_v2_problem():
    """
    Office-corridor hard-v2.

    This is a more constrained office-style corridor network. It is generated
    by carving corridors and rooms from an obstacle-filled map. The planner must
    leave a lower-left side office, move through a multi-turn corridor network,
    avoid blocked intersections, and reach an upper-right side office.
    """
    mask = np.zeros((120, 120), dtype=np.uint8)

    # Main corridor skeleton, intentionally narrow but feasible with clearance=2.
    _hv2_free(mask, 10, 24, 12, 108)     # left vertical corridor
    _hv2_free(mask, 52, 66, 12, 108)     # middle vertical corridor
    _hv2_free(mask, 94, 108, 12, 108)    # right vertical corridor
    _hv2_free(mask, 10, 108, 52, 66)     # middle horizontal corridor
    _hv2_free(mask, 10, 66, 96, 108)     # lower-left horizontal corridor
    _hv2_free(mask, 52, 108, 12, 24)     # upper-right horizontal corridor

    # Side rooms connected to the corridor network.
    _hv2_free(mask, 6, 34, 82, 112)      # start-side office
    _hv2_free(mask, 82, 114, 6, 36)      # goal-side office
    _hv2_free(mask, 28, 50, 34, 52)
    _hv2_free(mask, 68, 92, 68, 90)
    _hv2_free(mask, 28, 50, 68, 90)
    _hv2_free(mask, 70, 92, 34, 52)

    # Door connections from rooms to corridors.
    _hv2_free(mask, 22, 34, 94, 104)
    _hv2_free(mask, 88, 100, 14, 24)
    _hv2_free(mask, 46, 58, 42, 52)
    _hv2_free(mask, 62, 74, 76, 86)
    _hv2_free(mask, 46, 58, 76, 86)
    _hv2_free(mask, 86, 100, 42, 52)

    # Block selected intersections to prevent the easiest direct route and add traps.
    _hv2_obstacle(mask, 52, 66, 34, 50)   # middle vertical partial block
    _hv2_obstacle(mask, 94, 108, 76, 92)  # right vertical lower block
    _hv2_obstacle(mask, 32, 48, 52, 66)   # side obstruction near middle corridor
    _hv2_obstacle(mask, 72, 88, 52, 66)

    # Small dead-end branches.
    _hv2_free(mask, 10, 44, 28, 38)
    _hv2_free(mask, 76, 108, 84, 94)
    _hv2_obstacle(mask, 38, 46, 28, 38)
    _hv2_obstacle(mask, 76, 84, 84, 94)

    return _hv2_finalize(mask, start_xy=(18, 102), goal_xy=(104, 16), radius=5)


def make_warehouse_clutter_hard_v2_problem():
    """
    Warehouse-clutter hard-v2.

    Long shelves create narrow aisles. Several aisle switches are blocked so
    the route must move through specific openings instead of taking a straight
    diagonal or a single wide aisle.
    """
    mask = _hv2_empty_map(border=4)

    # Long vertical shelves.
    shelves = [
        (18, 30, 12, 54),
        (18, 30, 68, 108),
        (42, 54, 10, 44),
        (42, 54, 58, 108),
        (66, 78, 14, 64),
        (66, 78, 78, 108),
        (90, 102, 10, 42),
        (90, 102, 56, 106),
    ]
    for x0, x1, y0, y1 in shelves:
        _hv2_obstacle(mask, x0, x1, y0, y1)

    # Cross-aisle blockers and pallet clutter.
    blockers = [
        (8, 18, 48, 58),
        (30, 42, 50, 60),
        (54, 66, 28, 40),
        (78, 90, 66, 78),
        (102, 112, 36, 48),
        (56, 64, 96, 108),
        (8, 18, 92, 104),
        (102, 112, 84, 96),
    ]
    for x0, x1, y0, y1 in blockers:
        _hv2_obstacle(mask, x0, x1, y0, y1)

    # Local openings through shelves to keep the map connected but selective.
    _hv2_free(mask, 18, 30, 54, 68)
    _hv2_free(mask, 42, 54, 44, 58)
    _hv2_free(mask, 66, 78, 64, 78)
    _hv2_free(mask, 90, 102, 42, 56)

    return _hv2_finalize(mask, start_xy=(12, 104), goal_xy=(108, 16), radius=5)


def make_urban_blocks_hard_v2_problem():
    """
    Urban-blocks hard-v2.

    Grid-like city blocks are combined with blocked intersections, producing a
    road-network planning problem where the direct Manhattan route is partially
    closed and a longer detour is required.
    """
    mask = _hv2_empty_map(border=4)

    # Large city blocks in a 3x3 pattern.
    block_x = [(14, 34), (46, 68), (82, 106)]
    block_y = [(12, 34), (46, 70), (84, 108)]
    for x0, x1 in block_x:
        for y0, y1 in block_y:
            _hv2_obstacle(mask, x0, x1, y0, y1)

    # Block selected intersections / roads to avoid an easy boundary-only path.
    blockers = [
        (36, 44, 34, 46),
        (70, 82, 34, 46),
        (34, 46, 72, 84),
        (72, 84, 70, 84),
        (108, 116, 46, 82),
        (4, 12, 36, 76),
        (50, 62, 108, 116),
        (76, 88, 4, 12),
    ]
    for x0, x1, y0, y1 in blockers:
        _hv2_obstacle(mask, x0, x1, y0, y1)

    # Reopen a few controlled intersections.
    _hv2_free(mask, 36, 46, 54, 64)
    _hv2_free(mask, 70, 82, 88, 98)
    _hv2_free(mask, 108, 116, 84, 96)
    _hv2_free(mask, 4, 12, 78, 90)

    return _hv2_finalize(mask, start_xy=(40, 100), goal_xy=(76, 20), radius=5)


def make_hybrid_trap_hard_v2_problem():
    """
    Hybrid-trap hard-v2.

    Combines clutter, a central blocking wall, a false opening, and a U-shaped
    trap near the goal. The direct direction toward the goal is deliberately
    misleading; the feasible route requires moving around the trap structure.
    """
    mask = _hv2_empty_map(border=4)

    # Central barrier with one useful gate and one misleading blocked opening.
    _hv2_obstacle(mask, 56, 66, 12, 88)
    _hv2_free(mask, 56, 66, 26, 38)       # upper gate
    _hv2_free(mask, 56, 66, 76, 86)       # apparent lower gate
    _hv2_obstacle(mask, 66, 88, 76, 88)   # block after lower gate -> trap-like

    # U-shaped trap around the lower-right region.
    _hv2_obstacle(mask, 78, 108, 40, 52)
    _hv2_obstacle(mask, 78, 90, 40, 92)
    _hv2_obstacle(mask, 78, 108, 92, 104)
    _hv2_free(mask, 90, 108, 72, 88)      # one exit corridor inside the U

    # Additional clutter on the left and upper-right.
    obstacles = [
        (14, 30, 14, 30),
        (18, 34, 58, 76),
        (34, 48, 36, 54),
        (34, 50, 88, 104),
        (72, 88, 12, 30),
        (92, 112, 14, 30),
        (16, 28, 88, 106),
    ]
    for x0, x1, y0, y1 in obstacles:
        _hv2_obstacle(mask, x0, x1, y0, y1)

    return _hv2_finalize(mask, start_xy=(14, 104), goal_xy=(108, 82), radius=5)


# Preserve the active factory and extend it by redefining make_complex_problem.
# If the hard-query patch was applied earlier, _HARD_V2_BASE_MAKE_COMPLEX_PROBLEM
# will point to that already-extended factory, so all previous map names remain
# available.
_HARD_V2_BASE_MAKE_COMPLEX_PROBLEM = make_complex_problem


def make_complex_problem(map_name):
    """
    Factory for original, hard-query, and hard-v2 complex simulation maps.
    """
    if map_name == "multi_room_hard_v2":
        return make_multi_room_hard_v2_problem()

    if map_name == "office_corridor_hard_v2":
        return make_office_corridor_hard_v2_problem()

    if map_name == "warehouse_clutter_hard_v2":
        return make_warehouse_clutter_hard_v2_problem()

    if map_name == "urban_blocks_hard_v2":
        return make_urban_blocks_hard_v2_problem()

    if map_name == "hybrid_trap_hard_v2":
        return make_hybrid_trap_hard_v2_problem()

    return _HARD_V2_BASE_MAKE_COMPLEX_PROBLEM(map_name)

# === HARD_V2_BENCHMARK_EXTENSION_END ===


# ============================================================================
# Hard-v3 benchmark maps
# ============================================================================

def _hard_v3_rect(mask, x0, y0, x1, y1, value=1):
    """Carve or block an axis-aligned rectangle using x/y coordinates."""
    h, w = mask.shape
    x0 = max(0, int(round(x0)))
    x1 = min(w, int(round(x1)))
    y0 = max(0, int(round(y0)))
    y1 = min(h, int(round(y1)))
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = value


def make_office_corridor_hard_v3_problem():
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Guaranteed meandering route.
    _hard_v3_rect(binary_mask, 10, 96, 26, 110, 1)
    _hard_v3_rect(binary_mask, 10, 70, 22, 110, 1)
    _hard_v3_rect(binary_mask, 10, 70, 70, 82, 1)
    _hard_v3_rect(binary_mask, 58, 34, 70, 82, 1)
    _hard_v3_rect(binary_mask, 22, 34, 70, 46, 1)
    _hard_v3_rect(binary_mask, 22, 18, 34, 46, 1)
    _hard_v3_rect(binary_mask, 12, 18, 34, 30, 1)

    # Side rooms and dead-looking branches.
    _hard_v3_rect(binary_mask, 28, 92, 48, 106, 1)
    _hard_v3_rect(binary_mask, 30, 54, 46, 66, 1)
    _hard_v3_rect(binary_mask, 72, 66, 100, 78, 1)
    _hard_v3_rect(binary_mask, 78, 18, 106, 30, 1)
    _hard_v3_rect(binary_mask, 44, 14, 56, 28, 1)
    _hard_v3_rect(binary_mask, 84, 44, 106, 56, 1)
    _hard_v3_rect(binary_mask, 8, 42, 20, 58, 1)

    start = np.array([16, 102], dtype=float)
    goal = np.array([18, 24], dtype=float)
    return binary_mask, start, goal


def make_warehouse_clutter_hard_v3_problem():
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Guaranteed serpentine aisle.
    _hard_v3_rect(binary_mask, 8, 96, 24, 110, 1)
    _hard_v3_rect(binary_mask, 8, 14, 20, 110, 1)
    _hard_v3_rect(binary_mask, 8, 14, 54, 26, 1)
    _hard_v3_rect(binary_mask, 42, 14, 54, 96, 1)
    _hard_v3_rect(binary_mask, 42, 84, 86, 96, 1)
    _hard_v3_rect(binary_mask, 74, 32, 86, 96, 1)
    _hard_v3_rect(binary_mask, 74, 32, 110, 44, 1)
    _hard_v3_rect(binary_mask, 98, 32, 110, 66, 1)

    # Dead/side aisles.
    _hard_v3_rect(binary_mask, 24, 48, 36, 86, 1)
    _hard_v3_rect(binary_mask, 56, 16, 68, 44, 1)
    _hard_v3_rect(binary_mask, 60, 56, 72, 108, 1)
    _hard_v3_rect(binary_mask, 88, 60, 98, 78, 1)
    _hard_v3_rect(binary_mask, 22, 102, 38, 110, 1)

    start = np.array([14, 104], dtype=float)
    goal = np.array([104, 44], dtype=float)
    return binary_mask, start, goal


def make_urban_blocks_hard_v3_problem():
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Guaranteed route.
    _hard_v3_rect(binary_mask, 34, 96, 48, 110, 1)
    _hard_v3_rect(binary_mask, 34, 66, 48, 110, 1)
    _hard_v3_rect(binary_mask, 34, 66, 78, 80, 1)
    _hard_v3_rect(binary_mask, 66, 24, 78, 80, 1)
    _hard_v3_rect(binary_mask, 66, 24, 106, 38, 1)
    _hard_v3_rect(binary_mask, 94, 24, 106, 80, 1)

    # Additional disconnected / dead-end roads.
    _hard_v3_rect(binary_mask, 14, 14, 28, 34, 1)
    _hard_v3_rect(binary_mask, 14, 46, 28, 70, 1)
    _hard_v3_rect(binary_mask, 50, 14, 62, 38, 1)
    _hard_v3_rect(binary_mask, 14, 82, 34, 94, 1)
    _hard_v3_rect(binary_mask, 50, 90, 66, 110, 1)
    _hard_v3_rect(binary_mask, 78, 72, 92, 86, 1)

    start = np.array([42, 102], dtype=float)
    goal = np.array([100, 74], dtype=float)
    return binary_mask, start, goal


def make_multi_room_hard_v3_problem():
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Guaranteed multi-room route.
    _hard_v3_rect(binary_mask, 88, 84, 112, 108, 1)
    _hard_v3_rect(binary_mask, 96, 58, 108, 108, 1)
    _hard_v3_rect(binary_mask, 60, 58, 108, 70, 1)
    _hard_v3_rect(binary_mask, 60, 20, 72, 70, 1)
    _hard_v3_rect(binary_mask, 20, 20, 72, 32, 1)
    _hard_v3_rect(binary_mask, 20, 20, 32, 92, 1)
    _hard_v3_rect(binary_mask, 20, 84, 92, 96, 1)

    # Goal-side room and additional side rooms.
    _hard_v3_rect(binary_mask, 14, 22, 34, 40, 1)
    _hard_v3_rect(binary_mask, 42, 40, 56, 56, 1)
    _hard_v3_rect(binary_mask, 78, 18, 96, 34, 1)
    _hard_v3_rect(binary_mask, 76, 78, 90, 96, 1)
    _hard_v3_rect(binary_mask, 42, 76, 56, 96, 1)
    _hard_v3_rect(binary_mask, 8, 50, 22, 64, 1)

    start = np.array([104, 100], dtype=float)
    goal = np.array([26, 30], dtype=float)
    return binary_mask, start, goal


def make_hybrid_trap_hard_v3_problem():
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Guaranteed route.
    _hard_v3_rect(binary_mask, 10, 96, 28, 110, 1)
    _hard_v3_rect(binary_mask, 10, 70, 22, 110, 1)
    _hard_v3_rect(binary_mask, 10, 70, 54, 82, 1)
    _hard_v3_rect(binary_mask, 42, 52, 54, 82, 1)
    _hard_v3_rect(binary_mask, 42, 52, 84, 64, 1)
    _hard_v3_rect(binary_mask, 72, 34, 84, 64, 1)
    _hard_v3_rect(binary_mask, 72, 34, 106, 46, 1)
    _hard_v3_rect(binary_mask, 94, 34, 106, 86, 1)
    _hard_v3_rect(binary_mask, 86, 74, 106, 86, 1)

    # Local branches / traps.
    _hard_v3_rect(binary_mask, 22, 90, 44, 102, 1)
    _hard_v3_rect(binary_mask, 24, 16, 38, 32, 1)
    _hard_v3_rect(binary_mask, 56, 16, 70, 32, 1)
    _hard_v3_rect(binary_mask, 58, 76, 72, 100, 1)
    _hard_v3_rect(binary_mask, 78, 12, 100, 28, 1)
    _hard_v3_rect(binary_mask, 82, 54, 100, 66, 1)

    start = np.array([16, 104], dtype=float)
    goal = np.array([96, 80], dtype=float)
    return binary_mask, start, goal

# =============================================================================
# Topological benchmark maps
# =============================================================================
# These maps are designed to test topological planning difficulty rather than
# simply increasing obstacle count. Each map has a clear structural pattern:
# nested rings, spiral corridor, multi-gate rooms, braided route choices, and
# trap-with-bypass.
#
# Important:
#   binary_mask[y, x] = 1 -> free
#   binary_mask[y, x] = 0 -> obstacle
# =============================================================================

import numpy as np


def _topo_rect(binary_mask, x0, y0, x1, y1, value=0):
    h, w = binary_mask.shape
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    if x0 < x1 and y0 < y1:
        binary_mask[y0:y1, x0:x1] = int(value)


def _topo_boundary(binary_mask, thickness=4):
    h, w = binary_mask.shape
    t = int(thickness)
    _topo_rect(binary_mask, 0, 0, w, t, 0)
    _topo_rect(binary_mask, 0, h - t, w, h, 0)
    _topo_rect(binary_mask, 0, 0, t, h, 0)
    _topo_rect(binary_mask, w - t, 0, w, h, 0)


def _topo_vwall(binary_mask, x, y0, y1, thickness=5, gaps=None):
    # Draw a vertical wall and carve gap intervals along y.
    if gaps is None:
        gaps = []

    x = int(x)
    t = int(thickness)
    _topo_rect(binary_mask, x, y0, x + t, y1, 0)

    for gy0, gy1 in gaps:
        _topo_rect(binary_mask, x, gy0, x + t, gy1, 1)


def _topo_hwall(binary_mask, y, x0, x1, thickness=5, gaps=None):
    # Draw a horizontal wall and carve gap intervals along x.
    if gaps is None:
        gaps = []

    y = int(y)
    t = int(thickness)
    _topo_rect(binary_mask, x0, y, x1, y + t, 0)

    for gx0, gx1 in gaps:
        _topo_rect(binary_mask, gx0, y, gx1, y + t, 1)


def make_topo_nested_ring_problem():
    # Nested ring / hui-shaped maze.
    # The route must enter a sequence of nested rings through staggered gates.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _topo_boundary(binary_mask, thickness=4)

    # Outer ring: entrance on the left-lower side.
    _topo_hwall(binary_mask, y=16, x0=16, x1=104, thickness=5, gaps=[])
    _topo_hwall(binary_mask, y=99, x0=16, x1=104, thickness=5, gaps=[])
    _topo_vwall(binary_mask, x=16, y0=16, y1=104, thickness=5, gaps=[(82, 94)])
    _topo_vwall(binary_mask, x=99, y0=16, y1=104, thickness=5, gaps=[])

    # Middle ring: entrance on the right-upper side.
    _topo_hwall(binary_mask, y=32, x0=32, x1=88, thickness=5, gaps=[])
    _topo_hwall(binary_mask, y=83, x0=32, x1=88, thickness=5, gaps=[])
    _topo_vwall(binary_mask, x=32, y0=32, y1=88, thickness=5, gaps=[])
    _topo_vwall(binary_mask, x=83, y0=32, y1=88, thickness=5, gaps=[(42, 54)])

    # Inner ring: entrance on the lower side.
    _topo_hwall(binary_mask, y=48, x0=48, x1=72, thickness=5, gaps=[])
    _topo_hwall(binary_mask, y=69, x0=48, x1=72, thickness=5, gaps=[(56, 66)])
    _topo_vwall(binary_mask, x=48, y0=48, y1=74, thickness=5, gaps=[])
    _topo_vwall(binary_mask, x=67, y0=48, y1=74, thickness=5, gaps=[])

    start = np.array([10, 88], dtype=float)
    goal = np.array([60, 60], dtype=float)
    return binary_mask, start, goal


def make_topo_spiral_problem():
    # Spiral corridor.
    # The feasible route follows a spiral-like topology from outside to inside.
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    def carve(x0, y0, x1, y1):
        _topo_rect(binary_mask, x0, y0, x1, y1, 1)

    # A wide spiral corridor. Width is around 12 px, feasible under clearance=2.
    carve(6, 6, 114, 18)
    carve(102, 6, 114, 106)
    carve(18, 94, 114, 106)
    carve(18, 34, 30, 106)
    carve(18, 34, 90, 46)
    carve(78, 34, 90, 82)
    carve(46, 70, 90, 82)
    carve(46, 54, 58, 82)
    carve(46, 54, 70, 66)
    carve(62, 48, 76, 68)

    # Side chambers reduce the single-tunnel appearance while preserving topology.
    carve(34, 20, 48, 32)
    carve(92, 62, 100, 78)
    carve(8, 64, 18, 78)

    _topo_boundary(binary_mask, thickness=4)

    start = np.array([12, 12], dtype=float)
    goal = np.array([69, 58], dtype=float)
    return binary_mask, start, goal


def make_topo_multi_gate_problem():
    # Multi-gate room graph with staggered doorways.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _topo_boundary(binary_mask, thickness=4)

    _topo_vwall(binary_mask, x=38, y0=8, y1=112, thickness=5, gaps=[(18, 30), (66, 78)])
    _topo_vwall(binary_mask, x=78, y0=8, y1=112, thickness=5, gaps=[(24, 36), (82, 96)])
    _topo_hwall(binary_mask, y=38, x0=8, x1=112, thickness=5, gaps=[(16, 30), (84, 98)])
    _topo_hwall(binary_mask, y=78, x0=8, x1=112, thickness=5, gaps=[(42, 56), (88, 102)])

    _topo_rect(binary_mask, 52, 50, 66, 64, 0)
    _topo_rect(binary_mask, 14, 88, 28, 102, 0)
    _topo_rect(binary_mask, 92, 12, 106, 24, 0)

    start = np.array([20, 62], dtype=float)
    goal = np.array([100, 30], dtype=float)
    return binary_mask, start, goal


def make_topo_braided_paths_problem():
    # Braided route network with upper/middle/lower lanes and cross-links.
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    def carve(x0, y0, x1, y1):
        _topo_rect(binary_mask, x0, y0, x1, y1, 1)

    carve(8, 22, 112, 34)
    carve(8, 52, 112, 64)
    carve(8, 84, 112, 96)

    carve(20, 22, 32, 64)
    carve(50, 52, 62, 96)
    carve(82, 22, 94, 96)

    carve(36, 34, 48, 48)
    carve(96, 64, 108, 78)

    start = np.array([12, 90], dtype=float)
    goal = np.array([108, 28], dtype=float)
    return binary_mask, start, goal


def make_topo_trap_bypass_problem():
    # Direct start-goal direction enters a U-shaped trap; valid path bypasses it.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _topo_boundary(binary_mask, thickness=4)

    # U-shaped trap opening toward the start side.
    _topo_rect(binary_mask, 42, 32, 92, 42, 0)
    _topo_rect(binary_mask, 42, 78, 92, 88, 0)
    _topo_rect(binary_mask, 82, 32, 92, 88, 0)

    # Local obstacle inside the false pocket.
    _topo_rect(binary_mask, 56, 54, 70, 66, 0)

    # Structured clutter around the bypass.
    _topo_rect(binary_mask, 20, 18, 32, 30, 0)
    _topo_rect(binary_mask, 20, 90, 34, 102, 0)
    _topo_rect(binary_mask, 56, 14, 70, 26, 0)
    _topo_rect(binary_mask, 58, 94, 72, 106, 0)

    start = np.array([14, 60], dtype=float)
    goal = np.array([106, 60], dtype=float)
    return binary_mask, start, goal


_TOPOLOGICAL_PROBLEM_BUILDERS = {
    "topo_nested_ring": make_topo_nested_ring_problem,
    "topo_spiral": make_topo_spiral_problem,
    "topo_multi_gate": make_topo_multi_gate_problem,
    "topo_braided_paths": make_topo_braided_paths_problem,
    "topo_trap_bypass": make_topo_trap_bypass_problem,
}


# Preserve the original complex-map dispatcher and extend it.
try:
    _make_complex_problem_before_topological_benchmark
except NameError:
    _make_complex_problem_before_topological_benchmark = make_complex_problem


def make_complex_problem(map_name):
    name = str(map_name).strip()

    if name in _TOPOLOGICAL_PROBLEM_BUILDERS:
        return _TOPOLOGICAL_PROBLEM_BUILDERS[name]()

    return _make_complex_problem_before_topological_benchmark(map_name)

# =============================================================================
# Formal benchmark scenes v3 for RAG-RRT* experiments
# =============================================================================
# v3 keeps the overall v2 layout. It only adjusts the two maps that did not show
# validated paths in the preview:
#   - bench_complex_maze
#   - bench_cluttered
#
# The adjustment is intentionally small: widen several gates / remove one local
# block near the query endpoint / relax one clutter block near the start. The
# visual style and benchmark purpose remain unchanged.
#
# Important:
#   binary_mask[y, x] = 1 -> free
#   binary_mask[y, x] = 0 -> obstacle
# =============================================================================

import numpy as np


def _bench3_rect(binary_mask, x0, y0, x1, y1, value=0):
    h, w = binary_mask.shape
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    if x0 < x1 and y0 < y1:
        binary_mask[y0:y1, x0:x1] = int(value)


def _bench3_boundary(binary_mask, thickness=4):
    h, w = binary_mask.shape
    t = int(thickness)
    _bench3_rect(binary_mask, 0, 0, w, t, 0)
    _bench3_rect(binary_mask, 0, h - t, w, h, 0)
    _bench3_rect(binary_mask, 0, 0, t, h, 0)
    _bench3_rect(binary_mask, w - t, 0, w, h, 0)


def _bench3_vwall(binary_mask, x, y0, y1, thickness=7, gaps=None):
    if gaps is None:
        gaps = []
    _bench3_rect(binary_mask, x, y0, x + thickness, y1, 0)
    for gy0, gy1 in gaps:
        _bench3_rect(binary_mask, x, gy0, x + thickness, gy1, 1)


def _bench3_hwall(binary_mask, y, x0, x1, thickness=7, gaps=None):
    if gaps is None:
        gaps = []
    _bench3_rect(binary_mask, x0, y, x1, y + thickness, 0)
    for gx0, gx1 in gaps:
        _bench3_rect(binary_mask, gx0, y, gx1, y + thickness, 1)


def make_bench_simple_maze_problem_v3():
    # Same as v2.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench3_boundary(binary_mask, thickness=4)

    _bench3_vwall(binary_mask, x=34, y0=18, y1=102, thickness=7, gaps=[(72, 90)])
    _bench3_hwall(binary_mask, y=44, x0=34, x1=92, thickness=7, gaps=[(58, 72)])
    _bench3_vwall(binary_mask, x=86, y0=28, y1=94, thickness=7, gaps=[(50, 66)])

    _bench3_rect(binary_mask, 54, 82, 70, 96, 0)
    _bench3_rect(binary_mask, 12, 34, 22, 46, 0)

    start = np.array([14, 18], dtype=float)
    goal = np.array([104, 98], dtype=float)
    return binary_mask, start, goal


def make_bench_complex_maze_problem_v3():
    """
    Complex maze v3.

    Compared with v2:
        - slightly widened staggered gates;
        - removed the small block close to the upper-right goal region;
        - kept the same room/corridor topology and branch-choice purpose.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench3_boundary(binary_mask, thickness=4)

    _bench3_hwall(binary_mask, y=26, x0=8, x1=92, thickness=7, gaps=[(14, 38), (62, 82)])
    _bench3_vwall(binary_mask, x=28, y0=26, y1=112, thickness=7, gaps=[(44, 70), (90, 108)])
    _bench3_hwall(binary_mask, y=56, x0=28, x1=112, thickness=7, gaps=[(40, 62), (84, 106)])
    _bench3_vwall(binary_mask, x=64, y0=8, y1=96, thickness=7, gaps=[(14, 38), (64, 88)])
    _bench3_hwall(binary_mask, y=88, x0=8, x1=96, thickness=7, gaps=[(18, 42), (54, 78)])
    _bench3_vwall(binary_mask, x=94, y0=26, y1=112, thickness=7, gaps=[(10, 32), (38, 60), (72, 96)])

    for block in [
        (42, 10, 54, 22),
        (76, 34, 88, 46),
        (44, 72, 56, 82),
        (104, 94, 114, 106),
        (12, 66, 22, 78),
    ]:
        _bench3_rect(binary_mask, *block, value=0)

    start = np.array([14, 104], dtype=float)
    goal = np.array([108, 14], dtype=float)
    return binary_mask, start, goal


def make_bench_concave_trap_problem_v3():
    # Same as v2.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench3_boundary(binary_mask, thickness=4)

    _bench3_rect(binary_mask, 42, 28, 88, 38, 0)
    _bench3_rect(binary_mask, 42, 72, 98, 82, 0)
    _bench3_rect(binary_mask, 88, 28, 98, 82, 0)
    _bench3_rect(binary_mask, 52, 50, 70, 60, 0)

    _bench3_rect(binary_mask, 64, 12, 78, 24, 0)
    _bench3_rect(binary_mask, 102, 44, 112, 58, 0)

    _bench3_rect(binary_mask, 20, 88, 36, 102, 0)
    _bench3_rect(binary_mask, 56, 94, 72, 106, 0)

    start = np.array([14, 58], dtype=float)
    goal = np.array([108, 66], dtype=float)
    return binary_mask, start, goal


def make_bench_multi_gap_problem_v3():
    # Same as v2.
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench3_boundary(binary_mask, thickness=4)

    _bench3_vwall(binary_mask, x=28, y0=8, y1=112, thickness=7, gaps=[(24, 42), (90, 106)])
    _bench3_vwall(binary_mask, x=52, y0=8, y1=112, thickness=7, gaps=[(58, 76)])
    _bench3_vwall(binary_mask, x=78, y0=8, y1=112, thickness=7, gaps=[(30, 48), (82, 98)])

    _bench3_hwall(binary_mask, y=70, x0=52, x1=104, thickness=7, gaps=[(82, 98)])
    _bench3_rect(binary_mask, 10, 18, 20, 30, 0)
    _bench3_rect(binary_mask, 36, 44, 46, 56, 0)
    _bench3_rect(binary_mask, 96, 18, 108, 32, 0)
    _bench3_rect(binary_mask, 98, 96, 110, 108, 0)

    start = np.array([12, 62], dtype=float)
    goal = np.array([108, 52], dtype=float)
    return binary_mask, start, goal


def make_bench_cluttered_problem_v3():
    """
    Cluttered benchmark v3.

    Compared with v2:
        - keeps the cluster-style clutter;
        - removes the local block that made the start invalid under clearance=2;
        - retains medium collision-check pressure.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench3_boundary(binary_mask, thickness=4)

    blocks = [
        (18, 12, 30, 26), (40, 14, 52, 28), (72, 12, 84, 26), (92, 16, 106, 30),
        (12, 38, 26, 52), (34, 44, 48, 60), (64, 40, 78, 56), (92, 44, 106, 60),
        (22, 66, 36, 82), (52, 66, 66, 82), (80, 70, 94, 86),
        (40, 92, 54, 106), (68, 92, 82, 108), (96, 88, 110, 102),
        (56, 22, 66, 34), (30, 30, 40, 42), (78, 30, 88, 42), (44, 76, 54, 88),
    ]

    for block in blocks:
        _bench3_rect(binary_mask, *block, value=0)

    start = np.array([12, 92], dtype=float)
    goal = np.array([108, 20], dtype=float)
    return binary_mask, start, goal


_BENCHMARK_SCENE_BUILDERS_V3 = {
    "bench_simple_maze": make_bench_simple_maze_problem_v3,
    "bench_complex_maze": make_bench_complex_maze_problem_v3,
    "bench_concave_trap": make_bench_concave_trap_problem_v3,
    "bench_multi_gap": make_bench_multi_gap_problem_v3,
    "bench_cluttered": make_bench_cluttered_problem_v3,
}


try:
    _make_complex_problem_before_benchmark_scenes_v3
except NameError:
    _make_complex_problem_before_benchmark_scenes_v3 = make_complex_problem


def make_complex_problem(map_name):
    name = str(map_name).strip()

    if name in _BENCHMARK_SCENE_BUILDERS_V3:
        return _BENCHMARK_SCENE_BUILDERS_V3[name]()

    return _make_complex_problem_before_benchmark_scenes_v3(map_name)

# =============================================================================
# Formal benchmark scenes v4 for RAG-RRT* experiments
# =============================================================================
# v4 keeps the overall v3 geometry, but adjusts several start/goal placements
# after manual inspection of the validated previews:
#   - bench_complex_maze:  move goal to an upper-middle free pocket
#   - bench_concave_trap: move start into the central cavity region
#   - bench_multi_gap:    move goal to the lower-middle reachable chamber
#   - bench_cluttered:    move goal to an upper-middle free pocket
#
# Important:
#   binary_mask[y, x] = 1 -> free
#   binary_mask[y, x] = 0 -> obstacle
# =============================================================================

import numpy as np


def _bench4_rect(binary_mask, x0, y0, x1, y1, value=0):
    h, w = binary_mask.shape
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    if x0 < x1 and y0 < y1:
        binary_mask[y0:y1, x0:x1] = int(value)


def _bench4_boundary(binary_mask, thickness=4):
    h, w = binary_mask.shape
    t = int(thickness)
    _bench4_rect(binary_mask, 0, 0, w, t, 0)
    _bench4_rect(binary_mask, 0, h - t, w, h, 0)
    _bench4_rect(binary_mask, 0, 0, t, h, 0)
    _bench4_rect(binary_mask, w - t, 0, w, h, 0)


def _bench4_vwall(binary_mask, x, y0, y1, thickness=7, gaps=None):
    if gaps is None:
        gaps = []
    _bench4_rect(binary_mask, x, y0, x + thickness, y1, 0)
    for gy0, gy1 in gaps:
        _bench4_rect(binary_mask, x, gy0, x + thickness, gy1, 1)


def _bench4_hwall(binary_mask, y, x0, x1, thickness=7, gaps=None):
    if gaps is None:
        gaps = []
    _bench4_rect(binary_mask, x0, y, x1, y + thickness, 0)
    for gx0, gx1 in gaps:
        _bench4_rect(binary_mask, gx0, y, gx1, y + thickness, 1)


def make_bench_simple_maze_problem_v4():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench4_boundary(binary_mask, thickness=4)

    _bench4_vwall(binary_mask, x=34, y0=18, y1=102, thickness=7, gaps=[(72, 90)])
    _bench4_hwall(binary_mask, y=44, x0=34, x1=92, thickness=7, gaps=[(58, 72)])
    _bench4_vwall(binary_mask, x=86, y0=28, y1=94, thickness=7, gaps=[(50, 66)])

    _bench4_rect(binary_mask, 54, 82, 70, 96, 0)
    _bench4_rect(binary_mask, 12, 34, 22, 46, 0)

    start = np.array([14, 18], dtype=float)
    goal = np.array([104, 98], dtype=float)
    return binary_mask, start, goal


def make_bench_complex_maze_problem_v4():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench4_boundary(binary_mask, thickness=4)

    _bench4_hwall(binary_mask, y=26, x0=8, x1=92, thickness=7, gaps=[(14, 38), (62, 82)])
    _bench4_vwall(binary_mask, x=28, y0=26, y1=112, thickness=7, gaps=[(44, 70), (90, 108)])
    _bench4_hwall(binary_mask, y=56, x0=28, x1=112, thickness=7, gaps=[(40, 62), (84, 106)])
    _bench4_vwall(binary_mask, x=64, y0=8, y1=96, thickness=7, gaps=[(14, 38), (64, 88)])
    _bench4_hwall(binary_mask, y=88, x0=8, x1=96, thickness=7, gaps=[(18, 42), (54, 78)])
    _bench4_vwall(binary_mask, x=94, y0=26, y1=112, thickness=7, gaps=[(10, 32), (38, 60), (72, 96)])

    for block in [
        (42, 10, 54, 22),
        (76, 34, 88, 46),
        (44, 72, 56, 82),
        (104, 94, 114, 106),
        (12, 66, 22, 78),
    ]:
        _bench4_rect(binary_mask, *block, value=0)

    start = np.array([14, 104], dtype=float)
    goal = np.array([95, 28], dtype=float)
    return binary_mask, start, goal


def make_bench_concave_trap_problem_v4():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench4_boundary(binary_mask, thickness=4)

    _bench4_rect(binary_mask, 42, 28, 88, 38, 0)
    _bench4_rect(binary_mask, 42, 72, 98, 82, 0)
    _bench4_rect(binary_mask, 88, 28, 98, 82, 0)
    _bench4_rect(binary_mask, 52, 50, 70, 60, 0)

    _bench4_rect(binary_mask, 64, 12, 78, 24, 0)
    _bench4_rect(binary_mask, 102, 44, 112, 58, 0)

    _bench4_rect(binary_mask, 20, 88, 36, 102, 0)
    _bench4_rect(binary_mask, 56, 94, 72, 106, 0)

    start = np.array([80, 54], dtype=float)
    goal = np.array([108, 66], dtype=float)
    return binary_mask, start, goal


def make_bench_multi_gap_problem_v4():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench4_boundary(binary_mask, thickness=4)

    _bench4_vwall(binary_mask, x=28, y0=8, y1=112, thickness=7, gaps=[(24, 42), (90, 106)])
    _bench4_vwall(binary_mask, x=52, y0=8, y1=112, thickness=7, gaps=[(58, 76)])
    _bench4_vwall(binary_mask, x=78, y0=8, y1=112, thickness=7, gaps=[(30, 48), (82, 98)])

    _bench4_hwall(binary_mask, y=70, x0=52, x1=104, thickness=7, gaps=[(82, 98)])
    _bench4_rect(binary_mask, 10, 18, 20, 30, 0)
    _bench4_rect(binary_mask, 36, 44, 46, 56, 0)
    _bench4_rect(binary_mask, 96, 18, 108, 32, 0)
    _bench4_rect(binary_mask, 98, 96, 110, 108, 0)

    start = np.array([12, 62], dtype=float)
    goal = np.array([70, 86], dtype=float)
    return binary_mask, start, goal


def make_bench_cluttered_problem_v4():
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench4_boundary(binary_mask, thickness=4)

    blocks = [
        (18, 12, 30, 26), (40, 14, 52, 28), (72, 12, 84, 26), (92, 16, 106, 30),
        (12, 38, 26, 52), (34, 44, 48, 60), (64, 40, 78, 56), (92, 44, 106, 60),
        (22, 66, 36, 82), (52, 66, 66, 82), (80, 70, 94, 86),
        (40, 92, 54, 106), (68, 92, 82, 108), (96, 88, 110, 102),
        (56, 22, 66, 34), (30, 30, 40, 42), (78, 30, 88, 42), (44, 76, 54, 88),
    ]

    for block in blocks:
        _bench4_rect(binary_mask, *block, value=0)

    start = np.array([12, 92], dtype=float)
    goal = np.array([100, 12], dtype=float)
    return binary_mask, start, goal


_BENCHMARK_SCENE_BUILDERS_V4 = {
    "bench_simple_maze": make_bench_simple_maze_problem_v4,
    "bench_complex_maze": make_bench_complex_maze_problem_v4,
    "bench_concave_trap": make_bench_concave_trap_problem_v4,
    "bench_multi_gap": make_bench_multi_gap_problem_v4,
    "bench_cluttered": make_bench_cluttered_problem_v4,
}


try:
    _make_complex_problem_before_benchmark_scenes_v4
except NameError:
    _make_complex_problem_before_benchmark_scenes_v4 = make_complex_problem


def make_complex_problem(map_name):
    name = str(map_name).strip()

    if name in _BENCHMARK_SCENE_BUILDERS_V4:
        return _BENCHMARK_SCENE_BUILDERS_V4[name]()

    return _make_complex_problem_before_benchmark_scenes_v4(map_name)


# =============================================================================
# Structural challenge benchmark scenes v5 for RAG-RRT* experiments
# =============================================================================
# These scenes supplement benchmark scenes v4 with harder non-straight and
# bottleneck-dominated topologies. They are intended for feasibility preview and
# subsequent learned-prior benchmark extension, not as a replacement for all v4
# results.
#
# Maps:
#   bench_zigzag_corridor  : repeated serpentine / zigzag detours
#   bench_hourglass_narrow : narrow hourglass bottleneck
#   bench_classic_maze     : block-maze corridor with dead ends
#
# Important:
#   binary_mask[y, x] = 1 -> free
#   binary_mask[y, x] = 0 -> obstacle
# =============================================================================

import numpy as np


def _bench5_rect(binary_mask, x0, y0, x1, y1, value=0):
    """Set an axis-aligned rectangle using x/y coordinates."""
    h, w = binary_mask.shape
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(w, int(round(x1)))
    y1 = min(h, int(round(y1)))
    if x0 < x1 and y0 < y1:
        binary_mask[y0:y1, x0:x1] = int(value)


def _bench5_boundary(binary_mask, thickness=4):
    h, w = binary_mask.shape
    t = int(thickness)
    _bench5_rect(binary_mask, 0, 0, w, t, 0)
    _bench5_rect(binary_mask, 0, h - t, w, h, 0)
    _bench5_rect(binary_mask, 0, 0, t, h, 0)
    _bench5_rect(binary_mask, w - t, 0, w, h, 0)


def _bench5_hwall(binary_mask, y, x0, x1, thickness=6, gaps=None):
    """Draw a horizontal wall and reopen x-interval gaps."""
    if gaps is None:
        gaps = []
    _bench5_rect(binary_mask, x0, y, x1, y + thickness, 0)
    for gx0, gx1 in gaps:
        _bench5_rect(binary_mask, gx0, y, gx1, y + thickness, 1)


def _bench5_vwall(binary_mask, x, y0, y1, thickness=6, gaps=None):
    """Draw a vertical wall and reopen y-interval gaps."""
    if gaps is None:
        gaps = []
    _bench5_rect(binary_mask, x, y0, x + thickness, y1, 0)
    for gy0, gy1 in gaps:
        _bench5_rect(binary_mask, x, gy0, x + thickness, gy1, 1)


def _bench5_polygon(binary_mask, points, value=0):
    """Fill a polygon obstacle/free region using matplotlib.path."""
    try:
        from matplotlib.path import Path as _MplPath
    except Exception as exc:
        raise ImportError(
            "_bench5_polygon requires matplotlib. It is used only for map "
            "rasterization of the hourglass benchmark."
        ) from exc

    h, w = binary_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.vstack((xx.ravel(), yy.ravel())).T
    poly = _MplPath(np.asarray(points, dtype=float))
    inside = poly.contains_points(coords).reshape(h, w)
    binary_mask[inside] = int(value)


def _bench5_carve_segment(binary_mask, p0, p1, width=12, value=1):
    """Carve a thick segment between two xy points."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    half = float(width) / 2.0

    min_x = max(0, int(np.floor(min(p0[0], p1[0]) - half - 1)))
    max_x = min(binary_mask.shape[1], int(np.ceil(max(p0[0], p1[0]) + half + 2)))
    min_y = max(0, int(np.floor(min(p0[1], p1[1]) - half - 1)))
    max_y = min(binary_mask.shape[0], int(np.ceil(max(p0[1], p1[1]) + half + 2)))

    if min_x >= max_x or min_y >= max_y:
        return

    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    pts = np.stack([xx, yy], axis=-1).astype(float)

    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv <= 1e-12:
        dist = np.linalg.norm(pts - p0, axis=-1)
    else:
        t = ((pts - p0) @ v) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + t[..., None] * v
        dist = np.linalg.norm(pts - proj, axis=-1)

    binary_mask[min_y:max_y, min_x:max_x][dist <= half] = int(value)


def _bench5_carve_polyline(binary_mask, points, width=12, value=1):
    """Carve a thick polyline corridor."""
    points = [np.asarray(p, dtype=float) for p in points]
    for p0, p1 in zip(points[:-1], points[1:]):
        _bench5_carve_segment(binary_mask, p0, p1, width=width, value=value)

    # Round out bends/junctions.
    for p in points:
        _bench5_carve_segment(binary_mask, p, p, width=width, value=value)


def make_bench_zigzag_corridor_problem_v5():
    """
    Repeated zigzag / serpentine corridor benchmark.

    The map contains a sequence of horizontal walls with alternating left/right
    openings. The feasible path is forced to repeatedly switch sides rather than
    following the start-goal diagonal ellipse.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)
    _bench5_boundary(binary_mask, thickness=4)

    wall_thickness = 6
    # y, gap_x0, gap_x1. Gaps alternate right / left / right / left / right.
    wall_specs = [
        (18, 84, 112),
        (36, 8, 36),
        (54, 84, 112),
        (72, 8, 36),
        (90, 84, 112),
    ]

    for y, gx0, gx1 in wall_specs:
        _bench5_hwall(
            binary_mask,
            y=y,
            x0=4,
            x1=116,
            thickness=wall_thickness,
            gaps=[(gx0, gx1)],
        )

    # No extra blockers are added in the first version. The alternating wall
    # openings are already sufficient to force repeated side switches.

    start = np.array([14, 10], dtype=float)
    goal = np.array([106, 110], dtype=float)
    _ensure_start_goal_free(binary_mask, start, goal, radius=4)
    return binary_mask, start, goal


def make_bench_hourglass_narrow_problem_v5():
    """
    Rectangular minimum-passable narrow-passage benchmark.

    This version follows the user's revised sketch:
        - two large rectangular obstacles, one upper and one lower;
        - a very narrow horizontal passage between them;
        - start placed near the upper-left entrance region;
        - goal placed near the lower-right exit region.

    The passage is made close to the minimum feasible width under clearance=2.
    This scene mainly tests narrow-passage traversal and channel entry/exit,
    not long detour length.
    """
    height, width = 120, 120
    binary_mask = np.ones((height, width), dtype=np.uint8)

    _bench5_boundary(binary_mask, thickness=4)

    # Two large rectangular blocks, matching the red-box sketch.
    #
    # Top block:
    #   x = 12..108, y = 4..59
    # Bottom block:
    #   x = 12..108, y = 64..116
    #
    # Therefore the free horizontal channel is y = 59..63, i.e. 5 px high.
    # With clearance=2 and 8-connected A*, this is close to the narrowest
    # passable setting while still producing a validated path.
    _bench5_rect(binary_mask, 12, 4, 108, 59, 0)
    _bench5_rect(binary_mask, 12, 64, 108, 116, 0)

    # Start and goal are placed in the left and right side corridors,
    # respectively, as indicated by the red ellipses in the user's sketch.
    start = np.array([8, 8], dtype=float)
    goal = np.array([112, 112], dtype=float)

    # Small local clearing only. This avoids invalid start/goal cells without
    # widening the narrow horizontal passage.
    _ensure_start_goal_free(binary_mask, start, goal, radius=2)

    return binary_mask, start, goal

def make_bench_classic_maze_problem_v5():
    """
    Classic block-maze benchmark with dead-end branches.

    A deterministic 7 x 7 perfect-maze layout is rasterized into a 120 x 120
    grid. The start and goal are placed in opposite maze cells. The unique maze
    route has many turns and a high detour ratio, which makes the scenario less
    friendly to direct informed-ellipse sampling.
    """
    height, width = 120, 120
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    n_cells = 7
    cell_size = 10
    wall_size = 5
    offset = 4

    import random as _bench5_random
    rng = _bench5_random.Random(0)
    visited = set()
    edges = set()

    def cell_rect(i, j):
        x0 = offset + wall_size + i * (cell_size + wall_size)
        y0 = offset + wall_size + j * (cell_size + wall_size)
        return x0, y0, x0 + cell_size, y0 + cell_size

    def neighbors(cell):
        i, j = cell
        cand = []
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n_cells and 0 <= nj < n_cells:
                cand.append((ni, nj))
        rng.shuffle(cand)
        return cand

    def dfs(cell):
        visited.add(cell)
        for nb in neighbors(cell):
            if nb not in visited:
                edges.add(tuple(sorted([cell, nb])))
                dfs(nb)

    dfs((0, n_cells - 1))

    # Carve maze cells.
    for i in range(n_cells):
        for j in range(n_cells):
            x0, y0, x1, y1 = cell_rect(i, j)
            _bench5_rect(binary_mask, x0, y0, x1, y1, value=1)

    # Carve passages between connected neighboring cells.
    for a, b in edges:
        i, j = a
        k, l = b
        x0, y0, x1, y1 = cell_rect(i, j)
        x2, y2, x3, y3 = cell_rect(k, l)

        if i == k:
            # Vertical neighbors: open the wall between cell rows.
            _bench5_rect(binary_mask, x0, min(y1, y3), x1, max(y0, y2), value=1)
        else:
            # Horizontal neighbors: open the wall between cell columns.
            _bench5_rect(binary_mask, min(x1, x3), y0, max(x0, x2), y1, value=1)

    # Small side chambers make the maze visually less uniform without creating
    # shortcuts across the maze wall graph.
    side_chambers = [
        (6, 46, 16, 56),
        (38, 6, 48, 16),
        (68, 104, 80, 114),
        (98, 54, 112, 66),
    ]
    for block in side_chambers:
        _bench5_rect(binary_mask, *block, value=1)

    # Keep a solid outer boundary.
    _bench5_boundary(binary_mask, thickness=4)

    sx0, sy0, sx1, sy1 = cell_rect(0, n_cells - 1)
    gx0, gy0, gx1, gy1 = cell_rect(n_cells - 1, 0)
    start = np.array([(sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0], dtype=float)
    goal = np.array([(gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0], dtype=float)

    _ensure_start_goal_free(binary_mask, start, goal, radius=3)
    return binary_mask, start, goal


_BENCHMARK_STRUCTURAL_BUILDERS_V5 = {
    "bench_zigzag_corridor": make_bench_zigzag_corridor_problem_v5,
    "bench_hourglass_narrow": make_bench_hourglass_narrow_problem_v5,
    "bench_classic_maze": make_bench_classic_maze_problem_v5,
}


try:
    _make_complex_problem_before_structural_scenes_v5
except NameError:
    _make_complex_problem_before_structural_scenes_v5 = make_complex_problem


def make_complex_problem(map_name):
    name = str(map_name).strip()

    if name in _BENCHMARK_STRUCTURAL_BUILDERS_V5:
        return _BENCHMARK_STRUCTURAL_BUILDERS_V5[name]()

    return _make_complex_problem_before_structural_scenes_v5(map_name)

# === STRUCTURAL_CHALLENGE_BENCHMARK_V5_END ===
