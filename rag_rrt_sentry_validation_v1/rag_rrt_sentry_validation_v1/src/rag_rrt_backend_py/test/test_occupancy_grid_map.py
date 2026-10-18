"""Unit tests for map geometry, occupancy semantics, and planner conversion."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rag_rrt_backend_py.occupancy_grid_map import (
    MapGeometry,
    OccupancyGridMap,
    quaternion_to_yaw,
)


def test_translation_only_conversion_and_indexing() -> None:
    geometry = MapGeometry(
        width=4,
        height=3,
        resolution=0.5,
        origin_x=-1.0,
        origin_y=-2.0,
    )

    assert geometry.world_to_grid(-0.75, -1.75) == (0, 0)
    assert geometry.world_to_grid(0.25, -1.25) == (2, 1)
    assert geometry.grid_to_world(2, 1) == pytest.approx((0.25, -1.25))

    assert geometry.grid_to_index(2, 1) == 6
    assert geometry.index_to_grid(6) == (2, 1)


def test_map_boundaries_are_lower_inclusive_upper_exclusive() -> None:
    geometry = MapGeometry(
        width=2,
        height=2,
        resolution=1.0,
        origin_x=1.0,
        origin_y=2.0,
    )

    assert geometry.world_to_grid(1.0, 2.0) == (0, 0)
    assert geometry.world_to_grid(2.999999, 3.999999) == (1, 1)

    assert geometry.world_to_grid(0.999999, 2.0) is None
    assert geometry.world_to_grid(1.0, 1.999999) is None
    assert geometry.world_to_grid(3.0, 2.0) is None
    assert geometry.world_to_grid(1.0, 4.0) is None


def test_rotated_origin_ninety_degrees() -> None:
    geometry = MapGeometry(
        width=4,
        height=3,
        resolution=1.0,
        origin_x=10.0,
        origin_y=-2.0,
        origin_yaw=math.pi / 2.0,
    )

    world = geometry.grid_to_world(1, 0)
    assert world == pytest.approx((9.5, -0.5))
    assert geometry.world_to_grid(*world) == (1, 0)


def test_all_cell_centers_round_trip_with_arbitrary_rotation() -> None:
    geometry = MapGeometry(
        width=7,
        height=5,
        resolution=0.2,
        origin_x=-3.4,
        origin_y=1.7,
        origin_yaw=0.37,
    )

    for grid_y in range(geometry.height):
        for grid_x in range(geometry.width):
            world = geometry.grid_to_world(grid_x, grid_y)
            assert geometry.world_to_grid(*world) == (grid_x, grid_y)


def test_world_planner_round_trip_for_interior_point() -> None:
    geometry = MapGeometry(
        width=100,
        height=80,
        resolution=0.05,
        origin_x=-2.0,
        origin_y=1.5,
        origin_yaw=0.31,
    )

    world = geometry.local_to_world(2.137, 1.283)
    planner = geometry.world_to_planner(*world)
    assert planner is not None
    assert geometry.planner_to_world(*planner) == pytest.approx(world)


def test_integer_planner_coordinates_are_cell_centers() -> None:
    geometry = MapGeometry(
        width=10,
        height=10,
        resolution=0.2,
        origin_x=-1.0,
        origin_y=-1.0,
    )

    assert geometry.planner_to_world(3.0, 4.0) == pytest.approx((-0.3, -0.1))
    assert geometry.world_to_planner(-0.3, -0.1) == pytest.approx((3.0, 4.0))


def test_quaternion_to_yaw_normalizes_input() -> None:
    yaw = 0.8
    z = 2.0 * math.sin(yaw / 2.0)
    w = 2.0 * math.cos(yaw / 2.0)

    assert quaternion_to_yaw(0.0, 0.0, z, w) == pytest.approx(yaw)


def test_occupancy_threshold_unknown_policy_and_binary_mask() -> None:
    geometry = MapGeometry(
        width=3,
        height=2,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )

    conservative = OccupancyGridMap(
        geometry=geometry,
        data=[0, 49, -1, 50, 100, 1],
        occupied_threshold=50,
        unknown_is_occupied=True,
    )
    permissive = OccupancyGridMap(
        geometry=geometry,
        data=[0, 49, -1, 50, 100, 1],
        occupied_threshold=50,
        unknown_is_occupied=False,
    )

    np.testing.assert_array_equal(
        conservative.to_binary_mask(),
        np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        permissive.to_binary_mask(),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.uint8),
    )

    assert conservative.cell_is_free(0, 0)
    assert conservative.cell_is_free(1, 0)
    assert not conservative.cell_is_free(2, 0)
    assert permissive.cell_is_free(2, 0)


def test_binary_mask_is_not_vertically_flipped() -> None:
    geometry = MapGeometry(
        width=2,
        height=2,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )
    grid_map = OccupancyGridMap(
        geometry=geometry,
        data=[0, 100, 100, 0],
    )

    np.testing.assert_array_equal(
        grid_map.to_binary_mask(),
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )


def test_segment_check_detects_occupied_cell() -> None:
    geometry = MapGeometry(
        width=3,
        height=1,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )
    grid_map = OccupancyGridMap(
        geometry=geometry,
        data=[0, 100, 0],
    )

    is_free, checks, message = grid_map.segment_is_free(
        0.5,
        0.5,
        2.5,
        0.5,
    )

    assert not is_free
    assert checks >= 2
    assert "(1, 0)" in message


def test_from_message_reads_origin_rotation_and_frame() -> None:
    yaw = -0.45
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            width=2,
            height=2,
            resolution=0.25,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=-1.0, y=2.5),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            ),
        ),
        data=[0, 1, 50, -1],
    )

    grid_map = OccupancyGridMap.from_message(message)

    assert grid_map.frame_id == "map"
    assert grid_map.geometry.width == 2
    assert grid_map.geometry.height == 2
    assert grid_map.geometry.resolution == pytest.approx(0.25)
    assert grid_map.geometry.origin_x == pytest.approx(-1.0)
    assert grid_map.geometry.origin_y == pytest.approx(2.5)
    assert grid_map.geometry.origin_yaw == pytest.approx(yaw)


def test_invalid_geometry_and_data_are_rejected() -> None:
    with pytest.raises(ValueError):
        MapGeometry(
            width=1,
            height=1,
            resolution=0.0,
            origin_x=0.0,
            origin_y=0.0,
        )

    geometry = MapGeometry(
        width=2,
        height=2,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )

    with pytest.raises(ValueError):
        OccupancyGridMap(geometry=geometry, data=[0, 0, 0])
