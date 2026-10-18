#!/usr/bin/env python3
"""OccupancyGrid geometry, occupancy, and planner-coordinate helpers.

The classes in this module do not depend on rclpy or concrete ROS message
classes. ``OccupancyGridMap.from_message`` accepts an OccupancyGrid-like
object through duck typing, which keeps the conversion logic independently
testable.

Coordinate conventions
----------------------
* Grid cells are addressed as ``(grid_x, grid_y)``.
* OccupancyGrid storage is row-major:
  ``index = grid_y * width + grid_x``.
* NumPy arrays use ``array[grid_y, grid_x]``.
* No manual y-axis flip is performed.
* The legacy RRT* project treats integer planner coordinates as cell centers.
  Therefore world/planner conversion uses a half-cell offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]
PlannerPoint = Tuple[float, float]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a normalized quaternion."""

    values = (x, y, z, w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Quaternion components must be finite")

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-15:
        raise ValueError("Quaternion norm must be positive")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass(frozen=True)
class MapGeometry:
    """Geometry metadata for world, cell, and planner coordinates."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("height must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not math.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("resolution must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (self.origin_x, self.origin_y, self.origin_yaw)
        ):
            raise ValueError("map origin values must be finite")

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    def contains_grid(self, grid_x: int, grid_y: int) -> bool:
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height

    def world_to_local(self, world_x: float, world_y: float) -> WorldPoint:
        """Transform a map-frame world point into the unrotated map frame."""

        if not math.isfinite(world_x) or not math.isfinite(world_y):
            raise ValueError("world coordinates must be finite")

        dx = world_x - self.origin_x
        dy = world_y - self.origin_y

        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)

        # Inverse of the map-origin rigid transform.
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return local_x, local_y

    def local_to_world(self, local_x: float, local_y: float) -> WorldPoint:
        """Transform a point in the unrotated map frame into world coordinates."""

        if not math.isfinite(local_x) or not math.isfinite(local_y):
            raise ValueError("local coordinates must be finite")

        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)

        world_x = self.origin_x + cos_yaw * local_x - sin_yaw * local_y
        world_y = self.origin_y + sin_yaw * local_x + cos_yaw * local_y
        return world_x, world_y

    def world_to_grid(
        self,
        world_x: float,
        world_y: float,
    ) -> Optional[GridCell]:
        """Return the OccupancyGrid cell containing a world point."""

        try:
            local_x, local_y = self.world_to_local(world_x, world_y)
        except ValueError:
            return None

        grid_x = math.floor(local_x / self.resolution)
        grid_y = math.floor(local_y / self.resolution)

        if not self.contains_grid(grid_x, grid_y):
            return None
        return int(grid_x), int(grid_y)

    def grid_to_world(
        self,
        grid_x: int,
        grid_y: int,
        *,
        center: bool = True,
    ) -> WorldPoint:
        """Convert a valid OccupancyGrid cell to a world point."""

        if not self.contains_grid(grid_x, grid_y):
            raise IndexError(
                f"grid cell ({grid_x}, {grid_y}) is outside "
                f"{self.width}x{self.height}"
            )

        offset = 0.5 if center else 0.0
        local_x = (grid_x + offset) * self.resolution
        local_y = (grid_y + offset) * self.resolution
        return self.local_to_world(local_x, local_y)

    def world_to_planner(
        self,
        world_x: float,
        world_y: float,
    ) -> Optional[PlannerPoint]:
        """Convert world coordinates to legacy RRT* continuous coordinates.

        The external ``GridMap2D`` rounds a continuous planner point to the
        nearest integer cell. Integer planner coordinates therefore represent
        cell centers. The half-cell offset preserves that convention.
        """

        if self.world_to_grid(world_x, world_y) is None:
            return None

        local_x, local_y = self.world_to_local(world_x, world_y)
        planner_x = local_x / self.resolution - 0.5
        planner_y = local_y / self.resolution - 0.5

        # GridMap2D samples in [0, width-1] / [0, height-1]. Clamp only the
        # half-cell fringe at map boundaries; interior points are unchanged.
        planner_x = min(max(planner_x, 0.0), float(self.width - 1))
        planner_y = min(max(planner_y, 0.0), float(self.height - 1))
        return planner_x, planner_y

    def planner_to_world(
        self,
        planner_x: float,
        planner_y: float,
    ) -> WorldPoint:
        """Convert legacy RRT* continuous coordinates to world coordinates."""

        if not math.isfinite(planner_x) or not math.isfinite(planner_y):
            raise ValueError("planner coordinates must be finite")

        local_x = (planner_x + 0.5) * self.resolution
        local_y = (planner_y + 0.5) * self.resolution
        return self.local_to_world(local_x, local_y)

    def grid_to_index(self, grid_x: int, grid_y: int) -> int:
        """Return the row-major OccupancyGrid data index."""

        if not self.contains_grid(grid_x, grid_y):
            raise IndexError(
                f"grid cell ({grid_x}, {grid_y}) is outside "
                f"{self.width}x{self.height}"
            )
        return grid_y * self.width + grid_x

    def index_to_grid(self, index: int) -> GridCell:
        """Return ``(grid_x, grid_y)`` for a valid row-major index."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self.cell_count:
            raise IndexError(
                f"index {index} is outside data length {self.cell_count}"
            )
        grid_y, grid_x = divmod(index, self.width)
        return grid_x, grid_y


@dataclass
class OccupancyGridMap:
    """Validated occupancy data together with map geometry."""

    geometry: MapGeometry
    data: Sequence[int]
    occupied_threshold: int = 50
    unknown_is_occupied: bool = True
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if not 0 <= self.occupied_threshold <= 100:
            raise ValueError("occupied_threshold must be in [0, 100]")
        if len(self.data) != self.geometry.cell_count:
            raise ValueError(
                "OccupancyGrid data length does not match width * height: "
                f"{len(self.data)} != {self.geometry.cell_count}"
            )

    @classmethod
    def from_message(
        cls,
        message,
        *,
        occupied_threshold: int = 50,
        unknown_is_occupied: bool = True,
    ) -> "OccupancyGridMap":
        """Build a validated map from a ROS OccupancyGrid-like object."""

        info = message.info
        origin = info.origin
        orientation = origin.orientation

        geometry = MapGeometry(
            width=int(info.width),
            height=int(info.height),
            resolution=float(info.resolution),
            origin_x=float(origin.position.x),
            origin_y=float(origin.position.y),
            origin_yaw=quaternion_to_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        )

        return cls(
            geometry=geometry,
            data=message.data,
            occupied_threshold=occupied_threshold,
            unknown_is_occupied=unknown_is_occupied,
            frame_id=str(message.header.frame_id),
        )

    def cell_value(self, grid_x: int, grid_y: int) -> int:
        index = self.geometry.grid_to_index(grid_x, grid_y)
        return int(self.data[index])

    def cell_is_free(self, grid_x: int, grid_y: int) -> bool:
        if not self.geometry.contains_grid(grid_x, grid_y):
            return False

        value = self.cell_value(grid_x, grid_y)
        if value < 0:
            return not self.unknown_is_occupied
        return value < self.occupied_threshold

    def world_is_free(self, world_x: float, world_y: float) -> bool:
        grid = self.geometry.world_to_grid(world_x, world_y)
        return grid is not None and self.cell_is_free(*grid)

    def to_binary_mask(self) -> np.ndarray:
        """Return the legacy planner mask with shape ``(height, width)``.

        Values are ``1`` for free and ``0`` for non-free. Unknown cells follow
        ``unknown_is_occupied``. No vertical flip is applied.
        """

        values = np.asarray(self.data, dtype=np.int16).reshape(
            self.geometry.height,
            self.geometry.width,
        )

        known_free = (values >= 0) & (values < self.occupied_threshold)
        if self.unknown_is_occupied:
            free = known_free
        else:
            free = known_free | (values < 0)

        return free.astype(np.uint8, copy=False)

    def segment_is_free(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        *,
        minimum_spacing: float = 0.01,
    ) -> Tuple[bool, int, str]:
        """Check a world-frame segment using at most half-cell spacing."""

        coordinates = (start_x, start_y, goal_x, goal_y)
        if not all(math.isfinite(value) for value in coordinates):
            return False, 0, "Segment coordinates must be finite"
        if not math.isfinite(minimum_spacing) or minimum_spacing <= 0.0:
            raise ValueError("minimum_spacing must be finite and positive")

        distance = math.hypot(goal_x - start_x, goal_y - start_y)
        collision_spacing = max(
            self.geometry.resolution * 0.5,
            minimum_spacing,
        )
        subdivisions = max(
            1,
            math.ceil(distance / collision_spacing - 1.0e-12),
        )

        checks = 0
        for sample_index in range(subdivisions + 1):
            ratio = sample_index / subdivisions
            world_x = start_x + ratio * (goal_x - start_x)
            world_y = start_y + ratio * (goal_y - start_y)

            grid = self.geometry.world_to_grid(world_x, world_y)
            checks += 1

            if grid is None:
                return (
                    False,
                    checks,
                    "Segment leaves the map at world point "
                    f"({world_x:.3f}, {world_y:.3f})",
                )

            if not self.cell_is_free(*grid):
                return (
                    False,
                    checks,
                    f"Segment intersects a non-free cell at grid {grid}",
                )

        return True, checks, "Straight segment is collision-free"
