# envs/grid_map_2d.py
import math
import time
import numpy as np


class GridMap2D:
    """
    2D binary grid map.

    binary_mask:
        shape = (height, width)
        1 or True  : free space
        0 or False : obstacle
    """

    def __init__(self, binary_mask: np.ndarray, clearance: int = 0):
        if binary_mask.ndim != 2:
            raise ValueError("binary_mask must be a 2D array.")

        self.binary_mask = (binary_mask > 0).astype(np.uint8)
        self.height, self.width = self.binary_mask.shape
        self.clearance = int(clearance)

        self.x_range = (0.0, float(self.width - 1))
        self.y_range = (0.0, float(self.height - 1))

        self.collision_checks = 0
        self.collision_check_samples = 0
        self.collision_check_time = 0.0

    def reset_statistics(self):
        self.collision_checks = 0
        self.collision_check_samples = 0
        self.collision_check_time = 0.0

    def in_bounds(self, point) -> bool:
        x, y = float(point[0]), float(point[1])
        c = self.clearance
        return c <= x < self.width - c and c <= y < self.height - c

    def point_valid(self, point) -> bool:
        """
        Check whether a continuous point is inside the map and outside obstacles.
        The point is rounded to the nearest grid cell.
        """
        if not self.in_bounds(point):
            return False

        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        c = self.clearance
        x0, x1 = max(0, x - c), min(self.width, x + c + 1)
        y0, y1 = max(0, y - c), min(self.height, y + c + 1)

        return np.all(self.binary_mask[y0:y1, x0:x1] > 0)

    def edge_valid(self, start, end, resolution: float = 1.0) -> bool:
        """
        Check whether the line segment from start to end is collision-free
        by discretizing the segment.
        """
        check_start_time = time.perf_counter()
        self.collision_checks += 1

        try:
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)

            dist = np.linalg.norm(end - start)
            if dist < 1e-12:
                self.collision_check_samples += 1
                return self.point_valid(start)

            num_steps = max(2, int(math.ceil(dist / resolution)) + 1)

            for t in np.linspace(0.0, 1.0, num_steps):
                self.collision_check_samples += 1
                p = start + t * (end - start)
                if not self.point_valid(p):
                    return False

            return True
        finally:
            self.collision_check_time += time.perf_counter() - check_start_time

    def sample_free(self) -> np.ndarray:
        """
        Uniformly sample a valid point from free space.
        """
        while True:
            x = np.random.uniform(self.clearance, self.width - 1 - self.clearance)
            y = np.random.uniform(self.clearance, self.height - 1 - self.clearance)
            p = np.array([x, y], dtype=float)
            if self.point_valid(p):
                return p

    def path_cost(self, path) -> float:
        if path is None or len(path) == 0:
            return float("inf")
        path = np.asarray(path, dtype=float)
        return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
