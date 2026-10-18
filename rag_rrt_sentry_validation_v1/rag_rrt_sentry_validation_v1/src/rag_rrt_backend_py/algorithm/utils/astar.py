# utils/astar.py
import heapq
import math
from typing import Literal

import numpy as np

"""
    binary_mask 是核心输入, 代表了环境的地图信息, 其中1或True表示自由空间, 0或False表示障碍物
    
"""



class AStar2D:
    """
    A* search on a 2D binary grid map.

    binary_mask:
        shape = (height, width)
        1 or True  : free space
        0 or False : obstacle

    Coordinates:
        state = (x, y), where x is column index, y is row index.
    """

    def __init__(
        self,
        binary_mask: np.ndarray,
        clearance: int = 0,
        heuristic_type: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        if binary_mask.ndim != 2:
            raise ValueError("binary_mask must be a 2D array.")

        self.binary_mask = (binary_mask > 0).astype(np.uint8)
        self.height, self.width = self.binary_mask.shape
        self.clearance = int(clearance)
        self.heuristic_type = heuristic_type

        self.motion = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

    def search(self, start, goal):
        """
        Return:
            path: list[(x, y)], from start to goal.
                  Empty list if no path is found.
            visited: list[(x, y)]
        """
        start = self._to_grid_state(start)
        goal = self._to_grid_state(goal)

        if not self.is_state_valid(start) or not self.is_state_valid(goal):
            return [], []

        open_heap = []
        closed = []
        parent = {start: start}
        g = {start: 0.0}

        heapq.heappush(open_heap, (self.heuristic(start, goal), start))

        found = False

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            closed.append(current)

            if current == goal:
                found = True
                break

            for nxt in self.get_neighbors(current):
                edge_cost = self.edge_cost(current, nxt)
                if math.isinf(edge_cost):
                    continue

                new_cost = g[current] + edge_cost

                if nxt not in g or new_cost < g[nxt]:
                    g[nxt] = new_cost
                    parent[nxt] = current
                    f = new_cost + self.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))

        if not found:
            return [], closed

        return self.extract_path(parent, start, goal), closed

    def get_neighbors(self, state):
        x, y = state
        neighbors = []

        for dx, dy in self.motion:
            nxt = (x + dx, y + dy)
            if self.is_state_valid(nxt):
                neighbors.append(nxt)

        return neighbors

    def edge_cost(self, start, end):
        if not self.is_edge_valid(start, end):
            return math.inf
        return math.hypot(end[0] - start[0], end[1] - start[1])

    def is_state_valid(self, state) -> bool:
        x, y = state
        c = self.clearance

        if x < c or x >= self.width - c:
            return False
        if y < c or y >= self.height - c:
            return False

        x0, x1 = max(0, x - c), min(self.width, x + c + 1)
        y0, y1 = max(0, y - c), min(self.height, y + c + 1)

        return np.all(self.binary_mask[y0:y1, x0:x1] > 0)

    def is_edge_valid(self, start, end) -> bool:
        """
        For 8-connected A*, this mainly prevents diagonal corner cutting.
        """
        if not self.is_state_valid(start) or not self.is_state_valid(end):
            return False

        sx, sy = start
        ex, ey = end

        dx = ex - sx
        dy = ey - sy

        # If diagonal move, check the two side cells to avoid cutting through obstacle corners.
        if abs(dx) == 1 and abs(dy) == 1:
            if not self.is_state_valid((sx + dx, sy)):
                return False
            if not self.is_state_valid((sx, sy + dy)):
                return False

        return True

    def heuristic(self, state, goal):
        if self.heuristic_type == "manhattan":
            return abs(goal[0] - state[0]) + abs(goal[1] - state[1])
        return math.hypot(goal[0] - state[0], goal[1] - state[1])

    @staticmethod
    def extract_path(parent, start, goal):
        if goal not in parent:
            return []

        path = [goal]
        current = goal

        while current != start:
            current = parent[current]
            path.append(current)

        path.reverse()
        return path

    @staticmethod
    def _to_grid_state(point):
        return int(round(float(point[0]))), int(round(float(point[1])))


def compute_path_cost(path) -> float:
    if path is None or len(path) < 2:
        return float("inf")

    path_arr = np.asarray(path, dtype=float)
    return float(np.linalg.norm(path_arr[1:] - path_arr[:-1], axis=1).sum())