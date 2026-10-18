# samplers/astar_corridor_sampler.py
import random

import numpy as np

from utils.astar import AStar2D, compute_path_cost


class AStarCorridorSampler:
    """
    Guided sampler based on a coarse A* path.

    Sampling process:
        1. Randomly choose a point from the prior path.
        2. Add Gaussian noise around the selected point.
        3. Accept it if it is valid.
        4. If repeated attempts fail, return None.

    Backward-compatible behavior:
        - If prior_path / prior_result is not provided, the sampler internally
          runs A* and optionally applies path_shift / prior_failure_mode.
        - If prior_path / prior_result is provided, the sampler directly uses
          the external prior path and does not run A*, path_shift, or
          prior_failure_mode.
    """

    def __init__(
        self,
        grid_map,
        start,
        goal,
        sigma: float = 8.0,
        max_attempts: int = 50,
        clearance: int | None = None,
        heuristic_type: str = "euclidean",
        path_shift: tuple[float, float] | None = None,
        prior_failure_mode: str = "none",
        prior_failure_params: dict | None = None,
        prior_path=None,
        prior_result=None,
    ):
        self.grid_map = grid_map
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)

        self.sigma = float(sigma)
        self.max_attempts = int(max_attempts)

        if clearance is None:
            clearance = grid_map.clearance
        self.clearance = int(clearance)

        self.heuristic_type = heuristic_type

        self.path_shift = (
            np.asarray(path_shift, dtype=float)
            if path_shift is not None
            else np.zeros(2, dtype=float)
        )

        self.prior_failure_mode = (
            "none" if prior_failure_mode is None else str(prior_failure_mode)
        )

        self.prior_failure_params = (
            {} if prior_failure_params is None else dict(prior_failure_params)
        )

        self.original_path = []
        self.shifted_path = []
        self.path = []
        self.path_cost = float("inf")
        self.valid_prior = False

        self.prior_source = None
        self.prior_result = None

        # ------------------------------------------------------------------
        # External prior mode
        # ------------------------------------------------------------------
        # If prior_result or prior_path is provided, this sampler should only
        # use the given external prior path for guided sampling.
        #
        # In this mode, AStarCorridorSampler does NOT:
        #   - run A*
        #   - apply path_shift
        #   - apply prior_failure_mode
        #
        # This preserves the separation between:
        #   external prior generation
        # and
        #   guided corridor sampling.
        # ------------------------------------------------------------------
        external_prior_path = self._extract_external_prior_path(
            prior_path=prior_path,
            prior_result=prior_result,
        )

        if external_prior_path is not None:
            external_path_list = self._path_array_to_list(external_prior_path)

            self.original_path = list(external_path_list)
            self.shifted_path = list(external_path_list)
            self.path = list(external_path_list)

            self.path_cost = compute_path_cost(self.path)
            self.valid_prior = len(self.path) > 0

            self.prior_source = "external"
            self.prior_result = prior_result

            return

        # ------------------------------------------------------------------
        # Backward-compatible internal A* mode
        # ------------------------------------------------------------------
        self._build_prior_path()

        self.prior_source = "internal_astar"
        self.prior_result = None

    def _apply_prior_failure_mode(self, path):
        """
        Apply deterministic prior failure transformations.

        Supported modes:
            none
            partial_start
            partial_goal
            broken_middle
            endpoint_mismatch

        The returned path is the actual prior used by the guided sampler and
        by the connectivity diagnostic.
        """
        if path is None or len(path) == 0:
            return []

        arr = np.asarray(path, dtype=float)

        if len(arr) < 2:
            return [p.astype(float) for p in arr]

        mode = self.prior_failure_mode
        params = self.prior_failure_params

        if mode == "none":
            transformed = arr

        elif mode == "partial_start":
            keep_ratio = float(params.get("keep_ratio", 0.45))
            keep_ratio = float(np.clip(keep_ratio, 0.1, 0.9))

            k = max(2, int(len(arr) * keep_ratio))
            transformed = arr[:k]

        elif mode == "partial_goal":
            keep_ratio = float(params.get("keep_ratio", 0.45))
            keep_ratio = float(np.clip(keep_ratio, 0.1, 0.9))

            k = max(2, int(len(arr) * keep_ratio))
            transformed = arr[-k:]

        elif mode == "broken_middle":
            keep_ratio_each_side = float(
                params.get("keep_ratio_each_side", 0.30)
            )
            keep_ratio_each_side = float(
                np.clip(keep_ratio_each_side, 0.1, 0.45)
            )

            n = len(arr)
            k = max(2, int(n * keep_ratio_each_side))

            start_part = arr[:k]
            goal_part = arr[-k:]

            # Directly concatenating start and goal parts creates a large
            # invalid edge between them. The connectivity diagnostic should
            # detect this through edge validity / component structure.
            transformed = np.vstack([start_part, goal_part])

        elif mode == "endpoint_mismatch":
            shift = params.get("shift", (18.0, 18.0))
            shift = np.asarray(shift, dtype=float).reshape(2)
            transformed = arr + shift

        else:
            raise ValueError(f"Unknown prior_failure_mode: {mode}")

        return [np.asarray(p, dtype=float) for p in transformed]

    def _build_prior_path(self):
        astar = AStar2D(
            self.grid_map.binary_mask,
            clearance=self.clearance,
            heuristic_type=self.heuristic_type,
        )

        path, _ = astar.search(self.start, self.goal)

        if path is None or len(path) == 0:
            self.original_path = []
            self.shifted_path = []
            self.path = []
            self.path_cost = float("inf")
            self.valid_prior = False
            return

        self.original_path = [np.asarray(p, dtype=float) for p in path]

        shifted_path = []
        for p in self.original_path:
            shifted_path.append(p + self.path_shift)

        self.shifted_path = shifted_path

        transformed_path = self._apply_prior_failure_mode(self.shifted_path)

        self.path = transformed_path
        self.path_cost = compute_path_cost(self.path)

        # valid_prior means the prior path exists.
        # Even if the transformed prior is shifted, partial, broken, or endpoint
        # mismatched, it is still allowed to be used as guidance.
        self.valid_prior = len(self.path) > 0

    def sample(self):
        """
        Return:
            np.ndarray shape=(2,), or None if sampling fails.
        """
        if not self.valid_prior or len(self.path) == 0:
            return None

        for _ in range(self.max_attempts):
            center = random.choice(self.path)
            point = center + np.random.normal(loc=0.0, scale=self.sigma, size=2)

            if self.grid_map.point_valid(point):
                return point.astype(float)

        return None

    def get_prior_path(self):
        if not self.valid_prior:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.path, dtype=float)

    def get_shifted_path(self):
        if len(self.shifted_path) == 0:
            return np.empty((0, 2), dtype=float)

        return np.asarray(self.shifted_path, dtype=float)

    def get_original_path(self):
        if len(self.original_path) == 0:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.original_path, dtype=float)

    @staticmethod
    def _extract_external_prior_path(prior_path=None, prior_result=None):
        """
        Extract external prior path from either prior_path or prior_result.

        Return:
            np.ndarray with shape (N, 2), or None if no external prior is given.
        """
        if prior_result is not None:
            if hasattr(prior_result, "path_array"):
                path_arr = prior_result.path_array()
            elif hasattr(prior_result, "path"):
                path_arr = prior_result.path
            else:
                raise ValueError(
                    "prior_result is provided but does not contain path or path_array()."
                )

            return AStarCorridorSampler._coerce_prior_path(path_arr)

        if prior_path is not None:
            return AStarCorridorSampler._coerce_prior_path(prior_path)

        return None

    @staticmethod
    def _coerce_prior_path(path):
        """
        Convert a prior path to np.ndarray with shape (N, 2).

        Empty paths are allowed and will return an empty array.
        """
        if path is None:
            return np.empty((0, 2), dtype=float)

        arr = np.asarray(path, dtype=float)

        if arr.size == 0:
            return np.empty((0, 2), dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"prior_path must have shape (N, 2), got shape {arr.shape}."
            )

        return arr.reshape(-1, 2)

    @staticmethod
    def _path_array_to_list(path_arr):
        """
        Convert an array-like path to list[np.ndarray].

        The old sampler stores paths as lists of 2D numpy arrays. Keeping this
        internal representation avoids changing sample(), get_prior_path(),
        and downstream code.
        """
        path_arr = AStarCorridorSampler._coerce_prior_path(path_arr)
        return [np.asarray(p, dtype=float) for p in path_arr]

    @staticmethod
    def _compute_path_cost(path):
        """
        Compute Euclidean length of a path.

        This helper is kept for compatibility, but the sampler mainly uses
        utils.astar.compute_path_cost().
        """
        path = np.asarray(path, dtype=float)

        if path.ndim != 2 or path.shape[0] < 2:
            return float("inf")

        return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())