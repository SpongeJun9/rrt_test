"""Official PointNet++ guided baselines on the rg_rrt common RRT* backend.

This adapter deliberately separates two layers:

1. The guidance layer is taken from the official NIRRT* implementation:
   free-space point clouds, the official PointNet++ checkpoint and inference
   wrapper, optional Neural Connect (BFS), Neural Focus, and cost-triggered
   point-cloud refresh.
2. Tree expansion, rewiring, collision checking, stopping rules, and metrics
   use rg_rrt's existing RRT*/Informed-RRT* implementation.

The separation makes comparisons on rg_rrt maps use identical collision and
metric semantics while preserving the published neural-guidance mechanism.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from planners.informed_rrt_star_2d import InformedRRTStar2D
from planners.rrt_star_2d import RRTStar2D


_WRAPPER_CACHE: dict[tuple[str, str, bool], "_TimedOfficialPNGWrapper"] = {}


def _seed_torch(seed: int) -> None:
    """Seed PyTorch after model construction so cached/non-cached runs match."""
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _prepare_official_imports(nirrt_root: str | Path) -> Path:
    root = Path(nirrt_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NIRRT* repository root not found: {root}")

    checkpoint = (
        root
        / "results"
        / "model_training"
        / "pointnet2_2d"
        / "checkpoints"
        / "best_pointnet2_2d.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Official PointNet++ checkpoint not found: " + str(checkpoint)
        )

    root_text = str(root)
    if root_text not in sys.path:
        # Append instead of prepend so rg_rrt's own top-level modules keep
        # precedence.  The official packages used here have unique names.
        sys.path.append(root_text)
    return root


class _TimedOfficialPNGWrapper:
    """Small instrumentation layer around the official PointNet++ wrapper."""

    def __init__(self, nirrt_root: Path, device: str, use_connect: bool):
        self.nirrt_root = nirrt_root
        self.device = str(device)
        self.use_connect = bool(use_connect)

        module_name = (
            "wrapper.pointnet_pointnet2.pointnet2_wrapper_connect_bfs"
            if self.use_connect
            else "wrapper.pointnet_pointnet2.pointnet2_wrapper"
        )
        module = importlib.import_module(module_name)

        load_start = time.perf_counter()
        self.inner = module.PNGWrapper(
            root_dir=str(self.nirrt_root),
            device=self.device,
        )
        self.model_load_time_s = time.perf_counter() - load_start

        self._original_classify = self.inner.classify_path_points

        def timed_classify(*args, **kwargs):
            start = time.perf_counter()
            try:
                return self._original_classify(*args, **kwargs)
            finally:
                self.inference_count += 1
                self.inference_time_s += time.perf_counter() - start

        # Neural Connect calls self.classify_path_points internally. Replacing
        # the bound method here therefore counts every network forward pass,
        # including its iterative connection trials.
        self.inner.classify_path_points = timed_classify
        self.reset_run(seed=0)

    def reset_run(self, seed: int) -> None:
        self.inference_count = 0
        self.inference_time_s = 0.0
        self.connect_call_count = 0
        self.connect_time_s = 0.0
        _seed_torch(seed)

    def classify_path_points(self, *args, **kwargs):
        return self.inner.classify_path_points(*args, **kwargs)

    def generate_connected_path_points(self, *args, **kwargs):
        if not self.use_connect:
            raise RuntimeError("Neural Connect was not enabled for this wrapper.")
        start = time.perf_counter()
        try:
            return self.inner.generate_connected_path_points(*args, **kwargs)
        finally:
            self.connect_call_count += 1
            self.connect_time_s += time.perf_counter() - start

    def diagnostics(self) -> dict[str, Any]:
        return {
            "official_png_inference_count": int(self.inference_count),
            "official_png_inference_time_s": float(self.inference_time_s),
            "official_neural_connect_call_count": int(self.connect_call_count),
            "official_neural_connect_time_s": float(self.connect_time_s),
            "official_model_load_time_once_s": float(self.model_load_time_s),
        }


def _get_wrapper(
    nirrt_root: str | Path,
    device: str,
    use_connect: bool,
    seed: int,
) -> tuple[_TimedOfficialPNGWrapper, bool]:
    root = _prepare_official_imports(nirrt_root)
    key = (str(root), str(device), bool(use_connect))
    cache_hit = key in _WRAPPER_CACHE
    if not cache_hit:
        _WRAPPER_CACHE[key] = _TimedOfficialPNGWrapper(
            nirrt_root=root,
            device=device,
            use_connect=use_connect,
        )
    wrapper = _WRAPPER_CACHE[key]
    wrapper.reset_run(seed=seed)
    return wrapper, cache_hit


class _OfficialPointNetGuidanceMixin:
    """Shared official PointNet++ guidance used by NRRT* and NIRRT*."""

    def _init_official_guidance(
        self,
        nirrt_root: str | Path,
        device: str,
        seed: int,
        use_neural_connect: bool,
        pc_n_points: int,
        pc_over_sample_scale: int,
        pc_sample_rate: float,
        pc_update_cost_ratio: float,
        connect_max_trial_attempts: int,
    ) -> None:
        self.official_nirrt_root = str(Path(nirrt_root).expanduser().resolve())
        self.official_device = str(device)
        self.official_seed = int(seed)
        self.use_neural_connect = bool(use_neural_connect)
        self.pc_n_points = int(pc_n_points)
        self.pc_over_sample_scale = int(pc_over_sample_scale)
        self.pc_sample_rate = float(np.clip(pc_sample_rate, 0.0, 1.0))
        self.pc_update_cost_ratio = float(pc_update_cost_ratio)
        self.connect_max_trial_attempts = int(connect_max_trial_attempts)
        self.pc_neighbor_radius = float(self.step_len)

        if self.pc_n_points < 1024:
            raise ValueError(
                "The official PointNet++ architecture expects at least 1024 "
                "points; pc_n_points must be >= 1024."
            )
        if self.pc_over_sample_scale < 1:
            raise ValueError("pc_over_sample_scale must be >= 1.")
        if not 0.0 < self.pc_update_cost_ratio <= 1.0:
            raise ValueError("pc_update_cost_ratio must be in (0, 1].")

        self.png_wrapper, self.official_wrapper_cache_hit = _get_wrapper(
            nirrt_root=self.official_nirrt_root,
            device=self.official_device,
            use_connect=self.use_neural_connect,
            seed=self.official_seed,
        )
        point_cloud_module = importlib.import_module(
            "datasets.point_cloud_mask_utils"
        )
        self._generate_rectangle_point_cloud = (
            point_cloud_module.generate_rectangle_point_cloud
        )
        self._ellipsoid_point_cloud_sampling = (
            point_cloud_module.ellipsoid_point_cloud_sampling
        )
        self._get_point_cloud_mask_around_points = (
            point_cloud_module.get_point_cloud_mask_around_points
        )

        self._reset_guidance_state()

    def _reset_guidance_state(self) -> None:
        self.path_point_cloud_pred: np.ndarray | None = None
        self.current_point_cloud: np.ndarray | None = None
        self.last_pc_update_cost = float("inf")
        self.pc_update_count = 0
        self.pc_update_time_s = 0.0
        self.pc_generation_time_s = 0.0
        self.pc_last_over_sample_scale = self.pc_over_sample_scale
        self.guided_sample_attempts = 0
        self.guided_sample_successes = 0
        self.guided_empty_fallbacks = 0
        self.uniform_or_informed_samples = 0
        self.neural_connect_success_count = 0
        self.neural_connect_png_runs = 0
        self.png_wrapper.reset_run(seed=self.official_seed)

    def reset(self):
        super().reset()
        self._reset_guidance_state()

    def _rectangle_point_cloud(self) -> np.ndarray:
        binary_mask = np.asarray(self.map.binary_mask)
        free_fraction = float(np.mean(binary_mask > 0))
        if free_fraction <= 0.0:
            raise RuntimeError("The map contains no free cells.")

        # The official default scale=5 can yield fewer than 2048 raw free
        # points on low-free-volume maps (notably the hourglass scene). Increase
        # only the raw rejection-sampling budget; the network input remains the
        # official fixed 2048-point representation.
        adaptive_scale = max(
            self.pc_over_sample_scale,
            int(math.ceil(1.5 / free_fraction)),
        )
        last_error: Exception | None = None
        for _ in range(4):
            try:
                pc = self._generate_rectangle_point_cloud(
                    binary_mask,
                    self.pc_n_points,
                    adaptive_scale,
                )
                pc = np.asarray(pc, dtype=np.float64)
                if len(pc) == self.pc_n_points:
                    self.pc_last_over_sample_scale = adaptive_scale
                    return pc
                last_error = RuntimeError(
                    f"generated {len(pc)} points, expected {self.pc_n_points}"
                )
            except Exception as exc:  # retry with a larger raw sample budget
                last_error = exc
            adaptive_scale *= 2

        raise RuntimeError(
            "Unable to generate a full official rectangular point cloud; "
            f"last error: {last_error!r}"
        )

    def _informed_point_cloud(self, cmax: float, cmin: float) -> np.ndarray:
        if not np.isfinite(cmax) or cmin <= 1e-12:
            return self._rectangle_point_cloud()

        ratio = max(float(cmax / cmin), 1.0)
        raw_count = self.pc_n_points * self.pc_over_sample_scale
        last_pc = np.empty((0, 2), dtype=np.float64)
        for _ in range(5):
            pc = self._ellipsoid_point_cloud_sampling(
                self.start,
                self.goal,
                ratio,
                np.asarray(self.map.binary_mask),
                self.pc_n_points,
                n_raw_samples=raw_count,
            )
            last_pc = np.asarray(pc, dtype=np.float64)
            if len(last_pc) == self.pc_n_points:
                return last_pc
            raw_count *= 2

        raise RuntimeError(
            "Unable to generate a full informed point cloud: "
            f"generated {len(last_pc)}, expected {self.pc_n_points}."
        )

    def _update_point_cloud(self, cmax: float = float("inf")) -> None:
        update_start = time.perf_counter()
        generation_start = time.perf_counter()
        cmin = float(np.linalg.norm(self.goal - self.start))
        if np.isfinite(cmax):
            pc = self._informed_point_cloud(cmax=cmax, cmin=cmin)
        else:
            pc = self._rectangle_point_cloud()
        self.pc_generation_time_s += time.perf_counter() - generation_start

        if self.use_neural_connect:
            connection_success, num_png_runs, path_pred = (
                self.png_wrapper.generate_connected_path_points(
                    pc.astype(np.float32),
                    self.start.astype(np.float64),
                    self.goal.astype(np.float64),
                    {
                        "env_dims": tuple(np.asarray(self.map.binary_mask).shape),
                        "circle_obstacles": [],
                        "rectangle_obstacles": [],
                    },
                    neighbor_radius=self.pc_neighbor_radius,
                    max_trial_attempts=self.connect_max_trial_attempts,
                )
            )
            self.neural_connect_success_count += int(bool(connection_success))
            self.neural_connect_png_runs += int(num_png_runs)
        else:
            start_mask = self._get_point_cloud_mask_around_points(
                pc,
                self.start[np.newaxis, :],
                self.pc_neighbor_radius,
            )
            goal_mask = self._get_point_cloud_mask_around_points(
                pc,
                self.goal[np.newaxis, :],
                self.pc_neighbor_radius,
            )
            path_pred, _ = self.png_wrapper.classify_path_points(
                pc.astype(np.float32),
                start_mask.astype(np.float32),
                goal_mask.astype(np.float32),
            )

        path_pred = np.asarray(path_pred).reshape(-1)
        self.current_point_cloud = pc
        self.path_point_cloud_pred = pc[np.flatnonzero(path_pred > 0)]
        self.last_pc_update_cost = float(cmax)
        self.pc_update_count += 1
        self.pc_update_time_s += time.perf_counter() - update_start

    def _ensure_or_refresh_guidance(self, allow_refresh: bool) -> None:
        if self.current_point_cloud is None:
            self._update_point_cloud(cmax=float("inf"))
            return

        if (
            allow_refresh
            and np.isfinite(self.best_cost)
            and self.best_cost
            < self.pc_update_cost_ratio * self.last_pc_update_cost
        ):
            self._update_point_cloud(cmax=float(self.best_cost))

    def _sample_predicted_point(self) -> np.ndarray | None:
        self.guided_sample_attempts += 1
        if self.path_point_cloud_pred is None or len(self.path_point_cloud_pred) == 0:
            self.guided_empty_fallbacks += 1
            return None
        index = int(np.random.randint(0, len(self.path_point_cloud_pred)))
        self.guided_sample_successes += 1
        return self.path_point_cloud_pred[index].copy()

    def _attach_guidance_metadata(self, result, planner_identity: str):
        diagnostics = self.png_wrapper.diagnostics()
        result.metadata.update(diagnostics)
        result.metadata.update(
            {
                "external_baseline_identity": planner_identity,
                "external_baseline_backend": "rg_rrt_common_rrt_star",
                "official_guidance_source": "tedhuang96/nirrt_star",
                "official_pointnet2_checkpoint": str(
                    Path(self.official_nirrt_root)
                    / "results/model_training/pointnet2_2d/checkpoints/best_pointnet2_2d.pth"
                ),
                "official_wrapper_cache_hit": int(self.official_wrapper_cache_hit),
                "official_use_neural_connect": int(self.use_neural_connect),
                "official_pc_n_points": int(self.pc_n_points),
                "official_pc_over_sample_scale": int(self.pc_over_sample_scale),
                "official_pc_last_over_sample_scale": int(
                    self.pc_last_over_sample_scale
                ),
                "official_pc_sample_rate": float(self.pc_sample_rate),
                "official_pc_update_cost_ratio": float(
                    self.pc_update_cost_ratio
                ),
                "official_pc_update_count": int(self.pc_update_count),
                "official_pc_update_time_s": float(self.pc_update_time_s),
                "official_pc_generation_time_s": float(
                    self.pc_generation_time_s
                ),
                "official_guided_sample_attempts": int(
                    self.guided_sample_attempts
                ),
                "official_guided_sample_successes": int(
                    self.guided_sample_successes
                ),
                "official_guided_empty_fallbacks": int(
                    self.guided_empty_fallbacks
                ),
                "official_uniform_or_informed_samples": int(
                    self.uniform_or_informed_samples
                ),
                "official_neural_connect_success_count": int(
                    self.neural_connect_success_count
                ),
                "official_neural_connect_png_runs": int(
                    self.neural_connect_png_runs
                ),
                "uses_prior_path": 0,
                "uses_conninit": 0,
                "uses_bpc": 0,
                "use_topk_edge_checking": 0,
            }
        )
        return result


class OfficialNeuralRRTStar2D(_OfficialPointNetGuidanceMixin, RRTStar2D):
    """NRRT*: official PointNet++ guidance with the common RRT* backend."""

    def __init__(
        self,
        grid_map,
        start,
        goal,
        step_len: float = 5.0,
        search_radius: float = 15.0,
        iter_max: int = 5000,
        goal_radius: float | None = None,
        *,
        nirrt_root: str | Path,
        device: str = "cuda",
        seed: int = 0,
        pc_n_points: int = 2048,
        pc_over_sample_scale: int = 5,
        pc_sample_rate: float = 0.5,
    ):
        super().__init__(
            grid_map=grid_map,
            start=start,
            goal=goal,
            step_len=step_len,
            search_radius=search_radius,
            iter_max=iter_max,
            goal_radius=goal_radius,
        )
        self._init_official_guidance(
            nirrt_root=nirrt_root,
            device=device,
            seed=seed,
            use_neural_connect=False,
            pc_n_points=pc_n_points,
            pc_over_sample_scale=pc_over_sample_scale,
            pc_sample_rate=pc_sample_rate,
            pc_update_cost_ratio=0.9,
            connect_max_trial_attempts=1,
        )

    def sample(self):
        self._ensure_or_refresh_guidance(allow_refresh=False)
        if np.random.random() < self.pc_sample_rate:
            guided = self._sample_predicted_point()
            if guided is not None:
                return guided
        self.uniform_or_informed_samples += 1
        return self.map.sample_free()

    def solve(self, iter_after_initial=None, stop_at_first_solution=False):
        result = super().solve(
            iter_after_initial=iter_after_initial,
            stop_at_first_solution=stop_at_first_solution,
        )
        return self._attach_guidance_metadata(
            result,
            planner_identity="Neural RRT* (official PointNet++ guidance)",
        )


class OfficialNeuralInformedRRTStar2D(
    _OfficialPointNetGuidanceMixin,
    InformedRRTStar2D,
):
    """NIRRT*: Neural Focus plus optional official BFS Neural Connect."""

    def __init__(
        self,
        grid_map,
        start,
        goal,
        step_len: float = 5.0,
        search_radius: float = 15.0,
        iter_max: int = 5000,
        goal_radius: float | None = None,
        *,
        nirrt_root: str | Path,
        device: str = "cuda",
        seed: int = 0,
        pc_n_points: int = 2048,
        pc_over_sample_scale: int = 5,
        pc_sample_rate: float = 0.5,
        pc_update_cost_ratio: float = 0.9,
        connect_max_trial_attempts: int = 5,
        use_neural_connect: bool = True,
    ):
        super().__init__(
            grid_map=grid_map,
            start=start,
            goal=goal,
            step_len=step_len,
            search_radius=search_radius,
            iter_max=iter_max,
            goal_radius=goal_radius,
        )
        self._init_official_guidance(
            nirrt_root=nirrt_root,
            device=device,
            seed=seed,
            use_neural_connect=use_neural_connect,
            pc_n_points=pc_n_points,
            pc_over_sample_scale=pc_over_sample_scale,
            pc_sample_rate=pc_sample_rate,
            pc_update_cost_ratio=pc_update_cost_ratio,
            connect_max_trial_attempts=connect_max_trial_attempts,
        )

    def sample(self):
        self._ensure_or_refresh_guidance(allow_refresh=True)
        if np.random.random() < self.pc_sample_rate:
            guided = self._sample_predicted_point()
            if guided is not None:
                return guided
        self.uniform_or_informed_samples += 1
        return self.sample_informed_subset()

    def solve(self, iter_after_initial=None, stop_at_first_solution=False):
        result = super().solve(
            iter_after_initial=iter_after_initial,
            stop_at_first_solution=stop_at_first_solution,
        )
        return self._attach_guidance_metadata(
            result,
            planner_identity=(
                "Neural Informed RRT* (official PointNet++ + Neural Connect)"
                if self.use_neural_connect
                else "Neural Informed RRT* (official PointNet++)"
            ),
        )

