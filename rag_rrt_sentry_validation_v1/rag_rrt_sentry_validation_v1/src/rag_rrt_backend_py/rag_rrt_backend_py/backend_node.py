#!/usr/bin/env python3
"""Bundled Python planning backend for ROS 2 robot validation.

Supported modes at this stage:
* ``diagnostic_straight_line``: wiring and coordinate smoke test.
* ``rrt_star``: existing Python RRTStar2D implementation.
* ``irrt_star``: existing Python InformedRRTStar2D implementation.
* ``rag_bpc_conn_init_rel_only``: final RAG-RRT*-Rel controller with
  connectivity-based initial trust and diagnostic-only BPC.
* ``rag_bpc_conn_init_rel_only_topk_fallback``: the final efficiency
  variant using Parent TopK=8, Rewire TopK=12, and bounded fallback 4/6.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path as FilePath
from ament_index_python.packages import get_package_share_directory
import time
from typing import Iterable, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rag_rrt_interfaces.msg import PlannerMetrics
from rag_rrt_interfaces.srv import PlanPath
from visualization_msgs.msg import Marker, MarkerArray

from rag_rrt_backend_py.occupancy_grid_map import OccupancyGridMap
from rag_rrt_backend_py.rrt_star_adapter import (
    PLANNER_DISPLAY_NAMES,
    RAG_REL_PLANNER,
    RAG_REL_TOPK_FALLBACK_PLANNER,
    SUPPORTED_PLANNERS,
    RRTStarAdapter,
    RRTStarAdapterConfig,
)


class PlannerBackendNode(Node):
    """ROS 2 service backend used by the C++ planning gateway."""

    def __init__(self) -> None:
        super().__init__("planner_backend")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("service_name", "/rag_rrt/backend/compute_path")
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("unknown_is_occupied", True)
        self.declare_parameter("path_spacing", 0.10)

        self.declare_parameter("algorithm_project_root", str(FilePath(get_package_share_directory("rag_rrt_backend_py")) / "algorithm"))
        self.declare_parameter("robot_radius_m", 0.5)
        self.declare_parameter("rrt_step_len_cells", 5.0)
        self.declare_parameter("rrt_search_radius_cells", 15.0)
        self.declare_parameter("rrt_iter_max", 3000)
        self.declare_parameter("rrt_iter_after_initial", 300)
        self.declare_parameter("rrt_clearance_cells", 2)
        self.declare_parameter("irrt_max_informed_sample_attempts", 100)

        # Frozen RAG-RRT*-Rel parameters from the final 2-D configuration.
        self.declare_parameter("rag_prior_mode", "coarse_valid")
        self.declare_parameter("rag_prior_heuristic_type", "euclidean")
        self.declare_parameter("rag_guide_sigma_cells", 5.0)
        self.declare_parameter("rag_guide_max_attempts", 50)
        self.declare_parameter("rag_alpha_init", 0.5)
        self.declare_parameter("rag_alpha_min", 0.1)
        self.declare_parameter("rag_alpha_max", 0.8)
        self.declare_parameter("rag_window_size", 30)
        self.declare_parameter("rag_update_interval", 20)
        self.declare_parameter("rag_eta", 0.08)
        self.declare_parameter("rag_margin", 0.05)
        self.declare_parameter("rag_min_samples_to_update", 8)
        self.declare_parameter("rag_w_add", 0.2)
        self.declare_parameter("rag_w_progress", 1.0)
        self.declare_parameter("rag_w_collision", 1.5)
        self.declare_parameter("rag_w_improve", 2.0)
        self.declare_parameter("rag_stagnation_check_interval", 100)
        self.declare_parameter("rag_stagnation_start_iter", 100)
        self.declare_parameter("rag_stagnation_alpha_decay", 0.05)
        self.declare_parameter("rag_best_path_corridor_radius_cells", 10.0)
        self.declare_parameter("rag_min_consistency_samples", 8)
        self.declare_parameter("rag_consistency_low_threshold", 0.35)
        self.declare_parameter("rag_consistency_mid_threshold", 0.55)
        self.declare_parameter("rag_consistency_high_threshold", 0.70)
        self.declare_parameter("rag_connectivity_alpha_init_min", 0.30)
        self.declare_parameter("rag_connectivity_alpha_init_max", 0.50)
        self.declare_parameter("rag_parent_check_top_k", 8)
        self.declare_parameter("rag_rewire_check_top_k", 12)
        self.declare_parameter("rag_parent_fallback_check_k", 4)
        self.declare_parameter("rag_rewire_fallback_check_k", 6)

        self.declare_parameter("publish_debug_markers", True)
        self.declare_parameter(
            "debug_marker_topic",
            "/rag_rrt/debug_markers",
        )
        self.declare_parameter("debug_tree_line_width", 0.012)
        self.declare_parameter("debug_prior_line_width", 0.035)
        self.declare_parameter("debug_vertex_size", 0.035)
        self.declare_parameter("debug_start_goal_size", 0.14)
        self.declare_parameter("debug_marker_z", 0.035)

        self._map_topic = str(self.get_parameter("map_topic").value)
        self._service_name = str(self.get_parameter("service_name").value)
        self._occupied_threshold = int(
            self.get_parameter("occupied_threshold").value
        )
        self._unknown_is_occupied = bool(
            self.get_parameter("unknown_is_occupied").value
        )
        self._path_spacing = float(self.get_parameter("path_spacing").value)
        self._debug_markers_enabled = bool(
            self.get_parameter("publish_debug_markers").value
        )
        self._debug_marker_topic = str(
            self.get_parameter("debug_marker_topic").value
        )
        self._debug_tree_line_width = float(
            self.get_parameter("debug_tree_line_width").value
        )
        self._debug_prior_line_width = float(
            self.get_parameter("debug_prior_line_width").value
        )
        self._debug_vertex_size = float(
            self.get_parameter("debug_vertex_size").value
        )
        self._debug_start_goal_size = float(
            self.get_parameter("debug_start_goal_size").value
        )
        self._debug_marker_z = float(
            self.get_parameter("debug_marker_z").value
        )

        if not 0 <= self._occupied_threshold <= 100:
            raise ValueError("occupied_threshold must be in [0, 100]")
        if self._path_spacing <= 0.0:
            raise ValueError("path_spacing must be positive")
        for name, value in (
            ("debug_tree_line_width", self._debug_tree_line_width),
            ("debug_prior_line_width", self._debug_prior_line_width),
            ("debug_vertex_size", self._debug_vertex_size),
            ("debug_start_goal_size", self._debug_start_goal_size),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self._debug_marker_z):
            raise ValueError("debug_marker_z must be finite")

        adapter_config = RRTStarAdapterConfig(
            project_root=str(
                self.get_parameter("algorithm_project_root").value
            ),
            step_len_cells=float(
                self.get_parameter("rrt_step_len_cells").value
            ),
            search_radius_cells=float(
                self.get_parameter("rrt_search_radius_cells").value
            ),
            iter_max=int(self.get_parameter("rrt_iter_max").value),
            iter_after_initial=int(
                self.get_parameter("rrt_iter_after_initial").value
            ),
            clearance_cells=int(
                self.get_parameter("rrt_clearance_cells").value
            ),
            informed_max_sample_attempts=int(
                self.get_parameter("irrt_max_informed_sample_attempts").value
            ),
            rag_prior_mode=str(self.get_parameter("rag_prior_mode").value),
            rag_prior_heuristic_type=str(
                self.get_parameter("rag_prior_heuristic_type").value
            ),
            rag_guide_sigma_cells=float(
                self.get_parameter("rag_guide_sigma_cells").value
            ),
            rag_guide_max_attempts=int(
                self.get_parameter("rag_guide_max_attempts").value
            ),
            rag_alpha_init=float(self.get_parameter("rag_alpha_init").value),
            rag_alpha_min=float(self.get_parameter("rag_alpha_min").value),
            rag_alpha_max=float(self.get_parameter("rag_alpha_max").value),
            rag_window_size=int(self.get_parameter("rag_window_size").value),
            rag_update_interval=int(
                self.get_parameter("rag_update_interval").value
            ),
            rag_eta=float(self.get_parameter("rag_eta").value),
            rag_margin=float(self.get_parameter("rag_margin").value),
            rag_min_samples_to_update=int(
                self.get_parameter("rag_min_samples_to_update").value
            ),
            rag_w_add=float(self.get_parameter("rag_w_add").value),
            rag_w_progress=float(self.get_parameter("rag_w_progress").value),
            rag_w_collision=float(self.get_parameter("rag_w_collision").value),
            rag_w_improve=float(self.get_parameter("rag_w_improve").value),
            rag_stagnation_check_interval=int(
                self.get_parameter("rag_stagnation_check_interval").value
            ),
            rag_stagnation_start_iter=int(
                self.get_parameter("rag_stagnation_start_iter").value
            ),
            rag_stagnation_alpha_decay=float(
                self.get_parameter("rag_stagnation_alpha_decay").value
            ),
            rag_best_path_corridor_radius_cells=float(
                self.get_parameter("rag_best_path_corridor_radius_cells").value
            ),
            rag_min_consistency_samples=int(
                self.get_parameter("rag_min_consistency_samples").value
            ),
            rag_consistency_low_threshold=float(
                self.get_parameter("rag_consistency_low_threshold").value
            ),
            rag_consistency_mid_threshold=float(
                self.get_parameter("rag_consistency_mid_threshold").value
            ),
            rag_consistency_high_threshold=float(
                self.get_parameter("rag_consistency_high_threshold").value
            ),
            rag_connectivity_alpha_init_min=float(
                self.get_parameter("rag_connectivity_alpha_init_min").value
            ),
            rag_connectivity_alpha_init_max=float(
                self.get_parameter("rag_connectivity_alpha_init_max").value
            ),
            rag_parent_check_top_k=int(
                self.get_parameter("rag_parent_check_top_k").value
            ),
            rag_rewire_check_top_k=int(
                self.get_parameter("rag_rewire_check_top_k").value
            ),
            rag_parent_fallback_check_k=int(
                self.get_parameter("rag_parent_fallback_check_k").value
            ),
            rag_rewire_fallback_check_k=int(
                self.get_parameter("rag_rewire_fallback_check_k").value
            ),
        )
        self._rrt_adapter = RRTStarAdapter(adapter_config)
        self._grid_map: Optional[OccupancyGridMap] = None

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._map_sub = self.create_subscription(
            OccupancyGrid,
            self._map_topic,
            self._map_callback,
            map_qos,
        )
        self._service = self.create_service(
            PlanPath,
            self._service_name,
            self._plan_callback,
        )

        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._debug_marker_pub = self.create_publisher(
            MarkerArray,
            self._debug_marker_topic,
            marker_qos,
        )

        root_display = (
            self._rrt_adapter.configured_project_root or "<not configured>"
        )
        self.get_logger().info(
            "Planner backend ready: "
            f"map={self._map_topic} service={self._service_name} "
            f"rrt_project_root={root_display} "
            f"debug_markers={self._debug_markers_enabled} "
            f"debug_topic={self._debug_marker_topic}"
        )

    def _map_callback(self, msg: OccupancyGrid) -> None:
        try:
            grid_map = OccupancyGridMap.from_message(
                msg,
                occupied_threshold=self._occupied_threshold,
                unknown_is_occupied=self._unknown_is_occupied,
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"Rejected invalid OccupancyGrid: {exc}")
            return

        self._grid_map = grid_map
        geometry = grid_map.geometry

        self.get_logger().info(
            "Map received: "
            f"frame={grid_map.frame_id} "
            f"size={geometry.width}x{geometry.height} "
            f"resolution={geometry.resolution:.4f} "
            f"origin=({geometry.origin_x:.3f}, "
            f"{geometry.origin_y:.3f}, "
            f"yaw={geometry.origin_yaw:.6f})",
            once=True,
        )

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    def _clear_debug_markers(self) -> None:
        if not self._debug_markers_enabled:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = (
            self._grid_map.frame_id if self._grid_map else "map"
        )
        marker.action = Marker.DELETEALL
        self._debug_marker_pub.publish(MarkerArray(markers=[marker]))

    def _publish_rrt_tree_markers(
        self,
        adapter_result,
        request: PlanPath.Request,
        stamp,
    ) -> None:
        if not self._debug_markers_enabled or self._grid_map is None:
            return

        vertices = adapter_result.tree_world_vertices
        parents = adapter_result.tree_parents
        if vertices is None or parents is None:
            return

        vertices = np.asarray(vertices, dtype=float)
        parents = np.asarray(parents, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            self.get_logger().warning(
                f"Skipping malformed debug tree vertices: {vertices.shape}"
            )
            return

        if len(vertices) != len(parents):
            self.get_logger().warning(
                "Skipping debug tree because vertices and parents differ in length"
            )
            return

        frame_id = self._grid_map.frame_id
        markers = MarkerArray()

        tree = Marker()
        tree.header.stamp = stamp
        tree.header.frame_id = frame_id
        tree.ns = "rrt_tree_edges"
        tree.id = 0
        tree.type = Marker.LINE_LIST
        tree.action = Marker.ADD
        tree.pose.orientation.w = 1.0
        tree.scale.x = self._debug_tree_line_width
        tree.color.r = 0.15
        tree.color.g = 0.45
        tree.color.b = 1.0
        tree.color.a = 0.38

        valid_edge_count = 0
        for child_index in range(1, len(vertices)):
            parent_index = int(parents[child_index])
            if (
                parent_index < 0
                or parent_index >= len(vertices)
                or parent_index == child_index
            ):
                continue

            parent_point = Point()
            parent_point.x = float(vertices[parent_index, 0])
            parent_point.y = float(vertices[parent_index, 1])
            parent_point.z = self._debug_marker_z

            child_point = Point()
            child_point.x = float(vertices[child_index, 0])
            child_point.y = float(vertices[child_index, 1])
            child_point.z = self._debug_marker_z

            tree.points.extend((parent_point, child_point))
            valid_edge_count += 1

        vertices_marker = Marker()
        vertices_marker.header.stamp = stamp
        vertices_marker.header.frame_id = frame_id
        vertices_marker.ns = "rrt_tree_vertices"
        vertices_marker.id = 1
        vertices_marker.type = Marker.SPHERE_LIST
        vertices_marker.action = Marker.ADD
        vertices_marker.pose.orientation.w = 1.0
        vertices_marker.scale.x = self._debug_vertex_size
        vertices_marker.scale.y = self._debug_vertex_size
        vertices_marker.scale.z = self._debug_vertex_size
        vertices_marker.color.r = 0.10
        vertices_marker.color.g = 0.35
        vertices_marker.color.b = 0.95
        vertices_marker.color.a = 0.72

        for vertex in vertices:
            point = Point()
            point.x = float(vertex[0])
            point.y = float(vertex[1])
            point.z = self._debug_marker_z
            vertices_marker.points.append(point)

        start_marker = Marker()
        start_marker.header.stamp = stamp
        start_marker.header.frame_id = frame_id
        start_marker.ns = "rrt_start_goal"
        start_marker.id = 2
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = float(request.start.pose.position.x)
        start_marker.pose.position.y = float(request.start.pose.position.y)
        start_marker.pose.position.z = self._debug_marker_z
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = self._debug_start_goal_size
        start_marker.scale.y = self._debug_start_goal_size
        start_marker.scale.z = self._debug_start_goal_size
        start_marker.color.r = 0.0
        start_marker.color.g = 0.9
        start_marker.color.b = 0.15
        start_marker.color.a = 1.0

        goal_marker = Marker()
        goal_marker.header.stamp = stamp
        goal_marker.header.frame_id = frame_id
        goal_marker.ns = "rrt_start_goal"
        goal_marker.id = 3
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(request.goal.pose.position.x)
        goal_marker.pose.position.y = float(request.goal.pose.position.y)
        goal_marker.pose.position.z = self._debug_marker_z
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = self._debug_start_goal_size
        goal_marker.scale.y = self._debug_start_goal_size
        goal_marker.scale.z = self._debug_start_goal_size
        goal_marker.color.r = 0.95
        goal_marker.color.g = 0.1
        goal_marker.color.b = 0.1
        goal_marker.color.a = 1.0

        marker_items = [tree, vertices_marker, start_marker, goal_marker]

        prior_path = adapter_result.prior_world_path
        if prior_path is not None:
            prior_path = np.asarray(prior_path, dtype=float)
            if prior_path.ndim == 2 and prior_path.shape[1] == 2 and len(prior_path) >= 2:
                prior_marker = Marker()
                prior_marker.header.stamp = stamp
                prior_marker.header.frame_id = frame_id
                prior_marker.ns = "rag_external_prior"
                prior_marker.id = 4
                prior_marker.type = Marker.LINE_STRIP
                prior_marker.action = Marker.ADD
                prior_marker.pose.orientation.w = 1.0
                prior_marker.scale.x = self._debug_prior_line_width
                prior_marker.color.r = 0.95
                prior_marker.color.g = 0.35
                prior_marker.color.b = 0.05
                prior_marker.color.a = 0.92
                for waypoint in prior_path:
                    point = Point()
                    point.x = float(waypoint[0])
                    point.y = float(waypoint[1])
                    point.z = self._debug_marker_z + 0.01
                    prior_marker.points.append(point)
                marker_items.append(prior_marker)

        markers.markers.extend(marker_items)
        self._debug_marker_pub.publish(markers)
        self.get_logger().info(
            "Published static sampling-planner debug tree: "
            f"vertices={len(vertices)} edges={valid_edge_count} "
            f"topic={self._debug_marker_topic}"
        )

    def _empty_path(self, stamp=None) -> Path:
        path = Path()
        path.header.stamp = (
            self.get_clock().now().to_msg() if stamp is None else stamp
        )
        path.header.frame_id = (
            self._grid_map.frame_id if self._grid_map else ""
        )
        return path

    def _build_interpolated_path(
        self,
        start: PoseStamped,
        goal: PoseStamped,
        stamp,
    ) -> Path:
        path = self._empty_path(stamp)

        dx = goal.pose.position.x - start.pose.position.x
        dy = goal.pose.position.y - start.pose.position.y
        distance = math.hypot(dx, dy)
        count = max(
            1,
            math.ceil(distance / self._path_spacing - 1.0e-12),
        )
        heading = math.atan2(dy, dx) if distance > 1.0e-12 else 0.0

        for index in range(count + 1):
            ratio = index / count
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = start.pose.position.x + ratio * dx
            pose.pose.position.y = start.pose.position.y + ratio * dy
            pose.pose.position.z = 0.0
            pose.pose.orientation = self._yaw_to_quaternion(heading)
            path.poses.append(pose)

        if path.poses:
            path.poses[-1].pose.orientation = goal.pose.orientation
        return path

    def _build_path_from_world_points(
        self,
        world_points: Iterable[Iterable[float]],
        goal_orientation: Quaternion,
        stamp,
    ) -> Path:
        points = np.asarray(world_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("world path must have shape (N, 2), N >= 2")

        path = self._empty_path(stamp)
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0

            if index + 1 < len(points):
                dx = float(points[index + 1, 0] - point[0])
                dy = float(points[index + 1, 1] - point[1])
                heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-12 else 0.0
                pose.pose.orientation = self._yaw_to_quaternion(heading)
            else:
                pose.pose.orientation = goal_orientation

            path.poses.append(pose)

        return path

    @staticmethod
    def _path_length(world_points) -> float:
        points = np.asarray(world_points, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            return math.inf
        return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())

    def _make_metrics(
        self,
        request: PlanPath.Request,
        *,
        backend_mode: str,
        success: bool,
        message: str,
        prior_generation_time: float,
        planner_solve_time: float,
        path_conversion_time: float,
        total_planning_time: float,
        first_solution_iter: int,
        first_solution_cost: float,
        final_cost: float,
        collision_checks: int,
        num_vertices: int,
        planned_path_length: float,
        initial_alpha: float = math.nan,
        final_alpha: float = math.nan,
        connectivity_score: float = math.nan,
        bpc_consistency: float = math.nan,
        fallback_checked: int = 0,
        fallback_recoveries: int = 0,
    ) -> PlannerMetrics:
        metrics = PlannerMetrics()
        metrics.header.stamp = self.get_clock().now().to_msg()
        metrics.header.frame_id = (
            self._grid_map.frame_id if self._grid_map else ""
        )
        metrics.planner_name = request.planner_name
        metrics.backend_mode = backend_mode
        metrics.seed = request.seed
        metrics.planning_success = success
        metrics.prior_generation_time = float(prior_generation_time)
        metrics.planner_solve_time = float(planner_solve_time)
        metrics.path_conversion_time = float(path_conversion_time)
        metrics.total_planning_time = float(total_planning_time)
        metrics.first_solution_iter = int(first_solution_iter)
        metrics.first_solution_cost = float(first_solution_cost)
        metrics.final_cost = float(final_cost)
        metrics.collision_checks = int(collision_checks)
        metrics.num_vertices = int(num_vertices)
        metrics.planned_path_length = float(planned_path_length)
        metrics.initial_alpha = float(initial_alpha)
        metrics.final_alpha = float(final_alpha)
        metrics.connectivity_score = float(connectivity_score)
        metrics.bpc_consistency = float(bpc_consistency)
        metrics.fallback_checked = int(fallback_checked)
        metrics.fallback_recoveries = int(fallback_recoveries)
        metrics.status_message = message
        return metrics

    def _failure_response(
        self,
        request: PlanPath.Request,
        response: PlanPath.Response,
        *,
        message: str,
        callback_start: float,
        backend_mode: str,
        prior_generation_time: float = 0.0,
        planner_solve_time: float = 0.0,
        path_conversion_time: float = 0.0,
        collision_checks: int = 0,
        num_vertices: int = 0,
        first_solution_iter: int = -1,
        first_solution_cost: float = math.inf,
        final_cost: float = math.inf,
    ) -> PlanPath.Response:
        total_time = time.perf_counter() - callback_start
        response.success = False
        response.message = message
        response.path = self._empty_path()
        response.metrics = self._make_metrics(
            request,
            backend_mode=backend_mode,
            success=False,
            message=message,
            prior_generation_time=prior_generation_time,
            planner_solve_time=planner_solve_time,
            path_conversion_time=path_conversion_time,
            total_planning_time=total_time,
            first_solution_iter=first_solution_iter,
            first_solution_cost=first_solution_cost,
            final_cost=final_cost,
            collision_checks=collision_checks,
            num_vertices=num_vertices,
            planned_path_length=math.inf,
        )
        return response

    def _validate_request_frames(
        self,
        request: PlanPath.Request,
    ) -> Optional[str]:
        assert self._grid_map is not None
        map_frame = self._grid_map.frame_id
        for label, pose in (("start", request.start), ("goal", request.goal)):
            if pose.header.frame_id not in ("", map_frame):
                return (
                    f"{label} frame '{pose.header.frame_id}' "
                    f"does not match map frame '{map_frame}'"
                )
        return None

    def _run_diagnostic(
        self,
        request: PlanPath.Request,
        response: PlanPath.Response,
        callback_start: float,
    ) -> PlanPath.Response:
        assert self._grid_map is not None

        start_x = request.start.pose.position.x
        start_y = request.start.pose.position.y
        goal_x = request.goal.pose.position.x
        goal_y = request.goal.pose.position.y
        path_length = math.hypot(goal_x - start_x, goal_y - start_y)

        solve_start = time.perf_counter()
        segment_free, checks, message = self._grid_map.segment_is_free(
            start_x,
            start_y,
            goal_x,
            goal_y,
        )
        solve_time = time.perf_counter() - solve_start

        conversion_time = 0.0
        if segment_free:
            conversion_start = time.perf_counter()
            response.path = self._build_interpolated_path(
                request.start,
                request.goal,
                self.get_clock().now().to_msg(),
            )
            conversion_time = time.perf_counter() - conversion_start
            response.success = True
        else:
            response.path = self._empty_path()
            response.success = False

        total_time = time.perf_counter() - callback_start
        response.message = message
        response.metrics = self._make_metrics(
            request,
            backend_mode="diagnostic_straight_line",
            success=response.success,
            message=message,
            prior_generation_time=0.0,
            planner_solve_time=solve_time,
            path_conversion_time=conversion_time,
            total_planning_time=total_time,
            first_solution_iter=0 if response.success else -1,
            first_solution_cost=path_length if response.success else math.inf,
            final_cost=path_length if response.success else math.inf,
            collision_checks=checks,
            num_vertices=2 if response.success else 0,
            planned_path_length=path_length if response.success else math.inf,
        )
        return response

    @staticmethod
    def _last_finite(values, default=math.nan) -> float:
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return float(default)
        finite = array[np.isfinite(array)]
        return float(finite[-1]) if len(finite) else float(default)

    @staticmethod
    def _first_finite(values, default=math.nan) -> float:
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return float(default)
        finite = array[np.isfinite(array)]
        return float(finite[0]) if len(finite) else float(default)

    def _run_sampling_planner(
        self,
        request: PlanPath.Request,
        response: PlanPath.Response,
        callback_start: float,
    ) -> PlanPath.Response:
        assert self._grid_map is not None

        planner_name = request.planner_name.strip().lower()
        display_name = PLANNER_DISPLAY_NAMES.get(planner_name, planner_name)
        backend_modes = {
            "rrt_star": "python_rrt_star",
            "irrt_star": "python_irrt_star",
            RAG_REL_PLANNER: "python_rag_rrt_rel",
            RAG_REL_TOPK_FALLBACK_PLANNER: (
                "python_rag_rrt_rel_topk_fallback"
            ),
        }
        backend_mode = backend_modes.get(planner_name, "python_sampling_planner")

        try:
            robot_radius = float(self.get_parameter("robot_radius_m").value)
            if not math.isfinite(robot_radius) or robot_radius < 0.0:
                raise ValueError("robot_radius_m must be finite and nonnegative")
            effective_clearance = max(
                int(self.get_parameter("rrt_clearance_cells").value),
                math.ceil(robot_radius / self._grid_map.geometry.resolution),
            )
            self._rrt_adapter = RRTStarAdapter(replace(
                self._rrt_adapter.config, clearance_cells=effective_clearance
            ))
            adapter_result = self._rrt_adapter.plan(
                self._grid_map,
                (
                    float(request.start.pose.position.x),
                    float(request.start.pose.position.y),
                ),
                (
                    float(request.goal.pose.position.x),
                    float(request.goal.pose.position.y),
                ),
                int(request.seed),
                planner_name=planner_name,
                collect_debug_tree=self._debug_markers_enabled,
            )
        except Exception as exc:
            message = f"{display_name} adapter failed: {exc}"
            self.get_logger().error(message)
            return self._failure_response(
                request,
                response,
                message=message,
                callback_start=callback_start,
                backend_mode=backend_mode,
            )

        planner_result = adapter_result.planner_result
        metadata = getattr(planner_result, "metadata", {}) or {}
        resolution = self._grid_map.geometry.resolution
        first_iter_value = getattr(planner_result, "first_solution_iter", None)
        first_iter = -1 if first_iter_value is None else int(first_iter_value)

        first_cost_grid = float(
            getattr(planner_result, "first_solution_cost", math.inf)
        )
        final_cost_grid = float(getattr(planner_result, "final_cost", math.inf))
        first_cost_m = (
            first_cost_grid * resolution
            if math.isfinite(first_cost_grid)
            else math.inf
        )
        final_cost_m = (
            final_cost_grid * resolution
            if math.isfinite(final_cost_grid)
            else math.inf
        )

        collision_checks = int(getattr(planner_result, "collision_checks", 0))
        num_vertices = int(getattr(planner_result, "num_vertices", 0))

        alpha_history = getattr(planner_result, "alpha_history", [])
        initial_alpha = self._first_finite(alpha_history)
        final_alpha = self._last_finite(alpha_history)
        connectivity_score = float(
            metadata.get("prior_conn_connectivity_score", math.nan)
        )
        bpc_history = getattr(
            planner_result,
            "best_path_consistency_rate_history",
            metadata.get("best_path_consistency_rate_history", []),
        )
        bpc_consistency = self._last_finite(bpc_history)
        fallback_checked = int(
            metadata.get("topk_parent_fallback_checked_count", 0)
            + metadata.get("topk_rewire_fallback_checked_count", 0)
        )
        fallback_recoveries = int(
            metadata.get("topk_parent_fallback_success_count", 0)
            + metadata.get("topk_rewire_fallback_success_count", 0)
        )

        marker_stamp = self.get_clock().now().to_msg()
        self._publish_rrt_tree_markers(adapter_result, request, marker_stamp)

        if adapter_result.success and adapter_result.world_path is not None:
            try:
                response.path = self._build_path_from_world_points(
                    adapter_result.world_path,
                    request.goal.pose.orientation,
                    marker_stamp,
                )
            except Exception as exc:
                message = f"{display_name} path conversion failed: {exc}"
                return self._failure_response(
                    request,
                    response,
                    message=message,
                    callback_start=callback_start,
                    backend_mode=backend_mode,
                    prior_generation_time=adapter_result.prior_generation_time,
                    planner_solve_time=adapter_result.planner_solve_time,
                    path_conversion_time=adapter_result.path_conversion_time,
                    collision_checks=collision_checks,
                    num_vertices=num_vertices,
                    first_solution_iter=first_iter,
                    first_solution_cost=first_cost_m,
                    final_cost=final_cost_m,
                )
            response.success = True
            planned_path_length = self._path_length(adapter_result.world_path)
        else:
            response.path = self._empty_path()
            response.success = False
            planned_path_length = math.inf

        total_time = time.perf_counter() - callback_start
        response.message = adapter_result.message
        response.metrics = self._make_metrics(
            request,
            backend_mode=backend_mode,
            success=response.success,
            message=adapter_result.message,
            prior_generation_time=adapter_result.prior_generation_time,
            planner_solve_time=adapter_result.planner_solve_time,
            path_conversion_time=adapter_result.path_conversion_time,
            total_planning_time=total_time,
            first_solution_iter=first_iter,
            first_solution_cost=first_cost_m,
            final_cost=final_cost_m,
            collision_checks=collision_checks,
            num_vertices=num_vertices,
            planned_path_length=planned_path_length,
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            connectivity_score=connectivity_score,
            bpc_consistency=bpc_consistency,
            fallback_checked=fallback_checked,
            fallback_recoveries=fallback_recoveries,
        )
        return response

    def _plan_callback(
        self,
        request: PlanPath.Request,
        response: PlanPath.Response,
    ) -> PlanPath.Response:
        callback_start = time.perf_counter()
        self._clear_debug_markers()

        if self._grid_map is None:
            return self._failure_response(
                request,
                response,
                message=f"No valid OccupancyGrid received on {self._map_topic}",
                callback_start=callback_start,
                backend_mode="unavailable",
            )

        frame_error = self._validate_request_frames(request)
        if frame_error is not None:
            return self._failure_response(
                request,
                response,
                message=frame_error,
                callback_start=callback_start,
                backend_mode="request_validation",
            )

        planner_name = request.planner_name.strip().lower()
        if planner_name == "diagnostic_straight_line":
            response = self._run_diagnostic(
                request,
                response,
                callback_start,
            )
        elif planner_name in SUPPORTED_PLANNERS:
            response = self._run_sampling_planner(
                request,
                response,
                callback_start,
            )
        else:
            response = self._failure_response(
                request,
                response,
                message=(
                    f"Unsupported planner_name '{request.planner_name}'. "
                    "Supported values: diagnostic_straight_line, "
                    + ", ".join(SUPPORTED_PLANNERS)
                ),
                callback_start=callback_start,
                backend_mode="unsupported",
            )

        self.get_logger().info(
            f"Plan request: planner={request.planner_name} "
            f"seed={request.seed} success={response.success} "
            f"poses={len(response.path.poses)} "
            f"checks={response.metrics.collision_checks} "
            f"message={response.message}"
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
