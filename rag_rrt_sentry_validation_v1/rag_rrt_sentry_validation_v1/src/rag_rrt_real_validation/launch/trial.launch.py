import math
import os
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from rag_rrt_real_validation.entry_points import ENTRY_POINTS, verify_entry_points


def generate_launch_description():
    verify_entry_points({name: get_package_prefix(name) for name in ENTRY_POINTS})
    # RAG_TRIAL_CONFIG permits one explicit local site file; no device paths embedded.
    default = Path(get_package_share_directory('rag_rrt_real_validation')) / 'config/site.yaml'
    site_path = Path(os.environ.get('RAG_TRIAL_CONFIG', str(default))).expanduser()
    site = yaml.safe_load(site_path.read_text())
    radius = math.hypot(site['robot_length_m']/2+site['body_margin_m'],
                        site['robot_width_m']/2+site['body_margin_m'])
    if site.get('map_already_inflated', False):
        raise ValueError('Provide a raw static OccupancyGrid; this package applies its own clearance.')
    # Nonzero offsets require selecting a matching footprint-center TF frame.
    if site.get('robot_center_offset_x_m', 0.0) or site.get('robot_center_offset_y_m', 0.0):
        raise ValueError('Set base_frame to the body-center frame instead of using nonzero offsets.')
    config = Path(get_package_share_directory('rag_rrt_backend_py')) / 'config/backend.yaml'
    backend = Node(package='rag_rrt_backend_py', executable='planner_backend',
                   name='planner_backend', output='screen', parameters=[str(config), {
                       'use_sim_time': False,
                       'algorithm_project_root': str(Path(get_package_share_directory('rag_rrt_backend_py')) / 'algorithm'),
                       'map_topic': site['map_topic'], 'robot_radius_m': radius,
                       'service_name': '/rag_rrt/validation/compute_path',
                       'rrt_iter_max': int(site['rrt_iter_max']),
                       'rrt_iter_after_initial': int(site['rrt_iter_after_initial']),
                       'publish_debug_markers': False,
                   }])
    names = {'goal_x': '0.0', 'goal_y': '0.0', 'goal_yaw': '0.0',
             'goal_set': 'false', 'goal_from_click': 'false', 'goal_from_pose': 'false', 'execute': 'false', 'check_only': 'false',
             'method': site['method'], 'seed': str(site['seed']),
             'run_dir': '', 'output_dir': str(Path.home() / 'rag_rrt_real_results')}
    args = [DeclareLaunchArgument(n, default_value=v) for n,v in names.items()]
    params = dict(site)
    params.update({'use_sim_time': False, 'planner_service': '/rag_rrt/validation/compute_path',
                   'config_path': str(site_path)})
    for name in names:
        dtype = bool if name in ('goal_set','goal_from_click','goal_from_pose','execute','check_only') else int if name=='seed' else float if name.startswith('goal_') else str
        params[name] = ParameterValue(LaunchConfiguration(name), value_type=dtype)
    runner = Node(package='rag_rrt_real_validation', executable='trial',
                  name='rag_rrt_validation_trial', output='screen', parameters=[params])
    # Terminate only our launch processes when this one trial finishes.
    done = RegisterEventHandler(OnProcessExit(target_action=runner,
                on_exit=[EmitEvent(event=Shutdown(reason='single trial completed'))]))
    return LaunchDescription(args + [done, backend, runner])
