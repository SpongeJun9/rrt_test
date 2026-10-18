"""Real robot bringup plus one RViz goal and one RAG plan. No decision or BT."""
import os
from pathlib import Path
import tempfile
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from rag_rrt_real_validation.stack_config import prepare_stack
from rag_rrt_real_validation.entry_points import ENTRY_POINTS, verify_entry_points


def setup(context):
    verify_entry_points({name: get_package_prefix(name) for name in ENTRY_POINTS})
    value = lambda name: LaunchConfiguration(name).perform(context)
    nav_share = Path(get_package_share_directory('pb2025_nav_bringup'))
    own_share = Path(get_package_share_directory('rag_rrt_real_validation'))
    world = value('world')
    map_path = Path(value('map') or str(nav_share/'map/reality'/f'{world}.yaml')).expanduser().resolve()
    pcd_path = Path(value('prior_pcd_file') or str(nav_share/'pcd/reality'/f'{world}.pcd')).expanduser().resolve()
    nav_path = Path(value('params_file') or str(nav_share/'config/reality/nav2_params.yaml')).expanduser().resolve()
    site_path = Path(value('site_config') or str(own_share/'config/site.yaml')).expanduser().resolve()
    serial_share = Path(get_package_share_directory('core_serial'))
    serial_path = Path(value('serial_config') or str(serial_share/'config/core_serial_config.yaml')).expanduser().resolve()
    for path in (map_path, pcd_path, nav_path, site_path, serial_path):
        if not path.is_file():
            raise RuntimeError('Required real robot input does not exist: '+str(path))
    map_config = yaml.safe_load(map_path.read_text(encoding='utf-8'))
    image = Path(map_config['image'])
    if not image.is_absolute():
        image = map_path.parent/image
    if not image.is_file():
        raise RuntimeError('Map image does not exist: '+str(image))
    site = yaml.safe_load(site_path.read_text(encoding='utf-8'))
    site.update(goal_from_pose=True, goal_from_click=False, ready_timeout_sec=120.0)
    original = nav_path.read_text(encoding='utf-8')
    profile = value('nav_profile')
    nav, site = prepare_stack(yaml.safe_load(original.replace('<robot_namespace>', '')), site, profile)
    nav['map_server']['ros__parameters']['yaml_filename'] = str(map_path)
    # Persist exact generated configuration for this startup, not in original files.
    output = Path(value('output_dir')).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix='stack_', dir=output))
    generated_nav, generated_site = session/'navigation.yaml', session/'site.yaml'
    generated_nav.write_text(yaml.safe_dump(nav, allow_unicode=True), encoding='utf-8')
    generated_site.write_text(yaml.safe_dump(site, allow_unicode=True), encoding='utf-8')
    (session/'inputs.yaml').write_text(yaml.safe_dump(dict(map=str(map_path), prior_pcd_file=str(pcd_path),
        params_file=str(nav_path), serial_config=str(serial_path), nav_profile=profile), allow_unicode=True), encoding='utf-8')
    params = ParameterFile(str(generated_nav), allow_substs=True)
    def node(package, executable, name, **kwargs):
        return Node(package=package, executable=executable, name=name, output='screen',
                    parameters=[params], **kwargs)
    def include(path, arguments):
        return GroupAction([IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)),
                                                    launch_arguments=arguments.items())])
    actions = [SetEnvironmentVariable('RAG_TRIAL_CONFIG', str(generated_site))]
    if value('use_robot_state_pub').lower() == 'true':
        robot_share = Path(get_package_share_directory('pb2025_robot_description'))
        actions.append(include(robot_share/'launch/robot_description_launch.py',
            dict(namespace='', robot_name='pb2025_sentry_robot', use_sim_time='false', use_rviz='False',
                 params_file=str(robot_share/'params/robot_description.yaml'))))
    actions += [
        Node(package='core_serial', executable='core_serial_node', name='core_serial_node',
             parameters=[str(serial_path)], output='screen'),
        node('livox_ros_driver2', 'livox_ros_driver2_node', 'livox_ros_driver2'),
        include(nav_share/'launch/localization_launch.py', dict(namespace='', map=str(map_path),
            prior_pcd_file=str(pcd_path), params_file=str(generated_nav), use_sim_time='false',
            autostart='true', use_composition='False', use_respawn='False')),
        node('loam_interface', 'loam_interface_node', 'loam_interface'),
        node('sensor_scan_generation', 'sensor_scan_generation_node', 'sensor_scan_generation'),
        node('terrain_analysis', 'terrainAnalysis', 'terrain_analysis'),
        node('terrain_analysis_ext', 'terrainAnalysisExt', 'terrain_analysis_ext'),
        node('fake_vel_transform', 'fake_vel_transform_node', 'fake_vel_transform'),
        node('nav2_controller', 'controller_server', 'controller_server',
             remappings=[('cmd_vel', 'cmd_vel_controller')]),
        # Only supplies the original global costmap required by trial checks.
        # No BT navigator or other client invokes its Theta* planner.
        node('nav2_planner', 'planner_server', 'planner_server'),
        node('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother',
             remappings=[('cmd_vel', 'cmd_vel_controller'), ('cmd_vel_smoothed', 'cmd_vel_nav2_result')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_rag_control', output='screen', parameters=[
                 dict(use_sim_time=False, autostart=True,
                      node_names=['controller_server', 'planner_server', 'velocity_smoother'])]),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', str(own_share/'config/rag_single_plan.rviz')]),
        include(own_share/'launch/trial.launch.py', dict(goal_from_pose='true', goal_from_click='false',
            method=value('method'), seed=value('seed'), execute='true', output_dir=str(session))),
    ]
    print('RAG full stack configuration and results: '+str(session), flush=True)
    print('Navigation profile: '+profile+'; RAG clearance radius: '+str(site['planner_radius_m'])+' m', flush=True)
    return actions


def generate_launch_description():
    arguments = dict(world='', map='', prior_pcd_file='', params_file='', site_config='',
                     serial_config='', use_robot_state_pub='true', nav_profile='original',
                     method='rag_bpc_conn_init_rel_only_topk_fallback', seed='0',
                     output_dir=str(Path.home()/'rag_rrt_real_results'))
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k,v in arguments.items()]
                             + [OpaqueFunction(function=setup)])
