"""Prepare isolated trial settings without editing the robot's original YAML."""
from copy import deepcopy
import ast
import json
import math


def navigation_parameter_names(site):
    controller, goal = site['controller_id'], site['goal_checker_id']
    names = {
        site['controller_node']: ['use_sim_time', 'controller_plugins', 'goal_checker_plugins',
            controller+'.v_linear_max', controller+'.v_linear_min',
            controller+'.min_approach_linear_velocity', controller+'.enable_rotation',
            goal+'.xy_goal_tolerance', goal+'.yaw_goal_tolerance', goal+'.stateful'],
        site['velocity_smoother_node']: ['use_sim_time', 'max_velocity', 'min_velocity'],
    }
    for node in site['costmap_nodes']:
        names[node] = ['use_sim_time', 'robot_radius', 'footprint', 'robot_base_frame', 'global_frame']
    return names


def navigation_snapshot(nav, site):
    snapshot = {}
    for node, names in navigation_parameter_names(site).items():
        params = nav
        for part in node.strip('/').split('/'):
            params = params[part]
        params = params['ros__parameters']
        values = {}
        for name in names:
            if name == 'footprint':
                values[name] = params.get(name, '')
                continue
            value = params
            for part in name.split('.'):
                value = value[part]
            values[name] = deepcopy(value)
        snapshot[node] = values
    return snapshot


def configured_radius(params):
    footprint = params.get('footprint', '')
    vertices = ast.literal_eval(footprint) if isinstance(footprint, str) and footprint.strip() else footprint
    if vertices:
        if len(vertices) < 3:
            raise ValueError('original footprint needs at least three vertices')
        radius = max(math.hypot(float(p[0]), float(p[1])) for p in vertices)
    else:
        radius = float(params['robot_radius'])
    radius += float(params.get('footprint_padding', 0.0))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('original costmap radius must be finite and positive')
    return radius


def prepare_stack(original, site, profile='original'):
    """Preserve upstream settings or explicitly select the earlier trial overrides."""
    settings = deepcopy(site)
    if profile == 'validation':
        settings.update(navigation_profile='validation', planner_radius_m=-1.0,
                        expected_navigation_parameters='{}')
        return prepare_navigation(original, settings), settings
    if profile != 'original':
        raise ValueError('nav_profile must be original or validation')
    nav = deepcopy(original)
    global_map = nav['global_costmap']['global_costmap']['ros__parameters']
    controller = nav['controller_server']['ros__parameters']
    checker = controller[site['goal_checker_id']]
    smoother = nav['velocity_smoother']['ros__parameters']
    maxima, minima = smoother['max_velocity'], smoother['min_velocity']
    if len(maxima) != 3 or len(minima) != 3 or not all(
            math.isfinite(v) for v in maxima+minima):
        raise ValueError('invalid original velocity smoother limits')
    speed = math.hypot(*(max(abs(maxima[i]), abs(minima[i])) for i in (0, 1)))
    if speed <= 0 or any(minima[i] > 0 or maxima[i] < 0 for i in (0, 1)):
        raise ValueError('invalid original planar velocity bounds')
    settings.update(navigation_profile='original', planner_radius_m=configured_radius(global_map),
                    base_frame=global_map['robot_base_frame'], map_frame=global_map['global_frame'],
                    max_linear_speed_mps=speed,
                    endpoint_tolerance_m=float(checker['xy_goal_tolerance']),
                    yaw_tolerance_rad=float(checker['yaw_goal_tolerance']))
    settings['expected_navigation_parameters'] = json.dumps(navigation_snapshot(nav, settings))
    return nav, settings


def prepare_navigation(original, site):
    nav = deepcopy(original)
    cap = float(site['max_linear_speed_mps'])
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('max_linear_speed_mps must be positive')
    length, width, margin = (float(site[k]) for k in
                             ('robot_length_m', 'robot_width_m', 'body_margin_m'))
    if not all(math.isfinite(v) for v in (length, width, margin)) or min(length, width) <= 0 or margin < 0:
        raise ValueError('invalid trial body dimensions')
    hx, hy = length / 2 + margin, width / 2 + margin
    footprint = json.dumps([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    cp = nav['controller_server']['ros__parameters']
    cid, gid = site['controller_id'], site['goal_checker_id']
    cp[cid].update(v_linear_max=cap, v_linear_min=-cap,
                   min_approach_linear_velocity=min(.1, cap), enable_rotation=False)
    cp['min_y_velocity_threshold'] = .001
    cp[gid].update(xy_goal_tolerance=float(site['endpoint_tolerance_m']),
                   yaw_goal_tolerance=float(site['yaw_tolerance_rad']))
    # Per-axis limits must also bound diagonal speed by the trial's scalar cap.
    axis_cap = cap / math.sqrt(2)
    nav['velocity_smoother']['ros__parameters'].update(
        max_velocity=[axis_cap, axis_cap, 0.0], min_velocity=[-axis_cap, -axis_cap, 0.0])
    nav['fake_vel_transform']['ros__parameters']['init_spin_speed'] = 0.0
    for key in ('global_costmap', 'local_costmap'):
        nav[key][key]['ros__parameters'].update(footprint=footprint, footprint_padding=0.0)
    return nav
