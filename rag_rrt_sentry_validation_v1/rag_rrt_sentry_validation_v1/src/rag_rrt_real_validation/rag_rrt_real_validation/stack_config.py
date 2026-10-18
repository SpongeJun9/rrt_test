"""Prepare isolated trial settings without editing the robot's original YAML."""
from copy import deepcopy
import json
import math


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
