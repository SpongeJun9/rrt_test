"""Human-readable per-goal planning data, without changing planner behavior."""
import math


METRIC_FIELDS = (
    'prior_generation_time', 'planner_solve_time', 'path_conversion_time',
    'total_planning_time', 'planned_path_length', 'collision_checks', 'num_vertices',
    'first_solution_iter', 'initial_alpha', 'final_alpha', 'connectivity_score',
    'bpc_consistency', 'fallback_checked', 'fallback_recoveries',
)
INFLATION_PARAMETERS = (
    'inflation_layer.inflation_radius', 'inflation_layer.cost_scaling_factor',
    'footprint_padding',
)


def collect_metrics(metrics):
    # An unavailable/invalid metric remains null, rather than becoming a fake zero.
    result = {}
    for name in METRIC_FIELDS:
        value = getattr(metrics, name, None)
        result[name] = value if isinstance(value, (int, float)) and math.isfinite(value) else None
    return result


def build_summary(config, state):
    goal = state.get('actual_goal')
    if goal is None and config.get('goal_set'):
        goal = dict(x=config['goal_x'], y=config['goal_y'], yaw=config['goal_yaw'])
    maps = {}
    navigation = state.get('navigation_parameters', {})
    for node in config['costmap_nodes']:
        values = navigation.get(node, {})
        maps[node] = dict(
            inflation_radius_m=values.get('inflation_layer.inflation_radius'),
            cost_scaling_factor=values.get('inflation_layer.cost_scaling_factor'),
            robot_radius_m=values.get('robot_radius'),
            footprint_padding_m=values.get('footprint_padding'),
        )
    return dict(
        status=state['status'], stage=state['stage'], failure_reason=state.get('failure_reason'),
        method=config['method'], seed=config['seed'], map_frame=config['map_frame'],
        base_frame=config['base_frame'], start=state.get('actual_start'), goal=goal,
        coordinate_units='meters; yaw radians', navigation_profile=config['navigation_profile'],
        map_resolution_m=state.get('map_resolution_m'),
        planner_radius_m=state.get('planner_radius_m'),
        clearance_cells=state.get('effective_clearance_cells'),
        rrt_iter_max=config['rrt_iter_max'], rrt_iter_after_initial=config['rrt_iter_after_initial'],
        costmaps=maps, planning=state.get('planning_metrics', {}),
        planning_time_units='seconds; backend total excludes controller execution and operator waiting',
        rpc_wait_time_sec=state.get('request_wall_time_sec'),
        planner_request_attempts=state.get('planner_request_attempts', 0),
        backend_success=state.get('backend_success'), backend_message=state.get('backend_message'),
        backend_path_pose_count=state.get('backend_path_pose_count'),
        reference_length_m=state.get('reference_length_m'),
        reference_pose_count=state.get('reference_pose_count'),
    )


def number(value, digits=3, multiplier=1.0):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'N/A'
    return format(value*multiplier, '.'+str(digits)+'f')


def pose_text(pose):
    if pose is None:
        return 'N/A'
    yaw = pose['yaw']
    return ('x='+number(pose['x'])+' m, y='+number(pose['y'])+' m, yaw='+number(yaw)+
            ' rad ('+number(yaw, 2, 180/math.pi)+' deg)')


def count(value):
    return str(value) if value is not None else 'N/A'


def input_lines(summary):
    return [
        '方法='+summary['method']+', seed='+str(summary['seed'])+', profile='+summary['navigation_profile'],
        '起点 ['+summary['map_frame']+'; '+summary['base_frame']+']: '+pose_text(summary['start']),
        '终点 ['+summary['map_frame']+']: '+pose_text(summary['goal']),
        '地图分辨率='+number(summary['map_resolution_m'])+' m/格; RAG净空半径='+
        number(summary['planner_radius_m'])+' m; 净空='+count(summary['clearance_cells'])+
        ' 格; 迭代预算='+str(summary['rrt_iter_max'])+'; 首解后迭代='+str(summary['rrt_iter_after_initial']),
    ]


def navigation_lines(summary):
    return [
        node+': 膨胀半径 inflation_radius='+number(values['inflation_radius_m'])+
        ' m, 代价衰减系数 cost_scaling_factor='+number(values['cost_scaling_factor'])+
        ', robot_radius='+number(values['robot_radius_m'])+
        ' m, footprint_padding='+number(values['footprint_padding_m'])+' m'
        for node, values in summary['costmaps'].items()
    ]


def metric_lines(summary):
    metrics = summary['planning']
    ms = lambda name: number(metrics.get(name), 3, 1000)
    success = summary['backend_success']
    status = '成功' if success is True else ('失败' if success is False else 'N/A')
    return [
        '规划返回='+status+'; 路径长度='+number(metrics.get('planned_path_length'))+
        ' m; 原始路径点数='+count(summary['backend_path_pose_count']),
        '规划耗时: 后端总计='+ms('total_planning_time')+' ms, 先验生成='+ms('prior_generation_time')+
        ' ms, RAG求解='+ms('planner_solve_time')+' ms, 路径转换='+ms('path_conversion_time')+
        ' ms; RPC等待='+number(summary['rpc_wait_time_sec'], 3, 1000)+' ms',
        '搜索统计: 树节点='+count(metrics.get('num_vertices'))+', 碰撞检查='+count(metrics.get('collision_checks'))+
        ', 首解迭代='+count(metrics.get('first_solution_iter'))+
        ', 引导alpha='+number(metrics.get('initial_alpha'))+' -> '+number(metrics.get('final_alpha')),
    ]


def reference_line(summary):
    return '控制器参考路径: 长度='+number(summary['reference_length_m'])+' m, 点数='+count(summary['reference_pose_count'])


def all_lines(summary):
    lines = input_lines(summary)+navigation_lines(summary)+metric_lines(summary)
    lines.append(reference_line(summary))
    lines.append('任务状态='+summary['status']+'; 阶段='+summary['stage'])
    if not summary['planner_request_attempts']:
        lines.append('本次尚未调用 RAG；N/A 表示尚无对应数据。')
    if summary['failure_reason']:
        lines.append('失败原因='+summary['failure_reason'])
    return lines
