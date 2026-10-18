"""One explicitly requested trial. No automatic seed replacement or replanning."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import signal
import tempfile
import sys
import time
import uuid
import platform
import os
import importlib.metadata

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rcl_interfaces.srv import GetParameters
from action_msgs.msg import GoalStatusArray
from unique_identifier_msgs.msg import UUID
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from nav2_msgs.action import FollowPath
from rosidl_runtime_py.convert import message_to_ordereddict
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory
import yaml

from rag_rrt_interfaces.srv import PlanPath
from .geometry import METHODS, planning_radius, footprint_covers, yaw_of, wrap_yaw, resample, distance_to_path
from .stack_config import navigation_parameter_names


class TrialFailure(RuntimeError):
    pass


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, array.array)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parameter_value(v):
    names = {1:'bool_value', 2:'integer_value', 3:'double_value', 4:'string_value',
             5:'byte_array_value', 6:'bool_array_value', 7:'integer_array_value',
             8:'double_array_value', 9:'string_array_value'}
    return getattr(v, names[v.type]) if v.type in names else None


class Trial(Node):
    def __init__(self):
        super().__init__('rag_rrt_validation_trial')
        config = Path(get_package_share_directory('rag_rrt_real_validation')) / 'config/site.yaml'
        defaults = yaml.safe_load(config.read_text())
        defaults.update(goal_x=0.0, goal_y=0.0, goal_yaw=0.0, goal_set=False, execute=False,
                        check_only=False, run_dir='', output_dir=str(Path.home()/'rag_rrt_real_results'),
                        planner_service='/rag_rrt/validation/compute_path', config_path=str(config))
        for name,value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name,value)
        self.cfg = {k:self.get_parameter(k).value for k in defaults}
        output = Path(self.cfg['output_dir']).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
        if self.cfg['run_dir']:
            self.directory=Path(self.cfg['run_dir']).expanduser().resolve()
            if (self.directory/'metadata.json').exists():
                raise TrialFailure('run_dir already contains a result; refusing to overwrite')
            self.directory.mkdir(parents=True,exist_ok=True)
        else:
            self.directory = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_'),dir=output))
        self.state = dict(stage='initializing', status='RUNNING', method=self.cfg['method'], seed=self.cfg['seed'],
                          execute=self.cfg['execute'], parameters=self.cfg,
                          trajectory_source='normal localization TF estimate; not independent ground truth',
                          collision_evidence='not independently measured', software='rag_rrt_sentry_validation_v1')
        self.goal_point = None
        self.goal_pose = None
        self.state['runtime_versions']=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
            ros_distro=os.environ.get('ROS_DISTRO',''),ros_domain_id=os.environ.get('ROS_DOMAIN_ID','0'),
            rmw_implementation=os.environ.get('RMW_IMPLEMENTATION','default'))
        self.started = time.monotonic(); self.map = None; self.map_hash = None; self.frozen_map_hash = None
        self.odom = None; self.odom_received = 0.0; self.cmd = None; self.cmd_received = 0.0
        self.own_goal = None; self.pending_goal = None; self.result_future = None; self.action_terminal = False; self.feedback = []; self.statuses = {}
        self.motion_active = False; self.trajectory = []; self.record_trajectory = False; self.own_uuid = None
        self.last_sample = 0.0; self.last_motion_pose = None; self.reference = []; self.pose_history = []
        self.buffer=Buffer(); self.listener=TransformListener(self.buffer,self)
        latched=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PointStamped,self.cfg['goal_point_topic'],lambda msg:setattr(self,'goal_point',msg),10)
        self.create_subscription(PoseStamped,self.cfg['goal_pose_topic'],self.on_goal_pose,10)
        self.create_subscription(OccupancyGrid,self.cfg['map_topic'],self.on_map,latched)
        self.create_subscription(Odometry,self.cfg['odom_topic'],self.on_odom,qos_profile_sensor_data)
        self.create_subscription(Twist,self.cfg['cmd_vel_topic'],self.on_cmd,qos_profile_sensor_data)
        for action in [self.cfg['follow_path_action'], '/navigate_to_pose', '/navigate_through_poses']:
            self.create_subscription(GoalStatusArray,action+'/_action/status',
                                     lambda msg,key=action:self.statuses.__setitem__(key,msg),latched)
        self.zero_pub=self.create_publisher(Twist,self.cfg['cmd_vel_topic'],10)
        self.path_pub=self.create_publisher(RosPath,'/rag_rrt/validation/reference_path',latched)
        self.client=self.create_client(PlanPath,self.cfg['planner_service'])
        self.action=ActionClient(self,FollowPath,self.cfg['follow_path_action'])
        self.save_json('metadata.json',self.state)

    def save_json(self,name,value):
        (self.directory/name).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def on_goal_pose(self,msg):
        # Freeze the first goal. Later clicks never replace or replan this trial.
        if self.goal_pose is None:
            self.goal_pose = msg

    def on_map(self,msg):
        shape=(msg.info.width,msg.info.height,msg.info.resolution,message_to_ordereddict(msg.info.origin))
        digest=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()+array.array('b',msg.data).tobytes()).hexdigest()
        self.map,self.map_hash=msg,digest

    def on_odom(self,msg):
        self.odom,self.odom_received=msg,time.monotonic()

    def on_cmd(self,msg):
        self.cmd,self.cmd_received=msg,time.monotonic()

    def pose(self):
        transform=self.buffer.lookup_transform(self.cfg['map_frame'],self.cfg['base_frame'],Time())
        age=(self.get_clock().now().nanoseconds-Time.from_msg(transform.header.stamp).nanoseconds)/1e9
        if not math.isfinite(age) or age < -0.2 or age > self.cfg['tf_max_age_sec']:
            raise TrialFailure('localization/TF timestamp is stale or clock mismatch')
        p=transform.transform.translation;q=transform.transform.rotation
        values=(p.x,p.y,yaw_of(q))
        if not all(math.isfinite(v) for v in values):
            raise TrialFailure('nonfinite localization pose')
        return values,transform.header.stamp

    def spin(self,guard=True):
        if not rclpy.ok():
            raise TrialFailure('ROS context stopped')
        rclpy.spin_once(self,timeout_sec=0.05)
        now=time.monotonic()
        if now-self.last_sample >= 0.1:
            self.last_sample=now
            try:
                pose,stamp=self.pose()
            except Exception as exc:
                if guard and self.motion_active:
                    raise TrialFailure('localization unavailable during execution: '+str(exc)) from exc
            else:
                if guard and self.motion_active and self.last_motion_pose:
                    if math.hypot(pose[0]-self.last_motion_pose[0],pose[1]-self.last_motion_pose[1]) > self.cfg['localization_jump_m']:
                        raise TrialFailure('localization jump during execution')
                self.pose_history.append((now, pose))
                self.pose_history = self.pose_history[-10:]
                self.last_motion_pose=pose
                if self.record_trajectory:
                    row = dict(wall_elapsed_sec=now-self.started,ros_sec=stamp.sec+stamp.nanosec/1e9,
                                            x=pose[0],y=pose[1],yaw=pose[2])
                    self.trajectory.append(row)
                    partial = self.directory/'estimated_trajectory.partial.csv'
                    new_file = not partial.exists()
                    with partial.open('a', newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=['wall_elapsed_sec','ros_sec','x','y','yaw'])
                        if new_file:writer.writeheader()
                        writer.writerow(row)
        if guard and self.motion_active:
            self.assert_no_other_goal()
            if self.frozen_map_hash!=self.map_hash:
                raise TrialFailure('map changed during execution')
            if now-self.odom_received > self.cfg['sensor_max_age_sec']:
                raise TrialFailure('odometry stream is stale')
            if self.cmd is not None and now-self.cmd_received <= self.cfg['sensor_max_age_sec']:
                speed=math.hypot(self.cmd.linear.x,self.cmd.linear.y)
                if not math.isfinite(speed) or speed > self.cfg['max_linear_speed_mps']+0.05:
                    raise TrialFailure('observed command exceeds trial speed cap')

    def wait(self,predicate,timeout,description,guard=True):
        deadline=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>=deadline:
                raise TrialFailure(description+' timeout')
            self.spin(guard=guard)

    def future(self,future,timeout,description,guard=True):
        self.wait(future.done,timeout,description,guard)
        result=future.result()
        if result is None:
            raise TrialFailure(description+' returned no result')
        return result

    def assert_no_other_goal(self):
        for name,status in self.statuses.items():
            for item in status.status_list:
                if item.status not in (1,2,3):
                    continue
                ours = (name==self.cfg['follow_path_action'] and self.own_uuid is not None and
                        list(item.goal_info.goal_id.uuid)==self.own_uuid)
                if not ours:
                    raise TrialFailure('another navigation goal is active: '+name)

    def read_params(self,node,names):
        client=self.create_client(GetParameters,node.rstrip('/')+'/get_parameters')
        try:
            self.wait(client.service_is_ready,3.0,'parameter service '+node)
            request=GetParameters.Request();request.names=names
            response=self.future(client.call_async(request),3.0,'read parameters '+node)
            return {name:parameter_value(value) for name,value in zip(names,response.values)}
        finally:
            self.destroy_client(client)

    def navigation_checks(self):
        issues=[];report={}
        controller=self.cfg['controller_id'];goal=self.cfg['goal_checker_id']
        requests=navigation_parameter_names(self.cfg)
        for node,names in requests.items():
            try:
                report[node]=self.read_params(node,names)
            except Exception as exc:
                issues.append(str(exc))
        for node,params in report.items():
            if params.get('use_sim_time') is not False:
                issues.append(node+': real robot requires use_sim_time=false')
        cp=report.get(self.cfg['controller_node'],{})
        if controller not in (cp.get('controller_plugins') or []):issues.append('controller_id not configured')
        if goal not in (cp.get('goal_checker_plugins') or []):issues.append('goal_checker_id not configured')
        if self.cfg['navigation_profile']=='original':
            expected=json.loads(self.cfg['expected_navigation_parameters'])
            if set(expected)!=set(requests):issues.append('original navigation parameter snapshot is missing nodes')
            for node,names in requests.items():
                for name in names:
                    if name not in expected.get(node,{}):
                        issues.append(node+'.'+name+': missing original parameter snapshot')
                        continue
                    actual=report.get(node,{}).get(name)
                    wanted=expected[node][name]
                    if name=='footprint':
                        # Nav2 may expose an unspecified polygon as either empty string or [].
                        actual=actual or '';wanted=wanted or ''
                        if actual.strip()=='[]':actual=''
                        if wanted.strip()=='[]':wanted=''
                    # ROS double[] values are array.array; YAML sequences are lists.
                    if clean(actual)!=clean(wanted):issues.append(node+'.'+name+': differs from original navigation YAML')
            self.state['navigation_parameters']=report;self.state['preflight_issues']=issues
            self.state['footprint_source']='original navigation costmap configuration; physical dimensions not inferred'
            self.save_json('navigation_parameters.json',report)
            return issues
        if cp.get(controller+'.enable_rotation') is not False:issues.append('controller rotation must be disabled for this serial protocol')
        cap=self.cfg['max_linear_speed_mps']
        for key in [controller+'.v_linear_max',controller+'.v_linear_min',controller+'.min_approach_linear_velocity']:
            value=cp.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>cap+1e-8:
                issues.append(key+': missing or exceeds trial speed cap')
        sm=report.get(self.cfg['velocity_smoother_node'],{})
        for key in ['max_velocity','min_velocity']:
            value=sm.get(key)
            if value is None or len(value)<2 or any(not math.isfinite(v) or abs(v)>cap+1e-8 for v in value[:2]):
                issues.append('velocity_smoother.'+key+': missing or exceeds trial speed cap')
        for node in self.cfg['costmap_nodes']:
            values=report.get(node,{})
            try:
                covered=footprint_covers(values.get('footprint') or '',float(values.get('robot_radius') or 0),
                    self.cfg['robot_length_m'],self.cfg['robot_width_m'],self.cfg['body_margin_m'])
            except Exception:
                covered=False
            if not covered:issues.append(node+': footprint does not cover configured body and margin')
        self.state['navigation_parameters']=report;self.state['preflight_issues']=issues
        self.save_json('navigation_parameters.json',report)
        return issues

    def stopped(self):
        now=time.monotonic()
        if self.odom is None or now-self.odom_received>self.cfg['sensor_max_age_sec']:
            return False
        if len(self.pose_history)<3 or now-self.pose_history[-1][0]>0.3:
            return False
        recent=[p for p in self.pose_history if now-p[0]<=0.5]
        if len(recent)<3 or recent[-1][0]-recent[0][0]<0.2:
            return False
        latest=recent[-1][1]
        if any(math.dist(p[1][:2],latest[:2])>0.015 or abs(wrap_yaw(p[1][2]-latest[2]))>0.03 for p in recent):
            return False
        twist=self.odom.twist.twist
        if not all(math.isfinite(v) for v in [twist.linear.x,twist.linear.y,twist.angular.z]):
            return False
        if math.hypot(twist.linear.x,twist.linear.y)>0.03 or abs(twist.angular.z)>0.08:
            return False
        # A stale nonzero command is not evidence of stop; fresh odometry is required above.
        # Original fake_vel adds init_spin_speed to angular.z. The current serial
        # navigation packet sends only planar x/y; measured odometry above must
        # still confirm no actual rotation, for both profiles.
        check_command_yaw=self.cfg['navigation_profile']!='original'
        if self.cmd is not None and (not all(math.isfinite(v) for v in [self.cmd.linear.x,self.cmd.linear.y,self.cmd.angular.z]) or math.hypot(self.cmd.linear.x,self.cmd.linear.y)>0.03 or (check_command_yaw and abs(self.cmd.angular.z)>0.08)):
            return False
        return True

    def stop_and_cancel(self):
        errors=[];cancel_response=None
        if self.own_goal is None and self.pending_goal is not None:
            try:
                self.own_goal=self.future(self.pending_goal,self.cfg['cancel_timeout_sec'],'late goal acknowledgement',guard=False)
            except Exception as exc:errors.append(str(exc))
        if self.own_goal is not None and self.own_goal.accepted:
            try:
                if self.result_future is None:
                    self.result_future=self.own_goal.get_result_async()
                response=self.future(self.own_goal.cancel_goal_async(),self.cfg['cancel_timeout_sec'],'cancel',guard=False)
                cancel_response=message_to_ordereddict(response)
            except Exception as exc:errors.append(str(exc))
        deadline=time.monotonic()+self.cfg['stop_timeout_sec'];stable=0
        while rclpy.ok() and time.monotonic()<deadline:
            if self.cfg['publish_zero_on_abort']:
                self.zero_pub.publish(Twist())
            self.spin(guard=False)
            stable=stable+1 if self.stopped() else 0
            if self.result_future is not None and self.result_future.done():
                try:
                    terminal=self.result_future.result()
                    self.action_terminal=terminal is not None and terminal.status in (4,5,6)
                    if terminal is not None:self.save_json('abort_action_result.json',message_to_ordereddict(terminal))
                except Exception as exc:
                    errors.append(str(exc))
            if stable>=5 and self.action_terminal:break
        self.motion_active=False
        observed=stable>=5 and self.action_terminal
        self.state['abort_stop']=dict(cancel_response=cancel_response,errors=errors,stop_observed=observed,
                                      action_terminal=self.action_terminal,
                                      goal_acknowledgement_known=self.own_goal is not None)
        if not observed:self.get_logger().error('STOP NOT CONFIRMED: operator must use the original remote takeover/emergency stop.')

    def make_pose(self,pose):
        msg=PoseStamped();msg.header.frame_id=self.cfg['map_frame'];msg.header.stamp=self.get_clock().now().to_msg()
        msg.pose.position.x=pose[0];msg.pose.position.y=pose[1]
        msg.pose.orientation.z=math.sin(pose[2]/2);msg.pose.orientation.w=math.cos(pose[2]/2)
        return msg

    def run(self):
        self.state['stage']='preflight'
        radius=planning_radius(self.cfg)
        if self.cfg['navigation_profile'] not in ('original','validation'):raise TrialFailure('invalid navigation_profile')
        for key in ['max_linear_speed_mps','resample_spacing_m','max_path_length_m','start_tolerance_m',
                    'endpoint_tolerance_m','minimum_motion_m','tf_max_age_sec','sensor_max_age_sec','localization_jump_m',
                    'click_timeout_sec','ready_timeout_sec','planning_timeout_sec','execution_timeout_sec','cancel_timeout_sec','stop_timeout_sec']:
            if not math.isfinite(self.cfg[key]) or self.cfg[key]<=0:raise TrialFailure('invalid positive parameter: '+key)
        if not math.isfinite(self.cfg['yaw_tolerance_rad']) or self.cfg['yaw_tolerance_rad']<0:raise TrialFailure('invalid yaw tolerance')
        if self.cfg['method'] not in METHODS or self.cfg['seed']<0:raise TrialFailure('invalid method or seed')
        if self.cfg['goal_from_click'] and self.cfg['goal_from_pose']:raise TrialFailure('select only one RViz goal input')
        if not self.cfg['check_only'] and not self.cfg['goal_set'] and not self.cfg['goal_from_click'] and not self.cfg['goal_from_pose']:raise TrialFailure('goal is required; use --rviz-goal or --goal-x/--goal-y')
        if not all(math.isfinite(self.cfg[k]) for k in ['goal_x','goal_y','goal_yaw']):raise TrialFailure('goal must be finite')
        if self.cfg['map_already_inflated']:raise TrialFailure('raw static map is required')
        self.wait(lambda:self.map is not None and self.odom is not None,self.cfg['ready_timeout_sec'],'map and odometry')
        if self.cfg['goal_from_pose'] and not self.cfg['check_only']:
            self.get_logger().info('Use 2D Goal Pose in rag_single_plan.rviz; waiting on '+self.cfg['goal_pose_topic'])
            self.wait(lambda:self.goal_pose is not None,self.cfg['click_timeout_sec'],'RViz goal pose')
            msg=self.goal_pose
            if msg.header.frame_id!=self.cfg['map_frame']:raise TrialFailure('goal pose frame differs from map_frame')
            self.cfg['goal_x']=float(msg.pose.position.x);self.cfg['goal_y']=float(msg.pose.position.y)
            q=msg.pose.orientation
            norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
            if not math.isfinite(norm) or norm<=1e-12:raise TrialFailure('goal quaternion is invalid')
            x,y,z,w=q.x/norm,q.y/norm,q.z/norm,q.w/norm
            self.cfg['goal_yaw']=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            if not all(math.isfinite(self.cfg[k]) for k in ['goal_x','goal_y','goal_yaw']):raise TrialFailure('goal pose is not finite')
            self.cfg['goal_set']=True
            self.save_json('rviz_goal.json',message_to_ordereddict(msg))
        if self.cfg['goal_from_click'] and not self.cfg['check_only']:
            self.get_logger().info('Select a goal with RViz Publish Point (not 2D Goal Pose).')
            self.wait(lambda:self.goal_point is not None,self.cfg['click_timeout_sec'],'RViz clicked point')
            if self.goal_point.header.frame_id!=self.cfg['map_frame']:raise TrialFailure('clicked point frame differs from map_frame')
            self.cfg['goal_x']=float(self.goal_point.point.x);self.cfg['goal_y']=float(self.goal_point.point.y)
            if not all(math.isfinite(self.cfg[k]) for k in ['goal_x','goal_y']):raise TrialFailure('clicked goal is not finite')
            self.cfg['goal_set']=True
        start,stamp=self.pose()
        if self.map.header.frame_id!=self.cfg['map_frame']:raise TrialFailure('map frame differs from configured map_frame')
        if self.map.info.resolution<=0 or len(self.map.data)!=self.map.info.width*self.map.info.height:raise TrialFailure('invalid map metadata')
        self.frozen_map_hash=self.map_hash
        self.save_json('map.json',message_to_ordereddict(self.map))
        self.state['map_sha256']=self.map_hash;self.state['actual_start']={'x':start[0],'y':start[1],'yaw':start[2]}
        self.state['planner_radius_m']=radius
        self.state['effective_clearance_cells']=max(2,math.ceil(radius/self.map.info.resolution))
        self.state['collision_check_policy']='uniform conservative grid-cell crossing, shared by all three methods; square clearance neighborhood'
        self.state['ready_pose_ros_sec']=stamp.sec+stamp.nanosec/1e9
        issues=self.navigation_checks()
        if self.cfg['check_only']:
            self.state.update(status='CHECK_PASS' if not issues else 'CHECK_FAIL',stage='completed');return
        if self.cfg['execute']:
            if issues:raise TrialFailure('execution preflight failed: '+'; '.join(issues))
            self.wait(self.action.server_is_ready,self.cfg['ready_timeout_sec'],'FollowPath server')
            self.assert_no_other_goal()
            self.wait(self.stopped,self.cfg['ready_timeout_sec'],'robot stationary before planning')
        self.save_json('metadata.json',self.state)
        self.state['stage']='planning'
        self.wait(self.client.service_is_ready,self.cfg['ready_timeout_sec'],'planner service')
        request=PlanPath.Request();request.start=self.make_pose(start)
        request.goal=self.make_pose((self.cfg['goal_x'],self.cfg['goal_y'],self.cfg['goal_yaw']))
        request.planner_name=self.cfg['method'];request.seed=self.cfg['seed']
        started=time.monotonic();response=self.future(self.client.call_async(request),self.cfg['planning_timeout_sec'],'planning')
        self.state['request_wall_time_sec']=time.monotonic()-started
        self.save_json('planner_response.json',message_to_ordereddict(response))
        if not response.success:raise TrialFailure('planning failed: '+response.message)
        if response.metrics.planner_name!=self.cfg['method'] or response.metrics.seed!=self.cfg['seed']:
            raise TrialFailure('planner response method/seed do not match this request')
        if self.map_hash!=self.frozen_map_hash:raise TrialFailure('map changed during planning')
        if response.path.header.frame_id!=self.cfg['map_frame']:raise TrialFailure('path frame mismatch')
        points=[(p.pose.position.x,p.pose.position.y) for p in response.path.poses]
        sampled=resample(points,self.cfg['resample_spacing_m'])
        length=sum(math.dist(a,b) for a,b in zip(sampled,sampled[1:]))
        if length<self.cfg['minimum_motion_m']:raise TrialFailure('path is shorter than minimum motion; not executing')
        if length>self.cfg['max_path_length_m']:raise TrialFailure('path exceeds length limit; no truncation or execution')
        goal=(self.cfg['goal_x'],self.cfg['goal_y'])
        if math.dist(sampled[-1],goal)>1e-6:raise TrialFailure('planner path does not end at requested goal')
        self.reference=sampled;reference=RosPath();reference.header.frame_id=self.cfg['map_frame'];reference.header.stamp=self.get_clock().now().to_msg()
        for index,p in enumerate(sampled):
            yaw=math.atan2(sampled[index+1][1]-p[1],sampled[index+1][0]-p[0]) if index+1<len(sampled) else self.cfg['goal_yaw']
            reference.poses.append(self.make_pose((p[0],p[1],yaw)))
        self.save_json('reference_path.json',message_to_ordereddict(reference));self.path_pub.publish(reference)
        self.state['reference_length_m']=length
        if not self.cfg['execute']:
            self.state.update(status='PLANNED_ONLY',stage='completed');return
        current,_=self.pose()
        if math.dist(current[:2],sampled[0])>self.cfg['start_tolerance_m']:raise TrialFailure('robot moved away from planned start')
        if not self.stopped():raise TrialFailure('robot moved during planning')
        self.assert_no_other_goal()
        self.state['stage']='execution';self.record_trajectory=True;self.last_motion_pose=None
        initial,initial_stamp=self.pose()
        self.trajectory.append(dict(wall_elapsed_sec=time.monotonic()-self.started,
            ros_sec=initial_stamp.sec+initial_stamp.nanosec/1e9,x=initial[0],y=initial[1],yaw=initial[2]))
        action_goal=FollowPath.Goal();action_goal.path=reference
        action_goal.controller_id=self.cfg['controller_id'];action_goal.goal_checker_id=self.cfg['goal_checker_id']
        if hasattr(action_goal,'progress_checker_id'):action_goal.progress_checker_id=self.cfg['progress_checker_id']
        elif self.cfg['progress_checker_id']:raise TrialFailure('Humble cannot select progress_checker_id per goal')
        self.own_uuid=list(uuid.uuid4().bytes)
        self.state['goal_uuid']=self.own_uuid
        self.motion_active=True
        self.pending_goal=self.action.send_goal_async(action_goal,feedback_callback=lambda msg:self.feedback.append(
            dict(wall_elapsed_sec=time.monotonic()-self.started,data=message_to_ordereddict(msg.feedback))),
            goal_uuid=UUID(uuid=self.own_uuid))
        self.own_goal=self.future(self.pending_goal,5.0,'goal acknowledgement')
        if not self.own_goal.accepted:
            self.motion_active=False;raise TrialFailure('FollowPath goal rejected')
        self.save_json('metadata.json',self.state)
        execution_start=time.monotonic()
        self.result_future=self.own_goal.get_result_async()
        result=self.future(self.result_future,self.cfg['execution_timeout_sec'],'execution')
        self.action_terminal=result.status in (4,5,6)
        self.state['execution_wall_time_sec']=time.monotonic()-execution_start
        self.state['action_status']=result.status
        has_code=hasattr(result.result,'error_code')
        self.state['nav2_error_code_available']=has_code
        self.state['nav2_error_code']=int(result.result.error_code) if has_code else None
        self.state['nav2_error_msg']=getattr(result.result,'error_msg',None)
        self.save_json('action_result.json',message_to_ordereddict(result))
        if result.status!=4 or (has_code and result.result.error_code!=0):raise TrialFailure('FollowPath did not succeed')
        stable=0;deadline=time.monotonic()+self.cfg['stop_timeout_sec']
        while stable<5 and time.monotonic()<deadline:
            self.spin();stable=stable+1 if self.stopped() else 0
        if stable<5:raise TrialFailure('Action succeeded but stopped motion could not be confirmed')
        self.motion_active=False
        final,stamp=self.pose();self.trajectory.append(dict(wall_elapsed_sec=time.monotonic()-self.started,
            ros_sec=stamp.sec+stamp.nanosec/1e9,x=final[0],y=final[1],yaw=final[2]))
        position_error=math.dist(final[:2],goal);yaw_error=abs(wrap_yaw(final[2]-self.cfg['goal_yaw']))
        travel=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(self.trajectory,self.trajectory[1:]))
        errors=[distance_to_path((p['x'],p['y']),sampled) for p in self.trajectory]
        self.state.update(final_position_error_estimated_m=position_error,final_yaw_error_estimated_rad=yaw_error,
                          estimated_trajectory_length_m=travel,stop_observed=True,
                          estimated_tracking_rmse_m=math.sqrt(sum(v*v for v in errors)/len(errors)) if errors else None)
        if position_error>self.cfg['endpoint_tolerance_m'] or yaw_error>self.cfg['yaw_tolerance_rad'] or travel<self.cfg['minimum_motion_m']:
            raise TrialFailure('Action succeeded but independent execution criteria failed (localization estimate)')
        self.state.update(status='EXECUTION_PASS',stage='completed')

    def finish(self):
        self.save_json('metadata.json',self.state);self.save_json('feedback.json',self.feedback)
        if self.cmd is not None:self.save_json('last_command.json',message_to_ordereddict(self.cmd))
        with (self.directory/'estimated_trajectory.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['wall_elapsed_sec','ros_sec','x','y','yaw']);writer.writeheader();writer.writerows(self.trajectory)
        self.get_logger().info('RESULT: '+str(self.directory)+' status='+self.state['status'])


def main(args=None):
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
    def interrupted(signum,frame):raise KeyboardInterrupt()
    old_term=signal.signal(signal.SIGTERM,interrupted)
    node=None;code=2
    try:
        node=Trial();node.run()
        code=0 if node.state['status'] in ('PLANNED_ONLY','CHECK_PASS','EXECUTION_PASS') else 2
    except (KeyboardInterrupt,Exception) as exc:
        if node is not None:
            node.state.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',failure_reason=str(exc),failed_stage=node.state['stage'])
            if node.motion_active:
                try:node.stop_and_cancel()
                except Exception as stop_exc:node.state['abort_stop_error']=str(stop_exc)
            node.get_logger().error(str(exc) or 'interrupted')
        else:
            print('Trial initialization failed: '+str(exc),file=sys.stderr)
    finally:
        if node is not None:
            try:node.finish()
            finally:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        signal.signal(signal.SIGTERM,old_term)
    raise SystemExit(code)
