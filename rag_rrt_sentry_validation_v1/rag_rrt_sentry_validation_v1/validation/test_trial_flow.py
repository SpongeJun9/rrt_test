"""Exercise real trial control flow with fake ROS transport; not a ROS integration test."""
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
import yaml
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rag_rrt_real_validation'))
NS=types.SimpleNamespace

def module(name):
    obj=types.ModuleType(name);obj.__path__=[];sys.modules[name]=obj
    if '.' in name:
        parent,child=name.rsplit('.',1)
        if parent not in sys.modules:module(parent)
        setattr(sys.modules[parent],child,obj)
    return obj

class Stamp:
    sec=0;nanosec=0
class PoseStamped:
    def __init__(self):
        self.header=NS(frame_id='map',stamp=Stamp())
        self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
class RosPath:
    def __init__(self):self.header=NS(frame_id='map',stamp=Stamp());self.poses=[]
class Twist:
    def __init__(self):self.linear=NS(x=0.,y=0.,z=0.);self.angular=NS(x=0.,y=0.,z=0.)
class Node:
    def get_clock(self):return NS(now=lambda:NS(to_msg=lambda:Stamp()))
    def get_logger(self):return NS(info=lambda _:None,error=lambda _:None)
class Future:
    def __init__(self,value):self.value=value
    def done(self):return True
    def result(self):return self.value

module('rclpy').ok=lambda:True
module('rclpy.action').ActionClient=object
module('rclpy.node').Node=Node
q=module('rclpy.qos');q.QoSProfile=lambda **kw:kw;q.DurabilityPolicy=NS(TRANSIENT_LOCAL=1);q.ReliabilityPolicy=NS(RELIABLE=1);q.qos_profile_sensor_data=None
module('rclpy.signals').SignalHandlerOptions=NS(NO=0)
module('rclpy.time').Time=object
module('rcl_interfaces.srv').GetParameters=NS(Request=object)
module('action_msgs.msg').GoalStatusArray=object
module('unique_identifier_msgs.msg').UUID=lambda **kw:NS(**kw)
g=module('geometry_msgs.msg');g.PoseStamped=PoseStamped;g.PointStamped=object;g.Twist=Twist
n=module('nav_msgs.msg');n.Path=RosPath;n.OccupancyGrid=object;n.Odometry=object
module('nav2_msgs.action').FollowPath=NS(Goal=lambda:NS(path=None,controller_id='',goal_checker_id=''))
module('rosidl_runtime_py.convert').message_to_ordereddict=lambda _:dict(mock_transport=True)
t=module('tf2_ros');t.Buffer=object;t.TransformListener=object
module('ament_index_python.packages').get_package_share_directory=lambda name:str(ROOT/'src'/name)
module('rag_rrt_interfaces.srv').PlanPath=NS(Request=lambda:NS(start=None,goal=None,planner_name='',seed=0))
runner=importlib.import_module('rag_rrt_real_validation.runner')

def fixture(folder,execute=False):
    trial=runner.Trial.__new__(runner.Trial)
    trial.cfg=yaml.safe_load((ROOT/'src/rag_rrt_real_validation/config/site.yaml').read_text())
    trial.cfg.update(goal_x=1.,goal_y=0.,goal_yaw=0.,goal_set=True,execute=execute,check_only=False)
    trial.state=dict(stage='init',status='RUNNING');trial.directory=folder
    trial.map=NS(header=NS(frame_id='map'),info=NS(resolution=.05,width=20,height=20),data=[0]*400)
    trial.map_hash='frozen';trial.frozen_map_hash=None;trial.odom=NS(twist=NS(twist=Twist()))
    trial.odom_received=time.monotonic();trial.cmd=None;trial.started=time.monotonic()
    trial.trajectory=[];trial.pose_history=[];trial.reference=[];trial.feedback=[];trial.statuses={}
    trial.motion_active=False;trial.record_trajectory=False;trial.last_motion_pose=None;trial.own_uuid=None
    trial.own_goal=None;trial.pending_goal=None;trial.result_future=None;trial.action_terminal=False;trial.current_pose=(0.,0.,0.)
    trial.pose=lambda:(trial.current_pose,Stamp());trial.stopped=lambda:True
    def spin(guard=True):
        if trial.record_trajectory:
            p=trial.current_pose;trial.trajectory.append(dict(wall_elapsed_sec=time.monotonic()-trial.started,ros_sec=0.,x=p[0],y=p[1],yaw=p[2]))
    trial.spin=spin
    trial.path_pub=NS(publish=lambda msg:None);trial.zero_commands=[]
    trial.zero_pub=NS(publish=lambda msg:trial.zero_commands.append(msg))
    path=RosPath()
    for x in [0.,.5,1.]:
        p=PoseStamped();p.pose.position.x=x;path.poses.append(p)
    trial.response=NS(success=True,message='found path',path=path,metrics=NS(planner_name=trial.cfg['method'],seed=trial.cfg['seed']))
    trial.planning_calls=[]
    def plan(request):trial.planning_calls.append(request);return Future(trial.response)
    trial.client=NS(service_is_ready=lambda:True,call_async=plan)
    trial.goals=[];trial.cancels=[]
    trial.result=NS(status=4,result=NS())
    trial.handle=NS(accepted=True)
    def get_result():trial.current_pose=(1.,0.,0.);return Future(trial.result)
    def cancel():trial.cancels.append(True);return Future(NS(return_code=0))
    trial.handle.get_result_async=get_result;trial.handle.cancel_goal_async=cancel
    def send(goal,feedback_callback,goal_uuid):trial.goals.append(goal);return Future(trial.handle)
    trial.action=NS(server_is_ready=lambda:True,send_goal_async=send)
    cp={'use_sim_time':False,'controller_plugins':['FollowPath'],'goal_checker_plugins':['general_goal_checker'],
        'FollowPath.v_linear_max':.3,'FollowPath.v_linear_min':-.3,'FollowPath.min_approach_linear_velocity':.1,'FollowPath.enable_rotation':False}
    sm={'use_sim_time':False,'max_velocity':[.3,.3,0.],'min_velocity':[-.3,-.3,0.]}
    cm={'use_sim_time':False,'footprint':'','robot_radius':.5}
    trial.parameters={trial.cfg['controller_node']:cp,trial.cfg['velocity_smoother_node']:sm}
    for name in trial.cfg['costmap_nodes']:trial.parameters[name]=dict(cm)
    trial.read_params=lambda node,names:trial.parameters[node]
    return trial

class TrialFlowTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_plan_only_never_sends_goal(self):
        t=fixture(self.folder);t.run()
        self.assertEqual(t.state['status'],'PLANNED_ONLY');self.assertEqual(t.goals,[])
        self.assertTrue((self.folder/'planner_response.json').exists())

    def test_execute_humble_and_complete_target(self):
        t=fixture(self.folder,True);t.run()
        self.assertEqual(t.state['status'],'EXECUTION_PASS');self.assertEqual(len(t.goals),1)
        self.assertIsNone(t.state['nav2_error_code']);self.assertFalse(t.state['nav2_error_code_available'])
        self.assertEqual(t.goals[0].path.poses[-1].pose.position.x,1.)
        self.assertGreaterEqual(t.state['estimated_trajectory_length_m'],1.)

    def test_rviz_pose_plans_and_executes_exactly_once(self):
        t=fixture(self.folder,True)
        t.cfg.update(goal_set=False,goal_from_pose=True)
        t.goal_pose=None
        msg=PoseStamped();msg.pose.position.x=1.
        msg.pose.orientation.z=math.sin(.3);msg.pose.orientation.w=math.cos(.3)
        t.on_goal_pose(msg)
        replacement=PoseStamped();replacement.pose.position.x=9.
        t.on_goal_pose(replacement)
        t.run()
        self.assertEqual(len(t.planning_calls),1);self.assertEqual(len(t.goals),1)
        self.assertEqual(t.planning_calls[0].goal.pose.position.x,1.)
        self.assertAlmostEqual(t.cfg['goal_yaw'],.6)
        self.assertTrue((self.folder/'rviz_goal.json').exists())

    def test_rviz_invalid_goal_does_not_plan(self):
        t=fixture(self.folder,True);t.cfg.update(goal_set=False,goal_from_pose=True)
        t.goal_pose=PoseStamped();t.goal_pose.header.frame_id='odom'
        with self.assertRaisesRegex(runner.TrialFailure,'goal pose frame'):t.run()
        self.assertEqual(t.planning_calls,[]);self.assertEqual(t.goals,[])

    def test_rviz_zero_quaternion_does_not_plan(self):
        t=fixture(self.folder,True);t.cfg.update(goal_set=False,goal_from_pose=True)
        t.goal_pose=PoseStamped();t.goal_pose.pose.orientation.w=0.
        with self.assertRaisesRegex(runner.TrialFailure,'quaternion'):t.run()
        self.assertEqual(t.planning_calls,[])

    def test_new_interface_nonzero_error_fails(self):
        t=fixture(self.folder,True);t.result=NS(status=4,result=NS(error_code=101,error_msg='failed'))
        with self.assertRaises(runner.TrialFailure):t.run()
        self.assertEqual(t.state['nav2_error_code'],101);self.assertTrue(t.motion_active)

    def test_preflight_small_radius_blocks_before_planning(self):
        t=fixture(self.folder,True)
        t.parameters[t.cfg['costmap_nodes'][0]]['robot_radius']=.2
        with self.assertRaisesRegex(runner.TrialFailure,'footprint'):t.run()
        self.assertEqual(t.goals,[]);self.assertEqual(t.planning_calls,[])

    def test_planning_failure_does_not_execute_or_replace_seed(self):
        t=fixture(self.folder,True);t.response.success=False
        with self.assertRaisesRegex(runner.TrialFailure,'planning failed'):t.run()
        self.assertEqual(t.goals,[]);self.assertEqual(len(t.planning_calls),1)

    def test_active_other_goal_blocks(self):
        t=fixture(self.folder,True);t.statuses['/navigate_to_pose']=NS(status_list=[NS(status=2)])
        with self.assertRaisesRegex(runner.TrialFailure,'another navigation'):t.run()
        self.assertEqual(t.goals,[])

    def test_map_changed_after_planning_blocks(self):
        t=fixture(self.folder,True)
        def plan(request):t.map_hash='changed';return Future(t.response)
        t.client.call_async=plan
        with self.assertRaisesRegex(runner.TrialFailure,'map changed'):t.run()
        self.assertEqual(t.goals,[])

    def test_path_length_limit_does_not_truncate(self):
        t=fixture(self.folder,True);t.cfg['max_path_length_m']=.5
        with self.assertRaisesRegex(runner.TrialFailure,'length limit'):t.run()
        self.assertEqual(t.goals,[])

    def test_goal_rejection(self):
        t=fixture(self.folder,True);t.handle.accepted=False
        with self.assertRaisesRegex(runner.TrialFailure,'goal rejected'):t.run()
        self.assertFalse(t.motion_active)

    def test_abort_cancels_own_goal_and_sends_zero(self):
        t=fixture(self.folder,True);t.own_goal=t.handle;t.motion_active=True
        t.stop_and_cancel()
        self.assertEqual(len(t.cancels),1);self.assertGreater(len(t.zero_commands),0)
        self.assertTrue(t.state['abort_stop']['stop_observed'])

    def test_late_acknowledgement_is_cancelled(self):
        t=fixture(self.folder,True);t.pending_goal=Future(t.handle);t.motion_active=True
        t.stop_and_cancel()
        self.assertEqual(len(t.cancels),1)

    def test_execution_timeout_then_cancel(self):
        t=fixture(self.folder,True);t.cfg['execution_timeout_sec']=.01
        never=NS(done=lambda:False,result=lambda:None)
        t.handle.get_result_async=lambda:never
        with self.assertRaisesRegex(runner.TrialFailure,'execution timeout'):t.run()
        t.handle.get_result_async=lambda:Future(NS(status=5,result=NS()))
        t.result_future=Future(NS(status=5,result=NS()))
        t.stop_and_cancel();self.assertEqual(len(t.cancels),1)

    def test_own_goal_status_is_not_external_goal(self):
        t=fixture(self.folder,True);t.own_uuid=[1]*16
        item=NS(status=2,goal_info=NS(goal_id=NS(uuid=[1]*16)))
        t.statuses[t.cfg['follow_path_action']]=NS(status_list=[item])
        t.assert_no_other_goal()
        item.goal_info.goal_id.uuid=[2]*16
        with self.assertRaisesRegex(runner.TrialFailure,'another navigation'):t.assert_no_other_goal()

    def test_unknown_stop_not_claimed(self):
        t=fixture(self.folder,True);t.own_goal=t.handle;t.cfg['stop_timeout_sec']=.01;t.stopped=lambda:False
        t.spin=lambda guard=True:time.sleep(.002)
        t.stop_and_cancel();self.assertFalse(t.state['abort_stop']['stop_observed'])

    def test_stopped_requires_finite_velocity_and_tf_stability(self):
        t=fixture(self.folder);now=time.monotonic();t.pose_history=[(now-.4,(0.,0.,0.)),(now-.2,(0.,0.,0.)),(now,(0.,0.,0.))]
        self.assertTrue(runner.Trial.stopped(t))
        t.odom.twist.twist.linear.x=math.nan;self.assertFalse(runner.Trial.stopped(t))
        t.odom.twist.twist.linear.x=0.;t.pose_history[0]=(now-.4,(.1,0.,0.));self.assertFalse(runner.Trial.stopped(t))

if __name__=='__main__':unittest.main()
