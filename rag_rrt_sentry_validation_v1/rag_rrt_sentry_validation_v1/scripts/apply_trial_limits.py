#!/usr/bin/env python3
"""Explicit operator-invoked setup: modify runtime speed/footprint, never drive."""
import argparse
import math
import shlex
from pathlib import Path
import subprocess
import yaml

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'src/rag_rrt_real_validation/config/site.yaml'))
    parser.add_argument('--apply',action='store_true',help='Apply printed runtime parameter changes')
    args=parser.parse_args();site=yaml.safe_load(Path(args.config).expanduser().read_text())
    cap=float(site['max_linear_speed_mps']);x=float(site['robot_length_m'])/2+float(site['body_margin_m']);y=float(site['robot_width_m'])/2+float(site['body_margin_m'])
    if not all(math.isfinite(v) and v>0 for v in [cap,x,y]):parser.error('invalid speed/body geometry')
    prefix=site['controller_id'];controller=site['controller_node'];smoother=site['velocity_smoother_node']
    entries=[(controller,prefix+'.v_linear_max',cap),(controller,prefix+'.v_linear_min',-cap),
             (controller,prefix+'.min_approach_linear_velocity',min(0.1,cap)),
             (smoother,'max_velocity',[cap,cap,0.0]),(smoother,'min_velocity',[-cap,-cap,0.0])]
    for node in site['costmap_nodes']:
        entries.append((node,'footprint',str([[x,y],[x,-y],[-x,-y],[-x,y]])))
    print('这些设置仅在当前节点存活期间生效；原 YAML 文件不修改。先确保机器人停稳、无其他导航任务。')
    for node,key,value in entries:
        text=yaml.safe_dump(value,default_flow_style=True).strip().removesuffix('...').strip()
        # Footprint is a ROS string containing a polygon, not a double array.
        if key=='footprint':text='"'+value+'"'
        command=['ros2','param','set',node,key,text]
        print(shlex.join(command),flush=True)
        if args.apply:
            subprocess.run(command,check=True,timeout=10)
    command=['ros2','topic','pub','--once',site.get('spin_topic','/cmd_spin'),
             'example_interfaces/msg/Float32','{data: 0.0}']
    print(shlex.join(command),flush=True)
    if args.apply:subprocess.run(command,check=True,timeout=10)
    print('这条零值仅清除 fake_vel_transform 的旋转偏置；下位机自旋仍须按原系统方式关闭。')
    print('请运行 --check-only 读取并确认实际生效值；脚本不保证所有节点支持动态修改。')

if __name__=='__main__':main()
