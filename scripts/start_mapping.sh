#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$workspace_dir/install/setup.bash"
exec ros2 launch pb2025_nav_bringup rm_navigation_reality_launch.py \
  slam:=True use_sim_time:=False use_composition:=False use_robot_state_pub:=True "$@"
