#!/usr/bin/env bash
set -eo pipefail
validation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/humble/setup.bash || ! -x /usr/bin/colcon ]]; then
  echo '需要 ROS 2 Humble 和系统安装的 colcon；请按使用说明安装缺少的依赖。' >&2
  exit 2
fi
source /opt/ros/humble/setup.bash
if [[ -n "${SENTRY_UNDERLAY_SETUP:-}" ]]; then
  if [[ ! -f "$SENTRY_UNDERLAY_SETUP" ]]; then echo 'SENTRY_UNDERLAY_SETUP 文件不存在' >&2; exit 2; fi
  source "$SENTRY_UNDERLAY_SETUP"
fi
cd "$validation_dir"
/usr/bin/colcon build --symlink-install --event-handlers console_direct+ \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
echo '编译结束。此步骤没有启动机器人或发送运动指令。'
