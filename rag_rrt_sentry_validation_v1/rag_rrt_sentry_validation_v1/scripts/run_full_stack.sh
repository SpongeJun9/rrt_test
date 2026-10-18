#!/usr/bin/env bash
set -eo pipefail
validation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/humble/setup.bash || ! -f "$validation_dir/install/setup.bash" ]]; then
  echo '需要 ROS 2 Humble；先设置 SENTRY_UNDERLAY_SETUP 并运行 scripts/build.sh。' >&2; exit 2
fi
source /opt/ros/humble/setup.bash
if [[ -n "${SENTRY_UNDERLAY_SETUP:-}" ]]; then
  [[ -f "$SENTRY_UNDERLAY_SETUP" ]] || { echo 'SENTRY_UNDERLAY_SETUP 文件不存在' >&2; exit 2; }
  source "$SENTRY_UNDERLAY_SETUP"
fi
source "$validation_dir/install/setup.bash"
exec 9>"/tmp/rag_rrt_validation_domain_${ROS_DOMAIN_ID:-0}.lock"
flock -n 9 || { echo '同一 ROS Domain 已有验证程序运行。' >&2; exit 2; }
exec ros2 launch rag_rrt_real_validation full_stack.launch.py "$@"
