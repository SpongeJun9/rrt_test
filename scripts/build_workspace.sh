#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
cd "$workspace_dir"
targets=(pb2025_nav_bringup core_serial pb2025_robot_description pcd2pgm)
if [[ "${1:-}" == '--install-deps' ]]; then
  paths_file="$(mktemp)"
  trap 'rm -f "$paths_file"' EXIT
  colcon list --base-paths src --packages-up-to "${targets[@]}" --paths-only > "$paths_file"
  mapfile -t dependency_paths < "$paths_file"
  [[ ${#dependency_paths[@]} -gt 0 ]] || { echo '没有找到导航包' >&2; exit 2; }
  rosdep install --from-paths "${dependency_paths[@]}" --ignore-src -r -y --rosdistro humble
elif [[ $# -ne 0 ]]; then
  echo '用法：bash scripts/build_workspace.sh [--install-deps]' >&2; exit 2
fi
colcon build --base-paths src --packages-up-to "${targets[@]}" --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
export SENTRY_UNDERLAY_SETUP="$workspace_dir/install/setup.bash"
bash "$workspace_dir/rag_rrt_sentry_validation_v1/rag_rrt_sentry_validation_v1/scripts/build.sh"
echo '导航工作区和 RAG 验证工作区构建完成；尚未启动硬件。'
