#!/usr/bin/env bash
set -eo pipefail
validation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
goal_x=''; goal_y=''; goal_yaw='0'; method='rag_bpc_conn_init_rel_only_topk_fallback'; seed='0'; execute='false'; check='false'; clicked='false'
config_path="$validation_dir/src/rag_rrt_real_validation/config/site.yaml"
output_path="$HOME/rag_rrt_real_results"
rviz_goal='false'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) execute='true'; shift ;;
    --clicked-goal) clicked='true'; shift ;;
    --rviz-goal) rviz_goal='true'; shift ;;
    --check-only) check='true'; shift ;;
    --goal-x|--goal-y|--goal-yaw|--method|--seed|--config|--output)
      if [[ $# -lt 2 ]]; then echo "缺少 $1 的值" >&2; exit 2; fi
      case "$1" in
        --goal-x) goal_x="$2";; --goal-y) goal_y="$2";; --goal-yaw) goal_yaw="$2";;
        --method) method="$2";; --seed) seed="$2";; --config) config_path="$2";; --output) output_path="$2";;
      esac
      shift 2 ;;
    --help) echo 'run_trial.sh --check-only | --rviz-goal | --clicked-goal | --goal-x X --goal-y Y [--goal-yaw R] [--method NAME] [--seed N] [--execute] [--config FILE] [--output DIR]'; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
case "$method" in
  rrt) method='rrt_star';; informed) method='irrt_star';; rag) method='rag_bpc_conn_init_rel_only_topk_fallback';;
esac
if [[ "$check" == 'true' && "$execute" == 'true' ]]; then echo '--check-only 与 --execute 不能同时使用' >&2; exit 2; fi
if [[ "$rviz_goal" == 'true' && "$clicked" == 'true' ]]; then echo '只选择一种 RViz 目标输入' >&2; exit 2; fi
if [[ "$check" != 'true' && "$clicked" != 'true' && "$rviz_goal" != 'true' && ( -z "$goal_x" || -z "$goal_y" ) ]]; then
  echo '需要实际 map 坐标系中的 --goal-x 和 --goal-y；未启动任何节点。' >&2; exit 2
fi
if [[ ! -f "$config_path" || ! -f "$validation_dir/install/setup.bash" ]]; then
  echo '配置文件或编译结果不存在；先运行 build.sh。' >&2; exit 2
fi
export RAG_TRIAL_CONFIG="$config_path"
trial_lock="/tmp/rag_rrt_validation_domain_${ROS_DOMAIN_ID:-0}.lock"
exec 9>"$trial_lock"
if ! flock -n 9; then echo '同一 ROS Domain 的本机实验已在运行；未启动第二次实验。' >&2; exit 2; fi
source /opt/ros/humble/setup.bash
if [[ -n "${SENTRY_UNDERLAY_SETUP:-}" ]]; then
  if [[ ! -f "$SENTRY_UNDERLAY_SETUP" ]]; then echo 'SENTRY_UNDERLAY_SETUP 文件不存在' >&2; exit 2; fi
  source "$SENTRY_UNDERLAY_SETUP"
fi
source "$validation_dir/install/setup.bash"
args=("method:=$method" "seed:=$seed" "execute:=$execute" "check_only:=$check" "goal_from_click:=$clicked" "output_dir:=$output_path")
args+=("goal_from_pose:=$rviz_goal")
if [[ "$check" != 'true' && "$clicked" != 'true' && "$rviz_goal" != 'true' ]]; then args+=("goal_set:=true" "goal_x:=$goal_x" "goal_y:=$goal_y" "goal_yaw:=$goal_yaw"); fi
mkdir -p "$output_path"
trial_dir="$(mktemp -d "$output_path/trial_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
args+=("run_dir:=$trial_dir")
set +e
ros2 launch rag_rrt_real_validation trial.launch.py "${args[@]}" 2>&1 | tee "$trial_dir/launch.log"
launch_code=${PIPESTATUS[0]}
set -e
/usr/bin/python3 - "$trial_dir" "$launch_code" <<'PYRESULT'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); metadata=path/'metadata.json'
if not metadata.exists():
    print('未生成 metadata.json；请查看 '+str(path/'launch.log'))
    raise SystemExit(2)
status=json.loads(metadata.read_text())['status']
print('本次结果：'+status+'；目录：'+str(path))
raise SystemExit(0 if status in ['CHECK_PASS','PLANNED_ONLY','EXECUTION_PASS'] and int(sys.argv[2])==0 else 2)
PYRESULT
