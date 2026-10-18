#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SENTRY_UNDERLAY_SETUP="$workspace_dir/install/setup.bash"
exec bash "$workspace_dir/rag_rrt_sentry_validation_v1/rag_rrt_sentry_validation_v1/scripts/run_full_stack.sh" "$@"
