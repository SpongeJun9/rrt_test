#!/usr/bin/env bash
set -eo pipefail
validation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
if [[ -n "${SENTRY_UNDERLAY_SETUP:-}" ]]; then
  if [[ ! -f "$SENTRY_UNDERLAY_SETUP" ]]; then echo 'SENTRY_UNDERLAY_SETUP 文件不存在' >&2; exit 2; fi
  source "$SENTRY_UNDERLAY_SETUP"
fi
/usr/bin/python3 "$validation_dir/scripts/apply_trial_limits.py" "$@"
