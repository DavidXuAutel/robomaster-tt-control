#!/usr/bin/env bash
# Orchestrate the sim-capability verification and print the Fork verdict.
#
# Usage:
#   cp config.env.example config.env && edit it
#   ./run_all.sh                # sources ./config.env if present
#
# Runs: T0 connectivity -> T1 real render -> T2 capability probe -> verdict.
# Each probe merges into $OUT (default ./artifacts/sim_capability_report.json).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Load config if present (does not override already-exported vars).
if [[ -f "./config.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source ./config.env; set +a
fi

export OUT="${OUT:-$HERE/artifacts/sim_capability_report.json}"
export AIRSIM_HOST="${AIRSIM_HOST:-10.229.20.125}"
export AIRSIM_PORT="${AIRSIM_PORT:-41451}"
export ENV_NAME="${ENV_NAME:-env_airsim_16}"
PY="${PYTHON:-python3}"

mkdir -p "$(dirname "$OUT")"
echo "== sim_verify =="
echo "   AIRSIM        : $AIRSIM_HOST:$AIRSIM_PORT"
echo "   ENV_NAME      : $ENV_NAME"
echo "   OPENFLY_ROOT  : ${OPENFLY_ROOT:-<unset — T1 will skip>}"
echo "   report        : $OUT"
echo

echo "== preflight (环境优先检查) =="
if ! "$HERE/preflight.sh"; then
  echo
  echo "[run_all] 环境检查未通过。缺失项：系统前置手动装(SETUP.md §1.1)；"
  echo "          OpenFly 层跑  CONFIRM=1 ./setup_env.sh  自动补。"
  if [ "${FORCE:-0}" != "1" ]; then
    echo "[run_all] 已中止（探针会因缺依赖失败）。确要继续：FORCE=1 ./run_all.sh"
    exit 1
  fi
  echo "[run_all] FORCE=1 —— 忽略检查，继续跑探针……"
fi
echo

echo "-- T0 connectivity --"
"$PY" probes/t0_connectivity.py || echo "[T0] failed (see report)"

if [[ -n "${OPENFLY_ROOT:-}" ]]; then
  echo "-- T1 real render --"
  "$PY" probes/t1_render.py || echo "[T1] failed (see report)"
else
  echo "-- T1 real render -- SKIP (OPENFLY_ROOT unset)"
fi

echo "-- T2 capability probe --"
"$PY" probes/t2_capability.py || echo "[T2] finished with failures (see report)"

echo "-- verdict --"
"$PY" verdict.py
verdict_code=$?

echo
echo "Full report: $OUT"
exit $verdict_code
