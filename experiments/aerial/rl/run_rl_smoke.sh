#!/usr/bin/env bash
# One-shot smoke for the aerial RL skeleton. Three tiers, cheapest first:
#
#   Tier 0  offline unit tests (55 cases)     — always; no GPU / no network
#   Tier 1  mock corrector smoke              — always; no GPU / no network
#   Tier 2  live AirSim corrector smoke       — ONLY with --airsim; needs the
#           4090 renderer reachable + `airsim`/`cv2` installed on this host
#
# Tiers 0/1 run anywhere (CI, laptop, sandbox). Tier 2 validates the Plan-A
# ~30 Hz serial-real-env assumption on real hardware and is opt-in so the
# default invocation never hangs waiting on a renderer.
#
#   experiments/aerial/rl/run_rl_smoke.sh              # offline: tests + mock
#   experiments/aerial/rl/run_rl_smoke.sh --airsim     # + live 4090 smoke
#
# Env knobs (all optional):
#   RL_SMOKE_MAX_STEPS   steps per collection episode        (default 100)
#   RL_SMOKE_MIN_HZ      Tier-2 min achieved Hz to pass      (default 24 = .8*30)
#   PYTHON_BIN           python interpreter                  (default python3)
# AirSim host/port/camera are read from sim_verify/config.env if present, else
# fall back to the known 4090 defaults (10.229.20.125:41451).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_AIRSIM=0
for a in "$@"; do
  case "$a" in
    --airsim) RUN_AIRSIM=1 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $a (see --help)" >&2; exit 2 ;;
  esac
done

MAX_STEPS="${RL_SMOKE_MAX_STEPS:-100}"
MIN_HZ="${RL_SMOKE_MIN_HZ:-24}"

# --- Tier 0: offline unit tests -------------------------------------------
echo "[smoke] Tier 0 — offline unit tests"
"$PYTHON_BIN" -m pytest -q experiments/aerial/rl/tests/

# --- Tier 1: mock corrector smoke -----------------------------------------
echo "[smoke] Tier 1 — mock corrector smoke ($MAX_STEPS steps)"
"$PYTHON_BIN" -m experiments.aerial.rl._smoke --backend mock --max-steps "$MAX_STEPS"

# --- Tier 2: live AirSim corrector smoke (opt-in) -------------------------
if [ "$RUN_AIRSIM" != 1 ]; then
  echo "[smoke] Tier 2 — SKIPPED (pass --airsim on a 4090-reachable host)"
  echo "[smoke] done: offline tiers OK"
  exit 0
fi

CFG_ENV="$REPO_ROOT/experiments/aerial/sim_verify/config.env"
if [ -f "$CFG_ENV" ]; then
  echo "[smoke] sourcing $CFG_ENV"
  # shellcheck disable=SC1090
  source "$CFG_ENV"
fi
HOST="${AIRSIM_HOST:-10.229.20.125}"
PORT="${AIRSIM_PORT:-41451}"
CAMERA="${AIRSIM_CAMERA:-front_custom}"

if ! "$PYTHON_BIN" -c "import airsim, cv2" 2>/dev/null; then
  echo "[smoke] FAIL: --airsim needs 'airsim' and 'cv2' installed on this host" >&2
  exit 1
fi

echo "[smoke] Tier 2 — live AirSim smoke @ $HOST:$PORT (min ${MIN_HZ} Hz)"
"$PYTHON_BIN" -m experiments.aerial.rl._smoke \
  --backend airsim --max-steps "$MAX_STEPS" --min-hz "$MIN_HZ" \
  --host "$HOST" --port "$PORT" --camera "$CAMERA"

echo "[smoke] done: all tiers OK"
