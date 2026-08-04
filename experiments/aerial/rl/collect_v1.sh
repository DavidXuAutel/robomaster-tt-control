#!/usr/bin/env bash
# Two-tier V1 dataset collection for the aerial WAM RL skeleton.
#
# The V0 run (dataset_v0) commanded 12 Hz but achieved 7.1-8.3 (mean 7.9), so
# every action label encoded a step the loop couldn't hit — a ~17% per-step
# dt-desync across all 3138 steps. It's a smoke set, not a training set. This
# script collects a clean V1 set in two tiers, each run at a rate it can hit so
# commanded dt matches actual loop time:
#
#   Tier RGB    RGB-only @ 8 Hz  (grab_depth off)   -> dataset_v1_rgb
#     The bulk V1 training set. 8 Hz sits under the measured RGB-only closed-loop
#     floor (7.1-8.3), so labels stay consistent. This is what WM/dynamics and
#     policy train on.
#
#   Tier DEPTH  RGB+depth @ 3 Hz (grab_depth on)    -> dataset_v1_depth
#     A SMALL sample. Per-step DepthPlanar drops the closed loop to ~3 Hz, so it
#     runs at 3 Hz (again: commanded == achievable). This tier is supervision for
#     the v2 perception heads ([1b]/[1c]/[1d]) — RGB<->depth pairs over the scene
#     distribution. It is NOT dynamics/policy data: at a different dt the per-step
#     action cap differs, so its trajectories diverge from the RGB tier's. Exact
#     state correspondence with Tier RGB is neither expected nor needed.
#
# Both tiers share one annotation (start/goal poses) and route through
# collect_dataset.py, so both get the instant-crash quarantine + reset-collision
# guard + run-level fraction gate. A tier fails (nonzero exit) on a frozen
# renderer / black frames / dead control, or if >20% of episodes are quarantined.
#
#   # real collection on a 4090-reachable host:
#   ANNOTATION=path/to/seen_airsim16_m1a20.json \
#     experiments/aerial/rl/collect_v1.sh --airsim
#
#   # offline plumbing dry-run (mock env, no renderer / annotation):
#   experiments/aerial/rl/collect_v1.sh --mock
#
# Env knobs (all optional unless noted):
#   ANNOTATION       OpenFly annotation JSON (start/goal). REQUIRED for --airsim.
#   OUT_ROOT         parent dir for the two dataset dirs   (default artifacts/)
#   RGB_EPISODES     Tier RGB episode count                (default 20)
#   DEPTH_EPISODES   Tier DEPTH episode count (small!)     (default 4)
#   MAX_STEPS        steps per episode                     (default 200)
#   STEP_HZ_RGB      Tier RGB rate                         (default 8)
#   STEP_HZ_DEPTH    Tier DEPTH rate                       (default 3)
#   PYTHON_BIN       python interpreter                    (default python3)
# AirSim host/port/camera/vehicle are read from sim_verify/config.env if present,
# else fall back to the known 4090 defaults (10.229.20.125:41451).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKEND=""
for a in "$@"; do
  case "$a" in
    --airsim) BACKEND="airsim" ;;
    --mock)   BACKEND="mock" ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $a (see --help)" >&2; exit 2 ;;
  esac
done
if [ -z "$BACKEND" ]; then
  echo "specify --airsim (real collection) or --mock (offline dry-run); see --help" >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-experiments/aerial/rl/artifacts}"
RGB_EPISODES="${RGB_EPISODES:-20}"
DEPTH_EPISODES="${DEPTH_EPISODES:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
STEP_HZ_RGB="${STEP_HZ_RGB:-8}"
STEP_HZ_DEPTH="${STEP_HZ_DEPTH:-3}"

# Common args shared by both tiers. Real collection needs a renderer + annotation.
COMMON=(--backend "$BACKEND" --max-steps "$MAX_STEPS")
if [ "$BACKEND" = "airsim" ]; then
  CFG_ENV="$REPO_ROOT/experiments/aerial/sim_verify/config.env"
  if [ -f "$CFG_ENV" ]; then
    echo "[collect-v1] sourcing $CFG_ENV"
    # shellcheck disable=SC1090
    source "$CFG_ENV"
  fi
  if [ -z "${ANNOTATION:-}" ]; then
    echo "[collect-v1] FAIL: --airsim needs ANNOTATION=<start/goal JSON>" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import airsim, cv2" 2>/dev/null; then
    echo "[collect-v1] FAIL: --airsim needs 'airsim' and 'cv2' installed here" >&2
    exit 1
  fi
  COMMON+=(
    --host "${AIRSIM_HOST:-10.229.20.125}"
    --port "${AIRSIM_PORT:-41451}"
    --camera "${AIRSIM_CAMERA:-front_custom}"
    --vehicle "${AIRSIM_VEHICLE:-drone_1}"
    --annotation "$ANNOTATION"
  )
fi

# --- Tier RGB: bulk, RGB-only @ STEP_HZ_RGB -------------------------------
RGB_OUT="$OUT_ROOT/dataset_v1_rgb"
echo "[collect-v1] Tier RGB — $RGB_EPISODES eps @ ${STEP_HZ_RGB} Hz, RGB-only -> $RGB_OUT"
"$PYTHON_BIN" -m experiments.aerial.rl.collect_dataset "${COMMON[@]}" \
  --episodes "$RGB_EPISODES" --step-hz "$STEP_HZ_RGB" --out "$RGB_OUT"

# --- Tier DEPTH: small sample, RGB+depth @ STEP_HZ_DEPTH ------------------
DEPTH_OUT="$OUT_ROOT/dataset_v1_depth"
echo "[collect-v1] Tier DEPTH — $DEPTH_EPISODES eps @ ${STEP_HZ_DEPTH} Hz, +depth -> $DEPTH_OUT"
"$PYTHON_BIN" -m experiments.aerial.rl.collect_dataset "${COMMON[@]}" \
  --episodes "$DEPTH_EPISODES" --step-hz "$STEP_HZ_DEPTH" --grab-depth --out "$DEPTH_OUT"

echo "[collect-v1] done: RGB set in $RGB_OUT, depth sample in $DEPTH_OUT"
echo "[collect-v1] check each dir's QUALITY_SUMMARY.json (usable vs quarantined)"
