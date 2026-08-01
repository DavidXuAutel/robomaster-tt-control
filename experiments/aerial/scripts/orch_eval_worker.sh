#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/home/a25689/aerial_cache_shared/orchestration/eval_queue}"
EVAL_ENV_FILE="${EVAL_ENV_FILE:-/home/a25689/aerial_eval_cache/env.sh}"
EVAL_WORKER_LOCK="${EVAL_WORKER_LOCK:-/home/a25689/aerial_eval_cache/orchestration/eval_worker.lock}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Opt-in world-model video dump. DUMP_WM_FRAMES=1 also decodes the video-branch
# latents that conditioned each action into {prefix}_wm_step*.mp4 beside the
# ground-truth frames. WM_DUMP_EVERY>0 samples mid-rollout (0 = only step 0 per
# episode). Off by default so normal gate eval pays no extra VAE decode.
DUMP_WM_FRAMES="${DUMP_WM_FRAMES:-0}"
WM_DUMP_EVERY="${WM_DUMP_EVERY:-0}"
# Stage-0 diagnostic. ORACLE_STOP=1 terminates each episode as success the moment
# the drone enters SUCCESS_DIST of the ground-truth goal — measures the oracle-stop
# SR ceiling for the current (never-stops) policy. Off by default: normal gate eval
# uses the learned stop primitive. closest_approach / oracle_hit@20/30/40 are logged
# either way, so a plain gate run already gets the closest-approach diagnostic.
ORACLE_STOP="${ORACLE_STOP:-0}"
# Pin the queue at process start. ``source $EVAL_ENV_FILE`` (e.g. env.sh) often
# exports EVAL_QUEUE_DIR back to the *default* shared queue; without this pin,
# mark_done/mark_failed look in the wrong tree and the job stays stuck in the
# dedicated queue's running/ forever (and the next claim may steal default-queue
# b0v2 jobs onto this worker).
QUEUE_DIR="$EVAL_QUEUE_DIR"
RUN_ONCE=0
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --once) RUN_ONCE=1 ;;
  -h|--help)
    echo "Usage: $0 [--dry-run|--once]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$EVAL_WORKER_LOCK")"
if ! mkdir "$EVAL_WORKER_LOCK" 2>/dev/null; then
  echo "Eval worker already running (lock: $EVAL_WORKER_LOCK)" >&2
  exit 1
fi
trap 'rmdir "$EVAL_WORKER_LOCK"' EXIT

if (( DRY_RUN )); then
  printf '%s\n' \
    "AIRSIM_HOST=10.229.20.125 AIRSIM_PORT=41451 AIRSIM_ALLOW_LOCAL_LAUNCH=0 python3 -m experiments.aerial.eval.run_closed_loop --bridge openfly --policy fastwam --openfly-root /path/to/openfly --ann /path/to/seen.json --checkpoint /path/to/checkpoint.pt --out /path/to/metrics.json --max-episodes 20 --max-steps 100 --seed 42 --task aerial_joint_b1_joint --dump-frames /path/to/frames"
  exit 0
fi

json_field() {
  "$PYTHON_BIN" -c \
    'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' \
    "$1" "$2"
}

mark_failed_job() {
  local job_id="$1"
  local error="$2"
  "$PYTHON_BIN" -m experiments.aerial.orchestration.eval_queue \
    --queue-dir "$QUEUE_DIR" --mark-failed "$job_id" --error "$error"
}

run_job() {
  local job_json="$1"
  local job_id checkpoint out_metrics task ann openfly_root seed max_steps max_episodes
  job_id="$(json_field "$job_json" id)"
  checkpoint="$(json_field "$job_json" checkpoint)"
  out_metrics="$(json_field "$job_json" out_metrics)"
  task="$(json_field "$job_json" task)"
  ann="$(json_field "$job_json" ann)"
  openfly_root="$(json_field "$job_json" openfly_root)"
  seed="$(json_field "$job_json" seed)"
  max_steps="$(json_field "$job_json" max_steps)"
  max_episodes="$(json_field "$job_json" max_episodes)"

  if [[ ! -f "$EVAL_ENV_FILE" ]]; then
    mark_failed_job "$job_id" "missing environment file: $EVAL_ENV_FILE"
    return 2
  fi
  # shellcheck disable=SC1090
  source "$EVAL_ENV_FILE"
  # env.sh may have overwritten EVAL_QUEUE_DIR — restore the pinned worker queue.
  export EVAL_QUEUE_DIR="$QUEUE_DIR"
  export AIRSIM_HOST=10.229.20.125
  export AIRSIM_PORT=41451
  export AIRSIM_ALLOW_LOCAL_LAUNCH=0

  # Persist AirSim RGB frames beside metrics so closed-loop episodes can be
  # audited offline (synced back to the operator Mac after the queue drains).
  local dump_frames
  dump_frames="$(dirname "$out_metrics")/frames"
  mkdir -p "$dump_frames"

  local command=(
    "$PYTHON_BIN" -m experiments.aerial.eval.run_closed_loop
    --bridge openfly
    --policy fastwam
    --openfly-root "$openfly_root"
    --ann "$ann"
    --checkpoint "$checkpoint"
    --out "$out_metrics"
    --max-episodes "$max_episodes"
    --max-steps "$max_steps"
    --seed "$seed"
    --task "$task"
    --dump-frames "$dump_frames"
  )
  if [[ "$DUMP_WM_FRAMES" == "1" ]]; then
    command+=(--dump-wm-frames --wm-dump-every "$WM_DUMP_EVERY")
  fi
  if [[ "$ORACLE_STOP" == "1" ]]; then
    command+=(--oracle-stop)
  fi

  local eval_rc
  if (
    cd "$REPO_ROOT"
    PYTHONPATH=. "${command[@]}"
  ); then
    if "$PYTHON_BIN" -m experiments.aerial.orchestration.eval_queue \
      --queue-dir "$QUEUE_DIR" --mark-done "$job_id"; then
      return 0
    fi
    mark_failed_job "$job_id" "evaluation produced invalid or missing metrics"
    return 1
  else
    eval_rc=$?
    mark_failed_job "$job_id" "run_closed_loop exited $eval_rc"
    return "$eval_rc"
  fi
}

while true; do
  job_json="$("$PYTHON_BIN" -m experiments.aerial.orchestration.eval_queue \
    --queue-dir "$QUEUE_DIR" --claim)"
  if [[ -z "$job_json" ]]; then
    if (( RUN_ONCE )); then
      exit 0
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  if run_job "$job_json"; then
    job_rc=0
  else
    job_rc=$?
  fi
  if (( RUN_ONCE )); then
    exit "$job_rc"
  fi
done
