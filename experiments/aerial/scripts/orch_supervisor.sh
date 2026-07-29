#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
STATUS_PATH="${STATUS_PATH:-/home/a25689/aerial_cache/orchestration/status.json}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/home/a25689/aerial_cache_shared/orchestration/eval_queue}"
ORCH_ROOT="${ORCH_ROOT:-/home/a25689/aerial_cache/orchestration}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/a25689/aerial_cache_shared/orchestration/results}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_joint_b0_to_joint_video/m1b-${STAMP}/checkpoints/weights}"
CANDIDATE_STEPS="${CANDIDATE_STEPS:-1000,2000,3000,4000,5000}"
B1_TRAIN_PID_FILE="${B1_TRAIN_PID_FILE:-$ORCH_ROOT/b1_train.pid}"
B1_WATCH_PID_FILE="${B1_WATCH_PID_FILE:-$ORCH_ROOT/b1_watch.pid}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLL_S="${POLL_S:-30}"
DRY_RUN=0
ONCE=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --once) ONCE=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run|--once]"
    echo "Full B0→B1 supervisor: wait/enqueue → lock → gates → train/watch → S1 → DONE."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

supervisor() {
  PYTHONPATH=. "$PYTHON_BIN" -m experiments.aerial.orchestration.supervisor "$@"
}

read_phase() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase",""))' "$STATUS_PATH"
}

pid_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_b1_train_and_watch() {
  if ! pid_alive "$B1_TRAIN_PID_FILE"; then
    nohup "$SCRIPT_DIR/orch_b1_train.sh" >"$ORCH_ROOT/b1_train.log" 2>&1 &
    echo $! >"$B1_TRAIN_PID_FILE"
  fi
  if ! pid_alive "$B1_WATCH_PID_FILE"; then
    nohup "$SCRIPT_DIR/orch_ckpt_watch_enqueue.sh" >"$ORCH_ROOT/b1_watch.log" 2>&1 &
    echo $! >"$B1_WATCH_PID_FILE"
  fi
  supervisor --status "$STATUS_PATH" --stamp "$STAMP" --mark-b1-train-started
}

run_once() {
  local phase
  if [[ ! -f "$STATUS_PATH" ]]; then
    supervisor --status "$STATUS_PATH" --stamp "$STAMP" --init
  fi
  phase="$(read_phase)"

  if [[ "$phase" == "WAIT_B0_COMPLETE" ]]; then
    "$SCRIPT_DIR/orch_b0_wait_and_enqueue.sh"
    supervisor --status "$STATUS_PATH" --stamp "$STAMP" --set-phase EVAL_B0_CHECKPOINTS
    phase="EVAL_B0_CHECKPOINTS"
  fi

  if [[ "$phase" == "EVAL_B0_CHECKPOINTS" ]]; then
    supervisor \
      --status "$STATUS_PATH" \
      --stamp "$STAMP" \
      --advance-from-eval-queue \
      --queue-dir "$EVAL_QUEUE_DIR"
    phase="$(read_phase)"
  fi

  if [[ "$phase" == "LOCK_BASELINE" ]]; then
    supervisor \
      --status "$STATUS_PATH" \
      --stamp "$STAMP" \
      --lock-baseline \
      --weights-dir "$WEIGHTS_DIR" \
      --results-root "$RESULTS_ROOT" \
      --steps "$CANDIDATE_STEPS" \
      --out "$ORCH_ROOT/baseline_lock.manifest.json" || true
    phase="$(read_phase)"
  fi

  if [[ "$phase" == "B1_GATES" ]]; then
    "$SCRIPT_DIR/orch_b1_gates.sh" || true
    phase="$(read_phase)"
  fi

  if [[ "$phase" == "RUN_B1_TRAIN" ]]; then
    start_b1_train_and_watch || true
    phase="$(read_phase)"
  fi

  if [[ "$phase" == "EVAL_B1_CHECKPOINTS" ]]; then
    supervisor \
      --status "$STATUS_PATH" \
      --stamp "$STAMP" \
      --advance-b1-eval \
      --results-root "$RESULTS_ROOT" || true
    phase="$(read_phase)"
  fi

  if [[ "$phase" == "S1_REPORT" ]]; then
    "$SCRIPT_DIR/orch_s1_report.sh" || true
    phase="$(read_phase)"
  fi

  echo "$phase"
}

if (( DRY_RUN )); then
  cat <<EOF
WAIT_B0 → EVAL_B0 → LOCK → B1_GATES → RUN_B1_TRAIN(+watch) → EVAL_B1 → S1_REPORT → DONE
STATUS_PATH=$STATUS_PATH
EVAL_QUEUE_DIR=$EVAL_QUEUE_DIR
WEIGHTS_DIR=$WEIGHTS_DIR
RESULTS_ROOT=$RESULTS_ROOT
EOF
  exit 0
fi

cd "$REPO_ROOT"
mkdir -p "$ORCH_ROOT"

if (( ONCE )); then
  run_once
  exit 0
fi

while true; do
  phase="$(run_once)"
  case "$phase" in
    BLOCKED|FAILED|DONE)
      echo "supervisor stopped at phase=$phase"
      exit 0
      ;;
  esac
  sleep "$POLL_S"
done
