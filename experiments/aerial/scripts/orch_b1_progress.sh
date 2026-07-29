#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/home/a25689/aerial_ft_cache}"
ORCH_ROOT="${ORCH_ROOT:-/home/a25689/aerial_cache/orchestration}"
STATUS_PATH="${STATUS_PATH:-$ORCH_ROOT/status.json}"
RUN_DIR="${RUN_DIR:-$AERIAL_FT_CACHE/runs/b1-${STAMP}}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$RUN_DIR/checkpoints/weights}"
SHARED_WEIGHTS_DIR="${SHARED_WEIGHTS_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_b1_ft/m1b-${STAMP}/checkpoints/weights}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/home/a25689/aerial_cache_shared/orchestration/eval_queue}"
TRAIN_LOG="${TRAIN_LOG:-}"
if [[ -z "$TRAIN_LOG" ]]; then
  if [[ -f "$AERIAL_FT_CACHE/logs/ft/b1-5000-step.log" ]]; then
    TRAIN_LOG="$AERIAL_FT_CACHE/logs/ft/b1-5000-step.log"
  else
    TRAIN_LOG="$AERIAL_FT_CACHE/logs/ft/b1-1000-step.log"
  fi
fi
WATCH_LOG="${WATCH_LOG:-$ORCH_ROOT/b1_watch.log}"
PROGRESS_STEPS="${PROGRESS_STEPS:-1000,2000,3000,4000,5000}"
SKIP_PROCESS_CHECK="${SKIP_PROCESS_CHECK:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "${1:-}" in
  -h|--help)
    echo "Usage: STAMP=... $0"
    echo "Summarize B1 train / checkpoint / eval-queue progress."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

read_trim() {
  local path="$1"
  if [[ -f "$path" ]]; then
    tr -d '[:space:]' <"$path"
  else
    echo "missing"
  fi
}

count_queue() {
  local name="$1"
  local dir="$EVAL_QUEUE_DIR/$name"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

ckpt_flag() {
  local dir="$1"
  local step="$2"
  local path
  path="$(printf '%s/step_%06d.pt' "$dir" "$step")"
  if [[ -f "$path" ]]; then
    echo yes
  else
    echo no
  fi
}

alive_flag() {
  local pattern="$1"
  if pgrep -af "$pattern" >/dev/null 2>&1; then
    echo yes
  else
    echo no
  fi
}

echo "stamp=$STAMP"
echo "run_dir=$RUN_DIR"

if [[ -f "$STATUS_PATH" ]]; then
  "$PYTHON_BIN" - "$STATUS_PATH" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"phase={data.get('phase', 'missing')}")
print(f"gates_passed={str(bool(data.get('gates_passed'))).lower()}")
blocked = data.get("blocked_reason")
if blocked:
    print(f"blocked_reason={blocked}")
checked = data.get("checked_at")
if checked:
    print(f"checked_at={checked}")
PY
else
  echo "phase=missing"
  echo "gates_passed=missing"
fi

echo "ft.status=$(read_trim "$AERIAL_FT_CACHE/ft.status")"
echo "smoke.status=$(read_trim "$AERIAL_FT_CACHE/smoke.status")"
echo "ft_smoke.status=$(read_trim "$ORCH_ROOT/ft_smoke.status")"

if [[ -f "$TRAIN_LOG" ]]; then
  latest="$("$PYTHON_BIN" - "$TRAIN_LOG" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = list(re.finditer(r"step=(\d+/\d+)\s+loss=([^\s]+)", text))
if not matches:
    print("train_step=none")
else:
    m = matches[-1]
    print(f"train_step=step={m.group(1)}")
    print(f"train_loss=loss={m.group(2)}")
PY
)"
  printf '%s\n' "$latest"
else
  echo "train_step=none"
  echo "train_loss=none"
fi

IFS=',' read -r -a _progress_steps <<< "$PROGRESS_STEPS"
for step in "${_progress_steps[@]}"; do
  printf 'local_ckpt step_%06d=%s\n' "$step" "$(ckpt_flag "$WEIGHTS_DIR" "$step")"
done
for step in "${_progress_steps[@]}"; do
  printf 'shared_ckpt step_%06d=%s\n' "$step" "$(ckpt_flag "$SHARED_WEIGHTS_DIR" "$step")"
done
latest_local="$("$PYTHON_BIN" - "$WEIGHTS_DIR" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
steps = []
for path in root.glob("step_*.pt"):
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    if match:
        steps.append(int(match.group(1)))
print(max(steps) if steps else 0)
PY
)"
echo "latest_local_ckpt_step=$latest_local"

echo "queue pending=$(count_queue pending) running=$(count_queue running) done=$(count_queue done) failed=$(count_queue failed)"

if [[ "$SKIP_PROCESS_CHECK" != "1" ]]; then
  echo "proc_train=$(alive_flag "aerial_joint_b0_ft_dagger")"
  echo "proc_watch=$(alive_flag "experiments.aerial.orchestration.b1_discover")"
  # eval worker normally lives on the AirSim/eval host (:30682), not the train host.
  echo "proc_eval_worker_local=$(alive_flag "orch_eval_worker.sh")"
fi

if [[ "$SKIP_GPU_CHECK" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F',' '{gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); printf "gpu%s mem=%s util=%s\n", $1, $2, $3}'
fi

if [[ -f "$WATCH_LOG" ]]; then
  echo "watch_log_tail:"
  tail -n 5 "$WATCH_LOG" | sed 's/^/  /'
fi
