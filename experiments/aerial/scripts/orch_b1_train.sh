#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/home/a25689/aerial_ft_cache}"
STATUS_PATH="${STATUS_PATH:-/home/a25689/aerial_cache/orchestration/status.json}"
LOCK_PATH="${LOCK_PATH:-/home/a25689/aerial_cache/orchestration/baseline_lock.manifest.json}"
REPO_DIR="${REPO_DIR:-$AERIAL_FT_CACHE/repo}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$REPO_DIR/experiments/aerial/scripts/accelerate_zero2_no_offload_2proc.yaml}"
TASK="${TASK:-aerial_joint_b0_ft_dagger}"
RUN_DIR="${RUN_DIR:-$AERIAL_FT_CACHE/runs/b1-${STAMP}}"
STATUS_FILE="${STATUS_FILE:-$AERIAL_FT_CACHE/ft.status}"
MEMORY_LIMIT_MIB="${MEMORY_LIMIT_MIB:-72000}"
SKIP_MANIFEST_VERIFY="${SKIP_MANIFEST_VERIFY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

require_gates_passed() {
  "$PYTHON_BIN" - "$STATUS_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing orchestration status (gates): {path}")
data = json.loads(path.read_text(encoding="utf-8"))
phase = data.get("phase")
passed = bool(data.get("gates_passed"))
allowed = {"RUN_B1_TRAIN", "EVAL_B1_CHECKPOINTS"}
if phase == "BLOCKED" or not passed or phase not in allowed:
    raise SystemExit(
        f"refusing B1 train: gates not passed "
        f"(phase={phase!r}, gates_passed={passed}; need RUN_B1_TRAIN)"
    )
print("gates_ok")
PY
}

resolve_checkpoint() {
  "$PYTHON_BIN" - "$LOCK_PATH" "$AERIAL_FT_CACHE" <<'PY'
import json
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
cache = Path(sys.argv[2])
candidates = []
if lock_path.is_file():
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    ckpt = data.get("checkpoint")
    if ckpt:
        candidates.append(Path(str(ckpt)))
candidates.append(cache / "model" / "baseline.pt")
for path in candidates:
    if path.is_file():
        print(path)
        raise SystemExit(0)
raise SystemExit(f"missing baseline checkpoint from lock/cache: {candidates}")
PY
}

verify_manifest() {
  if [[ "$SKIP_MANIFEST_VERIFY" == "1" ]]; then
    echo "SKIP_MANIFEST_VERIFY=1"
    return 0
  fi
  "$PYTHON_BIN" - "$AERIAL_FT_CACHE" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = root / "SHA256SUMS"
if not manifest.is_file():
    raise SystemExit(f"missing SHA256 manifest: {manifest}")
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, relative = line.split(maxsplit=1)
    path = root / relative.strip()
    if not path.is_file():
        raise SystemExit(f"manifest file missing: {relative}")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != digest:
        raise SystemExit(f"SHA256 mismatch: {relative}")
print(f"SHA256 verified: {manifest}")
PY
}

resolve_resume() {
  # ZeRO-2 accelerate.save_state resume often fails with optimizer param-group
  # mismatch; continue from latest weights and advance the step counter from the
  # filename instead.
  local weights_dir="$RUN_DIR/checkpoints/weights"
  local latest=""
  if [[ -d "$weights_dir" ]]; then
    latest="$(find "$weights_dir" -mindepth 1 -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -1 || true)"
  fi
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi
  resolve_checkpoint
}

require_gates_passed
CHECKPOINT="$(resolve_checkpoint)"
RESUME="$(resolve_resume)"
[[ -d "$REPO_DIR" ]] || { echo "Missing synced repository: $REPO_DIR" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -e "$RESUME" ]] || { echo "Missing resume path: $RESUME" >&2; exit 2; }
[[ -f "$ACCEL_CONFIG" ]] || { echo "Missing Accelerate config: $ACCEL_CONFIG" >&2; exit 2; }
verify_manifest

TRAIN_OVERRIDES=(
  "task=$TASK"
  "mixed_precision=bf16"
  "resume=$RESUME"
  "max_steps=5000"
  "save_every=250"
  "learning_rate=1e-5"
  "model.loss.lambda_video=0.0"
  "model.loss.lambda_action=1.0"
  "eval_every=0"
)

if (( DRY_RUN )); then
  echo "DRY RUN: accelerate launch --config_file $ACCEL_CONFIG --num_processes 2 scripts/train.py output_dir=$RUN_DIR ${TRAIN_OVERRIDES[*]}"
  for step in 1000 2000 3000 4000 5000; do
    printf 'Expected: %s/checkpoints/weights/step_%06d.pt\n' "$RUN_DIR" "$step"
  done
  exit 0
fi

RUNTIME_DIR="$AERIAL_FT_CACHE/runtime"
LOG_DIR="$AERIAL_FT_CACHE/logs/ft"
mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$RUN_DIR"

printf 'RUNNING\n' > "$STATUS_FILE"
ft_finalize() {
  local rc=$?
  if [[ "$(tr -d '[:space:]' < "$STATUS_FILE")" == "RUNNING" ]]; then
    printf 'FAILED\n' > "$STATUS_FILE"
  fi
  return "$rc"
}
trap ft_finalize EXIT
trap 'exit 130' INT TERM

monitor_memory() {
  local output="$1"
  local stop="$2"
  : > "$output"
  while [[ ! -e "$stop" ]]; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$output" || true
    sleep 0.5
  done
}

validate_losses() {
  "$PYTHON_BIN" - "$1" <<'PY'
import math
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
values = [float(value) for value in re.findall(r"\bloss(?:_[A-Za-z0-9]+)?=([^\s,]+)", text)]
if not values:
    raise SystemExit("no training losses found in fine-tune log")
if not all(math.isfinite(value) for value in values):
    raise SystemExit("non-finite loss found in fine-tune log")
PY
}

LOG="$LOG_DIR/b1-5000-step.log"
MEMORY="$LOG_DIR/b1-5000-step.memory_mib"
STOP="$RUNTIME_DIR/memory-b1.stop"
rm -f "$STOP"
echo "B1 resume=$RESUME max_steps=5000 save_every=250"
monitor_memory "$MEMORY" "$STOP" &
MONITOR_PID=$!
set +e
(
  cd "$REPO_DIR"
  accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes 2 \
    scripts/train.py \
    "output_dir=$RUN_DIR" \
    "wandb.enabled=false" \
    "${TRAIN_OVERRIDES[@]}"
) 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
touch "$STOP"
wait "$MONITOR_PID" || true
(( RC == 0 )) || exit "$RC"
validate_losses "$LOG"

peak="$(awk 'BEGIN {m=0} /^[[:space:]]*[0-9]+[[:space:]]*$/ {if ($1>m) m=$1} END {print m}' "$MEMORY")"
[[ "$peak" =~ ^[0-9]+$ ]] || { echo "Could not measure GPU memory" >&2; exit 1; }
echo "Peak GPU memory: ${peak} MiB"
if (( peak >= MEMORY_LIMIT_MIB )); then
  echo "Peak memory gate failed: ${peak} MiB is not < ${MEMORY_LIMIT_MIB} MiB" >&2
  exit 1
fi
for step in $(seq 250 250 5000); do
  expected="$(printf '%s/checkpoints/weights/step_%06d.pt' "$RUN_DIR" "$step")"
  [[ -f "$expected" ]] || { echo "Missing required checkpoint: $expected" >&2; exit 1; }
done

printf 'COMPLETED\n' > "$STATUS_FILE"
printf '%s\n' "$RUN_DIR" > "$AERIAL_FT_CACHE/ft.run_dir"
trap - EXIT INT TERM
echo "B1 fine-tune completed: $RUN_DIR"
