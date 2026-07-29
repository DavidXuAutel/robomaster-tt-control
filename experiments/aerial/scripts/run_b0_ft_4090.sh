#!/usr/bin/env bash
set -euo pipefail

AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/tmp/aerial_ft_cache}"
REPO_DIR="$AERIAL_FT_CACHE/repo"
CHECKPOINT="$AERIAL_FT_CACHE/model/step_004000.pt"
ACCEL_CONFIG="$REPO_DIR/experiments/aerial/scripts/accelerate_zero2_opt_offload_2proc.yaml"
TASK="${TASK:-aerial_joint_b0_ft_dagger}"
RUN_DIR="${RUN_DIR:-$AERIAL_FT_CACHE/runs/b0-ft-1000}"
STATUS_FILE="$AERIAL_FT_CACHE/ft.status"
MEMORY_LIMIT_MIB=23552
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: [AERIAL_FT_CACHE=/tmp/aerial_ft_cache] $0 [--dry-run]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

verify_manifest() {
  python3 - "$AERIAL_FT_CACHE" <<'PY'
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

[[ -d "$REPO_DIR" ]] || { echo "Missing synced repository: $REPO_DIR" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ "$(basename "$CHECKPOINT")" == "step_004000.pt" ]] || {
  echo "Refusing checkpoint other than step_004000.pt" >&2
  exit 2
}
[[ -f "$ACCEL_CONFIG" ]] || { echo "Missing Accelerate config: $ACCEL_CONFIG" >&2; exit 2; }
[[ -f "$AERIAL_FT_CACHE/smoke.status" ]] || { echo "Smoke has not completed" >&2; exit 2; }
[[ "$(tr -d '[:space:]' < "$AERIAL_FT_CACHE/smoke.status")" == "COMPLETED" ]] || {
  echo "Smoke status is not COMPLETED" >&2
  exit 2
}
verify_manifest

RUNTIME_DIR="$AERIAL_FT_CACHE/runtime"
LOG_DIR="$AERIAL_FT_CACHE/logs/ft"
mkdir -p "$RUNTIME_DIR" "$LOG_DIR"
RETRY_DS_CONFIG="$RUNTIME_DIR/deepspeed_zero2_small_buckets.json"
RETRY_ACCEL_CONFIG="$RUNTIME_DIR/accelerate_zero2_small_buckets.yaml"

cat > "$RETRY_DS_CONFIG" <<'JSON'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "none"},
    "overlap_comm": false,
    "contiguous_gradients": false,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000
  }
}
JSON
cat > "$RETRY_ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: $RETRY_DS_CONFIG
  zero3_init_flag: false
machine_rank: 0
main_training_function: main
mixed_precision: null
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
use_cpu: false
EOF

echo "OOM retries allowed: 1"
echo "OOM retry buckets: reduce_bucket_size=50000000 allgather_bucket_size=50000000"
echo "peak memory gate: <${MEMORY_LIMIT_MIB} MiB/GPU"

if (( DRY_RUN )); then
  echo "DRY RUN: accelerate launch --config_file $ACCEL_CONFIG --num_processes 2 scripts/train.py task=$TASK resume=$CHECKPOINT max_steps=1000 save_every=250"
  for step in 250 500 1000; do
    printf 'Expected: %s/checkpoints/weights/step_%06d.pt\n' "$RUN_DIR" "$step"
  done
  exit 0
fi

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
  python3 - "$1" <<'PY'
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

run_attempt() {
  local config="$1"
  local output_dir="$2"
  local suffix="$3"
  local log="$LOG_DIR/1000-step-${suffix}.log"
  local memory="$LOG_DIR/1000-step-${suffix}.memory_mib"
  local stop="$RUNTIME_DIR/memory-ft-${suffix}.stop"
  rm -f "$stop"
  monitor_memory "$memory" "$stop" &
  local monitor_pid=$!
  set +e
  (
    cd "$REPO_DIR"
    accelerate launch \
      --config_file "$config" \
      --num_processes 2 \
      scripts/train.py \
      "output_dir=$output_dir" \
      "wandb.enabled=false" \
      "task=$TASK" \
      "mixed_precision=bf16" \
      "resume=$CHECKPOINT" \
      "max_steps=1000" \
      "save_every=250" \
      "eval_every=0"
  ) 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  touch "$stop"
  wait "$monitor_pid" || true
  if (( rc != 0 )); then
    if grep -Eiq 'out of memory|CUDA.*OOM|CUDA error: out of memory' "$log"; then
      return 42
    fi
    return "$rc"
  fi
  validate_losses "$log"
  local peak
  peak="$(awk 'BEGIN {m=0} /^[[:space:]]*[0-9]+[[:space:]]*$/ {if ($1>m) m=$1} END {print m}' "$memory")"
  [[ "$peak" =~ ^[0-9]+$ ]] || { echo "Could not measure GPU memory" >&2; return 1; }
  echo "Peak GPU memory: ${peak} MiB"
  if (( peak >= MEMORY_LIMIT_MIB )); then
    echo "Peak memory gate failed: ${peak} MiB is not < ${MEMORY_LIMIT_MIB} MiB" >&2
    return 1
  fi
  for step in 250 500 1000; do
    local expected
    expected="$(printf '%s/checkpoints/weights/step_%06d.pt' "$output_dir" "$step")"
    [[ -f "$expected" ]] || { echo "Missing required checkpoint: $expected" >&2; return 1; }
  done
}

set +e
run_attempt "$ACCEL_CONFIG" "$RUN_DIR" primary
rc=$?
set -e
FINAL_RUN_DIR="$RUN_DIR"
if (( rc == 42 )); then
  echo "OOM detected; using the single smaller-bucket retry from base weights" >&2
  FINAL_RUN_DIR="${RUN_DIR}-retry"
  set +e
  run_attempt "$RETRY_ACCEL_CONFIG" "$FINAL_RUN_DIR" retry
  rc=$?
  set -e
  if (( rc == 42 )); then
    echo "OOM retry failed: STOP. Escalate to ZeRO-3 review." >&2
    exit 1
  fi
fi
(( rc == 0 )) || exit "$rc"

printf 'COMPLETED\n' > "$STATUS_FILE"
printf '%s\n' "$FINAL_RUN_DIR" > "$AERIAL_FT_CACHE/ft.run_dir"
trap - EXIT INT TERM
echo "Fine-tune completed: $FINAL_RUN_DIR"
