#!/usr/bin/env bash
# Stage 0: oracle-stop closed-loop eval for collapse-fix ckpts (run on :30905).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

CKPT_ROOT="${CKPT_ROOT:-/home/a25689/aerial_cache_shared/runs/aerial_b0_v2/m1b-20260729-b0v2-10k-2gpu/checkpoints/weights}"
ANN="${ANN:-/home/a25689/aerial_cache_shared/orchestration/heldout/seen_airsim16_m1a20.json}"
OPENFLY_ROOT="${OPENFLY_ROOT:-/home/a25689/aerial_eval_cache/OpenFly-Platform}"
OUT_ROOT="${OUT_ROOT:-/home/a25689/aerial_cache_shared/orchestration/results/b0_v2_stage0_oracle}"
TASK="${TASK:-aerial_joint_1cam_1e-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-000500,001500,003500}"
MAX_EPISODES="${MAX_EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-100}"
SEED="${SEED:-42}"

mkdir -p "$OUT_ROOT"
IFS=',' read -r -a STEP_ARR <<< "$STEPS"

for step in "${STEP_ARR[@]}"; do
  ckpt="$CKPT_ROOT/step_${step}.pt"
  out_dir="$OUT_ROOT/step_${step}_seen20"
  mkdir -p "$out_dir/frames"
  echo "[stage0] oracle-stop eval step_${step}"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m experiments.aerial.eval.run_closed_loop \
    --bridge openfly \
    --policy fastwam \
    --openfly-root "$OPENFLY_ROOT" \
    --ann "$ANN" \
    --checkpoint "$ckpt" \
    --out "$out_dir/metrics.json" \
    --max-episodes "$MAX_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --task "$TASK" \
    --oracle-stop \
    --dump-frames "$out_dir/frames"
  echo "[stage0] wrote $out_dir/metrics.json"
done

echo "[stage0] DONE oracle-stop → $OUT_ROOT"
