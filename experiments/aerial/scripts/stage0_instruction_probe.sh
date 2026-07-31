#!/usr/bin/env bash
# Stage 0: wm_instruction_probe + verdict for collapse-fix (run on :30905).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

CKPT_ROOT="${CKPT_ROOT:-/home/a25689/aerial_cache_shared/runs/aerial_b0_v2/m1b-20260729-b0v2-10k-2gpu/checkpoints/weights}"
OUT_ROOT="${OUT_ROOT:-/home/a25689/aerial_cache_shared/orchestration/results/b0_v2_stage0_probe}"
INSTR="${INSTR:-$REPO_ROOT/experiments/aerial/collapse_fix/probe_instructions.json}"
TASK="${TASK:-aerial_joint_1cam_1e-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-001500,003500}"
SEED="${SEED:-42}"
# Fixed RGB: prefer a frame from prior eval dumps if present.
OBS="${OBS:-}"
if [[ -z "$OBS" ]]; then
  CAND=(
    /home/a25689/aerial_cache_shared/orchestration/results/b0_v2_20260729-b0v2-10k-2gpu/b1_20260729-b0v2-10k-2gpu/step_001500_seen20/frames/*_step0000.png
  )
  for c in "${CAND[@]}"; do
    if compgen -G "$c" > /dev/null; then
      OBS="$(ls $c | head -1)"
      break
    fi
  done
fi
if [[ -z "${OBS:-}" || ! -f "$OBS" ]]; then
  echo "Set OBS=/path/to/fixed.png (RGB from any eval frame)." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
IFS=',' read -r -a STEP_ARR <<< "$STEPS"

for step in "${STEP_ARR[@]}"; do
  ckpt="$CKPT_ROOT/step_${step}.pt"
  out_dir="$OUT_ROOT/step_${step}"
  mkdir -p "$out_dir"
  echo "[stage0] probe step_${step} obs=$OBS"
  "$PYTHON_BIN" -m experiments.aerial.eval.wm_instruction_probe \
    --checkpoint "$ckpt" \
    --obs "$OBS" \
    --instructions "$INSTR" \
    --out "$out_dir" \
    --task "$TASK" \
    --seed "$SEED"
  "$PYTHON_BIN" -m experiments.aerial.collapse_fix.probe_verdict \
    "$out_dir/summary.json" \
    --out "$out_dir/verdict.json"
done

echo "[stage0] DONE probe → $OUT_ROOT"
