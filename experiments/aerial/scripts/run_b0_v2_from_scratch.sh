#!/usr/bin/env bash
# Aerial WAM v2 — B0 from-scratch training launcher (1 host × 2×H100).
#
# Implements docs/design/2026-07-29-aerial-nav-wam-redesign.md §8.2-8.4:
#   preflight -> smoke (estimate budget) -> full from-scratch B0 train.
#
# From scratch means: Wan2.2 backbone + pretrained Action DiT, NO aerial
# checkpoint. `resume` is forced empty and the trainer guard rejects any
# pre-v2 aerial checkpoint. Run ON the training host (:31126), not from the
# Claude Code sandbox (which cannot reach the lab network).
#
# Usage:
#   run_b0_v2_from_scratch.sh preflight   # checks only
#   run_b0_v2_from_scratch.sh smoke       # 10-step dry run to size memory/throughput
#   run_b0_v2_from_scratch.sh train       # full from-scratch B0 run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ---- config (override via env) ----
ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/accelerate_zero2_no_offload_2proc.yaml}"
TASK="${TASK:-aerial_joint_1cam_1e-4}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAX_STEPS="${MAX_STEPS:-5000}"          # set from smoke throughput before the real run
SAVE_EVERY="${SAVE_EVERY:-500}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
TRAIN_SUBSET="${TRAIN_SUBSET:-./data/openfly_lerobot/train_subset}"
TEXT_EMBEDS="${TEXT_EMBEDS:-./data/text_embeds_cache/openfly}"
ACTION_DIT="${ACTION_DIT:-checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
LOG_DIR="${LOG_DIR:-logs/ft/b0_v2}"
mkdir -p "$LOG_DIR"

log() { echo "[b0-v2] $*"; }
die() { echo "[b0-v2][ERROR] $*" >&2; exit 1; }

preflight() {
  log "repo=$REPO_ROOT task=$TASK procs=$NUM_PROCESSES"
  [[ -f "$ACCEL_CONFIG" ]] || die "missing accelerate config: $ACCEL_CONFIG"
  [[ -f "scripts/train.py" ]] || die "missing scripts/train.py (wrong repo root?)"
  [[ -d "$TRAIN_SUBSET" ]] || die "missing train subset: $TRAIN_SUBSET"
  [[ -d "$TEXT_EMBEDS" ]] || die "missing text-embed cache: $TEXT_EMBEDS"
  [[ -f "$ACTION_DIT" ]] || die "missing pretrained Action DiT: $ACTION_DIT"

  # Expect exactly 2 visible H100s.
  local ngpu
  ngpu="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  [[ "$ngpu" == "$NUM_PROCESSES" ]] || die "expected $NUM_PROCESSES GPUs, torch sees $ngpu"

  # From-scratch invariant: no pre-v2 aerial checkpoint may leak in as resume.
  [[ -z "${RESUME:-}" ]] || die "RESUME must be empty for from-scratch B0 (got '$RESUME')"
  if [[ -n "${AERIAL_ALLOW_LEGACY_RESUME:-}" ]]; then
    die "AERIAL_ALLOW_LEGACY_RESUME is set; refuse — B0 v2 is from scratch"
  fi
  log "preflight OK (gpus=$ngpu, resume=null, guard armed)"
}

launch() {  # $1=max_steps $2=save_every $3=logfile
  local steps="$1" save="$2" logfile="$3"
  log "launch: max_steps=$steps save_every=$save -> $logfile"
  accelerate launch --config_file "$ACCEL_CONFIG" --num_processes "$NUM_PROCESSES" \
    scripts/train.py \
      task="$TASK" \
      resume=null \
      max_steps="$steps" \
      save_every="$save" \
      model.loss.lambda_video=1.0 \
      model.loss.lambda_action=1.0 \
    2>&1 | tee "$logfile"
}

cmd="${1:-preflight}"
case "$cmd" in
  preflight) preflight ;;
  smoke)
    preflight
    launch "$SMOKE_STEPS" "$SMOKE_STEPS" "$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
    log "smoke done — verify finite loss_action/loss_video, no NaN, peak mem <90%,"
    log "then set MAX_STEPS from observed throughput and run: $0 train"
    ;;
  train)
    preflight
    log "starting FULL from-scratch B0: max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
    launch "$MAX_STEPS" "$SAVE_EVERY" "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
    log "train finished — checkpoints under checkpoints/weights/step_*.pt"
    log "next: eval each on :30682 (design §8.5); lock baseline only if SR>0"
    ;;
  *) die "unknown command '$cmd' (use: preflight | smoke | train)" ;;
esac
