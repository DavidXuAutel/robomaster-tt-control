#!/usr/bin/env bash
# Aerial WAM v2 — collapse-fix retrain launcher (1 host × 2×H100).
#
# Implements docs/superpowers/plans/2026-07-30-aerial-collapse-fix-retrain-runbook.md.
# Scheme B (v3.2): keep ActionDiT flow-matching head + denoise loop, ADD a
# timestep-aware head_cls (10-class CE); closed-loop executes argmax(cls).
#
# THE THREE INVARIANTS (see runbook §Preconditions):
#   1. Opt-in / default no-op. None of this touches the base b0_v2 recipe.
#      `aerial_openfly.yaml`, `aerial_joint_1cam_1e-4.yaml`, and the raw
#      `train_subset` are untouched. This launcher only fires when you run it.
#   2. Training distribution/window changes come from TWO independent switches:
#        (a) re-converting data with --stop-relabel-radius (stop labels), and
#        (b) task=aerial_joint_collapse_fix (num_frames 9 / ratio 2 /
#            skip_padding + λ_ce=1.0 / λ_fm=0.1).
#      Flipping only (b) changes the window but leaves stop labels at the
#      ~2/2709 floor → policy still never terminates. You need BOTH.
#   3. Retrain prerequisite: `reconvert` MUST run before `train`, or the CE
#      head learns from ~0.07% stop labels and reproduces the b0_v2
#      never-terminate failure.
#
# NORMALIZER NOTE: re-converting recomputes the FastWAMProcessor min/max stats
# over the new (zero-heavy) action distribution. The relabeled subset's
# normalizer is therefore NOT interchangeable with old b0_v2 checkpoints — this
# run trains from scratch so that's fine, but never point an old ckpt at the
# relabeled data for closed-loop eval.
#
# Run ON the training host (:31126). The Claude Code sandbox cannot reach the
# lab network, re-convert the parquet data, or see the GPUs.
#
# Usage:
#   run_collapse_fix_retrain.sh reconvert   # regen train_subset_stop20 with stop labels
#   run_collapse_fix_retrain.sh preflight   # checks only (relabeled data, GPUs, cls wiring)
#   run_collapse_fix_retrain.sh smoke       # short cls-enabled run; verify finite loss_ce
#   run_collapse_fix_retrain.sh train       # full collapse-fix retrain
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

# ---- config (override via env) ----
ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/accelerate_zero2_no_offload_2proc.yaml}"
TASK="${TASK:-aerial_joint_collapse_fix}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAX_STEPS="${MAX_STEPS:-5000}"          # set from smoke throughput before the real run
SAVE_EVERY="${SAVE_EVERY:-500}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"

# Data: relabeled subset is written to a NEW dir so the raw train_subset stays
# intact (invariant #1). Launch overrides data.train.dataset_dirs to point here.
RAW_SUBSET="${RAW_SUBSET:-./data/openfly_lerobot/train_subset}"
RELABELED_SUBSET="${RELABELED_SUBSET:-./data/openfly_lerobot/train_subset_stop20}"
STOP_RADIUS="${STOP_RADIUS:-20.0}"      # = OPENFLY_SUCCESS_DIST_M (eval/metrics.py)
OPENFLY_ANN="${OPENFLY_ANN:-}"          # required for `reconvert`
OPENFLY_IMAGE_ROOT="${OPENFLY_IMAGE_ROOT:-}"  # required for `reconvert`

TEXT_EMBEDS="${TEXT_EMBEDS:-./data/text_embeds_cache/openfly}"
ACTION_DIT="${ACTION_DIT:-checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

# Loss recipe (also set in the task config; passed explicitly to be self-documenting).
LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-0.1}"   # λ_fm kept small; CE is the primary action objective
LAMBDA_CE="${LAMBDA_CE:-1.0}"

LOG_DIR="${LOG_DIR:-logs/ft/collapse_fix}"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Optional fixed output_dir so :30905 watcher can poll a known shared path
# (default Hydra stamp under ./runs/train/ is local-only on :31126).
OUTPUT_DIR="${OUTPUT_DIR:-}"

log() { echo "[collapse-fix] $*"; }
die() { echo "[collapse-fix][ERROR] $*" >&2; exit 1; }

reconvert() {
  [[ -n "$OPENFLY_ANN" ]] || die "OPENFLY_ANN must point to the OpenFly annotation json"
  [[ -n "$OPENFLY_IMAGE_ROOT" ]] || die "OPENFLY_IMAGE_ROOT must point to the OpenFly image root"
  [[ -f "$OPENFLY_ANN" ]] || die "annotation not found: $OPENFLY_ANN"
  [[ -d "$OPENFLY_IMAGE_ROOT" ]] || die "image root not found: $OPENFLY_IMAGE_ROOT"

  if [[ -e "$RELABELED_SUBSET" ]]; then
    [[ -n "${FORCE:-}" ]] || die "relabeled subset exists: $RELABELED_SUBSET (set FORCE=1 to overwrite)"
    log "FORCE=1 — removing existing $RELABELED_SUBSET"
    rm -rf "$RELABELED_SUBSET"
  fi

  log "re-converting OpenFly → $RELABELED_SUBSET with --stop-relabel-radius $STOP_RADIUS"
  "$PYTHON_BIN" experiments/aerial/convert_openfly_to_lerobot.py \
    --ann "$OPENFLY_ANN" \
    --image-root "$OPENFLY_IMAGE_ROOT" \
    --out "$RELABELED_SUBSET" \
    --stop-relabel-radius "$STOP_RADIUS"
  log "reconvert done. Stop supervision injected as zero-delta (primitive 0)."
  log "next: $0 preflight"
}

# Informational only: d_max forward-filter threshold. NOTE: model.action_cls_d_max
# is NOT yet plumbed through create_fastwam_joint (fastwam.py reads it via
# getattr(self,'action_cls_d_max',1e9)), so for retrain-1 forward filtering is
# OFF and minority classes are exempt regardless. Wire it only if p90 filtering
# is desired (see runbook §d_max).
show_dmax() {
  [[ -d "$RELABELED_SUBSET" ]] || { log "skip d_max (no relabeled subset yet)"; return 0; }
  "$PYTHON_BIN" experiments/aerial/collapse_fix/compute_dmax.py \
    --dataset "$RELABELED_SUBSET" --out artifacts/collapse_fix_dmax.json || \
    log "compute_dmax failed (non-fatal; d_max stays 1e9)"
}

preflight() {
  log "repo=$REPO_ROOT task=$TASK procs=$NUM_PROCESSES data=$RELABELED_SUBSET"
  [[ -f "$ACCEL_CONFIG" ]] || die "missing accelerate config: $ACCEL_CONFIG"
  [[ -f "scripts/train.py" ]] || die "missing scripts/train.py (wrong repo root?)"
  [[ -d "$RELABELED_SUBSET" ]] || die "missing relabeled subset: $RELABELED_SUBSET — run '$0 reconvert' first (invariant #3)"
  [[ -d "$TEXT_EMBEDS" ]] || die "missing text-embed cache: $TEXT_EMBEDS"
  [[ -f "$ACTION_DIT" ]] || die "missing pretrained Action DiT: $ACTION_DIT"

  # cls head must be present in the runtime.
  "$PYTHON_BIN" - <<'PY' || die "ActionDiT head_cls wiring missing (enable_action_cls path)"
import inspect
from fastwam.models.wan22.action_dit import ActionDiT
assert "enable_action_cls" in inspect.signature(ActionDiT.__init__).parameters
assert hasattr(ActionDiT, "classify_from_tokens")
print("[collapse-fix] head_cls wiring OK")
PY

  # Expect exactly NUM_PROCESSES visible GPUs.
  local ngpu
  ngpu="$("$PYTHON_BIN" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  [[ "$ngpu" == "$NUM_PROCESSES" ]] || die "expected $NUM_PROCESSES GPUs, torch sees $ngpu"

  # From-scratch invariant (same as b0_v2): no legacy aerial ckpt as resume.
  [[ -z "${RESUME:-}" ]] || die "RESUME must be empty for from-scratch retrain (got '$RESUME')"
  [[ -z "${AERIAL_ALLOW_LEGACY_RESUME:-}" ]] || die "AERIAL_ALLOW_LEGACY_RESUME set; refuse — retrain is from scratch"

  show_dmax
  log "preflight OK (gpus=$ngpu, resume=null, cls wiring armed)"
}

launch() {  # $1=max_steps $2=save_every $3=logfile
  local steps="$1" save="$2" logfile="$3"
  # Use on-host Wan checkpoints only — never ModelScope/HF re-download.
  # redirect_common_files=true remaps VAE/T5 to DiffSynth safetensors and
  # triggers a multi-GB download even when checkpoints/Wan-AI/*.pth exist.
  export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  log "launch: max_steps=$steps save_every=$save cls=on λ=(v=$LAMBDA_VIDEO,fm=$LAMBDA_ACTION,ce=$LAMBDA_CE) redirect_common_files=false skip_download=$DIFFSYNTH_SKIP_DOWNLOAD output_dir=${OUTPUT_DIR:-<hydra-default>} -> $logfile"
  local launch_args=(
    task="$TASK"
    resume=null
    max_steps="$steps"
    save_every="$save"
    "data.train.dataset_dirs=[$RELABELED_SUBSET]"
    +model.action_dit_config.enable_action_cls=true
    model.redirect_common_files=false
    model.loss.lambda_video="$LAMBDA_VIDEO"
    model.loss.lambda_action="$LAMBDA_ACTION"
    model.loss.lambda_ce="$LAMBDA_CE"
  )
  if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    launch_args+=("output_dir=$OUTPUT_DIR")
  fi
  accelerate launch --config_file "$ACCEL_CONFIG" --num_processes "$NUM_PROCESSES" \
    scripts/train.py \
      "${launch_args[@]}" \
    2>&1 | tee "$logfile"
}

cmd="${1:-preflight}"
case "$cmd" in
  reconvert) reconvert ;;
  preflight) preflight ;;
  smoke)
    preflight
    launch "$SMOKE_STEPS" "$SMOKE_STEPS" "$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
    log "smoke done — verify finite loss_video/loss_action/loss_ce, no NaN, peak mem <90%,"
    log "then set MAX_STEPS from observed throughput and run: $0 train"
    ;;
  train)
    preflight
    log "starting FULL collapse-fix retrain: max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
    launch "$MAX_STEPS" "$SAVE_EVERY" "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
    log "train finished — checkpoints under checkpoints/weights/step_*.pt"
    log "next: eval each on :30905 with ORACLE_STOP=0 (learned stop) + closest_approach diag"
    ;;
  *) die "unknown command '$cmd' (use: reconvert | preflight | smoke | train)" ;;
esac
