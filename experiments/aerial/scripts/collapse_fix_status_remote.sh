#!/usr/bin/env bash
# collapse_fix_status_remote.sh — run collapse_fix_status.sh ON a remote host over SSH.
#
# WHY: the train log, checkpoints and eval queue live on the lab hosts, not on
# your laptop. This wrapper SSHes in, cd's into the runtime checkout, and runs
# collapse_fix_status.sh there, forwarding the relevant env vars.
#
# Two hosts (per the current bring-up):
#   TARGET=train (default) -> 推理/训练机  10.239.121.21 : 31126
#   TARGET=eval            -> 测试/评测机  10.239.121.23 : 30905
# TARGET flips SSH_HOST and SSH_PORT together so you can't move one and forget the
# other. The train host sees the train log locally AND the shared ckpt/queue on
# Ceph, so ONE call there gives BOTH the [TRAIN] and [EVAL] sections in full.
#
# ALWAYS `--dry-run` first to confirm host/port/runtime — IP/port and the runtime
# path have changed across bring-ups before.
#
# Usage:
#   bash experiments/aerial/scripts/collapse_fix_status_remote.sh --dry-run   # print ssh cmd
#   bash experiments/aerial/scripts/collapse_fix_status_remote.sh             # one-shot (train host)
#   TARGET=eval bash experiments/aerial/scripts/collapse_fix_status_remote.sh # eval host
#   WATCH=1 bash experiments/aerial/scripts/collapse_fix_status_remote.sh     # loop on remote
#
# Env knobs:
#   TARGET     train (default) | eval  — picks the host/port pair below
#   SSH_HOST   explicit override (else derived from TARGET)
#   SSH_PORT   explicit override (else derived from TARGET)
#   RUNTIME    default /home/a25689/aerial_wam_runtime/robomaster-tt-control
#              (must contain experiments/aerial/scripts/collapse_fix_status.sh)
#   WATCH INTERVAL  forwarded to the remote status script
#   Any of: STAMP OUTPUT_DIR WEIGHTS_DIR EVAL_QUEUE_DIR LOG_DIR LOG
#           MAX_STEPS SAVE_EVERY STEPS STALL_S  — forwarded when set.
set -uo pipefail

TARGET="${TARGET:-train}"
case "$TARGET" in
  train) def_host="a25689@10.239.121.21"; def_port="31126" ;;
  eval)  def_host="a25689@10.239.121.23"; def_port="30905" ;;
  *) echo "Unknown TARGET: $TARGET (use train|eval, or set SSH_HOST/SSH_PORT)" >&2; exit 2 ;;
esac
SSH_HOST="${SSH_HOST:-$def_host}"
SSH_PORT="${SSH_PORT:-$def_port}"
RUNTIME="${RUNTIME:-/home/a25689/aerial_wam_runtime/robomaster-tt-control}"
STATUS_REL="experiments/aerial/scripts/collapse_fix_status.sh"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

# Forward only env vars that are actually set (non-empty), safely quoted.
FORWARD=(STAMP OUTPUT_DIR WEIGHTS_DIR EVAL_QUEUE_DIR LOG_DIR LOG \
         MAX_STEPS SAVE_EVERY STEPS STALL_S WATCH INTERVAL \
         PLOT PLOT_OUT PLOT_SMOOTH)
env_prefix=""
for k in "${FORWARD[@]}"; do
  v="${!k:-}"
  [[ -n "$v" ]] && env_prefix+="$k=$(printf '%q' "$v") "
done

mode="--once"
[[ "${WATCH:-0}" == "1" ]] && mode="--watch"

remote_cmd="cd $(printf '%q' "$RUNTIME") && ${env_prefix}bash $(printf '%q' "$STATUS_REL") $mode"

if (( DRY_RUN )); then
  echo "would run:"
  printf '  ssh -p %s %s %q\n' "$SSH_PORT" "$SSH_HOST" "$remote_cmd"
  echo
  echo "verify: TARGET=$TARGET SSH_HOST=$SSH_HOST SSH_PORT=$SSH_PORT RUNTIME=$RUNTIME"
  echo "        remote must have $STATUS_REL under RUNTIME."
  exit 0
fi

exec ssh -p "$SSH_PORT" "$SSH_HOST" "$remote_cmd"
