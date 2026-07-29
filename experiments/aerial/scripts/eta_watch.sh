#!/usr/bin/env bash
# eta_watch.sh — periodic watcher + ETA for the from-scratch B0 v2 train.
#
# Gate-driven training (see docs/design/2026-07-29-aerial-nav-wam-redesign.md §2/§6):
# we do NOT commit to a fixed max_steps. max_steps is just a ceiling; the real target
# is each save_every checkpoint, because every checkpoint is a Stage-1 gate-eval point
# (closed-loop seen-20 → lock baseline only when SR>0 and reproducible). So this watcher
# reports current step + throughput and computes the time to the NEXT checkpoint(s),
# not to some arbitrary max_steps. It also lists the checkpoints already on disk
# (the gate-eval candidates) and flags a stalled run.
#
# WHERE TO RUN: on the train host (:31126), alongside the training job:
#   cd "$RUNTIME" && nohup experiments/aerial/scripts/eta_watch.sh >/dev/null 2>&1 &
#   tail -f logs/ft/b0_v2/eta_watch.status
# One-shot (e.g. from a laptop over SSH, or a cron tick):
#   ssh -p 31126 a25689@10.239.121.21 \
#     'cd <RUNTIME> && ONESHOT=1 experiments/aerial/scripts/eta_watch.sh'
#
# Env knobs (all optional):
#   INTERVAL   poll seconds (default 300)
#   SAVE_EVERY checkpoint cadence == gate-eval cadence (default 500)
#   LOOKAHEAD  how many upcoming checkpoints to project (default 3)
#   TARGET     optional absolute target step for a total ETA (default 0 = off)
#   LOG        explicit train log path (default: newest train_*.log in LOG_DIR)
#   LOG_DIR    default logs/ft/b0_v2 (matches run_b0_v2_from_scratch.sh)
#   WEIGHTS_DIR default checkpoints/weights
#   STATUS     status file to append (default $LOG_DIR/eta_watch.status)
#   ONESHOT    1 = print once and exit (no loop)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-logs/ft/b0_v2}"
LOG="${LOG:-}"
INTERVAL="${INTERVAL:-300}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOOKAHEAD="${LOOKAHEAD:-3}"
TARGET="${TARGET:-0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-checkpoints/weights}"
STATUS="${STATUS:-$LOG_DIR/eta_watch.status}"
ONESHOT="${ONESHOT:-0}"
mkdir -p "$LOG_DIR"

pick_log() {
  if [[ -n "$LOG" ]]; then printf '%s\n' "$LOG"; return; fi
  ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1
}

hms() { local s=$1; (( s < 0 )) && s=0; printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

# wall-clock string for "now + N seconds" (GNU date, BSD date fallback)
at_time() { local s=$1; date -d "@$((s))" '+%m-%d %H:%M' 2>/dev/null || date -r "$s" '+%m-%d %H:%M' 2>/dev/null || echo '?'; }

pos() { awk -v v="$1" 'BEGIN{exit !(v+0>0)}'; }  # true if arg is a positive number

emit() { echo "$1"; printf '%s\n' "$1" >> "$STATUS"; }

prev_step=""; prev_ts=""
while :; do
  now=$(date +%s); nowh=$(date '+%Y-%m-%d %H:%M:%S')
  logf="$(pick_log)"

  if [[ -z "$logf" || ! -f "$logf" ]]; then
    emit "[$nowh] no train log in $LOG_DIR — train not started yet?"
    [[ "$ONESHOT" == 1 ]] && exit 0; sleep "$INTERVAL"; continue
  fi

  tl="$(grep -aE '\[train\] .*step=[0-9]+/' "$logf" | tail -1)"
  if [[ -z "$tl" ]]; then
    emit "[$nowh] $logf: no [train] step line yet (torch compile / warmup in progress)"
    [[ "$ONESHOT" == 1 ]] && exit 0; sleep "$INTERVAL"; continue
  fi

  cur="$(sed -nE 's/.*step=([0-9]+)\/.*/\1/p'        <<<"$tl")"
  mx="$( sed -nE 's/.*step=[0-9]+\/([0-9]+).*/\1/p'  <<<"$tl")"
  spd="$(sed -nE 's/.*speed=([0-9.]+) step\/s.*/\1/p' <<<"$tl")"
  loss="$(sed -nE 's/.* loss=([0-9.eE+-]+).*/\1/p'   <<<"$tl")"
  logeta="$(sed -nE 's/.*eta=([0-9:]+).*/\1/p'       <<<"$tl")"
  [[ -z "$spd" ]] && spd=0

  # cross-poll observed rate — more "current" than the log's cumulative speed=
  obs=""; stall=""
  if [[ -n "$prev_step" && -n "$prev_ts" ]]; then
    dstep=$((cur - prev_step)); dt=$((now - prev_ts))
    (( dt > 0 )) && obs=$(awk -v a="$dstep" -v b="$dt" 'BEGIN{printf "%.3f", a/b}')
    (( dstep == 0 )) && stall="  ** STALLED: step unchanged for ${dt}s — check AirSim/OOM/deadlock **"
  fi
  rate="$spd"; { [[ -n "$obs" ]] && pos "$obs"; } && rate="$obs"

  # ETA to the next LOOKAHEAD checkpoint boundaries (the gate-eval points)
  proj=""
  if pos "$rate"; then
    for i in $(seq 1 "$LOOKAHEAD"); do
      nb=$(( ( cur / SAVE_EVERY + i ) * SAVE_EVERY ))
      steps=$(( nb - cur ))
      secs=$(awk -v s="$steps" -v r="$rate" 'BEGIN{printf "%d", s/r}')
      proj+=$'\n'"    -> step_$(printf '%06d' "$nb"): +$(hms "$secs")  (~$(at_time $((now+secs))))"
    done
  fi

  tgt=""
  if (( TARGET > 0 )) && pos "$rate"; then
    ts=$(( TARGET - cur )); (( ts < 0 )) && ts=0
    tsec=$(awk -v s="$ts" -v r="$rate" 'BEGIN{printf "%d", s/r}')
    tgt="  target=$TARGET in +$(hms "$tsec") (~$(at_time $((now+tsec))))"
  fi

  ckpts="$(ls -1 "$WEIGHTS_DIR"/step_*.pt 2>/dev/null | wc -l | tr -d ' ')"

  emit "[$nowh] step=$cur/$mx loss=${loss:-?} rate=${rate} step/s (log_avg=${spd}) ckpts_on_disk=${ckpts}${tgt}${stall}${proj}"

  prev_step="$cur"; prev_ts="$now"
  [[ "$ONESHOT" == 1 ]] && exit 0
  sleep "$INTERVAL"
done
