#!/usr/bin/env bash
# collapse_fix_status.sh — one-shot progress report for the collapse-fix run.
#
# Prints TWO sections:
#   [TRAIN]  current step / losses (incl. loss_ce) from the newest train log,
#            checkpoints on disk, throughput + ETA to next/max step, dataset_stats.
#   [EVAL]   eval_queue_collapse_fix pending/running/done/failed counts, and a
#            per-checkpoint table of the learned-stop metrics that matter here:
#            SR / NE / SPL, n_stop_primitive (root cause #1: does it terminate?),
#            closest_approach_mean + oracle_hit@20 (root cause #2: goal-blindness).
#
# Host-agnostic by design:
#   - Checkpoints + eval queue live on shared Ceph → visible from :31126 and :30905.
#   - The train log is local to :31126. If it is not found (e.g. run from :30905),
#     the TRAIN section degrades to checkpoints-on-disk and says so.
#   - Throughput/ETA are derived from checkpoint mtimes (file-based), so they work
#     even when the log is absent.
#
# Usage:
#   bash experiments/aerial/scripts/collapse_fix_status.sh            # one-shot
#   WATCH=1 bash experiments/aerial/scripts/collapse_fix_status.sh    # loop (INTERVAL s)
#   watch -n 60 'bash experiments/aerial/scripts/collapse_fix_status.sh'  # external loop
#
# Env knobs (all optional; defaults match the runbook path table):
#   STAMP OUTPUT_DIR WEIGHTS_DIR EVAL_QUEUE_DIR LOG_DIR LOG
#   MAX_STEPS SAVE_EVERY STEPS INTERVAL WATCH STALL_S PYTHON_BIN
#   PLOT (default 1) -> render loss curve (incl. loss_ce) via plot_loss.py each
#            report; PLOT=0 to skip. PLOT_OUT (default $LOG_DIR/loss_curve.png),
#            PLOT_SMOOTH (default 21). Skips cleanly if matplotlib/log is absent.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

STAMP="${STAMP:-20260731-collapse-fix-1500}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$OUTPUT_DIR/checkpoints/weights}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/ft/collapse_fix}"
LOG="${LOG:-}"                       # explicit train log; else newest in LOG_DIR
MAX_STEPS="${MAX_STEPS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
STEPS="${STEPS:-500,1000,1500}"      # steps we expect to land / evaluate
INTERVAL="${INTERVAL:-60}"
WATCH="${WATCH:-0}"
STALL_S="${STALL_S:-1800}"           # warn if the train log has been idle this long
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLOT="${PLOT:-1}"                    # render loss curve (incl. loss_ce) by default; PLOT=0 to skip
PLOT_OUT="${PLOT_OUT:-$LOG_DIR/loss_curve.png}"
PLOT_SMOOTH="${PLOT_SMOOTH:-21}"

case "${1:-}" in
  --watch) WATCH=1 ;;
  --once) WATCH=0 ;;
  -h|--help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

pick_log() {
  [[ -n "$LOG" ]] && { printf '%s\n' "$LOG"; return; }
  local f
  f="$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)"
  [[ -z "$f" ]] && f="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)"
  printf '%s\n' "$f"
}

report() {
  echo "================ collapse-fix status @ $(date '+%Y-%m-%d %H:%M:%S') ================"
  echo "stamp=$STAMP  max_steps=$MAX_STEPS  save_every=$SAVE_EVERY"
  echo "weights=$WEIGHTS_DIR"
  echo "queue=$EVAL_QUEUE_DIR"
  echo

  # ---------------- TRAIN ----------------
  echo "---- [TRAIN] ----"
  local logf; logf="$(pick_log)"
  MAX_STEPS="$MAX_STEPS" SAVE_EVERY="$SAVE_EVERY" STALL_S="$STALL_S" \
  WEIGHTS_DIR="$WEIGHTS_DIR" OUTPUT_DIR="$OUTPUT_DIR" LOGF="$logf" \
  "$PYTHON_BIN" - <<'PY'
import os, re, glob, time, math

log = os.environ.get("LOGF") or ""
weights = os.environ["WEIGHTS_DIR"]
outdir = os.environ["OUTPUT_DIR"]
max_steps = int(os.environ["MAX_STEPS"])
save_every = int(os.environ["SAVE_EVERY"])
stall_s = int(os.environ["STALL_S"])

# --- checkpoints on disk (always available if weights dir is reachable) ---
cks = sorted(glob.glob(os.path.join(weights, "step_*.pt")))
def step_of(p):
    m = re.search(r"step_(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1
ck_steps = sorted(s for s in (step_of(p) for p in cks) if s >= 0)
latest_ck = ck_steps[-1] if ck_steps else None

# --- parse newest train log (tokenized like plot_loss/eta_watch) ---
TOKEN = re.compile(r"\b(step|loss_action|loss_video|loss_ce|loss|lr)=([-+0-9.eE]+)(?:/[0-9]+)?")
last = None
log_ok = bool(log) and os.path.isfile(log)
if log_ok:
    cur = {}
    with open(log, errors="replace") as fh:
        text = fh.read()
    for name, val in TOKEN.findall(text):
        if name == "step":
            if "step" in cur:
                last = cur
            cur = {"step": int(val)}
        elif "step" in cur:
            try: cur[name] = float(val)
            except ValueError: pass
    if "step" in cur:
        last = cur

if log_ok and last:
    s = last["step"]
    def g(k):
        v = last.get(k);
        return f"{v:.4f}" if isinstance(v, float) else "  -   "
    pct = 100.0 * s / max(max_steps, 1)
    print(f"log        : {os.path.basename(log)}")
    print(f"step       : {s}/{max_steps} ({pct:.0f}%)")
    print(f"loss       : total={g('loss')} video={g('loss_video')} "
          f"action={g('loss_action')} ce={g('loss_ce')}")
    age = time.time() - os.path.getmtime(log)
    flag = "  <-- STALLED?" if age > stall_s else ""
    print(f"log age    : {int(age)}s (idle){flag}")
    cur_step = s
elif log_ok:
    print(f"log        : {os.path.basename(log)} (no step= lines yet)")
    cur_step = latest_ck or 0
else:
    print(f"log        : not found under LOG_DIR (train host only) — using ckpts on disk")
    cur_step = latest_ck or 0

# --- checkpoints line ---
if ck_steps:
    print(f"checkpoints: {len(ck_steps)} on disk -> " + ", ".join(str(s) for s in ck_steps))
else:
    print(f"checkpoints: none yet under {weights}")

# --- dataset_stats.json (needed for eval denorm) ---
ds = os.path.join(outdir, "dataset_stats.json")
print(f"dataset_stats.json: {'present' if os.path.isfile(ds) else 'MISSING'} ({ds})")

# --- throughput + ETA from the two newest checkpoint mtimes (file-based) ---
if len(ck_steps) >= 2:
    p_last = os.path.join(weights, f"step_{ck_steps[-1]:06d}.pt")
    p_prev = os.path.join(weights, f"step_{ck_steps[-2]:06d}.pt")
    dstep = ck_steps[-1] - ck_steps[-2]
    dt = os.path.getmtime(p_last) - os.path.getmtime(p_prev)
    if dt > 0 and dstep > 0:
        sps = dstep / dt
        def hms(x):
            x = max(int(x), 0); return f"{x//3600:02d}:{(x%3600)//60:02d}:{x%60:02d}"
        nxt = ((cur_step // save_every) + 1) * save_every
        nxt = min(nxt, max_steps)
        eta_next = (nxt - cur_step) / sps if sps > 0 else 0
        eta_max = (max_steps - cur_step) / sps if sps > 0 else 0
        print(f"throughput : ~{sps:.3f} step/s (from ckpt mtimes)")
        print(f"ETA        : next ckpt(step {nxt}) in {hms(eta_next)} | "
              f"max({max_steps}) in {hms(eta_max)}")
    else:
        print("throughput : n/a (ckpt mtimes not monotonic)")
else:
    print("throughput : n/a (need >=2 checkpoints)")
PY

  # ---------------- LOSS CURVE (optional) ----------------
  if [[ "$PLOT" == "1" ]]; then
    if [[ -n "$logf" && -f "$logf" ]]; then
      "$PYTHON_BIN" "$SCRIPT_DIR/plot_loss.py" "$logf" \
        --out "$PLOT_OUT" --smooth "$PLOT_SMOOTH" 2>&1 | sed 's/^/plot       : /'
    else
      echo "plot       : skipped (no train log to plot; PLOT=1 needs the log on this host)"
    fi
  fi
  echo

  # ---------------- EVAL ----------------
  echo "---- [EVAL] (queue: $(basename "$EVAL_QUEUE_DIR")) ----"
  for st in pending running done failed; do
    local d="$EVAL_QUEUE_DIR/$st" c=0
    [[ -d "$d" ]] && c="$(find "$d" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    printf '%-8s %s\n' "$st:" "$c"
  done
  echo
  EVAL_QUEUE_DIR="$EVAL_QUEUE_DIR" STEPS="$STEPS" "$PYTHON_BIN" - <<'PY'
import os, json, glob, re

qdir = os.environ["EVAL_QUEUE_DIR"]
want = [int(x) for x in os.environ.get("STEPS", "").split(",") if x.strip().isdigit()]

def step_from(job, path):
    if isinstance(job.get("step"), int):
        return job["step"]
    ck = str(job.get("checkpoint", ""))
    m = re.search(r"step_(\d+)", ck)
    return int(m.group(1)) if m else -1

rows = []
for state in ("done", "running", "failed"):
    for jp in glob.glob(os.path.join(qdir, state, "*.json")):
        try:
            job = json.loads(open(jp).read())
        except Exception:
            continue
        step = step_from(job, jp)
        m = {}
        om = job.get("out_metrics")
        if state == "done" and om and os.path.isfile(om):
            try: m = json.loads(open(om).read())
            except Exception: m = {}
        rows.append((step, state, m))

if not rows:
    print("(no done/running/failed eval jobs yet)")
else:
    rows.sort(key=lambda r: (r[0], r[1]))
    hdr = f"{'step':>6} {'state':>7} {'SR':>6} {'NE':>7} {'SPL':>6} {'stop%':>6} {'close_m':>8} {'hit@20':>7}"
    print(hdr); print("-" * len(hdr))
    for step, state, m in rows:
        def f(k, w=6, p=3):
            v = m.get(k)
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        n = m.get("n") or 0
        nstop = m.get("n_stop_primitive")
        stop_pct = (100.0 * nstop / n) if isinstance(nstop,(int,float)) and n else None
        stop_s = f"{stop_pct:>5.0f}%" if stop_pct is not None else f"{'-':>6}"
        print(f"{step:>6} {state:>7} {f('SR')} {f('NE',7)} {f('SPL')} "
              f"{stop_s} {f('closest_approach_mean',8,2)} {f('oracle_hit@20',7)}")
    got = {s for s, st, _ in rows if st == "done"}
    miss = [s for s in want if s not in got]
    if miss:
        print(f"\npending evals for steps: {', '.join(map(str, miss))}")
PY
  echo "reference: b0_v2 SR=0 ; Stage-0 oracle-stop ceiling ~10% ; success dist=20m"
}

if [[ "$WATCH" == "1" ]]; then
  while true; do report; echo; sleep "$INTERVAL"; done
else
  report
fi
