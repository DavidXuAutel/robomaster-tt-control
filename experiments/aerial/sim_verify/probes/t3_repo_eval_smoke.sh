#!/usr/bin/env bash
# T3 (OPTIONAL) — end-to-end regression through the repo's closed-loop runner.
#
# Only relevant if you switched settings.json CV->Multirotor and want to confirm
# the existing teleport-style OpenFly eval still renders real frames. Requires
# the robomaster-tt-control repo (run_closed_loop.py). NOTE: there is NO
# --env / --env-name flag — the env name is derived per-episode from the
# annotation, so we only pass --openfly-root and --ann.
set -euo pipefail

: "${OPENFLY_ROOT:?set OPENFLY_ROOT}"
: "${ANN:?set ANN (path to seen_*.json annotation)}"
REPO="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo "")}"
if [[ -z "$REPO" || ! -f "$REPO/experiments/aerial/eval/run_closed_loop.py" ]]; then
  echo "[T3] SKIP: repo run_closed_loop.py not found (set REPO_ROOT to the repo checkout)"
  exit 0
fi

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
OUT_T3="${OUT_T3:-/tmp/t3_openfly.json}"
DUMP_T3="${DUMP_T3:-/tmp/t3_frames}"
mkdir -p "$DUMP_T3"

python3 -m experiments.aerial.eval.run_closed_loop \
  --bridge openfly --policy replay \
  --openfly-root "$OPENFLY_ROOT" \
  --ann "$ANN" \
  --max-episodes 1 --max-steps 20 \
  --dump-frames "$DUMP_T3" \
  --out "$OUT_T3"

echo "[T3] PASS if metrics written to $OUT_T3 and PNGs exist in $DUMP_T3:"
ls -la "$DUMP_T3" | head
