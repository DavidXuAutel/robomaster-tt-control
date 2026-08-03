#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STATUS_FILE="${STATUS_FILE:-/home/a25689/aerial_eval_cache/logs/eval/b0_seen_videos.status}"
COLLECTION_SOURCE="${COLLECTION_SOURCE:-/home/a25689/aerial_eval_cache/data/openfly_raw/Annotation/seen_airsim16_collection_source.json}"
COLLECTION_MANIFEST="${COLLECTION_MANIFEST:-/home/a25689/aerial_eval_cache/logs/eval/collection_manifest.json}"
COLLECTION_OUT="${COLLECTION_OUT:-/home/a25689/aerial_eval_cache/data/b0_dagger_correction}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${OPENFLY_ROOT:?Set OPENFLY_ROOT to the OpenFly-Platform checkout}"
: "${B0_CHECKPOINT:?Set B0_CHECKPOINT to the B0 step_004000.pt checkpoint}"

if [[ "$(basename "$B0_CHECKPOINT")" != "step_004000.pt" ]]; then
  echo "Refusing non-B0-final checkpoint: expected step_004000.pt" >&2
  exit 2
fi
if [[ ! -f "$B0_CHECKPOINT" ]]; then
  echo "Missing B0 checkpoint: $B0_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$COLLECTION_SOURCE" ]]; then
  echo "Blocking: collection source is not deployed: $COLLECTION_SOURCE" >&2
  echo "Do not substitute or derive this file from held-out annotations." >&2
  exit 2
fi
if [[ ! -f "$COLLECTION_MANIFEST" ]]; then
  echo "Missing oracle/pilot collection manifest: $COLLECTION_MANIFEST" >&2
  exit 2
fi

while true; do
  status=""
  if [[ -f "$STATUS_FILE" ]]; then
    status="$(tr -d '[:space:]' < "$STATUS_FILE")"
  fi
  case "$status" in
    COMPLETED)
      break
      ;;
    FAILED)
      echo "B0 video queue failed: $STATUS_FILE" >&2
      exit 1
      ;;
    *)
      sleep "$POLL_SECONDS"
      ;;
  esac
done

read -r takeover_m release_m abort_m < <(
  "$PYTHON_BIN" - "$COLLECTION_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
if not manifest.get("oracle_gate", {}).get("passed", False):
    raise SystemExit("oracle gate did not pass")
thresholds = manifest["thresholds"]
print(thresholds["takeover_m"], thresholds["release_m"], thresholds["abort_m"])
PY
)

mkdir -p "$COLLECTION_OUT"
cp "$COLLECTION_MANIFEST" "$COLLECTION_OUT/collection_manifest.json"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiments.aerial.eval.collect_dagger \
  --ann "$COLLECTION_SOURCE" \
  --out "$COLLECTION_OUT" \
  --bridge openfly \
  --policy fastwam \
  --openfly-root "$OPENFLY_ROOT" \
  --checkpoint "$B0_CHECKPOINT" \
  --task aerial_joint_b0_novideo \
  --max-episodes 40 \
  --takeover-m "$takeover_m" \
  --release-m "$release_m" \
  --abort-m "$abort_m"
