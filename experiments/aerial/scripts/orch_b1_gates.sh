#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
STATUS_PATH="${STATUS_PATH:-/home/a25689/aerial_cache/orchestration/status.json}"
ORCH_ROOT="${ORCH_ROOT:-/home/a25689/aerial_cache/orchestration}"
LOCK_PATH="${LOCK_PATH:-$ORCH_ROOT/baseline_lock.manifest.json}"
COLLECTION_SOURCE="${COLLECTION_SOURCE:-$ORCH_ROOT/collection_source.json}"
HELDOUT_ANN="${HELDOUT_ANN:-/home/a25689/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json}"
ORACLE_JSON="${ORACLE_JSON:-$ORCH_ROOT/oracle_gate.json}"
CORRECTION_ROOT="${CORRECTION_ROOT:-$ORCH_ROOT/correction_dataset}"
FT_MANIFEST="${FT_MANIFEST:-$ORCH_ROOT/ft_cache.sha256}"
SMOKE_STATUS="${SMOKE_STATUS:-$ORCH_ROOT/ft_smoke.status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run]"
    echo "Exits 0 only if all B1 gates pass; otherwise writes BLOCKED status and exits 2."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

command=(
  "$PYTHON_BIN" -m experiments.aerial.orchestration.supervisor
  --status "$STATUS_PATH"
  --stamp "$STAMP"
  --run-b1-gates
  --lock-path "$LOCK_PATH"
  --collection-source "$COLLECTION_SOURCE"
  --heldout-ann "$HELDOUT_ANN"
  --oracle-json "$ORACLE_JSON"
  --correction-root "$CORRECTION_ROOT"
  --ft-manifest "$FT_MANIFEST"
  --smoke-status "$SMOKE_STATUS"
)

if (( DRY_RUN )); then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO_ROOT"
PYTHONPATH=. "${command[@]}"
