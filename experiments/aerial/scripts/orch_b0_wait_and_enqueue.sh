#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_joint_b0_to_joint_video/m1b-${STAMP}/checkpoints/weights}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/home/a25689/aerial_cache_shared/orchestration/eval_queue}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/a25689/aerial_cache_shared/orchestration/results}"
ANN="${ANN:-/home/a25689/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json}"
OPENFLY_ROOT="${OPENFLY_ROOT:-/home/a25689/aerial_eval_cache/OpenFly-Platform}"
TASK="${TASK:-aerial_joint_b1_joint}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLL_S="${POLL_S:-60}"
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

command=(
  "$PYTHON_BIN" -m experiments.aerial.orchestration.b0_discover
  --weights-dir "$WEIGHTS_DIR"
  --queue-dir "$EVAL_QUEUE_DIR"
  --results-root "$RESULTS_ROOT"
  --ann "$ANN"
  --openfly-root "$OPENFLY_ROOT"
  --task "$TASK"
  --wait-final
  --poll-s "$POLL_S"
)

if (( DRY_RUN )); then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO_ROOT"
PYTHONPATH=. "${command[@]}"
