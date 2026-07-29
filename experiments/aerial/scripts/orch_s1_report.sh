#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
STATUS_PATH="${STATUS_PATH:-/home/a25689/aerial_cache/orchestration/status.json}"
ORCH_ROOT="${ORCH_ROOT:-/home/a25689/aerial_cache/orchestration}"
LOCK_PATH="${LOCK_PATH:-$ORCH_ROOT/baseline_lock.manifest.json}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/a25689/aerial_cache_shared/orchestration/results}"
REPORT_DIR="${REPORT_DIR:-$RESULTS_ROOT/b1_${STAMP}}"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/ft_selection_report.json}"
DIAGNOSIS_PATH="${DIAGNOSIS_PATH:-$REPORT_DIR/ft_s1_failure_diagnosis.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: STAMP=... LOCK_PATH=... $0 [--dry-run]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

candidate_args=()
for step in $(seq 1000 1000 5000); do
  metrics="$(printf '%s/b1_%s/step_%06d_seen20/metrics.json' "$RESULTS_ROOT" "$STAMP" "$step")"
  candidate_args+=(--candidate "$step=$metrics")
done

baseline_metrics="$("$PYTHON_BIN" - "$LOCK_PATH" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["metrics_path"])
PY
)"

compare_command=(
  "$PYTHON_BIN" -m experiments.aerial.eval.compare_finetune
  --baseline "$baseline_metrics"
  --lock-manifest "$LOCK_PATH"
  "${candidate_args[@]}"
  --out "$REPORT_PATH"
  --diagnosis-out "$DIAGNOSIS_PATH"
)

if (( DRY_RUN )); then
  printf '%q ' "${compare_command[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO_ROOT"
mkdir -p "$REPORT_DIR"
set +e
PYTHONPATH=. "${compare_command[@]}"
rc=$?
set -e

s1_pass=false
if (( rc == 0 )); then
  s1_pass=true
elif (( rc == 1 )); then
  s1_pass=false
else
  exit "$rc"
fi

PYTHONPATH=. "$PYTHON_BIN" -m experiments.aerial.orchestration.supervisor \
  --status "$STATUS_PATH" \
  --stamp "$STAMP" \
  --mark-s1-report \
  --s1-pass "$s1_pass" \
  --report-path "$REPORT_PATH"

echo "S1 report written: $REPORT_PATH (s1_pass=$s1_pass)"
exit 0
