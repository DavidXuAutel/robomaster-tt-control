#!/usr/bin/env bash
# Local CPU smoke: collapse-fix unit tests + existing Stage-0 oracle mocks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m pytest -q \
  experiments/aerial/tests/test_collapse_fix_labels.py \
  experiments/aerial/tests/test_run_closed_loop_mock.py

# ActionDiT head test needs torch (skip gracefully on Mac without aerial venv).
if "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
  "$PYTHON_BIN" -m pytest -q experiments/aerial/tests/test_action_cls_head.py
else
  echo "[smoke] skip test_action_cls_head.py (no torch)"
fi

echo "[smoke] collapse-fix unit tests OK"
