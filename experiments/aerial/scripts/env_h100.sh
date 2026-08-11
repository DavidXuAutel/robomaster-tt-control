#!/usr/bin/env bash
# Aerial WAM — H100 environment prep for the offline gates (①③) and the ②④
# rollout client. SOURCE it (don't execute) so the chosen python stays active:
#
#   source experiments/aerial/scripts/env_h100.sh
#
# Idempotent. It (1) activates a conda/venv env if one is configured or found,
# (2) auto-installs the LIGHT missing deps (pyyaml/numpy) into that python,
# (3) self-checks torch + CUDA and prints the verdict — heavy/version-pinned
# torch is NOT auto-installed (that must match the box's CUDA), it only reports.
#
# Override the interpreter/env without editing the file:
#   PYENV_ACTIVATE=/path/to/venv/bin/activate  source .../env_h100.sh
#   CONDA_ENV=aerial                            source .../env_h100.sh
#   AERIAL_PY=python3.10                        source .../env_h100.sh   # explicit interpreter
#
# After sourcing, use the printed interpreter (exported as $AERIAL_PY) for the gate:
#   "$AERIAL_PY" -m experiments.aerial.rl._v0_gate --signals 2,4 --rollout-eval ...

# --- 1) activate an env if one is configured / discoverable -------------------
if [ -n "${PYENV_ACTIVATE:-}" ] && [ -f "$PYENV_ACTIVATE" ]; then
  echo "[env] source venv: $PYENV_ACTIVATE"
  # shellcheck disable=SC1090
  source "$PYENV_ACTIVATE"
elif [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  echo "[env] conda activate $CONDA_ENV"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
else
  # No env configured — auto-discover a venv/conda env, else fall back to system.
  _found=""
  for c in "$HOME"/*venv*/bin/activate /tmp/*venv*/bin/activate \
           "$HOME"/.venv/bin/activate "$HOME"/venv/bin/activate; do
    [ -f "$c" ] && { _found="$c"; break; }
  done
  if [ -n "$_found" ]; then
    echo "[env] auto-found venv: $_found"
    # shellcheck disable=SC1090
    source "$_found"
  else
    echo "[env] no conda/venv found → using system python (set PYENV_ACTIVATE/CONDA_ENV to override)"
  fi
fi

# --- 2) pick the interpreter --------------------------------------------------
AERIAL_PY="${AERIAL_PY:-python}"
command -v "$AERIAL_PY" >/dev/null 2>&1 || AERIAL_PY="python3"
export AERIAL_PY
echo "[env] interpreter: $AERIAL_PY -> $(command -v "$AERIAL_PY")  ($($AERIAL_PY --version 2>&1))"

# --- 3) ensure light deps (safe to install; version-agnostic) -----------------
for mod in yaml numpy; do
  pkg="$mod"; [ "$mod" = yaml ] && pkg="pyyaml"
  if ! "$AERIAL_PY" -c "import $mod" >/dev/null 2>&1; then
    echo "[env] installing missing dep: $pkg"
    "$AERIAL_PY" -m pip install --quiet "$pkg" || echo "[env] WARNING: pip install $pkg failed"
  fi
done

# --- 4) self-check torch + CUDA (report only) ---------------------------------
"$AERIAL_PY" - <<'PY'
import importlib, sys
ok = True
try:
    import yaml, numpy  # noqa: F401
    print("[env] yaml, numpy: OK")
except Exception as e:  # noqa: BLE001
    ok = False; print("[env] yaml/numpy: MISSING ->", e)
try:
    import torch
    print(f"[env] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}"
          + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        ok = False; print("[env] WARNING: CUDA not available — gate --device cuda will fail.")
except Exception as e:  # noqa: BLE001
    ok = False
    print("[env] torch: MISSING ->", e)
    print("[env]   install a CUDA-matched torch, e.g. (adjust cuXXX to the box):")
    print("[env]   $AERIAL_PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
print("[env] READY" if ok else "[env] NOT READY — resolve the WARNING(s) above before the gate.")
sys.exit(0)
PY
