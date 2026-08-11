#!/usr/bin/env bash
# Aerial WAM — H100 environment for the offline gates (①③) and the ②④ rollout
# client. SOURCE it (don't execute) so the venv stays active in your shell:
#
#   source experiments/aerial/scripts/env_h100.sh              # activate + self-check
#   INSTALL=1 source experiments/aerial/scripts/env_h100.sh    # first time: build .venv
#
# The H100 boxes are ephemeral containers (env gets wiped on re-provision —
# 2026-08-10 move to .22 already needed a rebuild, and the current box had NO
# torch/pyyaml at all). This script makes the rebuild one command instead of a
# remembered pip incantation, and self-locates the repo via `git rev-parse` so
# the SAME file runs from any checkout.
#
# MINIMAL ②④/①③ gate deps only (NOT the full FastWAM stack — no deepspeed/
# transformers): torch cu128 + numpy + pyyaml + the airsim RPC client trio,
# PLUS einops + addict for the DA3 depth head.
# The DA3 depth head loads the vendored depth_anything_3 backbone (DinoV2 + DPT
# in third_party/), which hard-imports `einops` and `addict`. It does NOT need
# timm/transformers/depth_anything-pip, and `xformers` is optional (try/except
# fallback in swiglu_ffn). So the DA3 extras are just those two tiny pure-python
# packages — but they ARE required: without them the ②④ shield eval and the ①d
# depth gate crash on `ModuleNotFoundError: einops` when loading the depth ckpt.
#
# Overrides:
#   VENV=/path/to/venv        INSTALL=1 source .../env_h100.sh   # venv location
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128           # match the box CUDA

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV="${VENV:-$ROOT/.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# --- optional one-time bootstrap ---------------------------------------------
if [ "${INSTALL:-0}" = "1" ]; then
  echo "[env] bootstrap venv: $VENV"
  if [ ! -x "$VENV/bin/python" ]; then
    rm -rf "$VENV"
    if python3 -m venv "$VENV" 2>/dev/null; then
      : # normal venv (ensurepip present)
    else
      # Debian/Ubuntu system python often lacks ensurepip/python3-venv. Create a
      # pip-less venv and bootstrap pip via get-pip.py (SETUP.md §3.2 pattern) —
      # no sudo / no apt needed.
      echo "[env] ensurepip missing → venv --without-pip + get-pip.py"
      rm -rf "$VENV"
      python3 -m venv --without-pip "$VENV" \
        || { echo "[env] ✗ 'python3 -m venv --without-pip' failed"; return 1 2>/dev/null || exit 1; }
      # shellcheck disable=SC1090
      source "$VENV/bin/activate"
      GP="${TMPDIR:-/tmp}/get-pip.py"
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GP"
      elif command -v wget >/dev/null 2>&1; then
        wget -qO "$GP" https://bootstrap.pypa.io/get-pip.py
      else
        python - <<'PY'
import urllib.request, os
dst = os.path.join(os.environ.get("TMPDIR", "/tmp"), "get-pip.py")
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", dst)
print("[env] fetched get-pip.py via urllib:", dst)
PY
      fi
      python "$GP" || { echo "[env] ✗ get-pip.py failed (network to bootstrap.pypa.io?)"; return 1 2>/dev/null || exit 1; }
    fi
  fi
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  python -m pip install -U pip setuptools wheel
  echo "[env] installing torch cu128 (pinned) from $TORCH_INDEX ..."
  python -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url "$TORCH_INDEX"
  echo "[env] installing gate deps (numpy/pyyaml + airsim RPC client + DA3 einops/addict) ..."
  python -m pip install "numpy==1.26.4" "pyyaml" \
    "airsim>=1.8.1" "opencv-python-headless>=4.6" "msgpack-rpc-python>=0.4.1" \
    "einops" "addict"
  # airsim pulls the NON-headless opencv-python, which needs libGL.so.1 (absent
  # on GPU pods) and shadows the headless build. Force headless-only so `import
  # cv2` works without apt/libgl1.
  echo "[env] enforcing headless OpenCV (drop non-headless opencv that airsim pulled) ..."
  python -m pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
  python -m pip install --force-reinstall "opencv-python-headless>=4.6"
  echo "[env] bootstrap done."
fi

# --- activate (bootstrap venv, else a discoverable one, else system) ----------
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  echo "[env] activated venv: $VENV"
else
  echo "[env] no venv at $VENV → using system python (run 'INSTALL=1 source ...' to build it)"
fi

AERIAL_PY="${AERIAL_PY:-python}"
command -v "$AERIAL_PY" >/dev/null 2>&1 || AERIAL_PY="python3"
export AERIAL_PY
echo "[env] interpreter: $AERIAL_PY -> $(command -v "$AERIAL_PY")  ($($AERIAL_PY --version 2>&1))"

# --- self-check (report only) -------------------------------------------------
"$AERIAL_PY" - <<'PY'
ok = True
def probe(name, imp):
    global ok
    try:
        imp(); return True
    except Exception as e:  # noqa: BLE001
        ok = False; print(f"[env] {name}: MISSING -> {e}"); return False

if probe("yaml", lambda: __import__("yaml")) and probe("numpy", lambda: __import__("numpy")):
    print("[env] yaml, numpy: OK")
try:
    import torch
    cu = torch.cuda.is_available()
    print(f"[env] torch {torch.__version__}  cuda_available={cu}"
          + (f"  device={torch.cuda.get_device_name(0)}" if cu else ""))
    if not cu:
        ok = False; print("[env] WARNING: CUDA unavailable — gate --device cuda will fail.")
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"[env] torch: MISSING -> {e}")
    print("[env]   run:  INSTALL=1 source experiments/aerial/scripts/env_h100.sh")
for pkg in ("airsim", "cv2", "msgpackrpc", "einops", "addict"):
    probe(pkg, lambda p=pkg: __import__(p))
if all(m in globals() for m in ()):
    pass
print("[env] READY" if ok else "[env] NOT READY — resolve WARNING(s) above (likely: INSTALL=1 source ...).")
PY
