#!/usr/bin/env bash
# FastWAM 环境一键准备（Linux CUDA / macOS CPU|MPS）
# 用法：在仓库根目录执行：bash scripts/setup_fastwam_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() { echo "错误: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "未找到 python3。请先安装 Python 3.10+（推荐 Miniconda 或 python.org），macOS 需先安装 Xcode Command Line Tools。"

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
MAJOR="${PYVER%%.*}"
MINOR="${PYVER#*.}"
[[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]] || die "需要 Python >= 3.10，当前: $PYVER"

OS="$(uname -s)"
VENV="${VENV_PATH:-$ROOT/.venv}"

echo "==> FastWAM 根目录: $ROOT"
echo "==> Python: $(command -v python3) ($PYVER)"
echo "==> 虚拟环境: $VENV"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

install_non_torch_deps() {
  python <<'PY'
import os
import re
from pathlib import Path
import subprocess
import sys

root = Path(os.getcwd()).resolve()
text = (root / "pyproject.toml").read_text(encoding="utf-8")
m = re.search(r"dependencies\s*=\s*\[(.*?)\]\s*", text, re.S)
if not m:
    sys.exit("无法解析 pyproject.toml 中的 dependencies")
block = m.group(1)
deps = re.findall(r'"([^"]+)"', block)
deps = [d for d in deps if not d.startswith("torch==") and not d.startswith("torchvision==")]
if not deps:
    sys.exit("依赖列表为空")
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
}

if [[ "$OS" == "Darwin" ]]; then
  echo "==> macOS：安装 PyTorch（CPU/MPS 官方 wheel，不使用 CUDA +cu128）"
  pip install "torch==2.7.1" "torchvision==0.22.1"
  echo "==> 安装其余 pyproject 依赖（不含 torch/torchvision）"
  install_non_torch_deps
else
  echo "==> Linux：按官方 README 安装 CUDA 12.8 版 PyTorch（需 NVIDIA 驱动）"
  pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" \
    --extra-index-url "https://download.pytorch.org/whl/cu128"
  echo "==> 安装其余 pyproject 依赖（不含 torch/torchvision）"
  install_non_torch_deps
fi

echo "==> 以可编辑模式安装 fastwam 包（不重复解析依赖）"
pip install -e . --no-deps

mkdir -p checkpoints data runs evaluate_results

echo ""
echo "完成。请执行: source \"$VENV/bin/activate\""
echo "模型目录（可选）: export DIFFSYNTH_MODEL_BASE_PATH=\"$ROOT/checkpoints\""
echo "预处理 ActionDiT（需 GPU 时请使用 --device cuda；macOS 可用 --device cpu 试跑）见 README「Model Preparation」。"
