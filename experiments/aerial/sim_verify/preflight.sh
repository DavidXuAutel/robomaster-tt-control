#!/usr/bin/env bash
# 优先环境检查（只读，不安装任何东西）。跑 ./run_all.sh 前先跑这个。
#   exit 0 = 关键项齐全，可跑探针
#   exit 1 = 有关键项缺失（系统前置手动装；OpenFly 层可跑 ./setup_env.sh）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ -f ./config.env ] && { set -a; source ./config.env; set +a; }
export OPENFLY_ROOT="${OPENFLY_ROOT:-}"
export ENV_NAME="${ENV_NAME:-env_airsim_16}"
export AIRSIM_HOST="${AIRSIM_HOST:-127.0.0.1}"
export AIRSIM_PORT="${AIRSIM_PORT:-41451}"
# shellcheck source=lib/checks.sh
source lib/checks.sh

fail=0

echo "== 系统前置（缺则手动装，见 SETUP.md §1.1）=="
chk_os      || true            # 非致命：非 22.04 也可能能跑
chk_nvidia  || fail=1
chk_nvcc    || true            # 只 flash-attn 需要
chk_ros2    || fail=1
chk_conda   || fail=1

echo "== OpenFly 环境（缺可跑 ./setup_env.sh 自动补）=="
chk_conda_env    || fail=1
chk_openfly_clone|| fail=1
chk_toolws       || true       # colcon 未构建时部分功能仍可
chk_scene        || fail=1

echo "== apt 依赖 =="
for p in xvfb libgoogle-glog-dev ros-humble-pcl-ros nlohmann-json3-dev; do
  chk_apt "$p" || true
done

echo "== python 客户端（请在 openfly / venv 内运行本脚本）=="
chk_py airsim || fail=1
chk_py numpy  || fail=1
chk_py cv2    || true          # 仅 T1 需要

echo "== bridge 端口（未启动 bridge 时会 [--]，正常）=="
chk_port || true               # 非致命：探针 T0 会再报

echo
if [ "$fail" -eq 0 ]; then
  echo "[preflight] 关键项齐全 ✅  →  ./run_all.sh"
else
  echo "[preflight] 有关键项缺失 ❌"
  echo "            系统前置（驱动/CUDA/ROS2/conda）请按 SETUP.md §1.1 手动装；"
  echo "            OpenFly 层（env/clone/deps/scene）跑：CONFIRM=1 ./setup_env.sh"
fi
exit "$fail"
