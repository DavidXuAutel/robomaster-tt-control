#!/usr/bin/env bash
# 幂等安装：先检查，只对「不通过」的项下载/安装。可反复跑。
#
#   ./setup_env.sh            # 干跑：只打印将要安装的项，不执行
#   CONFIRM=1 ./setup_env.sh  # 真正执行安装
#
# 系统前置（NVIDIA 驱动 / CUDA / ROS2 Humble / conda 本体）不自动装 —— 太重且需
# 你自行决策；缺失则停并提示。OpenFly 层（conda env / clone / pip / apt / colcon /
# 客户端依赖）按需自动补。AirSim 场景因 HF 文件名多变，仅提示手动下载。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ -f ./config.env ] && { set -a; source ./config.env; set +a; }
: "${OPENFLY_ROOT:?set OPENFLY_ROOT in config.env}"
ENV_NAME="${ENV_NAME:-env_airsim_16}"
CONFIRM="${CONFIRM:-0}"
# shellcheck source=lib/checks.sh
source lib/checks.sh

if [ "$CONFIRM" != "1" ]; then
  echo "*** 干跑模式（DRY-RUN）：只打印将执行的安装，不落地。加 CONFIRM=1 才真装。 ***"
fi
echo

run(){   # 打印命令；仅 CONFIRM=1 时执行
  echo "  + $*"
  if [ "$CONFIRM" = "1" ]; then eval "$@"; fi
}
in_env(){ run "conda run -n openfly bash -lc \"$1\""; }

# ---- 系统前置：只检查，缺则停 ----
miss_sys=0
chk_conda >/dev/null 2>&1 || { echo "[stop] 缺 conda —— 见 SETUP.md §1.1"; miss_sys=1; }
chk_ros2  >/dev/null 2>&1 || { echo "[stop] 缺 ROS2 Humble —— 见 SETUP.md §1.1"; miss_sys=1; }
[ "$miss_sys" = 1 ] && { echo "先装系统前置再重跑。"; exit 1; }

# ---- 1) conda env openfly ----
if ! chk_conda_env >/dev/null 2>&1; then
  echo "[missing] conda env 'openfly'"; run "conda create -n openfly python=3.10 -y"
else echo "[ok] conda env 'openfly'"; fi

# ---- 2) OpenFly clone ----
if ! chk_openfly_clone >/dev/null 2>&1; then
  echo "[missing] OpenFly clone -> $OPENFLY_ROOT"
  run "git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git '$OPENFLY_ROOT'"
else echo "[ok] OpenFly clone"; fi

# ---- 3) pip 依赖（openfly 内）----
echo "[step] pip deps (idempotent)"
in_env "cd '$OPENFLY_ROOT' && pip install -r requirements.txt && pip install packaging ninja"
in_env "pip install 'flash-attn==2.5.5' --no-build-isolation || echo '[warn] flash-attn skipped (full-stack only)'"
in_env "python -c 'import dlimp' 2>/dev/null || { git clone https://github.com/kvablack/dlimp '$OPENFLY_ROOT/third_party_dlimp' 2>/dev/null; pip install -e '$OPENFLY_ROOT/third_party_dlimp'; }"

# ---- 4) apt 依赖 ----
for p in xvfb libgoogle-glog-dev ros-humble-pcl-ros nlohmann-json3-dev; do
  if ! chk_apt "$p" >/dev/null 2>&1; then echo "[missing] apt: $p"; run "sudo apt install -y $p"
  else echo "[ok] apt: $p"; fi
done

# ---- 5) colcon build tool_ws ----
if ! chk_toolws >/dev/null 2>&1; then
  echo "[missing] tool_ws build"
  run "bash -c 'source /opt/ros/humble/setup.bash && cd \"$OPENFLY_ROOT/tool_ws\" && colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3'"
else echo "[ok] tool_ws built"; fi

# ---- 6) 场景（不自动下，HF 文件名多变）----
if ! chk_scene >/dev/null 2>&1; then
  echo "[missing] scene $ENV_NAME —— 手动下载（SETUP.md §1.6）："
  echo "  huggingface-cli download IPEC-COMMUNITY/OpenFly_DataGen --repo-type dataset \\"
  echo "    --include 'airsim/${ENV_NAME}*' --local-dir /data/hf_openfly"
  echo "  然后 mv .../airsim/${ENV_NAME}* '$OPENFLY_ROOT/envs/airsim/'"
else echo "[ok] scene $ENV_NAME"; fi

# ---- 7) sim_verify 客户端依赖 ----
echo "[step] sim_verify client deps (openfly 内)"
in_env "pip install -r '$HERE/requirements.txt'"

echo
if [ "$CONFIRM" = "1" ]; then echo "[setup_env] 完成。跑 ./preflight.sh 复检，再 ./run_all.sh。"
else echo "[setup_env] 干跑结束。确认无误后：CONFIRM=1 ./setup_env.sh"; fi
