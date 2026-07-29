#!/usr/bin/env bash
# FastWAM / Genie G1：加载 ROS 2 Humble 全局环境变量（在当前 shell 中执行）
#
# 用法（在仓库根目录或任意目录）:
#   source /path/to/FastWAM/scripts/env_ros2_humble.sh
#
# 可选：叠加智元 GDK 等工作空间（需已 colcon build）
#   export GENIE_ROS_WS=/path/to/your_ws
#   source /path/to/FastWAM/scripts/env_ros2_humble.sh
#
# 常用覆盖（在 source 本脚本之前 export 即可）:
#   ROS_HUMBLE_PREFIX  默认 /opt/ros/humble
#   ROS_DISTRO         默认 humble
#   RMW_IMPLEMENTATION 默认 rmw_fastrtps_cpp
#   ROS_DOMAIN_ID      默认 0

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
export ROS_DISTRO

ROS_HUMBLE_PREFIX="${ROS_HUMBLE_PREFIX:-/opt/ros/${ROS_DISTRO}}"
export ROS_HUMBLE_PREFIX

_setup="${ROS_HUMBLE_PREFIX}/setup.bash"
if [[ ! -f "${_setup}" ]]; then
  echo "错误: 未找到 ${_setup}" >&2
  echo "请安装 ROS 2 Humble（例如 Ubuntu: sudo apt install ros-humble-desktop）" >&2
  echo "或设置 ROS_HUMBLE_PREFIX 指向你的 Humble 安装前缀。" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "${_setup}"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ -n "${GENIE_ROS_WS:-}" ]]; then
  _genie_setup="${GENIE_ROS_WS}/install/setup.bash"
  if [[ -f "${_genie_setup}" ]]; then
    # shellcheck source=/dev/null
    source "${_genie_setup}"
  else
    echo "警告: GENIE_ROS_WS=${GENIE_ROS_WS} 下未找到 install/setup.bash，已跳过叠加。" >&2
  fi
fi

echo "ROS 2 已加载: ROS_DISTRO=${ROS_DISTRO} ROS_HUMBLE_PREFIX=${ROS_HUMBLE_PREFIX} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
