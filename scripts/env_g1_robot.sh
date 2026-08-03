#!/usr/bin/env bash
# 为连接智元 G1 机器人配置 ROS 2 DDS 静态发现（默认 IP 10.229.66.60）。
#
# 必须先加载 Humble，再 source 本脚本：
#   source /path/to/FastWAM/scripts/env_ros2_humble.sh
#   source /path/to/FastWAM/scripts/env_g1_robot.sh
#
# 可选环境变量（在第二个 source 之前 export）:
#   G1_ROBOT_IP       默认 10.229.66.60
#   REMOTE_DDS        fastrtps | cyclonedds（默认 fastrtps，须与机器人 RMW 一致）
#   ROS_DOMAIN_ID     与 G1 上相同（默认 0）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

G1_ROBOT_IP="${G1_ROBOT_IP:-10.229.66.60}"
export G1_ROBOT_IP

REMOTE_DDS="${REMOTE_DDS:-fastrtps}"

_py=(python3)
if [[ -x "${ROOT}/.venv/bin/python3" ]]; then
  _py=("${ROOT}/.venv/bin/python3")
fi

_extra=()
if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
  _extra+=(--ros-domain-id "${ROS_DOMAIN_ID}")
fi

# shellcheck disable=SC2046
eval "$("${_py[@]}" "${ROOT}/experiments/genie_g1/g1_remote_link.py" \
  --print-shell-exports \
  --g1-ip "${G1_ROBOT_IP}" \
  --dds "${REMOTE_DDS}" \
  "${_extra[@]}")"

echo "G1 remote DDS 已配置: G1_ROBOT_IP=${G1_ROBOT_IP} REMOTE_DDS=${REMOTE_DDS} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
