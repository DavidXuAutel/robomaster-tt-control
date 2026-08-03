"""
ROS 2 Humble 相关「全局」默认值（与 ``scripts/env_ros2_humble.sh`` 对齐）。

在运行 ``run_g1_policy.py`` 等节点前，请在 **同一 shell** 中先执行::

    source /path/to/FastWAM/scripts/env_ros2_humble.sh

本模块仅同步 **当前进程** 中已由 shell 注入的环境变量；不会自动 source bash。
"""

from __future__ import annotations

import os

# --- 与 scripts/env_ros2_humble.sh 保持一致的缺省（可被 export 覆盖）---
ROS_DISTRO: str = os.environ.get("ROS_DISTRO", "humble")
ROS_HUMBLE_PREFIX: str = os.environ.get("ROS_HUMBLE_PREFIX", f"/opt/ros/{ROS_DISTRO}")
ROS_SETUP_BASH: str = os.path.join(ROS_HUMBLE_PREFIX, "setup.bash")

# 由 ``g1_remote_link`` / ``scripts/env_g1_robot.sh`` 在连接 G1 时设置（默认 10.229.66.60）
G1_ROBOT_IP: str | None = os.environ.get("G1_ROBOT_IP")

GENIE_ROS_WS: str | None = os.environ.get("GENIE_ROS_WS") or os.environ.get(
    "GENIE_ROS_WORKSPACE"
)

RMW_IMPLEMENTATION: str = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
ROS_DOMAIN_ID: str = os.environ.get("ROS_DOMAIN_ID", "0")

# AMENT_PREFIX_PATH 由 setup.bash 设置；未 source 时可能为空
AMENT_PREFIX_PATH: str | None = os.environ.get("AMENT_PREFIX_PATH")


def ros2_env_summary() -> str:
    """便于日志打印的一行摘要。"""
    return (
        f"ROS_DISTRO={ROS_DISTRO!r} ROS_HUMBLE_PREFIX={ROS_HUMBLE_PREFIX!r} "
        f"RMW_IMPLEMENTATION={RMW_IMPLEMENTATION!r} ROS_DOMAIN_ID={ROS_DOMAIN_ID!r} "
        f"G1_ROBOT_IP={G1_ROBOT_IP!r} GENIE_ROS_WS={GENIE_ROS_WS!r} "
        f"AMENT_PREFIX_PATH={'set' if AMENT_PREFIX_PATH else 'unset'}"
    )


def warn_if_ros_not_sourced() -> None:
    """若未 source Humble，提示用户（不抛异常）。"""
    if not AMENT_PREFIX_PATH:
        import warnings

        warnings.warn(
            "ROS 2 环境可能未加载：AMENT_PREFIX_PATH 为空。请先在同一 shell 中执行: "
            f"`source <FastWAM>/scripts/env_ros2_humble.sh`（期望存在 {ROS_SETUP_BASH}）。",
            stacklevel=2,
        )
