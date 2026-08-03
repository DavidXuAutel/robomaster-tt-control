#!/usr/bin/env python3
"""
Run FastWAM on 智元精灵 G01 over ROS2 (GDK topics documented in GDK v1.5.x PDF).

Prerequisites
--------------
- ROS 2 Humble 环境（推荐）: ``source scripts/env_ros2_humble.sh``（见仓库根目录）。
  或手动 ``source /opt/ros/humble/setup.bash``。
- 可选: ``export GENIE_ROS_WS=/path/to/genie_ws`` 再 source 同上脚本，叠加 ``genie_msgs`` 等。
- FastWAM Python 环境（torch / hydra）。

Example
-------
.. code-block:: bash

   cd /path/to/FastWAM
   source scripts/env_ros2_humble.sh
   # 与 G1（默认 10.229.66.60）跨机发现 DDS（与机器人同 ROS_DOMAIN_ID、同 RMW）
   source scripts/env_g1_robot.sh
   python experiments/genie_g1/run_g1_policy.py \\
     --ckpt /path/to/model.pt \\
     --dataset-stats /path/to/dataset_stats.json \\
     --instruction "pick up the cup" \\
     --sim-cfg-name sim_robotwin.yaml \\
     --g1-ip 10.229.66.60
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_g1_remote_link():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "g1_remote_link.py"
    spec = importlib.util.spec_from_file_location("fastwam_g1_remote_link", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load g1_remote_link from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ros2_humble_env():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "ros2_humble_env.py"
    spec = importlib.util.spec_from_file_location("fastwam_ros2_humble_env", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ros2_humble_env from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ros_g1_bridge():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "ros_g1_bridge.py"
    spec = importlib.util.spec_from_file_location("fastwam_ros_g1_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ros_g1_bridge from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_deploy_policy():
    path = PROJECT_ROOT / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py"
    spec = importlib.util.spec_from_file_location("fastwam_deploy_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load deploy_policy from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _optional_ros_topic(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastWAM deployment on Genie G01 (GDK ROS2).")
    p.add_argument("--ckpt", required=True, type=str, help="FastWAM checkpoint path (ckpt_setting).")
    p.add_argument(
        "--dataset-stats",
        required=True,
        type=str,
        help="dataset_stats.json matching training (same normalizer as checkpoint).",
    )
    p.add_argument(
        "--sim-cfg-name",
        default="sim_robotwin.yaml",
        help="Hydra config name under FastWAM/configs (default: sim_robotwin.yaml).",
    )
    p.add_argument(
        "--instruction",
        default="",
        help="Language instruction for the policy (empty uses DEFAULT_PROMPT formatting only).",
    )
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: from cfg).")
    p.add_argument("--mixed-precision", default=None, choices=("no", "fp16", "bf16"))
    p.add_argument("--replan-steps", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--action-horizon", type=int, default=None)

    p.add_argument("--left-arm-dof", type=int, default=7, help="Must match training action layout.")
    p.add_argument("--right-arm-dof", type=int, default=7, help="Must match training action layout.")
    p.add_argument("--gripper-open-mm", type=float, default=0.0)
    p.add_argument("--gripper-close-mm", type=float, default=120.0)
    p.add_argument("--wbc-max-delta-rad", type=float, default=0.25, help="Clamp vs /hal feedback (GDK ~0.2618).")
    p.add_argument("--arm-cmd-rate-hz", type=float, default=50.0, help="Sleep between arm commands.")

    p.add_argument("--topic-head-rgb", default="/camera/head_color")
    p.add_argument("--topic-left-rgb", default="/camera/hand_left_color")
    p.add_argument("--topic-right-rgb", default="/camera/hand_right_color")
    p.add_argument("--topic-arm-state", default="/hal/arm_joint_state")
    p.add_argument("--topic-arm-cmd", default="/wbc/arm_command")
    p.add_argument("--topic-left-ee-cmd", default="/wbc/left_ee_command")
    p.add_argument("--topic-right-ee-cmd", default="/wbc/right_ee_command")
    p.add_argument("--topic-left-ee-state", default="/hal/left_ee_data")
    p.add_argument("--topic-right-ee-state", default="/hal/right_ee_data")

    p.add_argument("--max-steps", type=int, default=10_000, help="Safety cap on policy steps.")
    p.add_argument("--spin-timeout-sec", type=float, default=30.0)
    p.add_argument(
        "--enter-servo-mode",
        action="store_true",
        help="Publish /wbc/set_control_mode MODE_SERVO once at start (GDK input_type=54).",
    )
    p.add_argument(
        "--g1-ip",
        default="10.229.66.60",
        help="G1 机器人 IP；用于生成 DDS 静态发现配置（默认 10.229.66.60）。",
    )
    p.add_argument(
        "--no-g1-remote",
        action="store_true",
        help="不配置远程 DDS（仅本机回环/已有组播环境时使用）。",
    )
    p.add_argument(
        "--remote-dds",
        choices=("fastrtps", "cyclonedds"),
        default="fastrtps",
        help="与机器人侧 RMW 一致；Humble 默认多为 fastrtps。",
    )
    p.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help="覆盖 ROS_DOMAIN_ID（须与 G1 上相同）；默认读环境变量或 0。",
    )
    return p.parse_args()


def _maybe_enter_servo_mode() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from genie_msgs.msg import SetControlMode  # type: ignore
    except Exception as exc:
        print("Skipping --enter-servo-mode (import failed):", exc, file=sys.stderr)
        return

    rclpy.init(args=sys.argv)
    node = Node("fastwam_set_servo_once")
    pub = node.create_publisher(SetControlMode, "/wbc/set_control_mode", 10)
    msg = SetControlMode()
    msg.header.frame_id = ""
    msg.input_type = 54
    msg.control_mode = 1  # MODE_SERVO
    time.sleep(0.5)
    pub.publish(msg)
    node.destroy_node()
    rclpy.shutdown()
    print("Published /wbc/set_control_mode MODE_SERVO (input_type=54).")


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    link = _load_g1_remote_link()
    if not args.no_g1_remote:
        ip = str(args.g1_ip or "").strip()
        if ip.lower() not in {"", "none", "local"}:
            applied = link.apply_remote_ros_env(
                ip,
                domain_id=args.ros_domain_id,
                dds=args.remote_dds,
            )
            print(link.summarize_connection(ip, applied), flush=True)

    if args.enter_servo_mode:
        _maybe_enter_servo_mode()

    dp = _load_deploy_policy()
    bridge_mod = _load_ros_g1_bridge()
    GenieG1TaskEnv = bridge_mod.GenieG1TaskEnv

    usr = {
        "ckpt_setting": args.ckpt,
        "dataset_stats_path": args.dataset_stats,
        "sim_cfg_name": args.sim_cfg_name,
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "replan_steps": args.replan_steps,
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": args.action_horizon,
    }
    usr = {k: v for k, v in usr.items() if v is not None}

    ros_env = _load_ros2_humble_env()
    ros_env.warn_if_ros_not_sourced()
    print(ros_env.ros2_env_summary(), flush=True)

    import rclpy

    rclpy.init(args=sys.argv)
    env = GenieG1TaskEnv(
        instruction=args.instruction or " ",
        topic_head_rgb=args.topic_head_rgb,
        topic_left_rgb=args.topic_left_rgb,
        topic_right_rgb=args.topic_right_rgb,
        topic_arm_joint_state=args.topic_arm_state,
        topic_arm_command=args.topic_arm_cmd,
        topic_left_ee_command=args.topic_left_ee_cmd,
        topic_right_ee_command=args.topic_right_ee_cmd,
        topic_left_ee_state=_optional_ros_topic(args.topic_left_ee_state),
        topic_right_ee_state=_optional_ros_topic(args.topic_right_ee_state),
        left_arm_dof=args.left_arm_dof,
        right_arm_dof=args.right_arm_dof,
        gripper_open_mm=args.gripper_open_mm,
        gripper_close_mm=args.gripper_close_mm,
        wbc_max_delta_rad=args.wbc_max_delta_rad,
        arm_command_rate_hz=args.arm_cmd_rate_hz,
    )

    try:
        print("Waiting for cameras + /hal/arm_joint_state ...")
        env.wait_for_observation(timeout_sec=args.spin_timeout_sec)
        policy = dp.get_model(usr)
        dp.reset_model(policy)
        print("Running policy loop. Ctrl+C to stop.")

        for _ in range(int(args.max_steps)):
            rclpy.spin_once(env, timeout_sec=0.0)
            need_obs = True
            if hasattr(policy, "should_request_observation"):
                need_obs = bool(policy.should_request_observation())
            obs = env.get_obs() if need_obs else None
            dp.eval(env, policy, obs)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
