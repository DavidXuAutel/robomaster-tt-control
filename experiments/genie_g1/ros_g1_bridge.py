"""
ROS2 bridge for 智元精灵 G01 GDK v1.5.x topics (see GDK 使用指南).

Implements the minimal ``TASK_ENV`` surface used by
``WorldActionRobotWinPolicy.step`` from ``deploy_policy``:

- ``get_instruction() -> str``
- ``get_obs()`` -> RoboTwin-shaped dict (head / left / right RGB + joint vector)
- ``take_action(action, action_type="qpos")``

Default topic names match the GDK PDF (ROS2):

- ``/camera/head_color``, ``/camera/hand_left_color``, ``/camera/hand_right_color``
- ``/hal/arm_joint_state`` (14 positions: left 7 + right 7, radians)
- Command: ``/wbc/arm_command`` (``sensor_msgs/JointState``), with optional
  per-step delta clamp (GDK notes ~0.2618 rad max step vs feedback).
- Grippers: ``/wbc/left_ee_command``, ``/wbc/right_ee_command`` (``JointState``;
  position in mm, 0 open .. 120 closed for parallel grippers per GDK).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import rclpy  # type: ignore[import-not-found]
    from rclpy.node import Node  # type: ignore[import-not-found]
    from rclpy.qos import (  # type: ignore[import-not-found]
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image, JointState  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - ROS optional at dev time
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[misc, assignment]
    Image = JointState = None  # type: ignore[assignment]
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = None  # type: ignore[assignment]
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None


def _require_ros() -> None:
    if _ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 Python bindings are required for Genie G1 deployment. "
            "Install packages (e.g. Ubuntu: `sudo apt install ros-humble-rclpy ros-humble-sensor-msgs`) "
            "and from the FastWAM repo root run: `source scripts/env_ros2_humble.sh`. "
            "The bridge uses rclpy, sensor_msgs (Image, JointState), and rclpy.qos (QoSProfile)."
        ) from _ROS_IMPORT_ERROR


def image_msg_to_rgb(msg: Any) -> np.ndarray:
    """Convert sensor_msgs/Image to HxWx3 uint8 RGB."""
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = (msg.encoding or "").lower()
    if enc in ("rgb8",):
        arr = buf.reshape((h, w, 3))
    elif enc in ("bgr8",):
        arr = buf.reshape((h, w, 3))[:, :, ::-1].copy()
    elif enc in ("mono8",):
        g = buf.reshape((h, w))
        arr = np.stack([g, g, g], axis=-1)
    else:
        raise ValueError(
            f"Unsupported image encoding '{msg.encoding}' "
            f"(supported: rgb8, bgr8, mono8)."
        )
    return np.ascontiguousarray(arr)


class GenieG1TaskEnv(Node):  # type: ignore[misc, valid-type]
    """
    Subscribes to GDK camera + arm state, publishes ``/wbc/arm_command`` and
    end-effector ``JointState`` topics for grippers.
    """

    def __init__(
        self,
        *,
        instruction: str,
        topic_head_rgb: str = "/camera/head_color",
        topic_left_rgb: str = "/camera/hand_left_color",
        topic_right_rgb: str = "/camera/hand_right_color",
        topic_arm_joint_state: str = "/hal/arm_joint_state",
        topic_arm_command: str = "/wbc/arm_command",
        topic_left_ee_command: str = "/wbc/left_ee_command",
        topic_right_ee_command: str = "/wbc/right_ee_command",
        topic_left_ee_state: Optional[str] = "/hal/left_ee_data",
        topic_right_ee_state: Optional[str] = "/hal/right_ee_data",
        left_arm_dof: int = 7,
        right_arm_dof: int = 7,
        gripper_open_mm: float = 0.0,
        gripper_close_mm: float = 120.0,
        wbc_max_delta_rad: float = 0.25,
        arm_command_rate_hz: float = 50.0,
        qos_depth: int = 5,
    ) -> None:
        _require_ros()
        super().__init__("fastwam_genie_g1_bridge")
        self._lock = threading.Lock()
        self._instruction = str(instruction)
        self._left_arm_dof = int(left_arm_dof)
        self._right_arm_dof = int(right_arm_dof)
        self._gripper_open_mm = float(gripper_open_mm)
        self._gripper_close_mm = float(gripper_close_mm)
        self._wbc_max_delta_rad = float(wbc_max_delta_rad)
        self._arm_cmd_period = 1.0 / max(1e-3, float(arm_command_rate_hz))

        self._rgb: Dict[str, Optional[np.ndarray]] = {
            "head": None,
            "left": None,
            "right": None,
        }
        self._arm_q: Optional[np.ndarray] = None  # length left+right (no gripper)
        self._left_grip_mm: Optional[float] = None
        self._right_grip_mm: Optional[float] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(qos_depth),
        )

        self.create_subscription(Image, topic_head_rgb, self._cb_head, qos)
        self.create_subscription(Image, topic_left_rgb, self._cb_left, qos)
        self.create_subscription(Image, topic_right_rgb, self._cb_right, qos)
        self.create_subscription(JointState, topic_arm_joint_state, self._cb_arm, qos)

        self._pub_arm = self.create_publisher(JointState, topic_arm_command, 10)
        self._pub_left_ee = self.create_publisher(JointState, topic_left_ee_command, 10)
        self._pub_right_ee = self.create_publisher(JointState, topic_right_ee_command, 10)

        self._ee_type = None
        if topic_left_ee_state:
            self._try_subscribe_ee_state(topic_left_ee_state, "left")
        if topic_right_ee_state:
            self._try_subscribe_ee_state(topic_right_ee_state, "right")

        self.get_logger().info(
            "GenieG1TaskEnv | head=%s left=%s right=%s arm_state=%s arm_cmd=%s",
            topic_head_rgb,
            topic_left_rgb,
            topic_right_rgb,
            topic_arm_joint_state,
            topic_arm_command,
        )

    def _try_subscribe_ee_state(self, topic: str, side: str) -> None:
        """Subscribe to genie_msgs/EndState if the message class exists."""
        try:
            from genie_msgs.msg import EndState  # type: ignore
        except Exception:
            self.get_logger().warning(
                "genie_msgs not found; %s gripper feedback will use last command until "
                "messages arrive. Install robot ROS overlay if needed.",
                side,
            )
            return

        def _cb(msg: Any, s: str = side) -> None:
            pos = self._extract_ee_position_mm(msg)
            if pos is None:
                return
            with self._lock:
                if s == "left":
                    self._left_grip_mm = pos
                else:
                    self._right_grip_mm = pos

        self.create_subscription(EndState, topic, _cb, 10)
        self._ee_type = EndState

    @staticmethod
    def _extract_ee_position_mm(msg: Any) -> Optional[float]:
        if hasattr(msg, "position") and msg.position is not None:
            try:
                seq = list(msg.position)
                if len(seq) > 0:
                    return float(seq[0])
            except (TypeError, ValueError, IndexError):
                pass
        return None

    def _cb_head(self, msg: Any) -> None:
        try:
            img = image_msg_to_rgb(msg)
        except ValueError:
            return
        with self._lock:
            self._rgb["head"] = img

    def _cb_left(self, msg: Any) -> None:
        try:
            img = image_msg_to_rgb(msg)
        except ValueError:
            return
        with self._lock:
            self._rgb["left"] = img

    def _cb_right(self, msg: Any) -> None:
        try:
            img = image_msg_to_rgb(msg)
        except ValueError:
            return
        with self._lock:
            self._rgb["right"] = img

    def _cb_arm(self, msg: Any) -> None:
        if not msg.position:
            return
        with self._lock:
            self._arm_q = np.asarray(msg.position, dtype=np.float64)

    def wait_for_observation(self, timeout_sec: float = 30.0) -> None:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self._lock:
                ok = (
                    self._rgb["head"] is not None
                    and self._rgb["left"] is not None
                    and self._rgb["right"] is not None
                    and self._arm_q is not None
                    and self._arm_q.size >= self._left_arm_dof + self._right_arm_dof
                )
            if ok:
                return
        raise TimeoutError(
            f"No full observation within {timeout_sec}s "
            "(need head/left/right RGB and arm joint state)."
        )

    def get_instruction(self) -> str:
        return self._instruction

    def get_obs(self) -> Dict[str, Any]:
        with self._lock:
            if (
                self._rgb["head"] is None
                or self._rgb["left"] is None
                or self._rgb["right"] is None
                or self._arm_q is None
            ):
                raise RuntimeError("Incomplete sensor streams; call wait_for_observation() first.")
            head = self._rgb["head"].copy()
            left = self._rgb["left"].copy()
            right = self._rgb["right"].copy()
            arm = self._arm_q.astype(np.float32).copy()

        la = self._left_arm_dof
        ra = self._right_arm_dof
        if arm.size < la + ra:
            raise RuntimeError(
                f"arm_joint_state.position too short: got {arm.size}, need >= {la + ra}"
            )
        left_arm = arm[:la]
        right_arm = arm[la : la + ra]

        # Gripper channels for policy vector: normalized [0,1] closedness (matches common RoboTwin-style datasets).
        lo, hi = self._gripper_open_mm, self._gripper_close_mm
        span = max(hi - lo, 1e-6)

        def _norm_grip(mm: Optional[float], fallback: float) -> float:
            if mm is None:
                return float(fallback)
            g = (float(mm) - lo) / span
            return float(np.clip(g, 0.0, 1.0))

        with self._lock:
            lg_mm, rg_mm = self._left_grip_mm, self._right_grip_mm

        left_g = _norm_grip(lg_mm, 0.0)
        right_g = _norm_grip(rg_mm, 0.0)

        vec = np.concatenate(
            [
                left_arm.astype(np.float32),
                np.asarray([left_g], dtype=np.float32),
                right_arm.astype(np.float32),
                np.asarray([right_g], dtype=np.float32),
            ],
            axis=0,
        )

        return {
            "observation": {
                "head_camera": {"rgb": head},
                "left_camera": {"rgb": left},
                "right_camera": {"rgb": right},
            },
            "joint_action": {"vector": vec},
        }

    def _clip_arm_targets(self, target14: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._arm_q is None or self._arm_q.size < target14.size:
                return target14
            cur = self._arm_q.astype(np.float64)[: target14.size].copy()
        delta = target14.astype(np.float64) - cur
        max_d = self._wbc_max_delta_rad
        delta = np.clip(delta, -max_d, max_d)
        return (cur + delta).astype(np.float32)

    def take_action(self, action: np.ndarray, action_type: str = "qpos") -> None:
        if action_type != "qpos":
            raise ValueError("GenieG1TaskEnv only supports action_type='qpos'.")

        vec = np.asarray(action, dtype=np.float32).ravel()
        la, ra = self._left_arm_dof, self._right_arm_dof
        expected = la + 1 + ra + 1
        if vec.size != expected:
            raise ValueError(
                f"Action dim mismatch: model produced {vec.size}, "
                f"expected {expected} (= left_arm_dof+1 + right_arm_dof+1). "
                "Match training embodiment / checkpoint."
            )

        left_arm = vec[:la]
        left_g = float(vec[la])
        right_arm = vec[la + 1 : la + 1 + ra]
        right_g = float(vec[la + 1 + ra])

        lo, hi = self._gripper_open_mm, self._gripper_close_mm
        left_mm = lo + float(np.clip(left_g, 0.0, 1.0)) * (hi - lo)
        right_mm = lo + float(np.clip(right_g, 0.0, 1.0)) * (hi - lo)

        arm_targets = np.concatenate([left_arm.astype(np.float64), right_arm.astype(np.float64)], axis=0)
        with self._lock:
            clipped = self._clip_arm_targets(arm_targets)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.position = [float(x) for x in clipped.tolist()]
        self._pub_arm.publish(js)

        jl = JointState()
        jl.header.stamp = js.header.stamp
        jl.position = [left_mm]
        self._pub_left_ee.publish(jl)

        jr = JointState()
        jr.header.stamp = js.header.stamp
        jr.position = [right_mm]
        self._pub_right_ee.publish(jr)

        with self._lock:
            self._left_grip_mm = left_mm
            self._right_grip_mm = right_mm
            if self._arm_q is not None and self._arm_q.size >= clipped.size:
                self._arm_q[: clipped.size] = clipped.astype(np.float64)

        time.sleep(self._arm_cmd_period)
