"""自动控制策略接缝(Phase 2 骨架)。

Policy.decide(frame, state) -> RcAxes：输入一帧图像 + 飞机状态，输出杆量。
全 0 视为悬停。App 在 AUTO 挂载且在飞时按固定频率调用它并下发 rc。

- MockAvoidPolicy   : 纯规则视觉避障，仅用于离线测通闭环。
- ScriptedPolicy    : 按时间走固定航线，用于可复现的 Phase 1 式飞行。
- ExternalModelPolicy: 预留给"别人的现成模型"，到手后在此实现 adapter。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tt_control.control import RcAxes

logger = logging.getLogger(__name__)


class Policy(ABC):
    name = "policy"

    @abstractmethod
    def decide(self, frame: np.ndarray, state: dict) -> RcAxes:
        """输入 BGR 帧 + 状态 dict，返回 RcAxes(roll,pitch,throttle,yaw)。"""

    def reset(self) -> None:
        """重新挂载 AUTO 时调用，清理内部计时/状态。"""


def _find_red_blob(frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
    """返回画面中最大红色块的 (cx, cy, area_ratio)，无则 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = frame.shape[:2]
    ratio = area / float(h * w)
    if ratio < 1e-4:
        return None
    mnt = cv2.moments(c)
    if mnt["m00"] == 0:
        return None
    cx = int(mnt["m10"] / mnt["m00"])
    cy = int(mnt["m01"] / mnt["m00"])
    return cx, cy, ratio


class MockAvoidPolicy(Policy):
    """规则避障:看到大的红色障碍就偏航+横滚绕开，否则缓慢前进。

    仅用于离线测通"帧→策略→杆量→下发→机体运动→孪生记录"这条闭环，
    不是真正的避障算法。
    """

    name = "mock"
    CRUISE = 25      # 无障碍时前进杆量
    AVOID = 40       # 避障杆量
    NEAR = 0.045     # 障碍面积比 > NEAR 视为逼近

    def decide(self, frame: np.ndarray, state: dict) -> RcAxes:
        blob = _find_red_blob(frame)
        if blob is None:
            return RcAxes(pitch=self.CRUISE)
        cx, _cy, ratio = blob
        w = frame.shape[1]
        left = cx < w / 2
        if ratio >= self.NEAR:
            # 逼近:向障碍反方向横滚 + 偏航，略微后撤
            sign = 1 if left else -1  # 障碍在左 → 向右(+)
            return RcAxes(roll=sign * self.AVOID, pitch=-10, yaw=sign * self.AVOID)
        # 远处:边前进边微调朝反方向
        sign = 1 if left else -1
        return RcAxes(roll=sign * (self.AVOID // 2), pitch=self.CRUISE)


class ScriptedPolicy(Policy):
    """按 [(时长秒, RcAxes), ...] 顺序走固定航线；跑完后悬停。"""

    name = "scripted"

    DEFAULT_PLAN: Sequence[Tuple[float, RcAxes]] = (
        (2.0, RcAxes(pitch=30)),                 # 前进
        (2.0, RcAxes(yaw=40)),                   # 原地右转
        (2.0, RcAxes(roll=30)),                  # 右移
        (2.0, RcAxes(pitch=-30)),                # 后退
        (1.5, RcAxes(throttle=30)),              # 上升
    )

    def __init__(self, plan: Optional[Sequence[Tuple[float, RcAxes]]] = None) -> None:
        self._plan: List[Tuple[float, RcAxes]] = list(plan or self.DEFAULT_PLAN)
        self._t0: Optional[float] = None

    def reset(self) -> None:
        self._t0 = None

    def decide(self, frame: np.ndarray, state: dict) -> RcAxes:
        if self._t0 is None:
            self._t0 = time.time()
        elapsed = time.time() - self._t0
        acc = 0.0
        for dur, axes in self._plan:
            acc += dur
            if elapsed < acc:
                return axes
        return RcAxes()  # 航线走完 → 悬停


class ExternalModelPolicy(Policy):
    """预留:接入"别人的现成模型"。

    到手后按其真实 I/O 在 decide 里实现:
        1. 预处理 frame(缩放/归一化/BGR->RGB 等)
        2. 组织模型输入(可能还需 state 里的位姿/速度)
        3. 调用模型推理(import 调用 / HTTP 服务 / ROS 话题)
        4. 把模型输出(检测框/航点/速度)映射为 RcAxes
    如果是分类/检测模型,可复用上面 _find_red_blob 的"居中程度→避障杆量"映射思路。
    """

    name = "external"

    def __init__(self, model_ref: str = "") -> None:
        self.model_ref = model_ref

    def decide(self, frame: np.ndarray, state: dict) -> RcAxes:
        raise NotImplementedError(
            "ExternalModelPolicy 尚未接入模型。拿到大众的模型 I/O 后在此实现 adapter。"
        )


def create_policy(name: str = "mock") -> Policy:
    name = (name or "mock").lower()
    if name in ("mock", "avoid"):
        return MockAvoidPolicy()
    if name in ("scripted", "script"):
        return ScriptedPolicy()
    if name in ("external", "model"):
        return ExternalModelPolicy()
    raise ValueError(f"未知策略: {name}（可选 mock / scripted / external）")
