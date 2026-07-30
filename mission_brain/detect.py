"""侦察侧轻量检测：颜色标记物（无新依赖）+ 可选 AprilTag。

AprilTag 需额外安装 pupil-apriltags；未安装时 RegionConfirmer 走颜色锚点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    cx: float
    cy: float
    area_ratio: float


@dataclass(frozen=True)
class AnchorHit:
    anchor_id: str
    age_ms: int = 0


def find_color_blob(
    frame: np.ndarray,
    *,
    label: str = "object_a",
    hsv_low1: Tuple[int, int, int] = (0, 90, 90),
    hsv_high1: Tuple[int, int, int] = (10, 255, 255),
    hsv_low2: Tuple[int, int, int] = (170, 90, 90),
    hsv_high2: Tuple[int, int, int] = (180, 255, 255),
    min_area_ratio: float = 1e-4,
) -> Optional[Detection]:
    """HSV 红色块检测（演示用物体 A / 锚点）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, hsv_low1, hsv_high1)
    m2 = cv2.inRange(hsv, hsv_low2, hsv_high2)
    mask = cv2.bitwise_or(m1, m2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    h, w = frame.shape[:2]
    ratio = area / float(h * w)
    if ratio < min_area_ratio:
        return None
    mnt = cv2.moments(c)
    if mnt["m00"] == 0:
        return None
    cx = float(mnt["m10"] / mnt["m00"])
    cy = float(mnt["m01"] / mnt["m00"])
    # 面积越大置信度越高，软封顶；演示阈值下至少 0.55 便于过 Brain 门
    conf = float(min(1.0, 0.55 + ratio / 0.05))
    return Detection(label=label, confidence=conf, cx=cx, cy=cy, area_ratio=ratio)


class RegionConfirmer:
    """连续 N 帧看到同一锚点才确认区域。

    - color 模式：anchor_id 由颜色标签映射（如 red→AX-01）
    - apriltag 模式：有 pupil_apriltags 时用 Tag ID
    """

    def __init__(
        self,
        anchor_to_region: Dict[str, str],
        *,
        need_frames: int = 3,
        mode: str = "color",
        color_anchor_id: str = "AX-01",
    ) -> None:
        self.anchor_to_region = dict(anchor_to_region)
        self.need_frames = int(need_frames)
        self.mode = mode
        self.color_anchor_id = color_anchor_id
        self._streak_id: Optional[str] = None
        self._streak_n = 0
        self._apriltag = None
        if mode == "apriltag":
            try:
                from pupil_apriltags import Detector  # type: ignore

                self._apriltag = Detector(families="tag36h11")
            except Exception:
                self._apriltag = None
                self.mode = "color"

    def reset(self) -> None:
        self._streak_id = None
        self._streak_n = 0

    def update(self, frame: np.ndarray) -> Optional[Tuple[str, AnchorHit]]:
        """返回 (region_id, AnchorHit) 或 None。"""
        hit = self._detect_anchor(frame)
        if hit is None:
            self._streak_id = None
            self._streak_n = 0
            return None
        if hit.anchor_id == self._streak_id:
            self._streak_n += 1
        else:
            self._streak_id = hit.anchor_id
            self._streak_n = 1
        if self._streak_n < self.need_frames:
            return None
        region_id = self.anchor_to_region.get(hit.anchor_id)
        if region_id is None:
            return None
        return region_id, hit

    def _detect_anchor(self, frame: np.ndarray) -> Optional[AnchorHit]:
        if self.mode == "apriltag" and self._apriltag is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self._apriltag.detect(gray)
            if not tags:
                return None
            tag = max(tags, key=lambda t: t.decision_margin)
            aid = f"TAG-{int(tag.tag_id)}"
            return AnchorHit(anchor_id=aid, age_ms=0)
        # color fallback：画面有足够大红块即视为当前 color_anchor_id
        blob = find_color_blob(frame, label="anchor", min_area_ratio=5e-4)
        if blob is None:
            return None
        return AnchorHit(anchor_id=self.color_anchor_id, age_ms=0)
