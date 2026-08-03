"""侦察侧轻量检测：颜色标记物（无新依赖）+ 显式 AprilTag。

锚点与物体 A 解耦。mode=apriltag 缺 pupil-apriltags 时 fail-fast，禁止静默回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


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


@dataclass(frozen=True)
class MarkerSpec:
    label: str
    kind: str = "color"  # color | apriltag
    hsv_ranges: Tuple[HsvRange, ...] = ()
    tag_ids: Tuple[int, ...] = ()
    min_area_ratio: float = 1e-4
    min_confidence: float = 0.55


# 默认演示约定：物体 A = 红；区域锚点（单点）= 蓝
PRESET_OBJECT_A_RED = MarkerSpec(
    label="object_a",
    kind="color",
    hsv_ranges=(
        ((0, 90, 90), (10, 255, 255)),
        ((170, 90, 90), (180, 255, 255)),
    ),
    min_area_ratio=1e-4,
    min_confidence=0.55,
)

PRESET_ANCHOR_BLUE = MarkerSpec(
    label="anchor_blue",
    kind="color",
    hsv_ranges=(((100, 90, 90), (130, 255, 255)),),
    min_area_ratio=5e-4,
    min_confidence=0.55,
)


class AprilTagDetector(Protocol):
    def detect(self, gray: np.ndarray) -> Sequence[Any]: ...


def detect_marker(frame: np.ndarray, spec: MarkerSpec) -> Optional[Detection]:
    if spec.kind == "color":
        return _detect_color(frame, spec)
    if spec.kind == "apriltag":
        raise ValueError("detect_marker(kind=apriltag) 请走 RegionConfirmer + detector")
    raise ValueError(f"unknown MarkerSpec.kind: {spec.kind}")


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
    """兼容旧调用：默认红色块。新代码请用 detect_marker + MarkerSpec。"""
    spec = MarkerSpec(
        label=label,
        kind="color",
        hsv_ranges=((hsv_low1, hsv_high1), (hsv_low2, hsv_high2)),
        min_area_ratio=min_area_ratio,
    )
    return detect_marker(frame, spec)


def _detect_color(frame: np.ndarray, spec: MarkerSpec) -> Optional[Detection]:
    if not spec.hsv_ranges:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for low, high in spec.hsv_ranges:
        m = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    assert mask is not None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    h, w = frame.shape[:2]
    ratio = area / float(h * w)
    if ratio < spec.min_area_ratio:
        return None
    mnt = cv2.moments(c)
    if mnt["m00"] == 0:
        return None
    cx = float(mnt["m10"] / mnt["m00"])
    cy = float(mnt["m01"] / mnt["m00"])
    conf = float(min(1.0, 0.55 + ratio / 0.05))
    if conf < spec.min_confidence:
        return None
    return Detection(
        label=spec.label, confidence=conf, cx=cx, cy=cy, area_ratio=ratio
    )


def _require_apriltag_detector(detector: Optional[AprilTagDetector]) -> AprilTagDetector:
    if detector is not None:
        return detector
    try:
        from pupil_apriltags import Detector  # type: ignore

        return Detector(families="tag36h11")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "mode=apriltag 需要 pupil-apriltags，或注入 detector；禁止静默回退 color"
        ) from exc


class RegionConfirmer:
    """连续 N 帧看到同一锚点才确认区域。

    - color：用 color_spec（默认蓝）检出后映射为 color_anchor_id
    - apriltag：Tag ID → TAG-{n}；缺库且无注入 detector → RuntimeError
    """

    def __init__(
        self,
        anchor_to_region: Dict[str, str],
        *,
        need_frames: int = 3,
        mode: str = "color",
        color_anchor_id: str = "AX-01",
        color_spec: MarkerSpec = PRESET_ANCHOR_BLUE,
        detector: Optional[AprilTagDetector] = None,
        allowed_tag_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if mode not in ("color", "apriltag"):
            raise ValueError("mode 只能是 color 或 apriltag（禁止 auto）")
        self.anchor_to_region = dict(anchor_to_region)
        self.need_frames = int(need_frames)
        self.mode = mode
        self.color_anchor_id = color_anchor_id
        self.color_spec = color_spec
        self._streak_id: Optional[str] = None
        self._streak_n = 0
        self._allowed_tag_ids = (
            set(int(x) for x in allowed_tag_ids) if allowed_tag_ids is not None else None
        )
        self._apriltag: Optional[AprilTagDetector] = None
        if mode == "apriltag":
            self._apriltag = _require_apriltag_detector(detector)

    def reset(self) -> None:
        self._streak_id = None
        self._streak_n = 0

    def update(self, frame: np.ndarray) -> Optional[Tuple[str, AnchorHit]]:
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
        if self.mode == "apriltag":
            assert self._apriltag is not None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = list(self._apriltag.detect(gray))
            if not tags:
                return None
            tag = max(tags, key=lambda t: getattr(t, "decision_margin", 0.0))
            tid = int(tag.tag_id)
            if self._allowed_tag_ids is not None and tid not in self._allowed_tag_ids:
                return None
            return AnchorHit(anchor_id=f"TAG-{tid}", age_ms=0)
        blob = detect_marker(frame, self.color_spec)
        if blob is None:
            return None
        return AnchorHit(anchor_id=self.color_anchor_id, age_ms=0)
