"""M1：蓝锚/红物体解耦 + AprilTag fail-fast + need_frames。"""

import cv2
import numpy as np
import pytest

from mission_brain.detect import (
    PRESET_ANCHOR_BLUE,
    PRESET_OBJECT_A_RED,
    RegionConfirmer,
    detect_marker,
)


def _blank(w=320, h=240, v=30):
    return np.full((h, w, 3), v, dtype=np.uint8)


def _paint_red(img, center, r=50):
    cv2.circle(img, center, r, (40, 40, 230), -1)


def _paint_blue(img, center, r=50):
    cv2.circle(img, center, r, (230, 80, 40), -1)  # BGR blue-ish


def test_detect_red_object_not_blue_anchor():
    img = _blank()
    _paint_red(img, (80, 120))
    assert detect_marker(img, PRESET_OBJECT_A_RED) is not None
    assert detect_marker(img, PRESET_ANCHOR_BLUE) is None


def test_detect_blue_anchor_not_red_object():
    img = _blank()
    _paint_blue(img, (240, 120))
    assert detect_marker(img, PRESET_ANCHOR_BLUE) is not None
    assert detect_marker(img, PRESET_OBJECT_A_RED) is None


def test_region_confirmer_needs_blue_and_streak():
    conf = RegionConfirmer(
        {"AX-01": "region_x"},
        need_frames=3,
        mode="color",
        color_anchor_id="AX-01",
    )
    red_only = _blank()
    _paint_red(red_only, (160, 120))
    for _ in range(5):
        assert conf.update(red_only) is None

    both = _blank()
    _paint_blue(both, (80, 120))
    _paint_red(both, (240, 120))
    assert conf.update(both) is None
    assert conf.update(both) is None
    hit = conf.update(both)
    assert hit is not None
    assert hit[0] == "region_x"
    assert hit[1].anchor_id == "AX-01"


def test_need_frames_param_honored():
    conf = RegionConfirmer(
        {"AX-01": "region_x"},
        need_frames=2,
        mode="color",
        color_anchor_id="AX-01",
    )
    both = _blank()
    _paint_blue(both, (80, 120))
    assert conf.update(both) is None
    assert conf.update(both) is not None


def test_apriltag_mode_fail_fast_without_library(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pupil_apriltags":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="apriltag"):
        RegionConfirmer({"TAG-0": "region_x"}, mode="apriltag", need_frames=1)


def test_apriltag_mode_with_injected_detector():
    class FakeTag:
        def __init__(self, tid):
            self.tag_id = tid
            self.decision_margin = 10.0

    class FakeDet:
        def detect(self, gray):
            return [FakeTag(0)]

    conf = RegionConfirmer(
        {"TAG-0": "region_x"},
        mode="apriltag",
        need_frames=1,
        detector=FakeDet(),
        allowed_tag_ids=[0],
    )
    hit = conf.update(_blank())
    assert hit is not None
    assert hit[1].anchor_id == "TAG-0"


def test_mode_auto_rejected():
    with pytest.raises(ValueError, match="color 或 apriltag"):
        RegionConfirmer({"AX-01": "r"}, mode="auto")
