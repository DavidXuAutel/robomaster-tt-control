"""真库 pupil-apriltags 路径（非 FakeDet）。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mission_brain.detect import RegionConfirmer

FIX = Path(__file__).resolve().parent / "fixtures" / "mission" / "apriltags"


def _scaled_tag(path: Path, canvas: int = 400, tag_px: int = 200) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert g is not None
    big = cv2.resize(g, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)
    out = np.full((canvas, canvas), 255, dtype=np.uint8)
    y0 = (canvas - tag_px) // 2
    out[y0 : y0 + tag_px, y0 : y0 + tag_px] = big
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


@pytest.mark.skipif(
    not FIX.joinpath("tag36h11_00000.png").is_file(),
    reason="apriltag fixture missing",
)
def test_real_apriltag_confirms_region():
    pytest.importorskip("pupil_apriltags")
    frame = _scaled_tag(FIX / "tag36h11_00000.png")
    conf = RegionConfirmer(
        {"TAG-0": "region_x"},
        mode="apriltag",
        need_frames=2,
        allowed_tag_ids=[0],
    )
    assert conf.update(frame) is None
    hit = conf.update(frame)
    assert hit is not None
    assert hit[0] == "region_x"
    assert hit[1].anchor_id == "TAG-0"


@pytest.mark.skipif(
    not FIX.joinpath("tag36h11_00001.png").is_file(),
    reason="apriltag fixture missing",
)
def test_real_apriltag_wrong_id_no_region():
    pytest.importorskip("pupil_apriltags")
    frame = _scaled_tag(FIX / "tag36h11_00001.png")
    conf = RegionConfirmer(
        {"TAG-0": "region_x"},
        mode="apriltag",
        need_frames=1,
        allowed_tag_ids=[0],
    )
    assert conf.update(frame) is None
