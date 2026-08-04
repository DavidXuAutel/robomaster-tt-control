"""M5：SharedMap 启动校验。"""

import pytest

from mission_brain.map_model import SharedMap


def test_example_map_loads():
    m = SharedMap.load("configs/mission/shared_map.example.json")
    m.validate()
    assert m.frame == "dog_map"


def test_missing_frame_rejected():
    with pytest.raises(ValueError, match="frame"):
        SharedMap.from_dict({"version": 1, "regions": {}})


def test_wrong_frame_rejected():
    with pytest.raises(ValueError, match="dog_map"):
        SharedMap.from_dict({"version": 1, "frame": "utm", "regions": {}})


def test_key_mismatch_rejected():
    with pytest.raises(ValueError, match="不一致"):
        SharedMap.from_dict(
            {
                "version": 1,
                "frame": "dog_map",
                "regions": {
                    "region_x": {
                        "region_id": "region_y",
                        "dog_goal_id": "wp",
                        "drone_route_id": "r",
                        "anchor_ids": ["A1"],
                    }
                },
            }
        )


def test_duplicate_anchor_rejected():
    with pytest.raises(ValueError, match="重复"):
        SharedMap.from_dict(
            {
                "version": 1,
                "frame": "dog_map",
                "regions": {
                    "region_x": {
                        "region_id": "region_x",
                        "dog_goal_id": "wp1",
                        "drone_route_id": "r1",
                        "anchor_ids": ["AX-01"],
                    },
                    "region_y": {
                        "region_id": "region_y",
                        "dog_goal_id": "wp2",
                        "drone_route_id": "r2",
                        "anchor_ids": ["AX-01"],
                    },
                },
            }
        )


def test_empty_anchors_rejected():
    with pytest.raises(ValueError, match="anchor_ids"):
        SharedMap.from_dict(
            {
                "version": 1,
                "frame": "dog_map",
                "regions": {
                    "region_x": {
                        "region_id": "region_x",
                        "dog_goal_id": "wp",
                        "drone_route_id": "r",
                        "anchor_ids": [],
                    }
                },
            }
        )
