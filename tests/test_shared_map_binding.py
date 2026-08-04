"""M8：SharedMap v2 的 platform_binding（方案 §5.2）。

要防的具体事故：平台 pointsId 形如 `快速打点-1785465994716`，重建图或重打点
后会失效。如果直接把它写进 region 配置，失效后软件层完全看不见 —— 派单
「成功」，狗不动，或者更糟：走到另一个点。
"""

import json

import pytest

from mission_brain.map_model import GoalBinding, PlatformBinding, SharedMap

V2 = {
    "version": 2,
    "frame": "dog_map",
    "regions": {
        "region_x": {
            "region_id": "region_x",
            "dog_goal_id": "wp_region_x_staging",
            "drone_route_id": "route_region_x_scout",
            "anchor_ids": ["AX-01"],
        }
    },
    "platform_binding": {
        "map_id": "map_demo_01",
        "map_version": "3",
        "exported_at": "2026-08-03T14:00:00+08:00",
        "goals": {
            "wp_region_x_staging": {
                "points_id": "快速打点-1785465994716",
                "points_name": "演示区X待机点",
                "x": 12.5,
                "y": -3.25,
                "th": 1.57,
            }
        },
    },
}


def test_v1_map_still_loads_without_binding():
    m = SharedMap.load("configs/mission/shared_map.example.json")
    assert m.platform_binding is None
    # v1 语义：标签即 ID，行为不变
    assert m.resolve_points_id("wp_region_x_staging") == "wp_region_x_staging"
    assert m.goal_pose("wp_region_x_staging") is None


def test_v1_map_rejects_binding_validation():
    m = SharedMap.load("configs/mission/shared_map.example.json")
    with pytest.raises(ValueError, match="platform_binding"):
        m.validate_binding()


def test_v2_resolves_points_id_and_pose():
    m = SharedMap.from_dict(V2)
    m.validate_binding(map_id="map_demo_01")
    assert m.resolve_points_id("wp_region_x_staging") == "快速打点-1785465994716"
    assert m.goal_pose("wp_region_x_staging") == (12.5, -3.25)


def test_dog_goal_label_accepted_as_alias():
    data = json.loads(json.dumps(V2))
    row = data["regions"]["region_x"]
    row.pop("dog_goal_id")
    row["dog_goal_label"] = "wp_region_x_staging"
    m = SharedMap.from_dict(data)
    assert m.get("region_x").dog_goal_id == "wp_region_x_staging"
    m.validate_binding()


def test_missing_binding_for_region_rejected():
    data = json.loads(json.dumps(V2))
    data["platform_binding"]["goals"] = {}
    m = SharedMap.from_dict(data)
    with pytest.raises(ValueError, match="缺少平台绑定"):
        m.validate_binding()
    assert m.resolve_points_id("wp_region_x_staging") is None


def test_map_id_mismatch_rejected():
    """地图换了而绑定没重导，比缺绑定更危险——会派单到错误的点。"""
    m = SharedMap.from_dict(V2)
    with pytest.raises(ValueError, match="重新导出绑定"):
        m.validate_binding(map_id="map_demo_02")


def test_duplicate_points_id_rejected():
    data = json.loads(json.dumps(V2))
    data["platform_binding"]["goals"]["wp_other"] = {
        "points_id": "快速打点-1785465994716"
    }
    with pytest.raises(ValueError, match="重复绑定"):
        SharedMap.from_dict(data)


def test_binding_without_points_id_rejected():
    data = json.loads(json.dumps(V2))
    data["platform_binding"]["goals"]["wp_bad"] = {"points_name": "无 ID"}
    with pytest.raises(ValueError, match="points_id"):
        SharedMap.from_dict(data)


def test_binding_must_be_object():
    data = json.loads(json.dumps(V2))
    data["platform_binding"] = ["nope"]
    with pytest.raises(ValueError, match="platform_binding"):
        SharedMap.from_dict(data)


def test_goals_must_be_object():
    data = json.loads(json.dumps(V2))
    data["platform_binding"]["goals"] = []
    with pytest.raises(ValueError, match="goals"):
        SharedMap.from_dict(data)


def test_roundtrip_preserves_binding():
    m = SharedMap.from_dict(V2)
    again = SharedMap.from_dict(m.to_dict())
    assert again.platform_binding is not None
    assert again.resolve_points_id("wp_region_x_staging") == "快速打点-1785465994716"
    assert again.goal_pose("wp_region_x_staging") == (12.5, -3.25)
    m.validate()


def test_pose_optional_in_binding():
    """点位没导出坐标时退回状态字符串判据，而不是报错。"""
    b = PlatformBinding.from_dict({"goals": {"wp": {"points_id": "p1"}}})
    assert b.goals["wp"] == GoalBinding(points_id="p1")
    data = json.loads(json.dumps(V2))
    data["platform_binding"] = {"goals": {"wp_region_x_staging": {"points_id": "p1"}}}
    m = SharedMap.from_dict(data)
    assert m.goal_pose("wp_region_x_staging") is None
    assert m.resolve_points_id("wp_region_x_staging") == "p1"


def test_resolvers_are_drop_in_for_topsee_nav():
    """SharedMap 的两个方法要能直接当 TopseeNav 的 resolver 用。"""
    from adapters.dog_topsee import TopseeNav

    m = SharedMap.from_dict(V2)
    assert callable(m.resolve_points_id) and callable(m.goal_pose)
    # 签名兼容性：TopseeNav 只要求 Callable[[str], Optional[...]]
    nav_kwargs = {
        "goal_resolver": m.resolve_points_id,
        "goal_pose_resolver": m.goal_pose,
    }
    assert set(nav_kwargs) <= set(TopseeNav.__init__.__code__.co_varnames)
