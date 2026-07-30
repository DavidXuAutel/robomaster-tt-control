"""Mission 事件契约（v1 冻结）。

公共信封: v, event_id, mission_id, source, sent_at, causation_id
禁止字段: drone 全局 pose、点云、协方差、实时视频流。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Set


SCHEMA_VERSION = 1

# 公共信封必填字段
ENVELOPE_KEYS = ("v", "event_id", "mission_id", "source", "sent_at", "type")

# v1 禁止出现在事件载荷中的键（防伪全局坐标 / 重载荷）
FORBIDDEN_KEYS = frozenset(
    {
        "global_pose",
        "pose_xyz",
        "point_cloud",
        "pointcloud",
        "covariance",
        "video_b64",
        "rgb_jpeg_b64",
        "depth_b64",
        "transform",
        "T_world",
    }
)


class EventType(str, Enum):
    MISSION_START = "mission.start"
    MISSION_ABORT = "mission.abort"
    MISSION_COMPLETED = "mission.completed"
    MISSION_FAILED = "mission.failed"

    DRONE_SCOUT = "drone.scout"
    DRONE_TARGET_FOUND = "drone.target_found"
    DRONE_SCOUT_FAILED = "drone.scout_failed"

    DOG_INSPECT = "dog.inspect"
    DOG_ARRIVED = "dog.arrived"
    DOG_TARGET_FOUND = "dog.target_found"
    DOG_INSPECT_FAILED = "dog.inspect_failed"

    GAS_SAMPLE = "gas.sample"
    GAS_COMPLETED = "gas.completed"
    GAS_FAILED = "gas.failed"

    HEARTBEAT = "agent.heartbeat"


# 各事件类型的必填业务字段（信封之外）
REQUIRED_FIELDS: Dict[EventType, Set[str]] = {
    EventType.MISSION_START: {"target_label", "region_ids", "deadline"},
    EventType.MISSION_ABORT: {"reason"},
    EventType.MISSION_COMPLETED: {"completed_at"},
    EventType.MISSION_FAILED: {"stage", "reason"},
    EventType.DRONE_SCOUT: {"region_id", "drone_route_id", "target_label", "deadline"},
    EventType.DRONE_TARGET_FOUND: {
        "region_id",
        "target_label",
        "confidence",
        "anchor_id",
        "anchor_age_ms",
        "observed_at",
        "evidence_uri",
    },
    EventType.DRONE_SCOUT_FAILED: {"region_id", "reason"},
    EventType.DOG_INSPECT: {
        "region_id",
        "dog_goal_id",
        "target_label",
        "evidence_uri",
        "deadline",
    },
    EventType.DOG_ARRIVED: {"region_id", "dog_goal_id", "arrived_at"},
    EventType.DOG_TARGET_FOUND: {
        "region_id",
        "target_label",
        "confidence",
        "evidence_uri",
    },
    EventType.DOG_INSPECT_FAILED: {"region_id", "stage", "reason"},
    EventType.GAS_SAMPLE: {
        "region_id",
        "target_label",
        "sample_window_s",
        "deadline",
    },
    EventType.GAS_COMPLETED: {
        "region_id",
        "target_label",
        "device_id",
        "sampled_at",
        "sample_window_s",
        "calibration_at",
        "readings",
    },
    EventType.GAS_FAILED: {"region_id", "reason"},
    EventType.HEARTBEAT: {"agent", "status"},
}


def new_event_id() -> str:
    return str(uuid.uuid4())


def make_event(
    event_type: EventType | str,
    *,
    mission_id: str,
    source: str,
    payload: Optional[Mapping[str, Any]] = None,
    causation_id: Optional[str] = None,
    event_id: Optional[str] = None,
    sent_at: Optional[float] = None,
) -> Dict[str, Any]:
    """构造带公共信封的事件 dict。"""
    if isinstance(event_type, EventType):
        type_str = event_type.value
    else:
        type_str = str(event_type)
        EventType(type_str)  # 校验合法

    body: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "event_id": event_id or new_event_id(),
        "mission_id": mission_id,
        "source": source,
        "sent_at": float(sent_at if sent_at is not None else time.time()),
        "type": type_str,
        "causation_id": causation_id,
    }
    if payload:
        for k, v in payload.items():
            if k in body:
                raise ValueError(f"payload 不可覆盖信封字段: {k}")
            body[k] = v
    validate_event(body)
    return body


def validate_event(event: Mapping[str, Any]) -> EventType:
    """校验信封 + 类型必填字段 + 禁止字段。返回 EventType。"""
    missing = [k for k in ENVELOPE_KEYS if k not in event]
    if missing:
        raise ValueError(f"缺少信封字段: {missing}")

    if int(event["v"]) != SCHEMA_VERSION:
        raise ValueError(f"不支持的 schema 版本: {event['v']}")

    try:
        et = EventType(str(event["type"]))
    except ValueError as exc:
        raise ValueError(f"未知事件类型: {event['type']}") from exc

    bad = FORBIDDEN_KEYS.intersection(event.keys())
    if bad:
        raise ValueError(f"v1 禁止字段: {sorted(bad)}")

    req = REQUIRED_FIELDS[et]
    miss_biz = sorted(k for k in req if k not in event)
    if miss_biz:
        raise ValueError(f"{et.value} 缺少字段: {miss_biz}")

    if et is EventType.MISSION_START:
        regions = event["region_ids"]
        if not isinstance(regions, Iterable) or isinstance(regions, (str, bytes)):
            raise ValueError("region_ids 必须是列表")
        if not list(regions):
            raise ValueError("region_ids 不能为空")

    if et is EventType.GAS_COMPLETED:
        readings = event["readings"]
        if not isinstance(readings, list) or not readings:
            raise ValueError("readings 必须是非空列表")
        for i, row in enumerate(readings):
            if not isinstance(row, Mapping):
                raise ValueError(f"readings[{i}] 必须是对象")
            for k in ("channel", "value", "unit", "alarm_state"):
                if k not in row:
                    raise ValueError(f"readings[{i}] 缺少 {k}")

    if et is EventType.DRONE_TARGET_FOUND:
        conf = float(event["confidence"])
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence 必须在 [0,1]")

    return et
