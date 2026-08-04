"""语义/拓扑共享地图（非稠密 SLAM）。

v2 新增 `platform_binding`：把「我们的稳定语义标签」与「厂商平台自动生成的
点位 ID」解耦。背景见 docs/design/2026-08-03-dog-integration-plan.md §5.2 ——
平台 pointsId 形如 `快速打点-1785465994716`，地图重建或重打点后会失效，
直接写进配置会造成软件层完全不可见的静默失效。

因此 `Region.dog_goal_id` 现在的语义是**稳定语义标签**（人写、进 git），
真正下发给平台的 pointsId 从 `platform_binding` 里查，由导出脚本维护。
v1 地图（没有 platform_binding）继续原样加载，此时标签即 ID。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Region:
    region_id: str
    dog_goal_id: str
    drone_route_id: str
    anchor_ids: tuple[str, ...]
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "dog_goal_id": self.dog_goal_id,
            "drone_route_id": self.drone_route_id,
            "anchor_ids": list(self.anchor_ids),
            "label": self.label,
        }


@dataclass(frozen=True)
class GoalBinding:
    """稳定语义标签 → 平台点位的一次绑定快照。"""

    points_id: str
    points_name: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    th: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"points_id": self.points_id}
        if self.points_name:
            out["points_name"] = self.points_name
        for k in ("x", "y", "th"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


@dataclass
class PlatformBinding:
    """平台侧绑定缓存。**可失效**，由 tools/export_dog_bindings.py 生成。

    `map_id` / `map_version` 用于启动期校验：绑定必须与机器人当前加载的
    地图一致，否则宁可拒绝启动，也不要拿着过期 pointsId 去派单。
    """

    map_id: str = ""
    map_version: str = ""
    exported_at: str = ""
    goals: Dict[str, GoalBinding] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformBinding":
        raw = data.get("goals", {})
        if not isinstance(raw, Mapping):
            raise ValueError("platform_binding.goals 必须是对象")
        goals: Dict[str, GoalBinding] = {}
        seen_points: Dict[str, str] = {}
        for label, row in raw.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"platform_binding.goals[{label}] 必须是对象")
            pid = row.get("points_id")
            if not pid:
                raise ValueError(f"platform_binding.goals[{label}] 缺少 points_id")
            pid = str(pid)
            if pid in seen_points:
                raise ValueError(
                    f"points_id {pid!r} 被重复绑定：{seen_points[pid]} 与 {label}"
                )
            seen_points[pid] = str(label)
            goals[str(label)] = GoalBinding(
                points_id=pid,
                points_name=str(row.get("points_name", "")),
                x=_opt_float(row.get("x")),
                y=_opt_float(row.get("y")),
                th=_opt_float(row.get("th")),
            )
        return cls(
            map_id=str(data.get("map_id", "")),
            map_version=str(data.get("map_version", "")),
            exported_at=str(data.get("exported_at", "")),
            goals=goals,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_id": self.map_id,
            "map_version": self.map_version,
            "exported_at": self.exported_at,
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
        }


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


@dataclass
class SharedMap:
    """versioned region_id → dog_goal / drone_route / anchors。"""

    version: int
    frame: str
    regions: Dict[str, Region] = field(default_factory=dict)
    notes: str = ""
    platform_binding: Optional[PlatformBinding] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedMap":
        if "version" not in data or "regions" not in data:
            raise ValueError("地图缺少 version 或 regions")
        if "frame" not in data:
            raise ValueError("地图缺少 frame（须为 dog_map）")
        frame = str(data["frame"])
        if frame != "dog_map":
            raise ValueError(f"frame 必须是 dog_map，收到: {frame}")

        regions: Dict[str, Region] = {}
        raw_regions = data["regions"]
        if not isinstance(raw_regions, Mapping):
            raise ValueError("regions 必须是对象")
        seen_anchors: Dict[str, str] = {}
        for rid, row in raw_regions.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"region {rid} 必须是对象")
            region_id = str(row.get("region_id", rid))
            if region_id != str(rid):
                raise ValueError(
                    f"region key {rid!r} 与内部 region_id {region_id!r} 不一致"
                )
            # v2 起推荐写 dog_goal_label（语义标签）；dog_goal_id 保留兼容
            dog_goal = row.get("dog_goal_id") or row.get("dog_goal_label")
            drone_route = row.get("drone_route_id")
            if not dog_goal or not drone_route:
                raise ValueError(f"region {region_id} 缺少 dog_goal_id/drone_route_id")
            anchors = tuple(str(a) for a in row.get("anchor_ids", []))
            if not anchors:
                raise ValueError(f"region {region_id} 的 anchor_ids 不能为空")
            for a in anchors:
                if a in seen_anchors:
                    raise ValueError(
                        f"anchor_id {a!r} 重复：{seen_anchors[a]} 与 {region_id}"
                    )
                seen_anchors[a] = region_id
            regions[region_id] = Region(
                region_id=region_id,
                dog_goal_id=str(dog_goal),
                drone_route_id=str(drone_route),
                anchor_ids=anchors,
                label=str(row.get("label", "")),
            )
        raw_binding = data.get("platform_binding")
        binding: Optional[PlatformBinding] = None
        if raw_binding is not None:
            if not isinstance(raw_binding, Mapping):
                raise ValueError("platform_binding 必须是对象")
            binding = PlatformBinding.from_dict(raw_binding)

        return cls(
            version=int(data["version"]),
            frame=frame,
            regions=regions,
            notes=str(data.get("notes", "")),
            platform_binding=binding,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SharedMap":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> None:
        """对已构造实例再跑一遍不变量（供测试/启动自检）。"""
        self.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "version": self.version,
            "frame": self.frame,
            "notes": self.notes,
            "regions": {rid: r.to_dict() for rid, r in self.regions.items()},
        }
        if self.platform_binding is not None:
            out["platform_binding"] = self.platform_binding.to_dict()
        return out

    # ---------- 平台绑定 ----------

    def validate_binding(self, *, map_id: Optional[str] = None) -> None:
        """启动期校验绑定完整性。任一 region 的标签没绑定就拒绝启动。

        map_id 传入机器人当前加载的地图编号时，会额外校验绑定是否属于该地图 ——
        地图换了而绑定没重新导出，比缺绑定更危险（会派单到一个不存在的点）。
        """
        if self.platform_binding is None:
            raise ValueError("地图没有 platform_binding，无法解析平台 pointsId")
        missing = sorted(
            r.dog_goal_id
            for r in self.regions.values()
            if r.dog_goal_id not in self.platform_binding.goals
        )
        if missing:
            raise ValueError(f"以下 dog_goal 缺少平台绑定: {missing}")
        if map_id is not None and self.platform_binding.map_id:
            if str(map_id) != self.platform_binding.map_id:
                raise ValueError(
                    f"绑定属于地图 {self.platform_binding.map_id!r}，"
                    f"机器人当前地图是 {map_id!r}，请重新导出绑定"
                )

    def resolve_points_id(self, dog_goal_label: str) -> Optional[str]:
        """标签 → 平台 pointsId。可直接作为 TopseeNav 的 goal_resolver。

        无 platform_binding 时（v1 地图）返回标签本身，保持旧行为。
        """
        if self.platform_binding is None:
            return dog_goal_label
        g = self.platform_binding.goals.get(dog_goal_label)
        return g.points_id if g is not None else None

    def goal_pose(self, dog_goal_label: str) -> Optional[Tuple[float, float]]:
        """标签 → 平台地图系下的 (x, y)。可作 TopseeNav 的 goal_pose_resolver。

        用于「距离到点」证据；缺坐标时返回 None，调用方应退回状态字符串判据。
        """
        if self.platform_binding is None:
            return None
        g = self.platform_binding.goals.get(dog_goal_label)
        if g is None or g.x is None or g.y is None:
            return None
        return float(g.x), float(g.y)

    def get(self, region_id: str) -> Region:
        try:
            return self.regions[region_id]
        except KeyError as exc:
            raise KeyError(f"未知 region_id: {region_id}") from exc

    def resolve_dog_goal(self, region_id: str) -> str:
        return self.get(region_id).dog_goal_id

    def resolve_drone_route(self, region_id: str) -> str:
        return self.get(region_id).drone_route_id

    def region_for_anchor(self, anchor_id: str) -> Optional[Region]:
        for r in self.regions.values():
            if anchor_id in r.anchor_ids:
                return r
        return None

    def region_ids(self) -> List[str]:
        return list(self.regions.keys())
