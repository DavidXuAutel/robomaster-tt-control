"""语义/拓扑共享地图（非稠密 SLAM）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


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


@dataclass
class SharedMap:
    """versioned region_id → dog_goal / drone_route / anchors。"""

    version: int
    frame: str
    regions: Dict[str, Region] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedMap":
        if "version" not in data or "regions" not in data:
            raise ValueError("地图缺少 version 或 regions")
        regions: Dict[str, Region] = {}
        raw_regions = data["regions"]
        if not isinstance(raw_regions, Mapping):
            raise ValueError("regions 必须是对象")
        for rid, row in raw_regions.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"region {rid} 必须是对象")
            region_id = str(row.get("region_id", rid))
            dog_goal = row.get("dog_goal_id")
            drone_route = row.get("drone_route_id")
            if not dog_goal or not drone_route:
                raise ValueError(f"region {region_id} 缺少 dog_goal_id/drone_route_id")
            anchors = tuple(str(a) for a in row.get("anchor_ids", []))
            regions[region_id] = Region(
                region_id=region_id,
                dog_goal_id=str(dog_goal),
                drone_route_id=str(drone_route),
                anchor_ids=anchors,
                label=str(row.get("label", "")),
            )
        return cls(
            version=int(data["version"]),
            frame=str(data.get("frame", "dog_map")),
            regions=regions,
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SharedMap":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "frame": self.frame,
            "notes": self.notes,
            "regions": {rid: r.to_dict() for rid, r in self.regions.items()},
        }

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
