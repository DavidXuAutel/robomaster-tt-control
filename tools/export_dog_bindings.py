#!/usr/bin/env python3
"""从平台导出点位绑定，写入 SharedMap 的 platform_binding（方案 §5.2、M8）。

要防的事故：平台 pointsId（形如 `快速打点-1785465994716`）在地图重建或重打点
后失效。手写进配置的话，失效后软件层完全看不见——派单「成功」但狗不动，
或者走到另一个点。所以绑定必须是**导出物**，并能定期 diff 出漂移。

两种用法：

    # 导出/刷新绑定
    python tools/export_dog_bindings.py --base-url ... --account ... \
        --robot-id B2000397 --map configs/mission/shared_map.example.json --write

    # CI / 飞行前自检：只 diff，有漂移就非零退出
    python tools/export_dog_bindings.py ... --map <地图> --check

标签与平台点位的对应关系靠点位名：默认要求平台点位名 == 我们的
`dog_goal_id`（语义标签）。名字对不上时用 `--alias` 提供映射表，
**不做模糊匹配**——猜错点位比报错危险得多。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.topsee_client import TopseeClient, TopseeError  # noqa: E402
from mission_brain.map_model import SharedMap  # noqa: E402

# 平台各接口字段命名不统一，取值一律走候选键名
_ID_KEYS = ("pointsId", "id", "pointId")
_NAME_KEYS = ("pointsName", "name", "pointName")
_MAP_ID_KEYS = ("mapId", "id", "mapKeyname", "keyname")
_MAP_VER_KEYS = ("version", "mapVersion", "updateTime")


def _pick(doc: Mapping[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = doc.get(k)
        if v not in (None, ""):
            return str(v)
    return None


def _walk_points(doc: Any) -> List[Dict[str, Any]]:
    """在任意嵌套结构里捞出「看起来像点位」的对象。

    G16：OpenAPI 把 getRobotMapAll 的响应标成了 ShowRobotTaskAllEntity，疑似
    导出错误，真实结构未知。所以这里做结构无关的递归收集，而不是硬编码路径。
    判据是「同时有点位 ID 和点位名」。
    """
    found: List[Dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            pid = _pick(node, _ID_KEYS)
            name = _pick(node, _NAME_KEYS)
            if pid and name:
                found.append(
                    {
                        "points_id": pid,
                        "points_name": name,
                        "x": _num(node.get("x")),
                        "y": _num(node.get("y")),
                        "th": _num(node.get("th")),
                    }
                )
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(doc)
    # 同一 pointsId 可能在多处出现（线路引用点位），去重保留首个
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in found:
        dedup.setdefault(row["points_id"], row)
    return list(dedup.values())


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _find_map_meta(doc: Any) -> Tuple[str, str]:
    if isinstance(doc, Mapping):
        return _pick(doc, _MAP_ID_KEYS) or "", _pick(doc, _MAP_VER_KEYS) or ""
    return "", ""


def build_binding(
    shared_map: SharedMap,
    map_all: Any,
    *,
    alias: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """返回 (binding_dict, 未匹配的标签, 平台上没被用到的点位名)。"""
    points = _walk_points(map_all)
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in points:
        by_name.setdefault(row["points_name"], row)

    alias = dict(alias or {})
    labels = sorted({r.dog_goal_id for r in shared_map.regions.values()})
    goals: Dict[str, Any] = {}
    unmatched: List[str] = []
    for label in labels:
        wanted = alias.get(label, label)
        row = by_name.get(wanted)
        if row is None:
            unmatched.append(label)
            continue
        entry: Dict[str, Any] = {
            "points_id": row["points_id"],
            "points_name": row["points_name"],
        }
        for k in ("x", "y", "th"):
            if row[k] is not None:
                entry[k] = row[k]
        goals[label] = entry

    used = {g["points_name"] for g in goals.values()}
    unused = sorted(n for n in by_name if n not in used)
    map_id, map_version = _find_map_meta(map_all)
    binding = {
        "map_id": map_id,
        "map_version": map_version,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "goals": goals,
    }
    return binding, unmatched, unused


def diff_binding(old: Optional[Mapping[str, Any]], new: Mapping[str, Any]) -> List[str]:
    """逐条列出绑定漂移。空列表表示一致。"""
    if old is None:
        return ["地图尚无 platform_binding（首次导出）"]
    drift: List[str] = []
    if str(old.get("map_id", "")) != str(new.get("map_id", "")):
        drift.append(
            f"map_id 变化: {old.get('map_id')!r} → {new.get('map_id')!r}（务必确认是否换图）"
        )
    old_goals = old.get("goals", {}) or {}
    new_goals = new.get("goals", {}) or {}
    for label in sorted(set(old_goals) | set(new_goals)):
        o = old_goals.get(label)
        n = new_goals.get(label)
        if o is None:
            drift.append(f"新增绑定: {label} → {n.get('points_id')}")
        elif n is None:
            drift.append(f"平台已找不到点位: {label}（原 {o.get('points_id')}）")
        elif str(o.get("points_id")) != str(n.get("points_id")):
            drift.append(
                f"pointsId 漂移: {label} {o.get('points_id')} → {n.get('points_id')}"
            )
        else:
            for k in ("x", "y", "th"):
                ov, nv = _num(o.get(k)), _num(n.get(k))
                if ov is not None and nv is not None and abs(ov - nv) > 0.05:
                    drift.append(f"{label}.{k} 坐标变化: {ov} → {nv}")
    return drift


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="导出/校验 SharedMap 的平台点位绑定")
    p.add_argument("--base-url", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--password", default=os.environ.get("TOPSEE_PASSWORD"))
    p.add_argument("--robot-id", required=True)
    p.add_argument("--map", required=True, help="SharedMap JSON 路径")
    p.add_argument("--alias", help="标签→平台点位名 的 JSON 映射表")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--write", action="store_true", help="把绑定写回地图文件")
    p.add_argument("--check", action="store_true", help="只 diff，有漂移则非零退出")
    p.add_argument("--from-json", help="离线模式：直接读一份 getRobotMapAll 响应，不联网")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    map_path = Path(args.map)
    shared_map = SharedMap.load(map_path)
    alias = json.loads(Path(args.alias).read_text(encoding="utf-8")) if args.alias else {}

    if args.from_json:
        map_all = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:
        if not args.password:
            print("错误：未提供密码（TOPSEE_PASSWORD 或 --password）", file=sys.stderr)
            return 2
        client = TopseeClient(
            args.base_url,
            account=args.account,
            password=args.password,
            timeout_s=args.timeout,
            insecure_tls=args.insecure,
        )
        try:
            client.login()
            map_all = client.get_robot_map_all(args.robot_id)
        except TopseeError as exc:
            print(f"平台交互失败：{exc}", file=sys.stderr)
            return 3

    binding, unmatched, unused = build_binding(shared_map, map_all, alias=alias)
    old = (
        shared_map.platform_binding.to_dict()
        if shared_map.platform_binding is not None
        else None
    )
    drift = diff_binding(old, binding)

    print(f"平台点位命中 {len(binding['goals'])} / {len(shared_map.regions)} 个 region 标签")
    if unmatched:
        print("\n以下标签在平台上找不到同名点位（不做模糊匹配，请用 --alias 明确指定）：")
        for label in unmatched:
            print(f"  - {label}")
    if unused:
        print(f"\n平台上还有 {len(unused)} 个未被引用的点位（前 10）：")
        for n in unused[:10]:
            print(f"  · {n}")
    print("\n=== 绑定漂移 ===")
    for line in drift or ["（无变化）"]:
        print(f"  {line}")

    if args.check:
        if unmatched:
            print("\n[check] 有标签未绑定，拒绝通过。", file=sys.stderr)
            return 5
        if drift:
            print("\n[check] 绑定已漂移，需重新导出。", file=sys.stderr)
            return 6
        print("\n[check] 绑定一致。")
        return 0

    if args.write:
        if unmatched:
            print("\n拒绝写入：仍有标签未绑定。", file=sys.stderr)
            return 5
        doc = shared_map.to_dict()
        doc["platform_binding"] = binding
        doc["version"] = max(2, int(doc.get("version", 1)))
        SharedMap.from_dict(doc)  # 写盘前先过一遍不变量
        map_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n已写入 {map_path}")
    else:
        print("\n（未加 --write，仅预览）")
        print(json.dumps(binding, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
