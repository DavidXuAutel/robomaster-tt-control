#!/usr/bin/env python3
"""拓普视平台只读探针 + E0/E2/E5–E10 验证实验（v3.1 §2；M6）。

**默认严格只读**。会让机器人动起来的实验（E1）必须显式加 `--allow-motion`，
且脚本会要求在命令行里再确认一次现场安全。这不是形式主义：E1 的动作就是让
狗真的走去一个点位。

实验编号以 `docs/design/2026-08-05-dog-deployment-loop-plan.md` §2 为准
（v2 §8 编号作废）。E3/E4 为人工/真机项，见 MANUAL_ONLY。

用法：

    python tools/topsee_probe.py \
        --base-url http://192.168.0.85:8001 \
        --account <账号> --robot-id B2000397 \
        --out artifacts/topsee_probe.json

密码从 `TOPSEE_PASSWORD` 环境变量读，避免进 shell 历史。

产物是一份 JSON 报告，六项回填写入 `configs/dog/topsee.json`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.topsee_client import (  # noqa: E402
    TopseeAuthError,
    TopseeClient,
    TopseeError,
    TopseeLicenseError,
)


class Probe:
    """跑一组实验并汇总成可回填方案的报告。"""

    def __init__(
        self,
        client: TopseeClient,
        *,
        robot_id: str,
        allow_motion: bool = False,
        samples: int = 20,
        interval_s: float = 1.0,
        points_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.robot_id = robot_id
        self.allow_motion = bool(allow_motion)
        self.samples = int(samples)
        self.interval_s = float(interval_s)
        self.points_id = points_id
        self.results: Dict[str, Dict[str, Any]] = {}

    # ---------- 框架 ----------

    def run(self, names: List[str]) -> Dict[str, Any]:
        for name in names:
            exp = EXPERIMENTS.get(name)
            if exp is None:
                self._record(name, "skipped", note=f"未知实验 {name}")
                continue
            if exp["motion"] and not self.allow_motion:
                self._record(
                    name,
                    "skipped",
                    note="该实验会让机器人动作/改平台状态，需 --allow-motion",
                )
                continue
            t0 = time.monotonic()
            try:
                data = exp["fn"](self)
                self._record(name, "ok", data=data, elapsed_s=time.monotonic() - t0)
            except TopseeError as exc:
                self._record(
                    name,
                    "failed",
                    note=f"{type(exc).__name__}: {exc}",
                    elapsed_s=time.monotonic() - t0,
                )
        return self.report()

    def _record(self, name: str, status: str, **kw: Any) -> None:
        row: Dict[str, Any] = {"status": status}
        row.update(kw)
        exp = EXPERIMENTS.get(name)
        if exp is not None:
            row["question"] = exp["question"]
            row["gap"] = exp["gap"]
        self.results[name] = row

    def report(self) -> Dict[str, Any]:
        return {
            "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": self.client.base_url,
            "robot_id": self.robot_id,
            "allow_motion": self.allow_motion,
            "results": self.results,
        }

    # ---------- 实验实现 ----------

    def e0_login(self) -> Dict[str, Any]:
        """E0：登录链路 + 授权状态。"""
        self.client.login()
        return {
            "login_ok": True,
            "deployment": self.client.deployment,
            "token_header_configured": self.client.token_header,
            "token_present": bool(self.client._token),
            "note": (
                "离线一体机：header=tuopushi_edge_token，字段=webToken。"
                if self.client.deployment == "offline"
                else "登录成功。云端 token header 名仍需抓包确认（G8），当前可靠 cookie 会话。"
            ),
        }

    def e1_navigate_produces_task(self) -> Dict[str, Any]:
        """E1（动作）：单点 sendNavigate 是否产生可查询任务？

        这是全套设计里最关键的假设。若答案为否，`NavBackend.is_arrived()` 就
        只能靠位姿距离判据，`configs` 里必须为每个目标点位补 x/y。
        """
        if not self.points_id:
            raise TopseeError("E1 需要 --points-id")
        before = self.client.get_current_task(self.robot_id)
        self.client.send_navigate(self.robot_id, self.points_id)
        seen: List[Any] = []
        for _ in range(self.samples):
            time.sleep(self.interval_s)
            seen.append(self.client.get_current_task(self.robot_id))
        tracked = any(isinstance(s, Mapping) and s for s in seen)
        return {
            "task_before": before,
            "task_tracked": tracked,
            "distinct_states": sorted(
                {
                    str(s.get("currentState"))
                    for s in seen
                    if isinstance(s, Mapping) and s.get("currentState") is not None
                }
            ),
            "samples": seen,
            "verdict": (
                "可查询：is_arrived 可用状态白名单"
                if tracked
                else "不可查询：必须改用位姿距离判据，且需补齐点位坐标"
            ),
        }

    def e2_state_enum(self) -> Dict[str, Any]:
        """E2：采样 currentState / totalState 的真实取值（G2 无枚举）。"""
        states: Dict[str, set] = {"currentState": set(), "totalState": set()}
        raw: List[Any] = []
        for _ in range(self.samples):
            task = self.client.get_current_task(self.robot_id)
            raw.append(task)
            if isinstance(task, Mapping):
                for k in states:
                    v = task.get(k)
                    if v is not None:
                        states[k].add(str(v))
            time.sleep(self.interval_s)
        return {
            "currentState_values": sorted(states["currentState"]),
            "totalState_values": sorted(states["totalState"]),
            "empty_responses": sum(1 for r in raw if not r),
            "note": "把这里的取值填进 TopseeNav(arrived_states=..., enroute_states=...)。",
        }

    def e5_map_structure(self) -> Dict[str, Any]:
        """E5：getRobotMapAll 的真实结构（G16 怀疑 OpenAPI 标注错误）。"""
        doc = self.client.get_robot_map_all(self.robot_id)
        return {
            "top_level_type": type(doc).__name__,
            "top_level_keys": sorted(doc.keys()) if isinstance(doc, Mapping) else None,
            "shape_sketch": _sketch(doc),
        }

    def e6_gas_time_format(self) -> Dict[str, Any]:
        """E6：气体历史的时间参数格式。逐个格式试，看哪个能回数据。"""
        now = time.time()
        candidates = {
            "space_seconds": "%Y-%m-%d %H:%M:%S",
            "iso_T": "%Y-%m-%dT%H:%M:%S",
            "date_only": "%Y-%m-%d",
        }
        out: Dict[str, Any] = {}
        for name, fmt in candidates.items():
            try:
                doc = self.client.get_gas_history(
                    robotId=self.robot_id,
                    startTime=time.strftime(fmt, time.localtime(now - 86400)),
                    endTime=time.strftime(fmt, time.localtime(now)),
                )
                out[name] = {"ok": True, "rows": _count(doc), "sketch": _sketch(doc)}
            except TopseeError as exc:
                out[name] = {"ok": False, "error": str(exc)}
        out["epoch_ms"] = _try(
            lambda: self.client.get_gas_history(
                robotId=self.robot_id,
                startTime=int((now - 86400) * 1000),
                endTime=int(now * 1000),
            )
        )
        return out

    def e7_stream_url(self) -> Dict[str, Any]:
        """E7：取流地址响应结构（五个参数全必填，deviceId/ip 需先摸清）。"""
        return {
            "note": "需要真实 deviceId 与 ip；此处只验证接口连通性与响应形状。",
            "probe": _try(
                lambda: self.client.get_stream_url(
                    device_id=self.robot_id, ip="0.0.0.0"
                )
            ),
        }

    def e8_session_lifetime(self) -> Dict[str, Any]:
        """E8：会话有效期（G8）。空转一段时间后看是否还认。"""
        idle_s = min(120.0, self.samples * self.interval_s)
        self.client.get_state_data()
        time.sleep(idle_s)
        before_logins = self.client.login_count
        self.client.get_state_data()
        return {
            "idle_s": idle_s,
            "relogin_triggered": self.client.login_count > before_logins,
            "note": "触发重登说明会话短于该空转时长；请加大 --samples 二分定位。",
        }

    def e9_alarm_structure(self) -> Dict[str, Any]:
        """E9：告警列表结构（作旁路证据用，字段名要确认）。"""
        doc = self.client.get_alarm_list(robotId=self.robot_id)
        return {"rows": _count(doc), "sketch": _sketch(doc)}

    def e10_state_data(self) -> Dict[str, Any]:
        """E10：实时状态列表结构 —— 电量字段名（preflight 门禁要用）。"""
        doc = self.client.get_state_data()
        return {
            "sketch": _sketch(doc),
            "note": "找出电量字段名，填进 DogControlArbiter(battery_provider=...)。",
        }


def _sketch(doc: Any, depth: int = 0) -> Any:
    """输出结构骨架而非全量数据，避免报告里塞满业务数据。"""
    if depth > 3:
        return "..."
    if isinstance(doc, Mapping):
        return {k: _sketch(v, depth + 1) for k, v in list(doc.items())[:25]}
    if isinstance(doc, list):
        return [_sketch(doc[0], depth + 1), f"...共 {len(doc)} 项"] if doc else []
    return type(doc).__name__


def _count(doc: Any) -> int:
    if isinstance(doc, list):
        return len(doc)
    if isinstance(doc, Mapping):
        for k in ("records", "list", "rows"):
            v = doc.get(k)
            if isinstance(v, list):
                return len(v)
    return 0


def _try(fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        return {"ok": True, "sketch": _sketch(fn())}
    except TopseeError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "E0": {
        "fn": Probe.e0_login,
        "motion": False,
        "question": "登录链路是否通、授权是否有效",
        "gap": "G8",
    },
    "E1": {
        "fn": Probe.e1_navigate_produces_task,
        "motion": True,
        "question": "单点 sendNavigate 是否产生可查询任务",
        "gap": "F8/G1",
    },
    "E2": {
        "fn": Probe.e2_state_enum,
        "motion": False,
        "question": "currentState / totalState 的真实取值枚举",
        "gap": "G2",
    },
    "E5": {
        "fn": Probe.e5_map_structure,
        "motion": False,
        "question": "getRobotMapAll 的真实响应结构",
        "gap": "G16",
    },
    "E6": {
        "fn": Probe.e6_gas_time_format,
        "motion": False,
        "question": "气体历史的时间参数格式",
        "gap": "G9",
    },
    "E7": {
        "fn": Probe.e7_stream_url,
        "motion": False,
        "question": "取流地址响应结构（≠深度可用；深度协议见 G12/D2b）",
        "gap": "G12",
    },
    "E8": {
        "fn": Probe.e8_session_lifetime,
        "motion": False,
        "question": "会话/token 有效期",
        "gap": "G8",
    },
    "E9": {
        "fn": Probe.e9_alarm_structure,
        "motion": False,
        "question": "告警列表字段结构",
        "gap": "alarm-schema",
    },
    "E10": {
        "fn": Probe.e10_state_data,
        "motion": False,
        "question": "实时状态字段名（电量等）",
        "gap": "battery-field",
    },
}

READ_ONLY = [k for k, v in EXPERIMENTS.items() if not v["motion"]]

# 只能靠抓包/真机的实验，脚本做不了，但必须出现在报告里以免被遗忘（v3.1 §2）
MANUAL_ONLY = {
    "E3": "手动/自动巡检模式切换（F14：无 HTTP 接口，须抓 Web 前端；人工切完 ack_human_mode_switch）",
    "E4": (
        "DDS 运动权限四步协议（G11）：①/sportmodestate 持续可读；"
        "②释放平台控制后极小幅 Move + 速度回读；③停发≥2s deadman 归零；"
        "④平台侧同时活动时是否抢权"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="拓普视平台只读探针（默认不发任何动作命令）")
    p.add_argument("--base-url", required=True, help="如 http://192.168.0.85:8001")
    p.add_argument("--account", required=True)
    p.add_argument(
        "--password",
        default=os.environ.get("TOPSEE_PASSWORD"),
        help="默认读 TOPSEE_PASSWORD 环境变量",
    )
    p.add_argument("--robot-id", required=True)
    p.add_argument(
        "--deployment",
        default="cloud",
        choices=("cloud", "offline"),
        help="cloud=:8001 OpenAPI；offline=:8888 一体机（JSON 登录 + tuopushi_edge_token）",
    )
    p.add_argument(
        "--token-header",
        default=None,
        help="覆盖 token header 名；offline 默认 tuopushi_edge_token",
    )
    p.add_argument("--points-id", help="E1 用的目标点位 ID")
    p.add_argument(
        "--experiments",
        default=",".join(READ_ONLY),
        help=f"逗号分隔，默认全部只读实验：{','.join(READ_ONLY)}",
    )
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--insecure", action="store_true", help="跳过 TLS 校验（自签证书现场）")
    p.add_argument(
        "--allow-motion",
        action="store_true",
        help="放开会让机器人动作的实验（E1）。现场必须有人看着、有急停手段。",
    )
    p.add_argument("--out", help="报告输出路径（JSON）")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.password:
        print("错误：未提供密码，请设置 TOPSEE_PASSWORD 或传 --password", file=sys.stderr)
        return 2

    names = [n.strip().upper() for n in args.experiments.split(",") if n.strip()]
    motion_wanted = [n for n in names if EXPERIMENTS.get(n, {}).get("motion")]
    if motion_wanted and args.allow_motion:
        print(f"以下实验会让机器人动作: {', '.join(motion_wanted)}")
        if input("确认现场已清空且有人手持急停？输入 GO 继续: ").strip() != "GO":
            print("已取消。")
            return 1

    client = TopseeClient(
        args.base_url,
        account=args.account,
        password=args.password,
        timeout_s=args.timeout,
        insecure_tls=args.insecure,
        deployment=args.deployment,
        token_header=args.token_header,
    )
    probe = Probe(
        client,
        robot_id=args.robot_id,
        allow_motion=args.allow_motion,
        samples=args.samples,
        interval_s=args.interval,
        points_id=args.points_id,
    )
    try:
        client.login()
    except TopseeLicenseError as exc:
        print(f"账号授权异常，需联系厂商续期：{exc}", file=sys.stderr)
        return 3
    except TopseeAuthError as exc:
        print(f"登录失败：{exc}", file=sys.stderr)
        return 3

    report = probe.run(names)
    report["manual_only"] = MANUAL_ONLY
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"报告已写入 {out}")
    else:
        print(text)

    print("\n=== 摘要 ===")
    for name, row in report["results"].items():
        print(f"{name:>4}  {row['status']:<8} {row.get('note', row.get('question', ''))}")
    print("\n以下只能靠抓包/实机，脚本做不了：")
    for name, why in MANUAL_ONLY.items():
        print(f"{name:>4}  {why}")
    return 0 if all(r["status"] != "failed" for r in report["results"].values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
