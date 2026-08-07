#!/usr/bin/env python3
"""D0 DDS 订阅链路自检：真实 CycloneDDS + 真实 unitree_go IDL，合成数据源。

## 这个工具证明什么、不证明什么

**证明**（此前只有 `LoopbackTransport` 桩覆盖，等于没测）：

1. `DdsTransport` 的 subscriber 能在**真实 DDS 传输**上收到报文；
2. `sport_state_to_dict` / `low_state_to_dict` 能正确解析**真实 IDL 对象**——
   在此之前所有单测喂的都是 dict，字段名拼错也发现不了；
3. 设备钟 `stamp` 能取出且单调；
4. `sample_age_s` / `is_dds_stale` 的断流判定在真实报文流上成立。

**不证明**（别拿这份产物去关 D0）：

- 真机时序：真狗的发布抖动、丢包、负载下退化，本工具一概测不出；
- 命令通道：`Move` / `StopMove` 的可控性与 deadman（E4-C），厂商未批准前禁做；
- 现场网络：改装机的网段隔离、广播风暴约束。

所以产物里的 `verdict` 只对「订阅链路软件正确性」负责，字段名就叫
`subscribe_path_ok`，不叫 `d0_ok`。

## 用法

    .venv/bin/python tools/dds_selfcheck.py --seconds 12 --hz 50

本机需要把 CycloneDDS 限定在回环，否则会去枚举物理网卡：

    export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>'\\
    '<NetworkInterface name="lo0" presence_required="false"/></Interfaces>'\\
    '<AllowMulticast>false</AllowMulticast></General></Domain></CycloneDDS>'
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.dog_unitree import (  # noqa: E402
    DDS_STALE_S,
    TOPIC_LOW_STATE,
    TOPIC_SPORT_STATE,
    DdsTransport,
)

# 真实报文必须带齐的字段。缺任何一项，D3 的本体感受数据就是残的。
REQUIRED_SPORT_KEYS = ("position", "velocity", "yaw_speed", "imu_state", "foot_force", "t_device")
REQUIRED_IMU_KEYS = ("rpy", "quaternion", "gyroscope", "accelerometer")
REQUIRED_MOTOR_KEYS = ("q", "dq", "tau_est", "temperature")


def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def observe(
    *, seconds: float, poll_hz: float, domain_id: int, family: str, interface: str = ""
) -> Dict[str, Any]:
    """只订阅、不发布：量别人发的流。

    这就是真机 D0 验收的形态——对着真狗（或宇树自己的 MuJoCo 桥）跑，
    我们只当观察者。发布端不是自己写的，才能排掉「两端一起理解错字段」的盲区。
    """
    transport = DdsTransport(interface=interface, domain_id=domain_id, family=family)
    transport.connect_readonly()

    deadline = time.monotonic() + seconds
    period = 1.0 / float(poll_hz)
    last_recv: Dict[str, float] = {}
    intervals: Dict[str, List[float]] = {TOPIC_SPORT_STATE: [], TOPIC_LOW_STATE: []}
    frames: Dict[str, int] = {TOPIC_SPORT_STATE: 0, TOPIC_LOW_STATE: 0}
    missing: List[str] = []
    device_stamps: List[float] = []

    while time.monotonic() < deadline:
        for topic in (TOPIC_SPORT_STATE, TOPIC_LOW_STATE):
            sample = transport.read(topic)
            if sample is None:
                continue
            t = float(sample["t_mono"])
            if last_recv.get(topic) == t:
                continue  # 同一帧被重复读到，不计数
            if topic in last_recv:
                intervals[topic].append(t - last_recv[topic])
            last_recv[topic] = t
            frames[topic] += 1
            if topic == TOPIC_SPORT_STATE:
                if sample.get("t_device") is not None:
                    device_stamps.append(float(sample["t_device"]))
                for k in REQUIRED_SPORT_KEYS:
                    if sample.get(k) is None:
                        missing.append(f"sport.{k}")
                for k in REQUIRED_IMU_KEYS:
                    if not (sample.get("imu_state") or {}).get(k):
                        missing.append(f"sport.imu_state.{k}")
            else:
                motors = sample.get("motor_state") or []
                if motors:
                    for k in REQUIRED_MOTOR_KEYS:
                        if k not in motors[0]:
                            missing.append(f"low.motor_state.{k}")
        time.sleep(period)

    def _stats(topic: str) -> Dict[str, Any]:
        gaps = intervals[topic]
        return {
            "frames": frames[topic],
            "effective_hz": round(frames[topic] / seconds, 2),
            "gap_p50_ms": round(statistics.median(gaps) * 1000, 2) if gaps else None,
            "gap_p99_ms": round(_pct(gaps, 0.99) * 1000, 2) if gaps else None,
            "gap_max_ms": round(max(gaps) * 1000, 2) if gaps else None,
        }

    missing_unique = sorted(set(missing))
    # 「单调」在全等序列上空洞成立。发布端不填 stamp（全 0）会假绿，
    # 所以必须另报「在推进」——宇树自己的 MuJoCo 桥就不填 stamp。
    distinct_stamps = len(set(device_stamps))
    notes: List[str] = []
    if device_stamps and distinct_stamps <= 1:
        notes.append(
            "device stamp 恒定（发布端未填 stamp）：t_device 校验在本次运行中无效"
        )
    result = {
        "tool": "dds_selfcheck",
        "mode": "observe",
        "scope": "只订阅外部发布端，量到帧率/间隔/字段完整性",
        "not_in_scope": ["命令通道 E4-C", "deadman", "真机时序"],
        "family": family,
        "domain_id": domain_id,
        "seconds": seconds,
        "poll_hz": poll_hz,
        "sportmodestate": _stats(TOPIC_SPORT_STATE),
        "lowstate": _stats(TOPIC_LOW_STATE),
        "device_stamp_count": len(device_stamps),
        "device_stamp_distinct": distinct_stamps,
        "device_stamp_advancing": distinct_stamps > 1,
        "device_stamp_monotonic": all(
            b >= a for a, b in zip(device_stamps, device_stamps[1:])
        ),
        "missing_fields": missing_unique,
        "notes": notes,
        "ts": time.time(),
    }
    result["subscribe_path_ok"] = bool(
        frames[TOPIC_SPORT_STATE] > 0 and frames[TOPIC_LOW_STATE] > 0 and not missing_unique
    )
    return result


def _build_publishers(domain_id: int):
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__LowState_,
        unitree_go_msg_dds__SportModeState_,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

    sport_pub = ChannelPublisher(TOPIC_SPORT_STATE, SportModeState_)
    sport_pub.Init()
    low_pub = ChannelPublisher(TOPIC_LOW_STATE, LowState_)
    low_pub.Init()
    return sport_pub, low_pub, unitree_go_msg_dds__SportModeState_, unitree_go_msg_dds__LowState_


def run(*, seconds: float, hz: float, domain_id: int, family: str) -> Dict[str, Any]:
    transport = DdsTransport(interface="", domain_id=domain_id, family=family)
    # 只读连接：本自检不碰命令通道，也确认 call() 在只读下确实被拒
    transport.connect_readonly()
    sport_pub, low_pub, make_sport, make_low = _build_publishers(domain_id)
    time.sleep(0.8)  # DDS 发现

    sport_msg, low_msg = make_sport(), make_low()
    period = 1.0 / float(hz)
    n = int(seconds * hz)
    base_sec = int(time.time())

    ages: List[float] = []
    device_stamps: List[float] = []
    sport_seen = low_seen = 0
    missing: List[str] = []

    for i in range(n):
        t = i * period
        sport_msg.stamp.sec = base_sec + int(t)
        sport_msg.stamp.nanosec = int((t % 1.0) * 1e9)
        sport_msg.position = [t * 0.5, 0.0, 0.30]
        sport_msg.velocity = [0.5, 0.0, 0.0]
        sport_msg.yaw_speed = 0.05
        sport_msg.imu_state.rpy = [0.0, 0.0, 0.1]
        sport_msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        sport_msg.imu_state.gyroscope = [0.01, 0.02, 0.03]
        sport_msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
        sport_msg.foot_force = [10, 20, 30, 40]
        sport_pub.Write(sport_msg)

        low_msg.tick = i
        low_msg.bms_state.soc = 77
        low_msg.foot_force = [10, 20, 30, 40]
        for m in low_msg.motor_state[:12]:
            m.q, m.dq, m.tau_est, m.temperature = 0.5, 0.1, 2.0, 40
        low_pub.Write(low_msg)

        time.sleep(period)

        sample = transport.read(TOPIC_SPORT_STATE)
        if sample is not None:
            sport_seen += 1
            age = transport.sample_age_s(TOPIC_SPORT_STATE)
            if age is not None:
                ages.append(age)
            if sample.get("t_device") is not None:
                device_stamps.append(float(sample["t_device"]))
            for k in REQUIRED_SPORT_KEYS:
                if sample.get(k) is None:
                    missing.append(f"sport.{k}")
            for k in REQUIRED_IMU_KEYS:
                if not (sample.get("imu_state") or {}).get(k):
                    missing.append(f"sport.imu_state.{k}")
        low = transport.read(TOPIC_LOW_STATE)
        if low is not None:
            low_seen += 1
            motors = low.get("motor_state") or []
            if motors:
                for k in REQUIRED_MOTOR_KEYS:
                    if k not in motors[0]:
                        missing.append(f"low.motor_state.{k}")

    # 断流：停发后 is_dds_stale 必须在阈值内翻真
    time.sleep(DDS_STALE_S + 0.3)
    stale_age = transport.sample_age_s(TOPIC_SPORT_STATE)
    stale_detected = stale_age is not None and stale_age > DDS_STALE_S

    # 只读连接下命令必须被拒（fail-closed）
    command_refused = False
    try:
        transport.call(1008, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
    except Exception as exc:  # noqa: BLE001
        command_refused = type(exc).__name__ == "UnitreeNotConnected"

    device_monotonic = all(b >= a for a, b in zip(device_stamps, device_stamps[1:]))
    missing_unique = sorted(set(missing))

    result: Dict[str, Any] = {
        "tool": "dds_selfcheck",
        "scope": "订阅链路软件正确性（真实 DDS + 真实 IDL，合成数据源）",
        "not_in_scope": ["真机时序/抖动", "命令通道 E4-C", "现场网络隔离"],
        "family": family,
        "domain_id": domain_id,
        "target_hz": hz,
        "seconds": seconds,
        # 采样年龄是「读取时刻 − 收到时刻」，本工具在 sleep 一个周期后才读，
        # 所以它主要反映轮询节拍，**不是 DDS 端到端时延**。别拿它当延迟指标。
        "age_note": "dominated by polling period, not DDS latency",
        "frames_published": n,
        "sport_samples_read": sport_seen,
        "low_samples_read": low_seen,
        "age_p50_ms": round(statistics.median(ages) * 1000, 2) if ages else None,
        "age_p95_ms": round(_pct(ages, 0.95) * 1000, 2) if ages else None,
        "age_p99_ms": round(_pct(ages, 0.99) * 1000, 2) if ages else None,
        "age_max_ms": round(max(ages) * 1000, 2) if ages else None,
        "device_stamp_count": len(device_stamps),
        "device_stamp_monotonic": device_monotonic,
        "missing_fields": missing_unique,
        "stale_detected_after_silence": stale_detected,
        "command_refused_in_readonly": command_refused,
        "ts": time.time(),
    }
    result["subscribe_path_ok"] = bool(
        sport_seen >= n * 0.95
        and low_seen >= n * 0.95
        and not missing_unique
        and device_monotonic
        and len(device_stamps) >= n * 0.95
        and stale_detected
        and command_refused
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--domain-id", type=int, default=7, help="本地自检用非 0 域，避开真机")
    ap.add_argument("--family", default="b2", choices=("b2", "go2"))
    ap.add_argument("--interface", default="", help="留空则交给 CYCLONEDDS_URI")
    ap.add_argument(
        "--observe",
        action="store_true",
        help="只订阅不发布，量外部发布端（真狗 / 宇树 MuJoCo 桥）。真机 D0 用这个",
    )
    ap.add_argument("--poll-hz", type=float, default=200.0, help="observe 模式的轮询频率")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.observe:
        result = observe(
            seconds=args.seconds,
            poll_hz=args.poll_hz,
            domain_id=args.domain_id,
            family=args.family,
            interface=args.interface,
        )
    else:
        result = run(
            seconds=args.seconds, hz=args.hz, domain_id=args.domain_id, family=args.family
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写入 {args.out}", file=sys.stderr)
    return 0 if result["subscribe_path_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
