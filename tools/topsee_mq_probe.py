#!/usr/bin/env python3
"""拓普视平台 STOMP-over-WebSocket 推送通道只读探针。

用途：验证 `docs/references/2026-08-08-topsee-interface-inventory.md` §3.4 逆向出的
推送通道，回答两个问题：

1. `service_robot_status` 是否携带位姿 / 定位置信度（决定 G4/G5 能否绕开 DDS 关闭）；
2. `service_robot_task_result` 的到点消息结构（决定到点检测能否从 HTTP 轮询改为推送）。

**安全约束（硬编码，不可通过参数放开）**：本工具只发 `CONNECT` / `SUBSCRIBE` /
心跳 / `DISCONNECT` 四种帧。`_transmit()` 对帧类型做白名单校验，任何控制指令
（如 `SEND` 到 `/edg` 的 `move-by-direction`）都发不出去。

零第三方依赖：自带最小 RFC6455 客户端，避免为一次探测引入新依赖。

凭据从环境变量读，不落盘、不进 shell 历史：
    export TOPSEE_MQ_USER=...
    export TOPSEE_MQ_PASS=...
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import ssl
import struct
import sys
import time
from collections import defaultdict
from typing import Any, Iterator
from urllib.parse import urlparse

# 逆向自平台前端的 STOMP 目的地（见接口清单 §3.4）。
# ⚠️ 关键：实测这些主题**必须拼 `.<robotNum>` 后缀**才收得到消息；
#    订裸主题会静默收到 0 帧且无 ERROR 帧，极易误判为「平台没在推」。
PER_ROBOT = {
    "status": "/topic/service_robot_status",
    "scan": "/topic/service_robot_scan",
    "velodyne": "/topic/service_robot_velodyne_base_laser",
    "task_result": "/topic/service_robot_task_result",
    "map": "/topic/service_robot_map_base64",
    "task_item": "/topic/service_robot_taskItem",
}
# 前端以 `.>` 通配订阅的主题，按原样订阅
GLOBAL = {
    "event": "/topic/service_robot_event.>",
    "alarm": "/topic/service_robot_alarm.>",
    "obstacle": "/topic/service_robot_obstacle_avoidance_topic",
}
DESTINATIONS = {**PER_ROBOT, **GLOBAL}

# 位姿 / 置信度候选字段名，用于回答 G4/G5
POSE_HINTS = re.compile(
    r"^(x|y|z|th|theta|yaw|pitch|roll|heading|orientation|pose|position|"
    r"confidence|score|quality|loc\w*|map_?x|map_?y|pos_?[xyz]|"
    r"linear|angular|twist|velocity|vel_?[xyz])$",
    re.I,
)

ALLOWED_COMMANDS = frozenset({"CONNECT", "SUBSCRIBE", "DISCONNECT"})


class WebSocket:
    """最小 RFC6455 客户端：只需文本帧收发 + ping/pong。"""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        u = urlparse(url)
        secure = u.scheme in ("wss", "https")
        port = u.port or (443 if secure else 80)
        path = u.path or "/"
        if u.query:
            path += "?" + u.query

        self._sock = socket.create_connection((u.hostname, port), timeout=timeout)
        if secure:
            ctx = ssl._create_unverified_context()
            self._sock = ctx.wrap_socket(self._sock, server_hostname=u.hostname)

        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {u.hostname}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._sock.sendall(req.encode())

        self._buf = b""
        while b"\r\n\r\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("握手期间连接关闭")
            self._buf += chunk
        head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in status:
            raise ConnectionError(f"WebSocket 升级失败: {status}")

    def send_text(self, payload: str) -> None:
        data = payload.encode()
        header = bytearray([0x81])  # FIN + text
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self._sock.sendall(bytes(header) + masked)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("连接关闭")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv_text(self) -> str | None:
        """返回一条文本消息；控制帧内部消化后返回 None。"""
        b0, b1 = self._recv_exact(2)
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._recv_exact(8))[0]
        payload = self._recv_exact(length) if length else b""

        if opcode == 0x9:  # ping -> pong
            self._sock.sendall(b"\x8a\x80" + os.urandom(4))
            return None
        if opcode == 0x8:  # close
            raise ConnectionError("服务端关闭连接")
        if opcode in (0x1, 0x0):
            return payload.decode(errors="replace")
        return None

    def close(self) -> None:
        try:
            self._sock.sendall(b"\x88\x80" + os.urandom(4))
        except OSError:
            pass
        self._sock.close()


class ReadOnlyStomp:
    """STOMP 客户端，帧类型白名单强制只读。"""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    def _transmit(self, command: str, headers: dict[str, str]) -> None:
        if command not in ALLOWED_COMMANDS:
            raise PermissionError(
                f"拒绝发送 {command}：本探针为只读工具，仅允许 {sorted(ALLOWED_COMMANDS)}"
            )
        lines = [command] + [f"{k}:{v}" for k, v in headers.items()]
        self._ws.send_text("\n".join(lines) + "\n\n\x00")

    def connect(self, login: str, passcode: str, host: str = "/") -> None:
        self._transmit(
            "CONNECT",
            {
                "accept-version": "1.0,1.1,1.2",
                "host": host,
                "login": login,
                "passcode": passcode,
                "heart-beat": "10000,10000",
            },
        )

    def subscribe(self, sub_id: str, destination: str) -> None:
        self._transmit("SUBSCRIBE", {"id": sub_id, "destination": destination, "ack": "auto"})

    def disconnect(self) -> None:
        self._transmit("DISCONNECT", {})

    def heartbeat(self) -> None:
        self._ws.send_text("\n")

    IDLE = "__IDLE__"

    def frames(self) -> Iterator[tuple[str, dict[str, str], str]]:
        """产出 STOMP 帧；读超时时产出 IDLE 让调用方有机会心跳/判超时。"""
        pending = ""
        while True:
            try:
                text = self._ws.recv_text()
            except socket.timeout:
                yield self.IDLE, {}, ""
                continue
            if text is None:
                continue
            pending += text
            while "\x00" in pending:
                raw, pending = pending.split("\x00", 1)
                raw = raw.lstrip("\n")
                if not raw:
                    continue
                head, _, body = raw.partition("\n\n")
                lines = head.split("\n")
                cmd = lines[0]
                headers = {}
                for line in lines[1:]:
                    k, _, v = line.partition(":")
                    headers[k] = v
                yield cmd, headers, body


def _liveness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """区分「字段存在」与「字段真的在动」——防 MuJoCo 式假绿。"""
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for field in ("t", "x", "y", "th", "matchProb", "seq"):
        vals = [r[field] for r in rows if r.get(field) is not None]
        if not vals:
            continue
        distinct = len(set(vals))
        out[field] = {
            "samples": len(vals),
            "distinct": distinct,
            "constant": distinct == 1,
            "first": vals[0],
            "last": vals[-1],
        }
    return out


def collect_keys(obj: Any, prefix: str = "", out: set[str] | None = None) -> set[str]:
    out = out if out is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            collect_keys(v, path, out)
    elif isinstance(obj, list) and obj:
        collect_keys(obj[0], f"{prefix}[]", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="平台地址，如 http://<主机>:8888")
    ap.add_argument("--path", default="/robot/mq/", help="STOMP WebSocket 路径")
    ap.add_argument("--robot-num", default=os.environ.get("TOPSEE_ROBOT_NUM"),
                    help="机器人 SN；按机器人分发的主题会自动拼为 <topic>.<robotNum>")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--topics", default="status,task_result,task_item,event,alarm",
                    help=f"逗号分隔，可选：{','.join(DESTINATIONS)}")
    ap.add_argument("--extra", default="", action="append" if False else "store",
                    help="额外订阅的原始目的地，逗号分隔。诊断用，例如 /topic/# 或 /exchange/amq.topic/#")
    ap.add_argument("--max-samples", type=int, default=2, help="每主题保留几条样本进报告")
    ap.add_argument("--out", default=None, help="报告输出路径（JSON）")
    args = ap.parse_args()

    user = os.environ.get("TOPSEE_MQ_USER")
    passwd = os.environ.get("TOPSEE_MQ_PASS")
    if not user or not passwd:
        print("错误：请设置 TOPSEE_MQ_USER / TOPSEE_MQ_PASS 环境变量", file=sys.stderr)
        return 2

    u = urlparse(args.base_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    ws_url = f"{scheme}://{u.netloc}{args.path}"

    wanted = [t.strip() for t in args.topics.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in DESTINATIONS]
    if unknown:
        print(f"错误：未知主题 {unknown}，可选 {list(DESTINATIONS)}", file=sys.stderr)
        return 2

    if not args.robot_num and any(t in PER_ROBOT for t in wanted):
        print(f"错误：{sorted(set(wanted) & set(PER_ROBOT))} 为按机器人分发的主题，"
              "必须提供 --robot-num（SN），否则必定 0 帧", file=sys.stderr)
        return 2

    destinations = {
        t: (f"{DESTINATIONS[t]}.{args.robot_num}" if t in PER_ROBOT else DESTINATIONS[t])
        for t in wanted
    }
    for raw in (d.strip() for d in args.extra.split(",")):
        if raw:
            destinations[raw] = raw
    wanted = list(destinations)

    print(f"连接 {ws_url} …", file=sys.stderr)
    ws = WebSocket(ws_url)
    stomp = ReadOnlyStomp(ws)
    stomp.connect(user, passwd)

    counts: dict[str, int] = defaultdict(int)
    keys: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[Any]] = defaultdict(list)
    first_ts: dict[str, float] = {}
    last_ts: dict[str, float] = {}
    non_json: dict[str, int] = defaultdict(int)
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    connected = False

    dest_to_name = {v: k for k, v in destinations.items()}
    deadline = time.monotonic() + args.seconds
    last_hb = time.monotonic()

    try:
        ws._sock.settimeout(1.0)
        for cmd, headers, body in stomp.frames():
            now = time.monotonic()
            if cmd == ReadOnlyStomp.IDLE:
                pass
            elif cmd == "CONNECTED":
                connected = True
                print("STOMP 已连接，订阅中…", file=sys.stderr)
                for i, t in enumerate(wanted):
                    stomp.subscribe(f"probe-{i}", destinations[t])
            elif cmd == "ERROR":
                errors.append(headers.get("message", "") or body[:200])
                break
            elif cmd == "MESSAGE":
                dest = headers.get("destination", "")
                name = dest_to_name.get(dest)
                if name is None:  # 通配符订阅，回落到前缀匹配
                    name = next((n for d, n in dest_to_name.items()
                                 if dest.startswith(d.rstrip(">"))), dest)
                counts[name] += 1
                first_ts.setdefault(name, now)
                last_ts[name] = now
                try:
                    doc = json.loads(body)
                except ValueError:
                    non_json[name] += 1
                else:
                    keys[name] |= collect_keys(doc)
                    if len(samples[name]) < args.max_samples:
                        samples[name].append(doc)
                    if isinstance(doc, dict):
                        pos = doc.get("position") or {}
                        series[name].append({
                            "t": doc.get("time"),
                            "x": pos.get("x"), "y": pos.get("y"), "th": pos.get("th"),
                            "matchProb": doc.get("matchProb"),
                            "seq": (doc.get("header") or {}).get("seq"),
                        })

            if now - last_hb > 8:
                stomp.heartbeat()
                last_hb = now
            if now > deadline:
                break
    except ConnectionError as exc:
        errors.append(str(exc))
    finally:
        try:
            stomp.disconnect()
        except Exception:  # noqa: BLE001
            pass
        ws.close()

    duration = args.seconds
    topics_report = {}
    for name in wanted:
        n = counts.get(name, 0)
        span = (last_ts.get(name, 0) - first_ts.get(name, 0)) or 0.0
        k = sorted(keys.get(name, ()))
        topics_report[name] = {
            "destination": destinations[name],
            "frames": n,
            "hz": round(n / duration, 3) if duration else None,
            "hz_within_span": round((n - 1) / span, 3) if n > 1 and span > 0 else None,
            "non_json_frames": non_json.get(name, 0),
            "keys": k,
            "pose_like_keys": [x for x in k if POSE_HINTS.match(x.split(".")[-1].rstrip("[]"))],
            "liveness": _liveness(series.get(name, [])),
            "samples": samples.get(name, []),
        }

    status = topics_report.get("status", {})
    report = {
        "tool": "topsee_mq_probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ws_path": args.path,
        "duration_s": duration,
        "stomp_connected": connected,
        "errors": errors,
        "topics": topics_report,
        "verdict": {
            "push_channel_ok": connected and sum(counts.values()) > 0,
            # G4/G5：位姿能否绕开 DDS
            "status_has_pose_like_fields": bool(status.get("pose_like_keys")),
            "task_result_pushes": counts.get("task_result", 0) > 0,
        },
        "notes": [
            "本探针只发 CONNECT/SUBSCRIBE/心跳/DISCONNECT，不具备下发控制指令的能力。",
            "pose_like_keys 为字段名启发式命中，仍需人工确认语义、坐标系与单位。",
            "task_result 为事件型主题，静止期无任务时 0 帧属正常，不可据此判定通道失效。",
        ],
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"报告已写入 {args.out}", file=sys.stderr)
    else:
        print(out)

    for name, info in topics_report.items():
        print(f"  {name:12s} {info['frames']:5d} 帧  {info['hz']:>7} Hz  "
              f"位姿候选={info['pose_like_keys'] or '无'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
