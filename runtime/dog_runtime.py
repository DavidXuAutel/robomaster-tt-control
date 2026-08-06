"""狗侧生产接线入口（v3.1 D0）。

装配 TopseeClient / TopseeNav / TopseeGas / Arbiter / DogSdkAdapter 的**唯一**
进程编排点。含健康检查、日志轮转、磁盘水位（<10% 停录制）。

`main.py` 故意不动——GUI 飞行栈与狗侧部署闭环分进程，避免互相拖垮。
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from adapters.dog_arbiter import DogControlArbiter
from adapters.dog_sdk import DogSdkAdapter
from adapters.dog_topsee import TopseeGas, TopseeNav, TopseePerception
from adapters.dog_unitree import LoopbackTransport, UnitreeSportClient
from adapters.gas_ledger import GasCalibrationLedger
from adapters.topsee_client import TopseeClient
from mission_brain.map_model import SharedMap

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/dog/topsee.json")


@dataclass
class DogRuntimeConfig:
    """`configs/dog/topsee.json` 的强类型视图；E 回填六项只从这里读。"""

    base_url: str
    account: str
    robot_id: str
    password_env: str = "TOPSEE_PASSWORD"
    deployment: str = "cloud"
    arrived_states: List[str] = field(default_factory=list)
    enroute_states: List[str] = field(default_factory=list)
    battery_field: str = ""
    time_format: str = ""
    token_header: str = ""
    alarm_fields: Dict[str, Any] = field(default_factory=dict)
    controller_state: Optional[str] = None
    controller_force: Optional[str] = None
    dds_interface: str = "eth0"
    dds_domain_id: int = 0
    transport: str = "loopback"
    min_confidence: float = 0.9
    min_battery_pct: float = 25.0
    disk_min_free_ratio: float = 0.10
    record_root: str = "data/dog_episodes"
    audit_dir: str = "logs/audit"
    log_dir: str = "logs/dog_runtime"
    log_max_bytes: int = 5_242_880
    log_backup_count: int = 5
    shared_map: str = "configs/mission/shared_map.example.json"
    gas_calibration: str = "configs/mission/gas_calibration.example.json"
    password: Optional[str] = None

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "DogRuntimeConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kw = {k: v for k, v in doc.items() if k in known}
        return cls(**kw)  # type: ignore[arg-type]


def load_topsee_config(path: Union[str, Path] = DEFAULT_CONFIG) -> DogRuntimeConfig:
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ValueError(f"配置必须是 JSON object: {p}")
    return DogRuntimeConfig.from_dict(doc)


class DogRuntime:
    """生产装配 + 健康门禁。Loopback 下可单测；真机改 transport=dds。"""

    def __init__(
        self,
        config: Union[DogRuntimeConfig, str, Path] = DEFAULT_CONFIG,
        *,
        emit: Optional[Callable[[Mapping[str, Any]], None]] = None,
        unitree: Optional[UnitreeSportClient] = None,
        topsee: Optional[TopseeClient] = None,
        battery_provider: Optional[Callable[[], Optional[float]]] = None,
        password: Optional[str] = None,
        autostart_poller: bool = False,
        setup_logging: bool = True,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_topsee_config(config)
        else:
            self.config = config
        if password is not None:
            self.config.password = password

        self.events: List[Mapping[str, Any]] = []
        self._emit = emit or self.events.append
        self.recording_enabled = True
        self._closed = False

        if setup_logging:
            self._configure_logging()

        self.shared_map = self._load_map()
        self.client = topsee or self._make_client()
        self.unitree = unitree or self._make_unitree()
        self.arbiter = DogControlArbiter(
            self.client,
            self.unitree,
            robot_id=self.config.robot_id,
            min_confidence=self.config.min_confidence,
            min_battery_pct=self.config.min_battery_pct,
            controller_state=self.config.controller_state,
            controller_force=self.config.controller_force,
            battery_provider=battery_provider or self._default_battery_provider,
            audit_dir=self.config.audit_dir,
        )
        self.nav = TopseeNav(
            self.client,
            robot_id=self.config.robot_id,
            arbiter=self.arbiter,
            goal_resolver=self.shared_map.resolve_points_id,
            goal_pose_resolver=self.shared_map.goal_pose,
            pose_source=self.unitree.pose_xy_yaw,
            arrived_states=tuple(self.config.arrived_states),
            enroute_states=tuple(self.config.enroute_states),
            autostart_poller=autostart_poller,
        )
        if self.config.time_format:
            # 回填入口只写配置；dog_topsee._fmt_time 在 D1 探针后消费
            logger.info("time_format 已配置: %s", self.config.time_format)
        ledger = GasCalibrationLedger.load(self.config.gas_calibration)
        self.gas = TopseeGas(self.client, robot_id=self.config.robot_id, ledger=ledger)
        self.perception = TopseePerception(
            self.client,
            robot_id=self.config.robot_id,
            mode="local_vision",
            detector=lambda _label: None,
            autostart_poller=False,
        )
        self.dog = DogSdkAdapter(
            self._emit,
            mode="backend",
            nav=self.nav,
            perception=self.perception,
            gas=self.gas,
            arbiter=self.arbiter,
        )
        logger.info(
            "DogRuntime ready robot=%s transport=%s record_root=%s",
            self.config.robot_id,
            self.config.transport,
            self.config.record_root,
        )

    # ---------- 装配辅助 ----------

    def _configure_logging(self) -> None:
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "dog_runtime.log",
            maxBytes=int(self.config.log_max_bytes),
            backupCount=int(self.config.log_backup_count),
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root = logging.getLogger()
        # 避免重复挂载同类 handler
        for h in list(root.handlers):
            if isinstance(h, logging.handlers.RotatingFileHandler) and getattr(
                h, "baseFilename", ""
            ).endswith("dog_runtime.log"):
                return
        root.addHandler(handler)
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)

    def _load_map(self) -> SharedMap:
        path = Path(self.config.shared_map)
        doc = json.loads(path.read_text(encoding="utf-8"))
        return SharedMap.from_dict(doc)

    def _password(self) -> str:
        if self.config.password:
            return str(self.config.password)
        env = os.environ.get(self.config.password_env, "")
        if not env:
            raise RuntimeError(
                f"未设置密码：请 export {self.config.password_env}=... 或传入 password="
            )
        return env

    def _make_client(self) -> TopseeClient:
        kw: Dict[str, Any] = {
            "account": self.config.account,
            "password": self._password(),
            "deployment": self.config.deployment or "cloud",
        }
        if self.config.token_header:
            kw["token_header"] = self.config.token_header
        return TopseeClient(self.config.base_url, **kw)

    def _make_unitree(self) -> UnitreeSportClient:
        mode = (self.config.transport or "loopback").lower()
        if mode == "dds":
            from adapters.dog_unitree import DdsTransport

            transport: Any = DdsTransport(
                interface=self.config.dds_interface,
                domain_id=self.config.dds_domain_id,
            )
        elif mode == "loopback":
            transport = LoopbackTransport()
        else:
            raise ValueError(f"未知 transport={self.config.transport!r}（loopback|dds）")
        client = UnitreeSportClient(transport)
        client.connect()
        return client

    def _default_battery_provider(self) -> Optional[float]:
        """从 /lowstate 读 soc；字段名由 battery_field 覆盖（E10 回填）。"""
        low = self.unitree.get_low_state()
        if not isinstance(low, Mapping):
            return None
        field_name = self.config.battery_field or "soc"
        bms = low.get("bms_state")
        if isinstance(bms, Mapping) and field_name in bms:
            try:
                return float(bms[field_name])
            except (TypeError, ValueError):
                return None
        if field_name in low:
            try:
                return float(low[field_name])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        # 常见默认
        if isinstance(bms, Mapping) and "soc" in bms:
            try:
                return float(bms["soc"])
            except (TypeError, ValueError):
                return None
        return None

    # ---------- 运行门禁 ----------

    def disk_free_ratio(self, path: Optional[Union[str, Path]] = None) -> float:
        target = Path(path or self.config.record_root)
        try:
            target.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(target))
        except OSError as exc:
            logger.error("disk_usage 失败，按 0 处理: %s", exc)
            return 0.0
        if usage.total <= 0:
            return 0.0
        return float(usage.free) / float(usage.total)

    def refresh_disk_gate(self) -> bool:
        """磁盘水位 < disk_min_free_ratio → 停录制，返回 False。"""
        ratio = self.disk_free_ratio()
        ok = ratio >= float(self.config.disk_min_free_ratio)
        if not ok and self.recording_enabled:
            logger.error(
                "磁盘水位过低 free=%.1f%% < %.1f%%，停录制",
                ratio * 100.0,
                float(self.config.disk_min_free_ratio) * 100.0,
            )
        self.recording_enabled = ok
        return ok

    def health_check(self) -> Dict[str, Any]:
        """装配/通道健康快照。不抛异常。

        transport=dds 时 dds_stale 计入 ok（断流＝不健康）；
        loopback 仅作软件自检，不因无真机样本判红。
        """
        try:
            disk_ok = self.refresh_disk_gate()
            disk_ratio = self.disk_free_ratio()
        except Exception as exc:  # noqa: BLE001 — 健康检查永不抛
            logger.exception("disk gate 失败: %s", exc)
            disk_ok = False
            disk_ratio = 0.0
            self.recording_enabled = False
        dds_stale = True
        try:
            dds_stale = self.unitree.is_dds_stale()
        except Exception as exc:  # noqa: BLE001
            logger.warning("dds stale 检查失败: %s", exc)
        transport = (self.config.transport or "loopback").lower()
        dds_ok = (not dds_stale) if transport == "dds" else True
        return {
            "ok": bool(disk_ok and dds_ok and not self._closed),
            "robot_id": self.config.robot_id,
            "transport": self.config.transport,
            "arbiter_state": self.arbiter.state.value,
            "recording_enabled": self.recording_enabled,
            "disk_free_ratio": disk_ratio,
            "disk_ok": disk_ok,
            "dds_stale": dds_stale,
            "dds_ok": dds_ok,
            "ts": time.time(),
        }

    def tick(self, now: Optional[float] = None) -> None:
        """单步推进 dog + arbiter（进程编排的周期入口）。"""
        t = float(now if now is not None else time.time())
        self.arbiter.tick(now=t)
        self.dog.tick(now=t)

    def run_forever(self, *, hz: float = 10.0, health_every_s: float = 5.0) -> int:
        """生产进程主循环：周期 tick + 健康检查；信号触发 abort/close。"""
        if hz <= 0:
            raise ValueError("hz 必须为正")
        period = 1.0 / float(hz)
        stop = {"flag": False}

        def _stop(_signum: int, _frame: Any) -> None:
            logger.warning("收到信号 %s，准备 abort/close", _signum)
            stop["flag"] = True

        prev_int = signal.signal(signal.SIGINT, _stop)
        prev_term = signal.signal(signal.SIGTERM, _stop)
        last_health = 0.0
        rc = 0
        try:
            while not stop["flag"]:
                t0 = time.monotonic()
                self.tick()
                now = time.monotonic()
                if now - last_health >= float(health_every_s):
                    h = self.health_check()
                    last_health = now
                    if not h["ok"]:
                        logger.error("health_check 失败: %s", h)
                        # 磁盘红停录制已在 refresh_disk_gate；DDS 断流走 safe_hold
                        if h.get("dds_stale") and (self.config.transport or "").lower() == "dds":
                            self.arbiter.safe_hold("dds_stale")
                slept = period - (time.monotonic() - t0)
                if slept > 0:
                    time.sleep(slept)
        except Exception:  # noqa: BLE001
            logger.exception("run_forever 异常")
            rc = 1
            try:
                self.abort("runtime_exception")
            except Exception:  # noqa: BLE001
                logger.exception("abort after exception failed")
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            self.close()
        return rc

    def abort(self, reason: str) -> None:
        """任务级停止：dog.abort 内部会 force_release。"""
        self.dog.abort(reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.arbiter.force_release("runtime_close")
        except Exception:  # noqa: BLE001
            logger.exception("close force_release failed")
        try:
            self.unitree.close()
        except Exception:  # noqa: BLE001
            logger.exception("unitree.close failed")
        stop = getattr(self.nav, "stop", None) or getattr(self.nav, "close", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                logger.exception("nav stop failed")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="狗侧生产接线入口（v3.1 D0）")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="configs/dog/topsee.json")
    p.add_argument(
        "--once-health",
        action="store_true",
        help="装配后打印一次 health_check JSON 后退出（默认）",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="进入周期 tick 主循环（需现场网络；默认 transport=loopback 可干跑）",
    )
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--health-every", type=float, default=5.0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # 默认 once-health；--run 才常驻
    rt = DogRuntime(args.config, autostart_poller=bool(args.run))
    if args.run:
        return rt.run_forever(hz=args.hz, health_every_s=args.health_every)
    h = rt.health_check()
    print(json.dumps(h, ensure_ascii=False, indent=2))
    rt.close()
    return 0 if h.get("ok") else 4


if __name__ == "__main__":
    sys.exit(main())
