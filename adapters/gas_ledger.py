"""气体传感器标定台账（人工维护、进 git）。

为什么需要它：平台 OpenAPI 全文检索「标定 / 校准 / calibr」命中 0 次（F13），
`gasJson` 是不透明 blob。`GasBackend.calibration_at()` 只剩两个坏选项——
返回当前时间（门禁形同虚设，安全后门）或返回很早时间（气检永久瘫痪）。
方案 §5.4 的决策是引入人工台账：运维填、进版本库、可审计。

用 JSON 而非方案里写的 YAML：`requirements.txt` 不含 PyYAML，
不为一个配置文件新增依赖。字段语义与方案一致。
"""

from __future__ import annotations

import json
import logging
import time
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

logger = logging.getLogger(__name__)

# 台账缺失或过期时 calibration_reason() 的取值（方案 §5.4）
REASON_SOURCE_UNAVAILABLE = "calibration_source_unavailable"
REASON_STALE = "calibration_stale"


class GasLedgerError(ValueError):
    """台账格式错误。启动期就该炸，不要拖到气检时才发现。"""


def parse_iso8601(text: str) -> float:
    """把 ISO8601 时间解析成 Unix 秒。

    naive 时间按本地时区解释（现场运维手填的多半是本地时间）。
    """
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise GasLedgerError(f"无法解析时间 {text!r}（需 ISO8601）") from exc
    if dt.tzinfo is None:
        return time.mktime(dt.timetuple()) + dt.microsecond / 1e6
    return timegm(dt.utctimetuple()) + dt.microsecond / 1e6


class GasCalibrationLedger:
    """按 robot_id 索引的标定记录。

    台账文件形如：

        {
          "max_age_s": 604800,
          "sensors": [
            {
              "robot_id": "B2000397",
              "device_id": "gas_rs485",
              "channels": ["CH4", "H2S", "CO", "O2"],
              "calibrated_at": "2026-07-20T09:30:00+08:00",
              "calibrated_by": "张某",
              "certificate": "docs/references/gas-cal/2026-07-20.pdf"
            }
          ]
        }

    `max_age_s` 可选，只用于 `reason()` 的过期判断；真正的门禁阈值由
    `DogSdkAdapter.calibration_max_age_s` 决定，两者应保持一致。
    """

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Any]],
        *,
        max_age_s: float = 7 * 24 * 3600,
    ) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in entries.items()
        }
        self.max_age_s = float(max_age_s)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GasCalibrationLedger":
        sensors = data.get("sensors")
        if not isinstance(sensors, list):
            raise GasLedgerError("台账缺少 sensors 列表")
        entries: Dict[str, Dict[str, Any]] = {}
        for i, row in enumerate(sensors):
            if not isinstance(row, Mapping):
                raise GasLedgerError(f"sensors[{i}] 必须是对象")
            robot_id = row.get("robot_id")
            if not robot_id:
                raise GasLedgerError(f"sensors[{i}] 缺少 robot_id")
            if not row.get("calibrated_at"):
                raise GasLedgerError(f"sensors[{i}] 缺少 calibrated_at")
            if str(robot_id) in entries:
                raise GasLedgerError(f"robot_id 重复: {robot_id}")
            entries[str(robot_id)] = {
                "device_id": str(row.get("device_id", "gas_rs485")),
                "channels": [str(c) for c in row.get("channels", [])],
                "calibrated_at": parse_iso8601(str(row["calibrated_at"])),
                "calibrated_by": str(row.get("calibrated_by", "")),
                "certificate": str(row.get("certificate", "")),
            }
        max_age = data.get("max_age_s", 7 * 24 * 3600)
        try:
            max_age_f = float(max_age)
        except (TypeError, ValueError) as exc:
            raise GasLedgerError(f"max_age_s 非数值: {max_age!r}") from exc
        if max_age_f <= 0:
            raise GasLedgerError("max_age_s 必须为正")
        return cls(entries, max_age_s=max_age_f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GasCalibrationLedger":
        p = Path(path)
        try:
            with p.open("r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError as exc:
            raise GasLedgerError(f"标定台账不存在: {p}") from exc
        except json.JSONDecodeError as exc:
            raise GasLedgerError(f"标定台账不是合法 JSON: {p}: {exc}") from exc

    def calibration_at(self, robot_id: str) -> float:
        """返回标定 Unix 秒；无记录返回 0.0（→ 门禁判 stale）。"""
        row = self._entries.get(str(robot_id))
        if row is None:
            logger.warning("标定台账无 robot_id=%s 的记录，按未标定处理", robot_id)
            return 0.0
        return float(row["calibrated_at"])

    def reason(self, robot_id: str, now: Optional[float] = None) -> Optional[str]:
        """无记录 → source_unavailable；过期 → stale；正常 → None。"""
        row = self._entries.get(str(robot_id))
        if row is None:
            return REASON_SOURCE_UNAVAILABLE
        t = float(now if now is not None else time.time())
        if (t - float(row["calibrated_at"])) > self.max_age_s:
            return REASON_STALE
        return None

    def entry(self, robot_id: str) -> Optional[Dict[str, Any]]:
        row = self._entries.get(str(robot_id))
        return dict(row) if row is not None else None

    def robot_ids(self) -> list[str]:
        return list(self._entries.keys())
