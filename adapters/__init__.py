"""平台适配器：统一 Scout / Dog 契约，对接 MissionBrain。

机器狗为分时双通道（见 docs/design/2026-08-03-dog-integration-plan.md）：
  慢通道 dog_topsee  —— 拓普视平台 HTTP，负责语义级派单/取证/气体
  快通道 dog_unitree —— 宇树 DDS，负责 WAM 的连续速度与高频状态
  两者由 dog_arbiter 互斥仲裁，任何时刻只有一个通道被授权下发。
"""

from adapters.dog_arbiter import ArbiterState, DogControlArbiter
from adapters.dog_stub import DogStubAdapter
from adapters.dog_topsee import NavStatus, TopseeGas, TopseeNav, TopseePerception
from adapters.dog_unitree import SpeedLimits, UnitreeSportClient
from adapters.drone_autel import AutelScoutAdapter
from adapters.drone_tello import TelloScoutAdapter
from adapters.gas_ledger import GasCalibrationLedger
from adapters.topsee_client import TopseeClient

__all__ = [
    "ArbiterState",
    "AutelScoutAdapter",
    "DogControlArbiter",
    "DogStubAdapter",
    "GasCalibrationLedger",
    "NavStatus",
    "SpeedLimits",
    "TelloScoutAdapter",
    "TopseeClient",
    "TopseeGas",
    "TopseeNav",
    "TopseePerception",
    "UnitreeSportClient",
]
