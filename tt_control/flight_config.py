"""飞行配置加载管线：JSON 配置文件 → dataclass 实例。

优先级：CLI 参数 > 配置文件 > dataclass 字段默认值。

用法:
    cfg = load_config("configs/my_test.json")
    avoid_params = build_avoid_params(cfg)
    orbit_params = build_orbit_params(cfg)
    fsm_params = build_fsm_params(cfg)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tt_control.avoidance import AvoidParams, OrbitParams
from tt_control.avoidance_fsm import FsmParams
from tt_control.wander import WanderParams

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 JSON 配置文件。

    路径为空或无文件时返回空字典（调用方将使用 dataclass 默认值）。
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            logger.info("默认配置文件 %s 不存在，使用 dataclass 默认值", path)
        else:
            logger.warning("配置文件 %s 不存在，使用 dataclass 默认值", path)
        return {}

    with open(path) as f:
        cfg: dict[str, Any] = json.load(f)

    logger.info("已加载配置文件 %s", path)
    return cfg


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """取配置节，过滤掉以 _ 或 // 为前缀的键（注释）。"""
    raw = cfg.get(name, {})
    return {k: v for k, v in raw.items() if not (k.startswith("_") or k.startswith("//"))}


def build_avoid_params(cfg: dict[str, Any] | None = None) -> AvoidParams:
    """从配置构建避障控制律参数，缺失字段使用 dataclass 默认值。"""
    return AvoidParams(**(_section(cfg or {}, "avoid")))


def build_orbit_params(cfg: dict[str, Any] | None = None) -> OrbitParams:
    """从配置构建环绕参数，缺失字段使用 dataclass 默认值。"""
    return OrbitParams(**(_section(cfg or {}, "orbit")))


def build_fsm_params(cfg: dict[str, Any] | None = None) -> FsmParams:
    """从配置构建状态机参数，缺失字段使用 dataclass 默认值。"""
    return FsmParams(**(_section(cfg or {}, "fsm")))


def build_wander_params(cfg: dict[str, Any] | None = None) -> WanderParams:
    """从配置构建漫游参数，缺失字段使用 dataclass 默认值。"""
    return WanderParams(**(_section(cfg or {}, "wander")))
