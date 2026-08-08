"""D0：DogRuntime 装配 / 健康检查 / 磁盘水位 / abort→force_release。"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.dog_arbiter import ArbiterState
from adapters.dog_unitree import LoopbackTransport, UnitreeSportClient
from adapters.topsee_client import TopseeClient
from runtime.dog_runtime import DogRuntime, DogRuntimeConfig, load_topsee_config
from tests.fixtures.topsee_fake import FakeTopseeServer

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "dog" / "topsee.json"


@pytest.fixture
def srv():
    with FakeTopseeServer() as s:
        yield s


def _runtime(srv, tmp_path, **kw):
    client = TopseeClient(
        srv.base_url, account=srv.state.account, password=srv.state.password, timeout_s=5.0
    )
    client.login()
    unitree = UnitreeSportClient(LoopbackTransport())
    unitree.connect()
    cfg = load_topsee_config(CFG)
    cfg.account = srv.state.account
    cfg.password = srv.state.password
    cfg.base_url = srv.base_url
    cfg.audit_dir = str(tmp_path / "audit")
    cfg.log_dir = str(tmp_path / "logs")
    cfg.record_root = str(tmp_path / "episodes")
    cfg.transport = "loopback"
    return DogRuntime(
        cfg,
        topsee=client,
        unitree=unitree,
        battery_provider=lambda: 88.0,
        setup_logging=True,
        **kw,
    )


def test_load_topsee_config_has_e_backfill_slots():
    cfg = load_topsee_config(CFG)
    for key in (
        "arrived_states",
        "enroute_states",
        "battery_field",
        "time_format",
        "token_header",
        "alarm_fields",
    ):
        assert hasattr(cfg, key)


def test_runtime_assembles_and_health_ok(srv, tmp_path):
    rt = _runtime(srv, tmp_path)
    h = rt.health_check()
    assert h["ok"] is True
    assert h["disk_ok"] is True
    assert h["recording_enabled"] is True
    assert h["arbiter_state"] == ArbiterState.IDLE.value
    assert h["robot_id"] == rt.config.robot_id
    # Loopback 有新鲜样本 → 非 stale
    assert h["dds_stale"] is False
    rt.close()


def test_disk_gate_stops_recording(srv, tmp_path, monkeypatch):
    rt = _runtime(srv, tmp_path)

    class _Usage:
        total = 100
        free = 5  # 5% < 10%

        used = 95

    monkeypatch.setattr("runtime.dog_runtime.shutil.disk_usage", lambda _p: _Usage())
    assert rt.refresh_disk_gate() is False
    assert rt.recording_enabled is False
    h = rt.health_check()
    assert h["disk_ok"] is False
    assert h["ok"] is False
    rt.close()


def test_abort_triggers_force_release(srv, tmp_path):
    rt = _runtime(srv, tmp_path)
    rt.arbiter.ack_confidence(0.95, by="tester")
    rt.arbiter.acquire_for_mission("m1")
    assert rt.arbiter.state is ArbiterState.MISSION_NAV
    rt.abort("mission_abort")
    assert rt.arbiter.state is ArbiterState.IDLE
    assert rt.arbiter.no_owner
    assert rt.dog.abort_count == 1
    rt.close()


def test_from_dict_ignores_unknown_keys():
    cfg = DogRuntimeConfig.from_dict(
        {
            "base_url": "http://x",
            "account": "a",
            "robot_id": "r",
            "extra_future_key": 1,
        }
    )
    assert cfg.robot_id == "r"
    assert not hasattr(cfg, "extra_future_key")


def test_tick_advances_dog_and_arbiter(srv, tmp_path):
    rt = _runtime(srv, tmp_path)
    rt.arbiter.ack_confidence(0.95, by="t")
    rt.arbiter.acquire_for_mission("m1")
    rt.tick()
    assert rt.arbiter.state is ArbiterState.MISSION_NAV
    rt.close()


def test_health_ok_requires_fresh_dds_when_transport_dds(srv, tmp_path):
    rt = _runtime(srv, tmp_path)
    rt.config.transport = "dds"
    # Loopback 仍有样本 → dds_stale false → ok
    h = rt.health_check()
    assert h["dds_ok"] is True and h["ok"] is True
    # 断开状态源后应判不健康
    rt.unitree.transport.state_available = False
    h2 = rt.health_check()
    assert h2["dds_stale"] is True and h2["dds_ok"] is False and h2["ok"] is False
    rt.close()


def test_probe_gap_labels_match_v31_section2():
    from tools import topsee_probe as probe

    assert probe.EXPERIMENTS["E7"]["gap"] == "G12"
    assert probe.EXPERIMENTS["E9"]["gap"] == "alarm-schema"
    assert probe.EXPERIMENTS["E10"]["gap"] == "battery-field"
    assert "四步协议" in probe.MANUAL_ONLY["E4"]
