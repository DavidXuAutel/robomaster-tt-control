"""契约扩展：DogSdkAdapter 的两个可选钩子 + 空 readings 处理。

三条都是为了「如实上报」而不是「静默降级」：
  1. nav.poll_fault()          → 平台语义对不上时报明确 reason，不装作在走路
  2. gas.calibration_reason()  → 区分「无标定数据源」与「台账过期」
  3. 空 readings               → GAS_FAILED(no_gas_data)，不编造读数

同时验证没有这些钩子的旧 backend（FakeNav / FakeGas）行为完全不变。
"""

import pytest

from adapters.dog_sdk import DogSdkAdapter
from adapters.fake_nav import FakeGas, FakeNav, FakePerception
from mission_brain.events import EventType, make_event


def _inspect(goal="wp_region_x_staging"):
    return make_event(
        EventType.DOG_INSPECT,
        mission_id="m",
        source="mission_brain",
        payload={
            "region_id": "region_x",
            "dog_goal_id": goal,
            "target_label": "object_a",
            "evidence_uri": "e",
            "deadline": 99.0,
        },
    )


def _gas_cmd():
    return make_event(
        EventType.GAS_SAMPLE,
        mission_id="m",
        source="mission_brain",
        payload={
            "region_id": "region_x",
            "target_label": "object_a",
            "sample_window_s": 30.0,
            "deadline": 99.0,
        },
    )


class FaultyNav(FakeNav):
    """带 poll_fault 的 nav：模拟 TopseeNav 报平台语义故障。"""

    def __init__(self, fault=None, **kw):
        super().__init__(**kw)
        self.fault = fault
        self.fault_polls = 0

    def poll_fault(self):
        self.fault_polls += 1
        f, self.fault = self.fault, None
        return f


class BoomNav(FakeNav):
    def poll_fault(self):
        raise RuntimeError("hook boom")


class ReasonGas(FakeGas):
    def __init__(self, reason=None, **kw):
        super().__init__(**kw)
        self._reason = reason

    def calibration_reason(self):
        return self._reason


class BoomGas(FakeGas):
    def calibration_reason(self):
        raise RuntimeError("hook boom")


def _dog(out, nav=None, gas=None, **kw):
    return DogSdkAdapter(
        out.append,
        mode="backend",
        nav=nav or FakeNav(),
        perception=FakePerception({"confidence": 0.9, "evidence_uri": "e"}),
        gas=gas or FakeGas(),
        **kw,
    )


class _FakeArbiter:
    def __init__(self):
        self.releases = []

    def force_release(self, reason, *, now=None):
        self.releases.append(reason)


def test_abort_calls_arbiter_force_release():
    """D0：abort 在持有 Arbiter 时必须 force_release。"""
    out = []
    arb = _FakeArbiter()
    dog = _dog(out, arbiter=arb)
    dog.begin_inspect(_inspect())
    dog.abort("mission_abort")
    assert arb.releases == ["mission_abort"]
    assert dog.abort_count == 1


def test_abort_on_abort_hook_does_not_bypass_arbiter():
    out = []
    seen = []
    arb = _FakeArbiter()
    dog = _dog(out, arbiter=arb, on_abort=seen.append)
    dog.abort("x")
    assert seen == ["x"]
    assert arb.releases == ["x"]  # 附加钩子，不能旁路 force_release


# ---------- nav.poll_fault ----------


def test_nav_fault_emits_inspect_failed_with_reason():
    out = []
    nav = FaultyNav(fault="nav_status_unrecognized", arrive_after_ticks=1)
    dog = _dog(out, nav=nav)
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    fails = [e for e in out if e["type"] == "dog.inspect_failed"]
    assert len(fails) == 1
    assert fails[0]["stage"] == "nav"
    assert fails[0]["reason"] == "nav_status_unrecognized"
    assert not any(e["type"] == "dog.arrived" for e in out)


def test_nav_fault_stops_further_arrival_attempts():
    out = []
    nav = FaultyNav(fault="nav_task_not_tracked", arrive_after_ticks=1)
    dog = _dog(out, nav=nav)
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    out.clear()
    dog.tick(now=2.0)
    dog.tick(now=3.0)
    assert out == []


def test_no_fault_means_normal_arrival():
    out = []
    nav = FaultyNav(fault=None, arrive_after_ticks=1)
    dog = _dog(out, nav=nav)
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    assert any(e["type"] == "dog.arrived" for e in out)
    assert nav.fault_polls == 1


def test_backend_without_hook_unchanged():
    """FakeNav 没有 poll_fault，行为必须与加钩子之前完全一致。"""
    out = []
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1))
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    assert any(e["type"] == "dog.arrived" for e in out)
    assert not any(e["type"] == "dog.inspect_failed" for e in out)


def test_hook_exception_does_not_break_main_flow():
    """诊断钩子炸了不许拖垮主流程。"""
    out = []
    dog = _dog(out, nav=BoomNav(arrive_after_ticks=1))
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    assert any(e["type"] == "dog.arrived" for e in out)


# ---------- gas.calibration_reason ----------


def test_calibration_reason_refines_gas_failed():
    out = []
    gas = ReasonGas(reason="calibration_source_unavailable", calibration_at=0.0)
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=gas)
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1_000_000.0)
    fails = [e for e in out if e["type"] == "gas.failed"]
    assert len(fails) == 1
    assert fails[0]["reason"] == "calibration_source_unavailable"


def test_calibration_reason_absent_falls_back_to_stale():
    out = []
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=FakeGas(calibration_at=0.0))
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1_000_000.0)
    fails = [e for e in out if e["type"] == "gas.failed"]
    assert fails[0]["reason"] == "calibration_stale"


def test_calibration_reason_hook_exception_falls_back():
    out = []
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=BoomGas(calibration_at=0.0))
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1_000_000.0)
    fails = [e for e in out if e["type"] == "gas.failed"]
    assert fails[0]["reason"] == "calibration_stale"


def test_fresh_calibration_still_completes():
    out = []
    gas = ReasonGas(reason=None, calibration_at=999.0)
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=gas)
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1000.0)
    done = [e for e in out if e["type"] == "gas.completed"]
    assert len(done) == 1
    assert done[0]["calibration_at"] == 999.0


# ---------- 空 readings ----------


def test_empty_readings_emit_no_gas_data_not_crash():
    """平台只能按窗口回查历史，窗口内无数据是正常结果，不许编造读数凑校验。"""
    out = []
    gas = FakeGas(calibration_at=999.0, readings=[])
    gas._readings = []  # FakeGas 的默认值会填一条，这里显式清空
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=gas)
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1000.0)
    fails = [e for e in out if e["type"] == "gas.failed"]
    assert len(fails) == 1
    assert fails[0]["reason"] == "no_gas_data"
    assert not any(e["type"] == "gas.completed" for e in out)


def test_gas_sample_emitted_once_only():
    out = []
    gas = FakeGas(calibration_at=999.0)
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=gas)
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    for t in (1000.0, 1001.0, 1002.0):
        dog.tick(now=t)
    assert len([e for e in out if e["type"] == "gas.completed"]) == 1
    assert gas.sample_calls == 1


def test_disconnected_sensor_still_reported_first():
    out = []
    gas = FakeGas(connected=False, calibration_at=999.0)
    dog = _dog(out, nav=FakeNav(arrive_after_ticks=1), gas=gas)
    dog.begin_inspect(_inspect())
    dog.begin_gas_sample(_gas_cmd())
    dog.tick(now=1000.0)
    fails = [e for e in out if e["type"] == "gas.failed"]
    assert fails[0]["reason"] == "sensor_disconnected"
    assert gas.sample_calls == 0
