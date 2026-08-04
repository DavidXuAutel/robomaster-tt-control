"""M3：FakeNav + DogSdk 显式 mode（软件预埋，非真机 G1）。"""

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


def test_mode_backend_requires_all():
    with pytest.raises(ValueError, match="缺少 backend"):
        DogSdkAdapter(lambda e: None, mode="backend", nav=FakeNav())


def test_mode_stub_explicit():
    out = []
    dog = DogSdkAdapter(out.append, mode="stub")
    assert dog.mode == "stub"
    dog.begin_inspect(_inspect())
    dog.tick(now=1.0)
    assert any(e["type"] == "dog.arrived" for e in out)


def test_d01_ten_gotos():
    out = []
    nav = FakeNav(arrive_after_ticks=1)
    perc = FakePerception(
        {"confidence": 0.9, "evidence_uri": "dog://e"}
    )
    gas = FakeGas()
    dog = DogSdkAdapter(
        out.append, mode="backend", nav=nav, perception=perc, gas=gas
    )
    for i in range(10):
        out.clear()
        dog.begin_inspect(_inspect(goal=f"wp_{i}"))
        dog.tick(now=float(i))
        assert any(e["type"] == "dog.arrived" for e in out)
    assert nav.goto_calls == 10
    assert nav.last_goal_id == "wp_9"


def test_d02_goto_rejected():
    out = []
    nav = FakeNav(reject_goto=True)
    dog = DogSdkAdapter(
        out.append,
        mode="backend",
        nav=nav,
        perception=FakePerception(),
        gas=FakeGas(),
    )
    dog.begin_inspect(_inspect())
    fails = [e for e in out if e["type"] == "dog.inspect_failed"]
    assert len(fails) == 1
    assert fails[0]["stage"] == "nav"


def test_d03_abort_cancel_no_arrived():
    out = []
    nav = FakeNav(arrive_after_ticks=1)
    dog = DogSdkAdapter(
        out.append,
        mode="backend",
        nav=nav,
        perception=FakePerception({"confidence": 0.9, "evidence_uri": "e"}),
        gas=FakeGas(),
    )
    dog.begin_inspect(_inspect())
    dog.abort("stop")
    assert nav.cancel_calls == 1
    dog.tick(now=1.0)
    dog.tick(now=2.0)
    assert not any(e["type"] == "dog.arrived" for e in out)


def test_d04_cancel_raises_still_latched():
    out = []
    nav = FakeNav(arrive_after_ticks=1, raise_on_cancel=True)
    dog = DogSdkAdapter(
        out.append,
        mode="backend",
        nav=nav,
        perception=FakePerception(),
        gas=FakeGas(),
    )
    dog.begin_inspect(_inspect())
    dog.abort("x")
    assert dog._aborted is True
    dog.tick(now=1.0)
    assert not any(e["type"] == "dog.arrived" for e in out)
