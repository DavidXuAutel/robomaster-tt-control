"""狗 stub / sdk 回退 + 气检失败路径。"""

import time

from adapters.dog_sdk import DogSdkAdapter
from adapters.dog_stub import DogStubAdapter
from mission_brain.events import EventType, make_event


def test_dog_stub_happy_path():
    out = []
    dog = DogStubAdapter(
        out.append, nav_delay_s=0.0, search_delay_s=0.0, sample_delay_s=0.0
    )
    dog.begin_inspect(
        make_event(
            EventType.DOG_INSPECT,
            mission_id="m",
            source="mission_brain",
            payload={
                "region_id": "region_x",
                "dog_goal_id": "wp_region_x_staging",
                "target_label": "object_a",
                "evidence_uri": "e",
                "deadline": 99.0,
            },
        )
    )
    t = time.time()
    dog.tick(now=t)
    dog.tick(now=t + 0.01)
    assert any(e["type"] == "dog.arrived" for e in out)
    assert any(e["type"] == "dog.target_found" for e in out)
    dog.begin_gas_sample(
        make_event(
            EventType.GAS_SAMPLE,
            mission_id="m",
            source="mission_brain",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "sample_window_s": 1.0,
                "deadline": 99.0,
            },
        )
    )
    dog.tick(now=t + 0.02)
    gas = [e for e in out if e["type"] == "gas.completed"]
    assert len(gas) == 1
    assert gas[0]["readings"][0]["channel"] == "CH4"


def test_gas_sensor_disconnect_fails():
    out = []
    dog = DogStubAdapter(
        out.append,
        nav_delay_s=0.0,
        search_delay_s=0.0,
        sample_delay_s=0.0,
        force_sensor_disconnect=True,
    )
    dog.begin_inspect(
        make_event(
            EventType.DOG_INSPECT,
            mission_id="m",
            source="brain",
            payload={
                "region_id": "region_x",
                "dog_goal_id": "wp",
                "target_label": "object_a",
                "evidence_uri": "e",
                "deadline": 9.0,
            },
        )
    )
    t = 1000.0
    dog.tick(now=t)
    dog.tick(now=t + 0.1)
    dog.begin_gas_sample(
        make_event(
            EventType.GAS_SAMPLE,
            mission_id="m",
            source="brain",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "sample_window_s": 1.0,
                "deadline": 9.0,
            },
        )
    )
    dog.tick(now=t + 0.2)
    assert any(e["type"] == "gas.failed" for e in out)
    assert any(e.get("reason") == "sensor_disconnected" for e in out)


def test_dog_sdk_explicit_stub_mode():
    out = []
    sdk = DogSdkAdapter(out.append, mode="stub")
    sdk.begin_inspect(
        make_event(
            EventType.DOG_INSPECT,
            mission_id="m",
            source="brain",
            payload={
                "region_id": "region_x",
                "dog_goal_id": "wp",
                "target_label": "object_a",
                "evidence_uri": "e",
                "deadline": 9.0,
            },
        )
    )
    t = 1000.0
    sdk.tick(now=t)
    sdk.tick(now=t + 0.1)
    assert any(e["type"] == "dog.target_found" for e in out)
