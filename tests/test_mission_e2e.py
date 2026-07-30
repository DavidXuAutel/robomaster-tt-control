"""端到端：Brain + TelloScout + DogStub。"""

from mission_brain.map_model import SharedMap
from mission_brain.runner import run_demo_mission


def test_demo_mission_completes():
    m = SharedMap.load("configs/mission/shared_map.example.json")
    result = run_demo_mission(m, region_ids=["region_x"])
    assert result["state"] == "COMPLETE", result
    assert "drone.target_found" in result["events"]
    assert "dog.inspect" in result["events"]
    assert "gas.completed" in result["events"]
    assert "mission.completed" in result["events"]
