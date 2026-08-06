"""端到端组合：SharedMap v2 → Arbiter → TopseeNav/Gas → DogSdkAdapter → 事件。

单测各自绿不代表拼起来能跑。这里验证真实装配路径：
  1. 语义标签经 platform_binding 解析成平台 pointsId 才派单
  2. Arbiter 未授权时慢通道一条命令都发不出去
  3. 到点/气检事件能通过 v1 事件契约的校验（含 FORBIDDEN_KEYS）
  4. 慢通道 → WAM 快通道的分时交接不会出现双源下发
"""

import pytest

from adapters.dog_arbiter import ArbiterState, DogControlArbiter
from adapters.dog_sdk import DogSdkAdapter
from adapters.dog_topsee import TopseeGas, TopseeNav, TopseePerception
from adapters.dog_unitree import LoopbackTransport, UnitreeSportClient
from adapters.gas_ledger import GasCalibrationLedger
from adapters.topsee_client import TopseeClient
from mission_brain.events import EventType, make_event, validate_event
from mission_brain.map_model import SharedMap
from tests.fixtures.topsee_fake import FakeTopseeServer

ROBOT = "B2000397"
PID = "快速打点-1785465994716"
LABEL = "wp_region_x_staging"

MAP_V2 = {
    "version": 2,
    "frame": "dog_map",
    "regions": {
        "region_x": {
            "region_id": "region_x",
            "dog_goal_id": LABEL,
            "drone_route_id": "route_region_x_scout",
            "anchor_ids": ["AX-01"],
        }
    },
    "platform_binding": {
        "map_id": "map_demo_01",
        "goals": {LABEL: {"points_id": PID, "points_name": "演示区X待机点", "x": 5.0, "y": 5.0}},
    },
}


class Stack:
    """完整装配，模拟 main 里该怎么接线。"""

    def __init__(self, srv, *, now=1_000_000.0):
        self.srv = srv
        self.events = []
        self.now = now
        self.shared_map = SharedMap.from_dict(MAP_V2)
        self.shared_map.validate_binding(map_id="map_demo_01")

        self.client = TopseeClient(
            srv.base_url,
            account=srv.state.account,
            password=srv.state.password,
            timeout_s=5.0,
        )
        self.client.login()

        self.unitree = UnitreeSportClient(LoopbackTransport())
        self.unitree.connect()
        self.arbiter = DogControlArbiter(
            self.client,
            self.unitree,
            robot_id=ROBOT,
            controller_state="02",
            battery_provider=lambda: 90.0,
        )
        self.nav = TopseeNav(
            self.client,
            robot_id=ROBOT,
            arbiter=self.arbiter,
            goal_resolver=self.shared_map.resolve_points_id,
            goal_pose_resolver=self.shared_map.goal_pose,
            pose_source=self.unitree.pose_xy_yaw,
            arrived_states=("已到达",),
            enroute_states=("行进中",),
            autostart_poller=False,
        )
        ledger = GasCalibrationLedger.from_dict(
            {
                "max_age_s": 10 ** 9,
                "sensors": [{"robot_id": ROBOT, "calibrated_at": "2026-07-20T09:30:00+08:00"}],
            }
        )
        self.gas = TopseeGas(self.client, robot_id=ROBOT, ledger=ledger)
        self.perception = TopseePerception(
            self.client,
            robot_id=ROBOT,
            mode="local_vision",
            detector=lambda label: {"confidence": 0.91, "evidence_uri": "file://a.jpg"},
        )
        self.dog = DogSdkAdapter(
            self.events.append,
            mode="backend",
            nav=self.nav,
            perception=self.perception,
            gas=self.gas,
            calibration_max_age_s=10 ** 9,
            arbiter=self.arbiter,
        )

    def inspect_cmd(self):
        return make_event(
            EventType.DOG_INSPECT,
            mission_id="m1",
            source="mission_brain",
            payload={
                "region_id": "region_x",
                "dog_goal_id": self.shared_map.resolve_dog_goal("region_x"),
                "target_label": "object_a",
                "evidence_uri": "e",
                "deadline": self.now + 300,
            },
        )

    def gas_cmd(self):
        return make_event(
            EventType.GAS_SAMPLE,
            mission_id="m1",
            source="mission_brain",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "sample_window_s": 30.0,
                "deadline": self.now + 300,
            },
        )

    def tick(self):
        self.nav.cache.refresh_once()
        self.arbiter.tick(now=self.now)
        self.dog.tick(now=self.now)
        self.now += 1.0

    def types(self):
        return [e["type"] for e in self.events]

    def platform_task(self, state):
        """模拟平台侧任务状态。

        必须在派单之后调用：preflight 为了避免双调度源会先 stopTask，
        假平台会把 current_task 一并清空。
        """
        self.srv.state.current_task = {"pointsId": PID, "currentState": state}

    def start_mission(self):
        self.arbiter.ack_confidence(0.95, by="tester")
        self.arbiter.acquire_for_mission("m1", now=self.now)
        self.dog.on_brain_event(self.inspect_cmd())


@pytest.fixture
def srv():
    with FakeTopseeServer() as s:
        yield s


def test_full_inspect_and_gas_flow(srv):
    srv.state.gas_rows = [{"type": "CH4", "value": 1.5, "unit": "%LEL", "alarmState": "ok"}]
    st = Stack(srv)
    st.start_mission()
    # 语义标签必须被翻译成平台 pointsId 才派单
    assert srv.state.navigate_calls == [(ROBOT, PID)]

    st.platform_task("行进中")
    st.tick()
    assert "dog.arrived" not in st.types()  # 还在路上

    st.platform_task("已到达")
    st.tick()
    assert "dog.arrived" in st.types()
    assert "dog.target_found" in st.types()

    st.dog.on_brain_event(st.gas_cmd())
    st.tick()
    assert "gas.completed" in st.types()

    done = next(e for e in st.events if e["type"] == "gas.completed")
    assert done["readings"] == [
        {"channel": "CH4", "value": 1.5, "unit": "%LEL", "alarm_state": "ok"}
    ]
    # 每一条事件都要能过 v1 契约（含禁止字段）校验
    for e in st.events:
        validate_event(e)


def test_no_pose_leaks_into_events(srv):
    """红线：DDS 位姿只准进 WAM 落盘通道，绝不进事件。"""
    st = Stack(srv)
    st.start_mission()
    st.platform_task("已到达")
    st.tick()
    assert st.events
    blob = repr(st.events)
    for banned in ("pose_xyz", "global_pose", "point_cloud", "T_world"):
        assert banned not in blob


def test_distance_evidence_works_when_state_unknown(srv):
    """E2 枚举没落地时，位姿距离判据必须能独立完成到点判定。"""
    st = Stack(srv)
    st.start_mission()
    st.platform_task("平台某个未知状态")
    st.tick()
    assert "dog.arrived" not in st.types()

    st.unitree.transport.teleport(5.1, 4.95)  # type: ignore[attr-defined]
    st.tick()
    assert "dog.arrived" in st.types()


def test_arbiter_veto_blocks_whole_slow_channel(srv):
    """Arbiter 没授权（还在 IDLE）时，dog.inspect 必须直接失败而不是偷偷派单。"""
    st = Stack(srv)
    assert st.arbiter.state is ArbiterState.IDLE
    st.dog.on_brain_event(st.inspect_cmd())
    assert srv.state.navigate_calls == []
    fails = [e for e in st.events if e["type"] == "dog.inspect_failed"]
    assert len(fails) == 1
    assert fails[0]["reason"] == "goto_goal_rejected"


def test_e1_failure_surfaces_as_named_reason(srv):
    """平台查不到任务时，必须报 nav_task_not_tracked，不许拖到超时。"""
    st = Stack(srv)
    st.nav.unknown_tolerance = 2
    st.start_mission()
    assert srv.state.current_task is None  # 平台压根没建任务
    for _ in range(3):
        st.tick()
    fails = [e for e in st.events if e["type"] == "dog.inspect_failed"]
    assert fails and fails[0]["reason"] == "nav_task_not_tracked"


def test_handoff_to_wam_is_time_multiplexed(srv):
    """慢通道 → 快通道交接期间，任何时刻只有一个通道被授权。"""
    st = Stack(srv)
    st.start_mission()
    st.platform_task("已到达")
    st.tick()
    assert "dog.arrived" in st.types()
    assert st.arbiter.allow_topsee_cmd() is True

    st.arbiter.request_wam(now=st.now)
    assert st.arbiter.allow_topsee_cmd() is False
    assert st.arbiter.unitree_cmd_enabled is False  # 等人工切模式期间两边都不许动

    token = st.arbiter.ack_human_mode_switch(by="tester", now=st.now)
    assert st.arbiter.allow_topsee_cmd() is False
    assert st.arbiter.unitree_cmd_enabled is True
    st.arbiter.move(0.4, 0.0, 0.1, token=token)
    assert st.unitree.move_calls == 1

    # WAM 段中慢通道派单必须被拒
    assert st.nav.goto_goal(LABEL) is False

    st.arbiter.begin_handover_to_mission(now=st.now)
    assert st.arbiter.state is ArbiterState.IDLE
    assert st.arbiter.has_single_owner


def test_abort_path_stops_both_channels(srv):
    st = Stack(srv)
    st.start_mission()
    st.platform_task("行进中")
    # D0：dog.abort 必须自行 force_release，不再依赖调用方二次释放
    st.dog.abort("mission_abort")
    assert ROBOT in srv.state.stop_calls
    assert st.arbiter.no_owner
    assert st.unitree.last_cmd == (0.0, 0.0, 0.0)
    st.tick()
    assert "dog.arrived" not in st.types()
