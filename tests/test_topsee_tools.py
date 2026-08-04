"""M6/M8：探针与绑定导出工具对假平台的端到端行为。

探针的第一条纪律是「默认只读」——这条必须有测试守着，否则某次改动让它
默默开始给狗派单，现场就会出事。
"""

import json

import pytest

from tests.fixtures.topsee_fake import FakeTopseeServer
from tools import export_dog_bindings as exp
from tools import topsee_probe as probe

ROBOT = "B2000397"


@pytest.fixture
def srv():
    with FakeTopseeServer() as s:
        yield s


def _argv(srv, *extra):
    return [
        "--base-url",
        srv.base_url,
        "--account",
        srv.state.account,
        "--password",
        srv.state.password,
        "--robot-id",
        ROBOT,
        *extra,
    ]


# ---------- 探针 ----------


def test_probe_default_experiments_are_all_read_only():
    assert probe.READ_ONLY
    assert all(not probe.EXPERIMENTS[n]["motion"] for n in probe.READ_ONLY)
    assert "E1" not in probe.READ_ONLY


def test_probe_skips_motion_experiment_without_flag(srv, capsys):
    """没加 --allow-motion 时，E1 必须被跳过且一条命令都不发。"""
    rc = probe.main(_argv(srv, "--experiments", "E0,E1", "--samples", "1"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "--allow-motion" in out
    assert srv.state.navigate_calls == []


def test_probe_e2_collects_state_enum(srv, capsys):
    srv.state.current_task = {"pointsId": "p1", "currentState": "行进中", "totalState": "执行中"}
    rc = probe.main(
        _argv(srv, "--experiments", "E2", "--samples", "2", "--interval", "0.01")
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out.split("\n=== 摘要 ===")[0])
    e2 = report["results"]["E2"]
    assert e2["status"] == "ok"
    assert e2["data"]["currentState_values"] == ["行进中"]
    assert e2["data"]["totalState_values"] == ["执行中"]


def test_probe_writes_report_file(srv, tmp_path):
    out = tmp_path / "sub" / "report.json"
    rc = probe.main(
        _argv(
            srv, "--experiments", "E0,E10", "--samples", "1", "--interval", "0.01",
            "--out", str(out),
        )
    )
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["robot_id"] == ROBOT
    assert doc["allow_motion"] is False
    assert set(doc["manual_only"]) == {"E3", "E4"}


def test_probe_records_failure_without_crashing(srv, capsys):
    srv.state.force_error["/robot/state/getStateData"] = (1, "boom")
    rc = probe.main(
        _argv(srv, "--experiments", "E10", "--samples", "1", "--interval", "0.01")
    )
    assert rc == 4  # 有实验失败 → 非零退出
    assert "failed" in capsys.readouterr().out


def test_probe_login_failure_is_distinguished(srv, capsys):
    srv.state.license_expired = True
    rc = probe.main(_argv(srv, "--experiments", "E0"))
    assert rc == 3
    assert "授权" in capsys.readouterr().err


def test_probe_requires_password(srv, capsys):
    rc = probe.main(
        [
            "--base-url", srv.base_url,
            "--account", "a",
            "--password", "",
            "--robot-id", ROBOT,
        ]
    )
    assert rc == 2
    assert "密码" in capsys.readouterr().err


def test_probe_e6_tries_multiple_time_formats(srv, capsys):
    srv.state.gas_rows = [{"type": "CH4", "value": 1.0, "unit": "%LEL"}]
    rc = probe.main(_argv(srv, "--experiments", "E6", "--samples", "1"))
    assert rc == 0
    report = json.loads(capsys.readouterr().out.split("\n=== 摘要 ===")[0])
    data = report["results"]["E6"]["data"]
    assert set(data) >= {"space_seconds", "iso_T", "date_only", "epoch_ms"}
    assert data["space_seconds"]["rows"] == 1


def test_probe_unknown_experiment_is_skipped_not_fatal(srv, capsys):
    rc = probe.main(_argv(srv, "--experiments", "E99"))
    assert rc == 0
    assert "未知实验" in capsys.readouterr().out


# ---------- 绑定导出 ----------

MAP_ALL = {
    "mapId": "map_demo_01",
    "version": "3",
    "lines": [
        {
            "lineName": "巡检线1",
            "points": [
                {
                    "pointsId": "快速打点-1785465994716",
                    "pointsName": "wp_region_x_staging",
                    "x": 12.5,
                    "y": -3.25,
                    "th": 1.57,
                },
                {"pointsId": "快速打点-1785465994999", "pointsName": "无关点位"},
            ],
        }
    ],
    # 同一点位在线路与点位表里重复出现，导出必须去重
    "points": [
        {
            "pointsId": "快速打点-1785465994716",
            "pointsName": "wp_region_x_staging",
            "x": 12.5,
            "y": -3.25,
        }
    ],
}


def _v1_map(tmp_path, label="wp_region_x_staging"):
    p = tmp_path / "map.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "frame": "dog_map",
                "regions": {
                    "region_x": {
                        "region_id": "region_x",
                        "dog_goal_id": label,
                        "drone_route_id": "r1",
                        "anchor_ids": ["AX-01"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


def test_walk_points_dedups_across_nesting():
    rows = exp._walk_points(MAP_ALL)
    ids = sorted(r["points_id"] for r in rows)
    assert ids == ["快速打点-1785465994716", "快速打点-1785465994999"]


def test_export_writes_binding_and_bumps_version(tmp_path, capsys):
    m = _v1_map(tmp_path)
    raw = tmp_path / "mapall.json"
    raw.write_text(json.dumps(MAP_ALL, ensure_ascii=False), encoding="utf-8")
    rc = exp.main(
        [
            "--base-url", "http://unused",
            "--account", "a",
            "--robot-id", ROBOT,
            "--map", str(m),
            "--from-json", str(raw),
            "--write",
        ]
    )
    assert rc == 0
    doc = json.loads(m.read_text(encoding="utf-8"))
    assert doc["version"] == 2
    g = doc["platform_binding"]["goals"]["wp_region_x_staging"]
    assert g["points_id"] == "快速打点-1785465994716"
    assert g["x"] == 12.5
    assert doc["platform_binding"]["map_id"] == "map_demo_01"


def test_export_refuses_write_when_label_unmatched(tmp_path, capsys):
    """猜错点位比报错危险得多，所以不做模糊匹配、不许部分写入。"""
    m = _v1_map(tmp_path, label="不存在的标签")
    raw = tmp_path / "mapall.json"
    raw.write_text(json.dumps(MAP_ALL, ensure_ascii=False), encoding="utf-8")
    rc = exp.main(
        [
            "--base-url", "http://unused", "--account", "a", "--robot-id", ROBOT,
            "--map", str(m), "--from-json", str(raw), "--write",
        ]
    )
    assert rc == 5
    assert "platform_binding" not in json.loads(m.read_text(encoding="utf-8"))


def test_alias_resolves_name_mismatch(tmp_path):
    m = _v1_map(tmp_path, label="wp_staging")
    raw = tmp_path / "mapall.json"
    raw.write_text(json.dumps(MAP_ALL, ensure_ascii=False), encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.write_text(
        json.dumps({"wp_staging": "wp_region_x_staging"}, ensure_ascii=False),
        encoding="utf-8",
    )
    rc = exp.main(
        [
            "--base-url", "http://unused", "--account", "a", "--robot-id", ROBOT,
            "--map", str(m), "--from-json", str(raw), "--alias", str(alias), "--write",
        ]
    )
    assert rc == 0
    doc = json.loads(m.read_text(encoding="utf-8"))
    assert doc["platform_binding"]["goals"]["wp_staging"]["points_id"].endswith("4716")


def test_check_mode_detects_points_id_drift(tmp_path, capsys):
    m = _v1_map(tmp_path)
    raw = tmp_path / "mapall.json"
    raw.write_text(json.dumps(MAP_ALL, ensure_ascii=False), encoding="utf-8")
    exp.main(
        [
            "--base-url", "http://unused", "--account", "a", "--robot-id", ROBOT,
            "--map", str(m), "--from-json", str(raw), "--write",
        ]
    )
    # 平台重打点：pointsId 变了
    drifted = json.loads(json.dumps(MAP_ALL))
    drifted["points"][0]["pointsId"] = "快速打点-1799999999999"
    drifted["lines"][0]["points"][0]["pointsId"] = "快速打点-1799999999999"
    raw.write_text(json.dumps(drifted, ensure_ascii=False), encoding="utf-8")
    rc = exp.main(
        [
            "--base-url", "http://unused", "--account", "a", "--robot-id", ROBOT,
            "--map", str(m), "--from-json", str(raw), "--check",
        ]
    )
    assert rc == 6
    assert "漂移" in capsys.readouterr().out


def test_check_mode_passes_when_consistent(tmp_path):
    m = _v1_map(tmp_path)
    raw = tmp_path / "mapall.json"
    raw.write_text(json.dumps(MAP_ALL, ensure_ascii=False), encoding="utf-8")
    args = [
        "--base-url", "http://unused", "--account", "a", "--robot-id", ROBOT,
        "--map", str(m), "--from-json", str(raw),
    ]
    exp.main(args + ["--write"])
    assert exp.main(args + ["--check"]) == 0


def test_diff_flags_map_id_change():
    old = {"map_id": "A", "goals": {"wp": {"points_id": "p1"}}}
    new = {"map_id": "B", "goals": {"wp": {"points_id": "p1"}}}
    drift = exp.diff_binding(old, new)
    assert any("map_id" in d for d in drift)


def test_diff_flags_disappeared_point():
    old = {"map_id": "A", "goals": {"wp": {"points_id": "p1"}}}
    new = {"map_id": "A", "goals": {}}
    assert any("找不到点位" in d for d in exp.diff_binding(old, new))


def test_export_from_platform_over_http(srv, tmp_path):
    """联网路径也要走一遍，确认 login + getRobotMapAll 串得通。"""
    srv.state.map_all = MAP_ALL
    m = _v1_map(tmp_path)
    rc = exp.main(
        [
            "--base-url", srv.base_url,
            "--account", srv.state.account,
            "--password", srv.state.password,
            "--robot-id", ROBOT,
            "--map", str(m),
            "--write",
        ]
    )
    assert rc == 0
    doc = json.loads(m.read_text(encoding="utf-8"))
    assert doc["platform_binding"]["goals"]["wp_region_x_staging"]["points_id"].endswith(
        "4716"
    )
