"""M9：气体标定人工台账（方案 §5.4）。"""

import json
import time
from datetime import datetime, timezone

import pytest

from adapters.gas_ledger import (
    REASON_SOURCE_UNAVAILABLE,
    REASON_STALE,
    GasCalibrationLedger,
    GasLedgerError,
    parse_iso8601,
)


def test_example_ledger_loads():
    led = GasCalibrationLedger.load("configs/mission/gas_calibration.example.json")
    assert "B2000397" in led.robot_ids()
    assert led.max_age_s == 604800


REF_UTC = datetime(2026, 7, 20, 1, 30, tzinfo=timezone.utc).timestamp()


def test_parse_iso8601_with_offset():
    assert parse_iso8601("2026-07-20T09:30:00+08:00") == pytest.approx(REF_UTC)


def test_parse_iso8601_with_z():
    assert parse_iso8601("2026-07-20T01:30:00Z") == pytest.approx(REF_UTC)


def test_offset_and_z_forms_agree():
    """+08:00 与 Z 必须落到同一时刻，否则台账时间会差 8 小时。"""
    assert parse_iso8601("2026-07-20T09:30:00+08:00") == parse_iso8601(
        "2026-07-20T01:30:00Z"
    )


def test_naive_time_interpreted_as_local():
    naive = parse_iso8601("2026-07-20T09:30:00")
    expected = datetime(2026, 7, 20, 9, 30).astimezone().timestamp()
    assert naive == pytest.approx(expected)


def test_parse_iso8601_rejects_garbage():
    with pytest.raises(GasLedgerError, match="ISO8601"):
        parse_iso8601("上周三")


def test_missing_robot_is_source_unavailable():
    """宁可气检失败，也不能给个默认时间让门禁失效。"""
    led = GasCalibrationLedger.from_dict({"sensors": []})
    assert led.calibration_at("nobody") == 0.0
    assert led.reason("nobody") == REASON_SOURCE_UNAVAILABLE


def test_stale_vs_fresh():
    now = time.time()
    led = GasCalibrationLedger.from_dict(
        {
            "max_age_s": 3600,
            "sensors": [
                {
                    "robot_id": "fresh",
                    "calibrated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(now - 60)
                    ),
                },
                {
                    "robot_id": "old",
                    "calibrated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(now - 7200)
                    ),
                },
            ],
        }
    )
    assert led.reason("fresh", now=now) is None
    assert led.reason("old", now=now) == REASON_STALE


@pytest.mark.parametrize(
    "bad,msg",
    [
        ({}, "sensors"),
        ({"sensors": {}}, "sensors"),
        ({"sensors": ["x"]}, "必须是对象"),
        ({"sensors": [{"calibrated_at": "2026-01-01"}]}, "robot_id"),
        ({"sensors": [{"robot_id": "a"}]}, "calibrated_at"),
        (
            {
                "sensors": [
                    {"robot_id": "a", "calibrated_at": "2026-01-01"},
                    {"robot_id": "a", "calibrated_at": "2026-01-02"},
                ]
            },
            "重复",
        ),
        ({"max_age_s": 0, "sensors": []}, "max_age_s"),
        ({"max_age_s": "abc", "sensors": []}, "max_age_s"),
    ],
)
def test_malformed_ledger_rejected(bad, msg):
    with pytest.raises(GasLedgerError, match=msg):
        GasCalibrationLedger.from_dict(bad)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(GasLedgerError, match="不存在"):
        GasCalibrationLedger.load(tmp_path / "nope.json")


def test_broken_json_raises_clear_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(GasLedgerError, match="合法 JSON"):
        GasCalibrationLedger.load(p)


def test_entry_exposes_audit_fields(tmp_path):
    p = tmp_path / "led.json"
    p.write_text(
        json.dumps(
            {
                "sensors": [
                    {
                        "robot_id": "B2",
                        "channels": ["CH4", "O2"],
                        "calibrated_at": "2026-07-20T09:30:00+08:00",
                        "calibrated_by": "张某",
                        "certificate": "docs/x.pdf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    e = GasCalibrationLedger.load(p).entry("B2")
    assert e is not None
    assert e["calibrated_by"] == "张某"
    assert e["channels"] == ["CH4", "O2"]
    assert e["certificate"] == "docs/x.pdf"
