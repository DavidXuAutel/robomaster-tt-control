from experiments.aerial.orchestration.state import Phase, read_status, write_status


def test_write_status_atomic_roundtrip(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, {"phase": Phase.WAIT_B0_COMPLETE.value, "stamp": "x"})
    assert read_status(path)["phase"] == "WAIT_B0_COMPLETE"


def test_read_missing_returns_empty(tmp_path):
    assert read_status(tmp_path / "missing.json") == {}
