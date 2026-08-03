from pathlib import Path

from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint


def test_complete_requires_sha_and_stable_size(tmp_path):
    pt = tmp_path / "step_001000.pt"
    pt.write_bytes(b"abc")
    assert is_complete_checkpoint(pt, settle_s=0.0, min_bytes=1) is False
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")
    assert is_complete_checkpoint(pt, settle_s=0.0, min_bytes=1) is True


def test_production_min_bytes_default(tmp_path):
    pt = tmp_path / "step_001000.pt"
    pt.write_bytes(b"x")
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")
    assert is_complete_checkpoint(pt, settle_s=0.0) is False

    with pt.open("wb") as handle:
        handle.truncate(1_000_000_000)
    assert is_complete_checkpoint(pt, settle_s=0.0) is True


def test_settle_requires_stable_size(monkeypatch, tmp_path):
    pt = tmp_path / "step_001000.pt"
    with pt.open("wb") as handle:
        handle.truncate(1_000_000_000)
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")

    monkeypatch.setattr(
        "experiments.aerial.orchestration.checkpoint.time.sleep", lambda _: None
    )
    assert is_complete_checkpoint(pt, settle_s=0.1, min_bytes=1) is True

    def grow_during_sleep(_):
        with pt.open("ab") as handle:
            handle.write(b"x")

    monkeypatch.setattr(
        "experiments.aerial.orchestration.checkpoint.time.sleep", grow_during_sleep
    )
    assert is_complete_checkpoint(pt, settle_s=0.1, min_bytes=1) is False


def test_settle_returns_false_if_checkpoint_removed(monkeypatch, tmp_path):
    pt = tmp_path / "step_001000.pt"
    with pt.open("wb") as handle:
        handle.truncate(1_000_000_000)
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")

    def delete_pt_during_sleep(_):
        pt.unlink()

    monkeypatch.setattr(
        "experiments.aerial.orchestration.checkpoint.time.sleep", delete_pt_during_sleep
    )
    assert is_complete_checkpoint(pt, settle_s=0.1, min_bytes=1) is False


def test_settle_returns_false_if_sidecar_removed(monkeypatch, tmp_path):
    pt = tmp_path / "step_001000.pt"
    sha = tmp_path / "step_001000.pt.sha256"
    with pt.open("wb") as handle:
        handle.truncate(1_000_000_000)
    sha.write_text("deadbeef  step_001000.pt\n")

    def delete_sha_during_sleep(_):
        sha.unlink()

    monkeypatch.setattr(
        "experiments.aerial.orchestration.checkpoint.time.sleep", delete_sha_during_sleep
    )
    assert is_complete_checkpoint(pt, settle_s=0.1, min_bytes=1) is False


def test_stat_race_returns_false(monkeypatch, tmp_path):
    pt = tmp_path / "step_001000.pt"
    with pt.open("wb") as handle:
        handle.truncate(1_000_000_000)
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")

    real_stat = Path.stat
    initial_pt_stats = {"n": 0}

    def stat_disappears_before_initial_read(self):
        if self == pt:
            initial_pt_stats["n"] += 1
            if initial_pt_stats["n"] == 2:
                raise FileNotFoundError(self)
        return real_stat(self)

    monkeypatch.setattr(Path, "stat", stat_disappears_before_initial_read)
    assert is_complete_checkpoint(pt, settle_s=0.0, min_bytes=1) is False

    post_settle_pt_stats = {"n": 0}

    def stat_disappears_before_post_settle_read(self):
        if self == pt:
            post_settle_pt_stats["n"] += 1
            if post_settle_pt_stats["n"] == 4:
                raise FileNotFoundError(self)
        return real_stat(self)

    monkeypatch.setattr(Path, "stat", stat_disappears_before_post_settle_read)
    monkeypatch.setattr(
        "experiments.aerial.orchestration.checkpoint.time.sleep", lambda _: None
    )
    assert is_complete_checkpoint(pt, settle_s=0.1, min_bytes=1) is False
