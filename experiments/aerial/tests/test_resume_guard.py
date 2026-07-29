from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam.trainer import Wan22Trainer


def _bare_trainer(cfg=None):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.cfg = cfg if cfg is not None else SimpleNamespace()
    return trainer


def test_rejects_residual_pre_v2_seed(monkeypatch):
    monkeypatch.delenv("AERIAL_ALLOW_LEGACY_RESUME", raising=False)
    trainer = _bare_trainer()
    residual = Path(
        "/home/a25689/aerial_cache_shared/runs/aerial_joint_b0/"
        "m1b-20260722-012926/checkpoints/weights/step_000500.pt"
    )
    with pytest.raises(RuntimeError, match="pre-v2 aerial checkpoint"):
        trainer._assert_resume_allowed(residual)


def test_allows_v2_checkpoint(monkeypatch):
    monkeypatch.delenv("AERIAL_ALLOW_LEGACY_RESUME", raising=False)
    trainer = _bare_trainer()
    # A v2-produced checkpoint path is not on the deny list -> allowed.
    trainer._assert_resume_allowed(
        Path("/home/a25689/aerial_ft_cache/runs/b1-v2-20260801/weights/step_000250.pt")
    )


def test_env_override_bypasses_guard(monkeypatch):
    monkeypatch.setenv("AERIAL_ALLOW_LEGACY_RESUME", "1")
    trainer = _bare_trainer()
    # Even the residual seed is allowed once the override is set.
    trainer._assert_resume_allowed(Path("/x/m1b-20260722-012926/step_000500.pt"))


def test_cfg_override_bypasses_guard(monkeypatch):
    monkeypatch.delenv("AERIAL_ALLOW_LEGACY_RESUME", raising=False)
    trainer = _bare_trainer(SimpleNamespace(allow_legacy_resume=True))
    trainer._assert_resume_allowed(Path("/x/step_000500.pt"))


def test_positive_provenance_requires_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("AERIAL_ALLOW_LEGACY_RESUME", raising=False)
    trainer = _bare_trainer(SimpleNamespace(require_v2_resume_provenance=True))

    unmarked = tmp_path / "step_000250.pt"
    unmarked.write_bytes(b"w")
    with pytest.raises(RuntimeError, match="v2 provenance"):
        trainer._assert_resume_allowed(unmarked)

    # A `.v2` sidecar satisfies the provenance requirement.
    (tmp_path / "step_000250.pt.v2").write_text("ok")
    trainer._assert_resume_allowed(unmarked)
