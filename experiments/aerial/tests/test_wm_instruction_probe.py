import json
from pathlib import Path

import numpy as np

from experiments.aerial.eval import wm_instruction_probe as probe


# Mirrors FastWAMAerialPolicy's world-model-frame contract: when dump_video is
# on, stash a clip on last_generated_frames. Here the imagined clip and the
# chosen primitive both depend on the instruction, so the probe can tell a
# live text-conditioning path from a dead one.
class _InstrSensitivePolicy:
    def __init__(self) -> None:
        self.dump_video = False
        self.last_generated_frames = None
        self.seen: list[tuple[str, tuple]] = []

    def predict_primitive(self, obs_rgb, state, instruction):
        self.seen.append((instruction, tuple(np.asarray(state).ravel().tolist())))
        n = len(instruction) % 4  # instruction-dependent, deterministic
        self.last_generated_frames = (
            [np.full((4, 4, 3), 10 * n, dtype=np.uint8) for _ in range(n)]
            if self.dump_video and n
            else None
        )
        return n


def test_load_instructions_json_list(tmp_path):
    path = tmp_path / "instr.json"
    path.write_text(json.dumps(["go left", "go right"]), encoding="utf-8")
    assert probe.load_instructions(path) == ["go left", "go right"]


def test_load_instructions_plain_text(tmp_path):
    path = tmp_path / "instr.txt"
    path.write_text("go left\n\n  go right  \n", encoding="utf-8")
    assert probe.load_instructions(path) == ["go left", "go right"]


def test_probe_holds_obs_and_state_fixed(tmp_path, monkeypatch):
    policy = _InstrSensitivePolicy()
    monkeypatch.setattr(
        probe, "_save_wm_clip", lambda out, prefix, frames, **kw: out / f"{prefix}.mp4"
    )
    obs = np.zeros((8, 8, 3), dtype=np.uint8)
    state = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    instructions = ["a", "bb", "ccc", ""]

    records = probe.probe_instructions(policy, obs, state, instructions, tmp_path)

    # Same obs+state for every instruction — only the text varied.
    assert [instr for instr, _ in policy.seen] == instructions
    assert {st for _, st in policy.seen} == {(1.0, 2.0, 3.0, 0.5)}
    # Records carry the per-instruction primitive + frame count.
    assert [r["primitive"] for r in records] == [1, 2, 3, 0]
    assert [r["n_frames"] for r in records] == [1, 2, 3, 0]
    # Instructions that produced frames get a clip; the empty one (0 frames) does not.
    assert records[0]["clip"] is not None
    assert records[3]["clip"] is None


def test_main_writes_summary(tmp_path, monkeypatch):
    obs_path = tmp_path / "obs.png"
    try:
        from PIL import Image
    except ImportError:
        import pytest

        pytest.skip("needs Pillow to write the fixed observation png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(obs_path)

    instr_path = tmp_path / "instr.json"
    instr_path.write_text(json.dumps(["left", "right", ""]), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        probe, "build_policy", lambda *a, **kw: _InstrSensitivePolicy()
    )
    monkeypatch.setattr(
        probe, "_save_wm_clip", lambda out, prefix, frames, **kw: out / f"{prefix}.mp4"
    )

    rc = probe.main(
        [
            "--checkpoint",
            str(tmp_path / "unused.pt"),
            "--obs",
            str(obs_path),
            "--instructions",
            str(instr_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert [r["instruction"] for r in summary] == ["left", "right", ""]
    assert all("primitive" in r for r in summary)
