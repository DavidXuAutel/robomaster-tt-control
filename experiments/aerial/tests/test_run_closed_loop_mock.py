import json
from pathlib import Path

import numpy as np
import pytest

from experiments.aerial.eval.run_closed_loop import (
    MockBridge,
    ReplayPolicy,
    _assemble_episode_mp4,
    _save_wm_clip,
    apply_body_delta,
    eval_hydra_overrides,
    evaluate_episodes,
    load_annotation,
    run_episode,
    write_metrics,
)
from experiments.aerial.openfly_actions import primitive_to_delta


# A non-mock bridge (kinematics reused) so run_episode's dump gate fires.
class _NonMockBridge:
    def __init__(self) -> None:
        self._inner = MockBridge(seed=3)

    def reset(self, episode):
        self._inner.reset(episode)

    def render(self):
        return np.full((48, 64, 3), (10, 20, 30), dtype=np.uint8)

    def state(self):
        return self._inner.state()

    def step(self, primitive_id):
        self._inner.step(primitive_id)

    def close(self):
        self._inner.close()

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_openfly" / "seen_mini.json"


def test_apply_body_delta_forward():
    pos = np.array([0.0, 0.0, 1.0])
    delta = primitive_to_delta(1)
    new_pos, new_yaw = apply_body_delta(pos, 0.0, delta)
    assert np.allclose(new_pos, [3.0, 0.0, 1.0], atol=1e-6)
    assert new_yaw == pytest.approx(0.0)


def test_apply_body_delta_wraps_yaw():
    """Recorded yaw feeds the state normalizer; unwrapped drift wrecks its min/max."""
    yaw = 0.0
    for _ in range(83):
        _, yaw = apply_body_delta(np.zeros(3), yaw, primitive_to_delta(2))
    assert -np.pi < yaw <= np.pi

    _, wrapped = apply_body_delta(np.zeros(3), np.pi - 0.1, np.array([0.0, 0.0, 0.0, 0.3]))
    assert wrapped == pytest.approx(np.pi - 0.1 + 0.3 - 2 * np.pi)


def test_mock_bridge_replay_episode_success():
    episode = load_annotation(FIXTURE)[0]
    bridge = MockBridge(seed=7)
    policy = ReplayPolicy(episode["action"])
    result = run_episode(bridge, policy, episode, max_steps=20)
    assert result.success
    assert result.navigation_error == pytest.approx(0.0, abs=1e-5)
    assert result.path_length == pytest.approx(6.0, abs=1e-5)


def test_evaluate_episodes_mock_writes_finite_metrics(tmp_path):
    episodes = load_annotation(FIXTURE)
    out = tmp_path / "metrics_mock.json"
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=17,
    )
    write_metrics(metrics, out)

    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["n"] == float(len(episodes))
    for key in ("SR", "NE", "SPL"):
        assert key in loaded
        assert np.isfinite(float(loaded[key]))


def test_evaluate_episodes_preserves_per_episode_metrics():
    episodes = load_annotation(FIXTURE)[:2]
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=42,
    )
    assert len(metrics["episodes"]) == 2
    assert set(metrics["episodes"][0]) == {
        "episode_id",
        "success",
        "NE",
        "path_length",
        "shortest_length",
        "steps",
        "closest_approach",
        "terminated_by",
        "oracle_hit@20",
        "oracle_hit@30",
        "oracle_hit@40",
    }


def _have_image_writer() -> bool:
    try:
        import PIL  # type: ignore  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import cv2  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_image_writer(), reason="needs Pillow or opencv-python")
def test_dump_frames_writes_pngs_for_real_bridge(tmp_path, monkeypatch):
    episode = load_annotation(FIXTURE)[0]
    dump_dir = tmp_path / "frames"
    bridge = _NonMockBridge()
    policy = ReplayPolicy(episode["action"])
    assembled: list[tuple[str, str]] = []

    def fake_assemble(dump, prefix, *, fps=10.0):
        assembled.append((str(dump), prefix))
        out = Path(dump) / f"{prefix}.mp4"
        out.write_bytes(b"fake-mp4")
        return out

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._assemble_episode_mp4",
        fake_assemble,
    )
    result = run_episode(
        bridge,
        policy,
        episode,
        max_steps=20,
        dump_dir=dump_dir,
        frame_prefix="000_r0",
    )
    pngs = sorted(dump_dir.glob("000_r0_step*.png"))
    # Render happens once per loop iteration, including the final iteration that
    # reads the stop primitive — so the goal frame is captured: steps + 1.
    assert len(pngs) == result.steps + 1
    assert pngs[0].name == "000_r0_step0000.png"
    assert assembled == [(str(dump_dir), "000_r0")]
    assert (dump_dir / "000_r0.mp4").is_file()


def test_assemble_episode_mp4_encodes_sorted_pngs(tmp_path, monkeypatch):
    dump_dir = tmp_path / "frames"
    dump_dir.mkdir()
    # Minimal 2x2 RGB PNGs via Pillow when available, else raw via imageio mock only.
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("needs Pillow to create input PNGs")

    for step, color in ((0, (255, 0, 0)), (1, (0, 255, 0)), (2, (0, 0, 255))):
        Image.fromarray(
            __import__("numpy").full((2, 2, 3), color, dtype="uint8")
        ).save(dump_dir / f"epA_step{step:04d}.png")

    written: list[object] = []

    class _Writer:
        def append_data(self, frame):
            written.append(frame)

        def close(self):
            return None

    def fake_get_writer(path, **kwargs):
        assert path.endswith("epA.mp4")
        assert kwargs.get("fps") == 10.0
        return _Writer()

    class _FakeImageio:
        @staticmethod
        def get_writer(path, **kwargs):
            return fake_get_writer(path, **kwargs)

        @staticmethod
        def imread(path):
            from PIL import Image
            import numpy as np

            return np.asarray(Image.open(path))

    import sys
    import types

    fake_mod = types.ModuleType("imageio")
    fake_v2 = types.ModuleType("imageio.v2")
    fake_v2.get_writer = _FakeImageio.get_writer
    fake_v2.imread = _FakeImageio.imread
    monkeypatch.setitem(sys.modules, "imageio", fake_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", fake_v2)

    out = _assemble_episode_mp4(dump_dir, "epA", fps=10.0)
    assert out == dump_dir / "epA.mp4"
    assert len(written) == 3


def test_dump_frames_skipped_for_mock_bridge(tmp_path):
    episode = load_annotation(FIXTURE)[0]
    dump_dir = tmp_path / "frames"
    bridge = MockBridge(seed=7)
    policy = ReplayPolicy(episode["action"])
    run_episode(bridge, policy, episode, max_steps=20, dump_dir=dump_dir)
    # Pseudo-RGB must never be dumped; dir stays empty (or is never created).
    assert not dump_dir.exists() or not any(dump_dir.iterdir())


# A policy that mimics FastWAMAerialPolicy's world-model-frame contract: when
# dump_video is on at predict time, it stashes a clip on last_generated_frames.
class _WMPolicy:
    def __init__(self, actions) -> None:
        self._inner = ReplayPolicy(actions)
        self.dump_video = False
        self.last_generated_frames = None

    def reset(self):
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def predict_primitive(self, obs_rgb, state, instruction):
        if self.dump_video:
            self.last_generated_frames = [
                np.full((8, 8, 3), 40 * i, dtype=np.uint8) for i in range(3)
            ]
        else:
            self.last_generated_frames = None
        return self._inner.predict_primitive(obs_rgb, state, instruction)


@pytest.mark.skipif(not _have_image_writer(), reason="needs Pillow or opencv-python")
def test_save_wm_clip_writes_pngs_and_mp4(tmp_path, monkeypatch):
    frames = [np.full((8, 8, 3), 40 * i, dtype=np.uint8) for i in range(3)]
    appended: list[object] = []

    class _Writer:
        def append_data(self, frame):
            appended.append(frame)

        def close(self):
            return None

    import sys
    import types

    fake_v2 = types.ModuleType("imageio.v2")
    fake_v2.get_writer = lambda path, **kw: _Writer()
    monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    monkeypatch.setitem(sys.modules, "imageio.v2", fake_v2)

    out = _save_wm_clip(tmp_path, "000_r0_wm_step0000", frames)
    assert out == tmp_path / "000_r0_wm_step0000.mp4"
    assert len(appended) == 3
    pngs = sorted(tmp_path.glob("000_r0_wm_step0000_f*.png"))
    assert [p.name for p in pngs] == [
        "000_r0_wm_step0000_f00.png",
        "000_r0_wm_step0000_f01.png",
        "000_r0_wm_step0000_f02.png",
    ]


def test_save_wm_clip_noop_on_empty(tmp_path):
    assert _save_wm_clip(tmp_path, "x_wm_step0000", []) is None
    assert not any(tmp_path.iterdir())


def test_run_episode_dumps_wm_clip_step0_only(tmp_path, monkeypatch):
    episode = load_annotation(FIXTURE)[0]
    dump_dir = tmp_path / "frames"
    bridge = _NonMockBridge()
    policy = _WMPolicy(episode["action"])

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._assemble_episode_mp4",
        lambda dump, prefix, **kw: None,
    )
    wm_calls: list[str] = []
    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._save_wm_clip",
        lambda dump, prefix, frames, **kw: wm_calls.append(prefix),
    )

    run_episode(
        bridge,
        policy,
        episode,
        max_steps=20,
        dump_dir=dump_dir,
        frame_prefix="000",
        dump_wm=True,
    )
    # Default cadence: exactly one world-model dump, at step 0.
    assert wm_calls == ["000_wm_step0000"]
    # dump_video must be turned back off after the step-0 capture.
    assert policy.dump_video is False


def test_run_episode_wm_dump_every_samples_midrollout(tmp_path, monkeypatch):
    episode = load_annotation(FIXTURE)[0]
    dump_dir = tmp_path / "frames"
    bridge = _NonMockBridge()
    policy = _WMPolicy(episode["action"])

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._assemble_episode_mp4",
        lambda dump, prefix, **kw: None,
    )
    wm_calls: list[str] = []
    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._save_wm_clip",
        lambda dump, prefix, frames, **kw: wm_calls.append(prefix),
    )

    run_episode(
        bridge,
        policy,
        episode,
        max_steps=20,
        dump_dir=dump_dir,
        frame_prefix="000",
        dump_wm=True,
        wm_dump_every=2,
    )
    assert "000_wm_step0000" in wm_calls
    assert "000_wm_step0002" in wm_calls
    assert "000_wm_step0001" not in wm_calls


def test_run_episode_no_wm_dump_when_disabled(tmp_path, monkeypatch):
    episode = load_annotation(FIXTURE)[0]
    dump_dir = tmp_path / "frames"
    bridge = _NonMockBridge()
    policy = _WMPolicy(episode["action"])

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._assemble_episode_mp4",
        lambda dump, prefix, **kw: None,
    )
    wm_calls: list[str] = []
    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop._save_wm_clip",
        lambda dump, prefix, frames, **kw: wm_calls.append(prefix),
    )

    run_episode(
        bridge,
        policy,
        episode,
        max_steps=20,
        dump_dir=dump_dir,
        frame_prefix="000",
        dump_wm=False,
    )
    assert wm_calls == []
    assert policy.dump_video is False


def test_eval_overrides_use_local_wan22_and_text_encoder(monkeypatch):
    monkeypatch.delenv("AERIAL_EVAL_HYDRA_OVERRIDES", raising=False)
    overrides = eval_hydra_overrides("aerial_joint_1cam_1e-4")
    assert overrides[0] == "task=aerial_joint_1cam_1e-4"
    # redirect=true resolves the VAE to converted safetensors that are not on
    # the eval hosts, which made every B0 v2 checkpoint eval fail.
    assert "model.redirect_common_files=false" in overrides
    assert "model.load_text_encoder=true" in overrides


def test_eval_overrides_append_env_extras(monkeypatch):
    monkeypatch.setenv(
        "AERIAL_EVAL_HYDRA_OVERRIDES", "model.foo=1, data.train.num_frames=9 ,"
    )
    overrides = eval_hydra_overrides("aerial_joint_1cam_1e-4")
    assert overrides[-2:] == ["model.foo=1", "data.train.num_frames=9"]


def test_evaluate_episodes_builds_policy_once(monkeypatch):
    """Regression: rebuilding Wan2.2 every episode OOMs on episode 2."""
    episodes = load_annotation(FIXTURE)
    assert len(episodes) >= 2

    calls: list[int] = []

    def fake_build_policy(policy_name, episode, **kwargs):
        calls.append(1)
        return ReplayPolicy(episode.get("action", []) or [0])

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop.build_policy",
        fake_build_policy,
    )
    metrics = evaluate_episodes(
        episodes[:2],
        bridge_name="mock",
        policy_name="fastwam",
        max_steps=20,
        checkpoint=Path("/tmp/unused.pt"),
        seed=0,
    )
    assert len(calls) == 1
    assert metrics["n"] == 2.0


def test_mock_metrics_reproducible_with_seed():
    episodes = load_annotation(FIXTURE)
    first = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=99,
    )
    second = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=99,
    )
    assert first == second


# ---------------------------------------------------------------------------
# Stage-0 oracle-stop + closest-approach diagnostics
# ---------------------------------------------------------------------------

# Never stops on its own: emits the same forward primitive forever. Reproduces
# the b0_v2 failure mode (fwd-only, flies past the goal) so the oracle-stop and
# closest-approach diagnostics have something realistic to measure.
class _ForwardPolicy:
    def __init__(self, primitive: int = 1) -> None:
        self._primitive = int(primitive)

    def reset(self) -> None:
        return None

    def predict_primitive(self, obs_rgb, state, instruction):
        del obs_rgb, state, instruction
        return self._primitive


def _line_episode(goal_x: float) -> dict:
    """Goal straight ahead on +x; fwd3 (primitive 1) marches toward it."""
    return {
        "pos": [[0.0, 0.0, 0.0], [goal_x, 0.0, 0.0]],
        "yaw": [0.0, 0.0],
        "gpt_instruction": "fly forward",
    }


def _sideways_episode(goal_y: float) -> dict:
    """Goal off to the side; a fwd-only (+x) policy never approaches it."""
    return {
        "pos": [[0.0, 0.0, 0.0], [0.0, goal_y, 0.0]],
        "yaw": [0.0, 0.0],
        "gpt_instruction": "fly forward",
    }


def test_oracle_stop_terminates_within_success_dist():
    # Goal at x=30; fwd3 steps: dist 27,24,21,18(<20). Stops the moment it
    # enters SUCCESS_DIST (20 m), succeeds, and reports why it stopped.
    episode = _line_episode(30.0)
    result = run_episode(
        MockBridge(seed=1), _ForwardPolicy(), episode, max_steps=100, oracle_stop=True
    )
    assert result.terminated_by == "oracle_stop"
    assert result.success
    assert result.steps == 4
    assert result.closest_approach == pytest.approx(18.0, abs=1e-6)


def test_without_oracle_stop_flies_past_but_closest_approach_records_the_miss():
    # Same fwd-only policy, no oracle stop: it flies straight through the goal
    # to x=300, so NE is huge and it FAILS — yet closest_approach ~= 0 proves
    # the trajectory did pass through. This is the never-stops diagnostic.
    episode = _line_episode(30.0)
    result = run_episode(
        MockBridge(seed=1), _ForwardPolicy(), episode, max_steps=100, oracle_stop=False
    )
    assert result.terminated_by == "max_steps"
    assert not result.success
    assert result.navigation_error == pytest.approx(270.0, abs=1e-6)
    assert result.closest_approach == pytest.approx(0.0, abs=1e-6)
    assert result.steps == 100


def test_closest_approach_records_heading_miss_when_goal_never_approached():
    # Goal 100 m to the side; a fwd-only policy only increases the distance.
    # closest_approach stays at the start distance -> oracle stop can't help.
    # This is the heading/goal-conditioning failure (root cause #2).
    episode = _sideways_episode(100.0)
    result = run_episode(
        MockBridge(seed=1), _ForwardPolicy(), episode, max_steps=50, oracle_stop=True
    )
    assert result.terminated_by == "max_steps"
    assert not result.success
    assert result.closest_approach == pytest.approx(100.0, abs=1e-6)


def test_evaluate_episodes_emits_stage0_diagnostics(monkeypatch):
    episodes = [_line_episode(30.0), _sideways_episode(100.0)]

    monkeypatch.setattr(
        "experiments.aerial.eval.run_closed_loop.build_policy",
        lambda *a, **kw: _ForwardPolicy(),
    )
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="fastwam",
        max_steps=100,
        checkpoint=Path("/tmp/unused.pt"),
        seed=0,
        oracle_stop=True,
    )

    assert metrics["protocol_version"] == "stage0-oracle-v1"
    assert metrics["oracle_stop"] is True
    # One episode enters 20 m (oracle stop), the other never does (times out).
    assert metrics["oracle_hit@20"] == pytest.approx(0.5)
    assert metrics["n_oracle_stop"] == 1.0
    assert metrics["n_max_steps"] == 1.0
    assert metrics["n_oracle_stop"] + metrics["n_stop_primitive"] + metrics[
        "n_max_steps"
    ] == metrics["n"]
    assert np.isfinite(metrics["closest_approach_mean"])
    assert np.isfinite(metrics["closest_approach_median"])


def test_write_metrics_preserves_protocol_version_string(tmp_path):
    episodes = [_line_episode(30.0)]
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=0,
        oracle_stop=True,
    )
    out = tmp_path / "metrics.json"
    write_metrics(metrics, out)
    loaded = json.loads(out.read_text())
    # String / bool keys must survive the float coercion in write_metrics.
    assert loaded["protocol_version"] == "stage0-oracle-v1"
    assert loaded["oracle_stop"] is True
    assert isinstance(loaded["closest_approach_mean"], float)
    assert loaded["episodes"][0]["terminated_by"] in {
        "oracle_stop",
        "stop_primitive",
        "max_steps",
    }


def test_default_behavior_unchanged_without_oracle_stop():
    # Regression guard: the replay success path must be identical to before —
    # oracle_stop defaults to off and does not perturb NE / steps / success.
    episode = load_annotation(FIXTURE)[0]
    result = run_episode(MockBridge(seed=7), ReplayPolicy(episode["action"]), episode, max_steps=20)
    assert result.success
    assert result.navigation_error == pytest.approx(0.0, abs=1e-5)
    assert result.terminated_by == "stop_primitive"
    assert result.closest_approach == pytest.approx(0.0, abs=1e-5)
