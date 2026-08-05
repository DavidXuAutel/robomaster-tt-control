"""Mac (torch-free) tests for the V0 ②/④ rollout runners + gate assembly.

② is exercised on the mock env (goal-seeking ``HeuristicPolicy`` genuinely
closes distance, random does not). The mock has no obstacles, so ④ is exercised
on a purpose-built wall stub with a pessimistic GT-proxy depth predictor — the
point of the Mac test is the *wiring* (predictor → obs.info → shield override →
fewer near-collision steps), not a renderer-grade physics check. The real ④
pass happens on airsim with a trained depth head.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.rl import v0_metrics as metrics
from experiments.aerial.rl import v0_rollout_eval as rollout
from experiments.aerial.rl._v0_gate import assemble_verdict
from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv, MockEnvConfig
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.train_rl import HeuristicPolicy


# --------------------------------------------------------------------------- #
# ② progress-vs-random on the mock env                                         #
# --------------------------------------------------------------------------- #
def test_signal2_heuristic_beats_random_on_mock():
    env = MockAirSimDroneEnv(MockEnvConfig(step_hz=5.0))
    starts = rollout.make_start_episodes(16, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    rnd = rollout.RandomActionPolicy(seed=0)
    prog = rollout.run_progress_eval(env, policy, rnd, starts, max_steps=200)

    s2 = metrics.check_progress_vs_random(
        prog["policy_progress_sums"], prog["random_progress_sums"],
        prog["policy_final_dists"], prog["random_final_dists"],
    )
    assert s2["ok"], s2
    # Sanity: the goal-seeker really does close more distance than random.
    assert s2["mean_progress_policy"] > s2["mean_progress_random"]
    assert s2["mean_final_dist_policy"] < s2["mean_final_dist_random"]


# --------------------------------------------------------------------------- #
# ④ shield-on cuts near-collision on a wall stub                               #
# --------------------------------------------------------------------------- #
class _WallEnv:
    """Minimal obstacle env: a wall at ``wall_x``; GT depth = distance to it.

    Kinematics are 1-D along +x (yaw fixed): each step advances by the commanded
    forward delta, clamped so the vehicle cannot pass through the wall. ``depth``
    is a full field whose min equals ``wall_x - pos_x`` (clipped ≥ 0.05), so the
    GT near-collision mask and a GT-proxy predictor can both read it.
    """

    def __init__(self, *, wall_x: float = 10.0, step_hz: float = 5.0, size: int = 8) -> None:
        self.config = type("C", (), {"step_hz": float(step_hz)})()
        self._wall_x = float(wall_x)
        self._size = int(size)
        self._pos = np.zeros(3, dtype=np.float64)
        self._goal = np.array([30.0, 0.0, 0.0], dtype=np.float64)

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    def _depth_min(self) -> float:
        return max(self._wall_x - float(self._pos[0]), 0.05)

    def _observe(self) -> Observation:
        d = np.full((self._size, self._size), self._depth_min(), dtype=np.float32)
        collided = bool(self._pos[0] >= self._wall_x)
        state = np.array(
            [self._pos[0], self._pos[1], self._pos[2], 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return Observation(rgb=np.zeros((self._size, self._size, 3), dtype=np.uint8),
                           state=state, depth=d, collided=collided, info={})

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        if episode is not None:
            self._pos = np.asarray(episode["pos"], dtype=np.float64)[0].copy()
            self._goal = np.asarray(episode["pos"], dtype=np.float64)[-1].copy()
        else:
            self._pos = np.zeros(3, dtype=np.float64)
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        # Advance forward (body +x with yaw 0), clamp at the wall.
        self._pos[0] = min(self._pos[0] + float(action[0]), self._wall_x)
        return self._observe(), {"cmd": np.asarray(action, dtype=np.float64).tolist()}


class _PessimisticGTDepthPredictor:
    """GT-proxy predictor with a lookahead margin (stops before the near zone).

    Returns ``min(GT depth) - margin`` so the shield (threshold 1.5) triggers
    while the true depth is still well above 1.5 — the way a real depth head +
    safety margin would brake *before* the vehicle enters the near-collision
    band. Reads GT depth only inside the runner's scoring path (never a policy
    input), which is exactly what the V0 supervision boundary allows.
    """

    def __init__(self, margin: float = 1.6) -> None:
        self.margin = float(margin)

    def reset(self) -> None:
        return None

    def predict_min(self, obs: Observation) -> Optional[float]:
        d = np.asarray(obs.depth, dtype=np.float64)
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return None
        return float(np.min(finite)) - self.margin


def test_signal4_shield_reduces_near_collision_on_wall():
    env = _WallEnv(wall_x=10.0, step_hz=5.0)
    starts = rollout.make_start_episodes(8, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=1.6)

    masks = rollout.run_shield_eval(
        env, policy, predictor, starts,
        near_collision_depth_m=1.5, max_steps=60,
    )
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
    )
    assert s4["ok"], s4
    # Shield-off plows into the wall band; shield-on brakes before it.
    assert s4["near_coll_rate_off"] > 0.0
    assert s4["near_coll_rate_on"] < s4["near_coll_rate_off"]


def test_episode_masks_collided_reads_post_step_obs():
    """④ ``collided`` is a post-step event: it lands on ``next_obs`` of the
    terminal transition, never on any ``obs``. Reading pre-step ``tr.obs`` (the
    old bug) made the mask all-False and vacuously passed intervention-before-
    contact. The mask must be True exactly on the terminal step."""
    from experiments.aerial.rl.buffer import Transition

    def _obs(collided: bool) -> Observation:
        return Observation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            state=np.zeros(7, dtype=np.float32),
            depth=np.full((2, 2), 5.0, dtype=np.float32),
            collided=collided,
            info={},
        )

    # 3 steps; contact happens on the last step (only next_obs carries it).
    ep = [
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=False,
                   next_obs=_obs(False), info={"intervention": True}),
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=False,
                   next_obs=_obs(False), info={"intervention": True}),
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=True,
                   next_obs=_obs(True), info={"intervention": False}),
    ]
    masks = rollout._episode_masks(ep, near_collision_depth_m=1.5)
    assert masks["collided"] == [False, False, True], masks["collided"]
    # With a real contact on step 2 and interventions on steps 0-1, the ④
    # intervention-before-contact sub-metric is now exercised (not vacuous).
    s4 = metrics.check_shield_effectiveness(
        interventions_on=[masks["intervention"]],
        collided_on=[masks["collided"]],
        near_coll_on=[masks["near"]],
        near_coll_off=[masks["near"]],
    )
    assert s4["n_contact_episodes"] == 1, s4
    assert s4["intervention_before_contact_frac"] == 1.0, s4


def test_signal4_degenerate_on_obstacle_free_mock():
    """Mock has no obstacles (depth ramp min ≈ 1.0 always < 1.5) → ④ must NOT
    spuriously pass: on/off near-rates are ~equal so the ratio fails."""
    env = MockAirSimDroneEnv(MockEnvConfig(step_hz=5.0))
    starts = rollout.make_start_episodes(4, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=0.0)  # honest, but no obstacle

    masks = rollout.run_shield_eval(
        env, policy, predictor, starts, near_collision_depth_m=1.5, max_steps=20,
    )
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
    )
    assert not s4["ok"], s4  # honest degenerate: no real obstacle to avoid


# --------------------------------------------------------------------------- #
# gate assembly: depth pillar cannot be bypassed                               #
# --------------------------------------------------------------------------- #
def _ok() -> Dict[str, Any]:
    return {"ok": True}


def test_assemble_verdict_requires_depth_pillar():
    green = assemble_verdict(s1abc=_ok(), s1d=_ok(), s2=_ok(), s3=_ok(), s4=_ok())
    assert green["ok"], green
    # ①a–c green but ①d (depth AbsRel) failing → whole gate fails.
    no_d = assemble_verdict(
        s1abc=_ok(), s1d={"ok": False, "reason": "no depth head"},
        s2=_ok(), s3=_ok(), s4=_ok(),
    )
    assert not no_d["ok"], no_d
    assert no_d["passed"]["1"] is False


# --------------------------------------------------------------------------- #
# split evaluation (plan B): --signals subsets → --emit partials → --merge     #
# --------------------------------------------------------------------------- #
def test_merge_partials_reproduces_single_host_verdict(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    sig1 = {"ok": True, "abc": _ok(), "d": _ok()}
    part_13 = {"partial": True, "signals": {"1": sig1, "3": _ok()}}
    part_24 = {"partial": True, "signals": {"2": _ok(), "4": _ok()}}
    p13 = tmp_path / "part_13.json"
    p24 = tmp_path / "part_24.json"
    p13.write_text(json.dumps(part_13))
    p24.write_text(json.dumps(part_24))

    merged = gate._merge_partials([p13, p24])
    assert set(merged) == {"1", "2", "3", "4"}
    verdict = metrics.aggregate_v0_verdict(merged)
    assert verdict["ok"], verdict


def test_merge_missing_signal_is_not_a_pass(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    p13 = tmp_path / "part_13.json"
    p13.write_text(json.dumps({"signals": {"1": {"ok": True}, "3": {"ok": True}}}))
    merged = gate._merge_partials([p13])
    verdict = metrics.aggregate_v0_verdict(merged)
    assert not verdict["ok"], verdict  # ②/④ absent → cannot pass


def test_aggregate_rejects_non_bool_ok():
    """``bool("false")`` is True. aggregate_v0_verdict must not coerce a
    string/int 'ok' into a pass — a partial round-tripped through a hand-edited
    JSON with ok="false" would otherwise flip a failing signal green."""
    all_true = {k: {"ok": True} for k in ("1", "2", "3", "4")}
    assert metrics.aggregate_v0_verdict(all_true)["ok"] is True

    with_string = dict(all_true)
    with_string["3"] = {"ok": "false"}  # truthy string
    v = metrics.aggregate_v0_verdict(with_string)
    assert v["ok"] is False, v
    assert "non-bool" in v["reason"], v


def test_parse_signals_subset_and_default():
    from experiments.aerial.rl import _v0_gate as gate

    assert gate._parse_signals(None) == {"1", "2", "3", "4"}
    assert gate._parse_signals("1,3") == {"1", "3"}
    assert gate._parse_signals(" 2 , 4 ") == {"2", "4"}


def test_merge_cli_exits_nonzero_when_incomplete(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    p = tmp_path / "part.json"
    p.write_text(json.dumps({"signals": {"1": {"ok": True}, "3": {"ok": True}}}))
    assert gate.main(["--merge", str(p)]) == 1


def test_rollout_signals_fail_closed_on_non_airsim_backend(tmp_path):
    """②/④ must not be scored authoritatively on a mock/analytic env: the
    goal-seeker trivially beats random and there are no real obstacles, which is
    the false-pass class that invalidated the single-pillar checkpoint. Default
    config ships backend:mock → ②/④ come back FAIL with an 'airsim' reason and
    the CLI exits non-zero, without ever building the env."""
    from experiments.aerial.rl import _v0_gate as gate

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("env:\n  backend: mock\n")
    out = tmp_path / "part2.json"
    rc = gate.main(["--signals", "2", "--config", str(cfg), "--emit", str(out)])
    assert rc == 1
    blob = json.loads(out.read_text())
    s2 = blob["signals"]["2"]
    assert s2["ok"] is False
    assert "airsim" in s2["reason"]
    assert s2["backend"] == "mock"


def test_signal1abc_fails_on_missing_recon_ent_keys(tmp_path):
    """①a–c is recon-monotonicity ∧ no-collapse ∧ loss-drop. A learning log with
    only ``loss`` (recon/entropy keys absent) must FAIL — the old pass-safe
    defaults (recon 0, ent 1) let a WM green-light ①a–c on the loss drop alone,
    the exact single-pillar shortcut that invalidated wm_step_5000."""
    from experiments.aerial.rl import _v0_gate as gate

    log = tmp_path / "loss_only.jsonl"
    log.write_text("\n".join(
        json.dumps({"loss": 1.0 - 0.05 * i}) for i in range(20)
    ))
    res = gate._signal1abc_from_log(log, metrics.DEFAULT_THRESHOLDS)
    assert res["ok"] is False, res
    assert "missing" in res["reason"], res

    # Full log (loss + recon + ent) still evaluates on the real curves.
    full = tmp_path / "full.jsonl"
    full.write_text("\n".join(
        json.dumps({"loss": 1.0 - 0.04 * i, "recon_err": 1.0 - 0.03 * i,
                    "post_entropy_frac": 0.9})
        for i in range(20)
    ))
    ok = gate._signal1abc_from_log(full, metrics.DEFAULT_THRESHOLDS)
    assert "missing" not in ok.get("reason", ""), ok


# --------------------------------------------------------------------------- #
# ③ diagnostic: forward-motion window selection (pure math)                     #
# --------------------------------------------------------------------------- #
def test_forwardness_separates_forward_lateral_climb():
    from experiments.aerial.rl._v0_gate import _forwardness

    # heading = +x (yaw 0) for all three windows, L=4.
    yaw = np.zeros((3, 4), dtype=np.float64)
    dvec = np.array(
        [
            [5.0, 0.0, 0.0],   # forward along heading  → |cos| ≈ 1
            [0.0, 5.0, 0.0],   # pure lateral (strafe)  → |cos| ≈ 0
            [0.0, 0.0, 5.0],   # pure climb             → |cos| ≈ 0
        ],
        dtype=np.float64,
    )
    f = _forwardness(dvec, yaw)
    assert f[0] > 0.95
    assert f[1] < 0.05
    assert f[2] < 0.05


def test_forwardness_backward_is_axis_aligned():
    from experiments.aerial.rl._v0_gate import _forwardness

    # moving backward along heading still changes |Δ median depth| → keep it.
    f = _forwardness(np.array([[-5.0, 0.0, 0.0]]), np.zeros((1, 4)))
    assert f[0] > 0.95
