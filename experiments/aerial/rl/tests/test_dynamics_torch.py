"""Phase 2b torch-RSSM WM tests — SKIP when torch is absent (dev host is GPU-less).

These run on the H100 (torch 2.7.1+cu128) on tiny CPU tensors. Two jobs:
  1. Pin the torch primitives to the :mod:`dreamer_recipe` numpy reference
     (symlog / two-hot / categorical-KL) — the §1.5 single-source-of-truth check.
  2. Smoke the model end-to-end (build → training_loss → backward → update →
     encode/step packing) so the data plumbing and shapes are de-risked.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402  (skip whole module off-H100)

from experiments.aerial.rl import dreamer_recipe as ref  # noqa: E402
from experiments.aerial.rl.buffer import Transition  # noqa: E402
from experiments.aerial.rl.dynamics_torch import (  # noqa: E402
    TorchRSSMDynamics,
    _categorical_kl,
    _symexp,
    _symlog,
    _twohot_decode,
    _twohot_targets,
)
from experiments.aerial.rl.env.obs import Observation  # noqa: E402


# image_size must be a multiple of 16 (encoder does 4 stride-2 downsamples; the
# decoder reconstructs via image_size//16). 16 is the smallest fast tile.
def _obs(pos=(0.0, 0.0, 0.0), collided=False, size=16):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Observation(rgb=rgb, state=state, collided=collided)


def _windows(batch=2, length=3, size=16):
    ws = []
    for b in range(batch):
        ep = []
        for t in range(length):
            ep.append(Transition(
                obs=_obs(pos=(float(t), float(b), 0.0), collided=(t == length - 1)),
                action=np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                reward=float(t) - 0.5,
                done=(t == length - 1),
            ))
        ws.append(ep)
    return ws


def _tiny_model(size=8):
    # Small dims + tiny image so the whole thing runs fast on CPU.
    return TorchRSSMDynamics(
        image_size=size, recurrent_dim=16, stoch_dim=4, stoch_classes=4,
        hidden_dim=16, num_bins=41, device="cpu", torch_dtype=torch.float32,
    )


# -- primitives match the numpy reference (§1.5 single source of truth) ------
def test_symlog_symexp_match_reference():
    x = np.linspace(-30.0, 30.0, 17)
    xt = torch.tensor(x, dtype=torch.float64)
    np.testing.assert_allclose(_symlog(xt).numpy(), ref.symlog(x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_symexp(xt).numpy(), ref.symexp(x), rtol=1e-10, atol=1e-10)


def test_twohot_targets_match_reference():
    bins_np = ref.make_bins(41, -10.0, 10.0)
    bins_t = torch.tensor(bins_np, dtype=torch.float64)
    x = np.array([-12.0, -3.3, 0.0, 1.7, 9.9], dtype=np.float64)
    got = _twohot_targets(torch.tensor(x, dtype=torch.float64), bins_t).numpy()
    exp = ref.two_hot_encode(x, bins_np)
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)
    # decode round-trips the expectation
    np.testing.assert_allclose(
        _twohot_decode(torch.tensor(exp), bins_t).numpy(),
        ref.two_hot_decode(exp, bins_np), rtol=1e-10, atol=1e-10,
    )


def test_categorical_kl_matches_reference():
    rng = np.random.default_rng(1)
    post = rng.dirichlet(np.ones(5), size=6)
    prior = rng.dirichlet(np.ones(5), size=6)
    got = _categorical_kl(torch.tensor(post), torch.tensor(prior)).numpy()
    np.testing.assert_allclose(got, ref.categorical_kl(post, prior), rtol=1e-8, atol=1e-8)


# -- model end-to-end --------------------------------------------------------
def test_training_loss_finite_and_backprops():
    m = _tiny_model()
    from experiments.aerial.rl.wm_data import windows_to_arrays
    sample = m._arrays_to_tensors(windows_to_arrays(_windows()))
    loss, ld = m.training_loss(sample)
    assert torch.isfinite(loss)
    for k in ("loss", "loss_pred", "loss_dyn", "loss_rep", "recon_err", "post_entropy"):
        assert k in ld and np.isfinite(ld[k])
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_update_returns_updated_shape_no_skipped():
    m = _tiny_model()
    out = m.update(_windows())
    assert "skipped" not in out                    # corrector must report "updated"
    for k in ("loss", "loss_pred", "loss_dyn", "loss_rep", "recon_err", "grad_norm"):
        assert k in out and np.isfinite(out[k])


def test_free_bits_floor_holds_on_kl_terms():
    m = _tiny_model()
    from experiments.aerial.rl.wm_data import windows_to_arrays
    sample = m._arrays_to_tensors(windows_to_arrays(_windows()))
    _, ld = m.training_loss(sample)
    # loss_dyn/rep = beta * clamp_min(KL, free_nats); never below beta*free_nats.
    assert ld["loss_dyn"] >= m.beta_dyn * m.free_nats - 1e-6
    assert ld["loss_rep"] >= m.beta_rep * m.free_nats - 1e-6


def test_encode_step_latent_packing():
    m = _tiny_model()
    z = m.encode(_obs())
    assert z.shape == (m.latent_dim,)
    assert m.latent_dim == m.recurrent_dim + m.stoch_dim * m.stoch_classes
    out = m.step(z, np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32))
    assert out.z_next.shape == (m.latent_dim,)
    assert 0.0 <= out.p_coll <= 1.0
    assert isinstance(out.done, bool)


def test_from_config_reads_world_model_block():
    cfg = {
        "recurrent_dim": 16, "stoch_dim": 4, "stoch_classes": 4, "num_bins": 41,
        "bin_lo": -10.0, "bin_hi": 10.0, "free_bits": 1.0,
        "loss_scales": {"pred": 1.0, "dyn": 1.0, "rep": 0.1},
        "lr": 1e-4, "grad_clip": 1000.0, "image_size": 8,
        "decoder": {"train_only": True}, "device": "cpu",
    }
    m = TorchRSSMDynamics.from_config(cfg)
    assert m.recurrent_dim == 16 and m.stoch_dim == 4 and m.stoch_classes == 4
    assert m.beta_rep == pytest.approx(0.1)
    assert m.latent_dim == 16 + 4 * 4


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny_model()
    m.update(_windows())
    p = str(tmp_path / "wm.pt")
    m.save_checkpoint(p, step=1)
    m2 = _tiny_model()
    payload = m2.load_checkpoint(p)
    assert payload["step"] == 1
    for (n1, a), (n2, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        assert torch.allclose(a, b)
