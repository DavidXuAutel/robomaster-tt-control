"""Tests for the DreamerV3 numpy reference primitives (``dreamer_recipe``).

These pin the exact math the torch RSSM trainer (Phase 2b) must reproduce:
symlog/symexp invertibility, two-hot targets whose expectation equals the label,
categorical KL, free-bits clamping, and the balanced-KL loss combine.
"""
import numpy as np
import pytest

from experiments.aerial.rl import dreamer_recipe as dr


# -- symlog / symexp ----------------------------------------------------------

def test_symlog_symexp_round_trip():
    x = np.array([-1000.0, -12.5, -1.0, 0.0, 1.0, 3.3, 250.0, 1e5])
    np.testing.assert_allclose(dr.symexp(dr.symlog(x)), x, rtol=1e-9, atol=1e-6)


def test_symlog_zero_and_sign():
    assert dr.symlog(0.0) == 0.0
    assert dr.symlog(5.0) > 0 and dr.symlog(-5.0) < 0
    # compresses magnitude: |symlog(x)| << |x| for large x
    assert abs(float(dr.symlog(1000.0))) < 10.0


# -- two-hot ------------------------------------------------------------------

def test_two_hot_sums_to_one_and_bracackets():
    bins = dr.make_bins(11, -5.0, 5.0)          # centres at -5,-4,...,5
    probs = dr.two_hot_encode(2.3, bins)
    assert probs.shape == (11,)
    assert float(probs.sum()) == pytest.approx(1.0)
    # only the two bracketing bins (2.0 and 3.0) carry mass
    nz = np.nonzero(probs > 1e-12)[0]
    assert set(nz.tolist()) == {7, 8}           # bins[7]=2.0, bins[8]=3.0


def test_two_hot_decode_recovers_value_in_range():
    bins = dr.make_bins(101, -10.0, 10.0)
    xs = np.array([-9.99, -3.14, 0.0, 0.5, 7.7, 9.99])
    got = dr.two_hot_decode(dr.two_hot_encode(xs, bins), bins)
    np.testing.assert_allclose(got, xs, atol=1e-9)


def test_two_hot_clamps_out_of_range():
    bins = dr.make_bins(5, -2.0, 2.0)
    # below/above the grid clamp to the boundary bin value
    assert dr.two_hot_decode(dr.two_hot_encode(-100.0, bins), bins) == pytest.approx(-2.0)
    assert dr.two_hot_decode(dr.two_hot_encode(100.0, bins), bins) == pytest.approx(2.0)


def test_two_hot_at_bin_centre_is_one_hot():
    bins = dr.make_bins(7, -3.0, 3.0)
    probs = dr.two_hot_encode(bins[4], bins)     # exactly a centre
    assert float(probs[4]) == pytest.approx(1.0)
    assert float(probs.sum()) == pytest.approx(1.0)


def test_two_hot_encode_preserves_leading_shape():
    bins = dr.make_bins(21, -10.0, 10.0)
    x = np.zeros((3, 4))
    probs = dr.two_hot_encode(x, bins)
    assert probs.shape == (3, 4, 21)


def test_symlog_twohot_round_trip():
    bins = dr.make_bins(255)                      # default symlog grid [-20, 20]
    # symlog(500) ~= 6.2, well inside the grid, so the head recovers 500 exactly
    xs = np.array([-500.0, -3.0, 0.0, 12.0, 500.0])
    got = dr.twohot_symexp_decode(dr.twohot_symlog_encode(xs, bins), bins)
    np.testing.assert_allclose(got, xs, rtol=1e-6, atol=1e-4)


def test_decode_bins_mismatch_raises():
    with pytest.raises(ValueError):
        dr.two_hot_decode(np.ones((3,)), dr.make_bins(5))


# -- categorical KL / free bits / balance ------------------------------------

def test_categorical_kl_zero_when_equal():
    p = np.array([0.2, 0.3, 0.5])
    assert dr.categorical_kl(p, p) == pytest.approx(0.0, abs=1e-6)


def test_categorical_kl_positive_and_batched():
    post = np.array([[0.9, 0.1], [0.5, 0.5]])
    prior = np.array([[0.5, 0.5], [0.5, 0.5]])
    kl = dr.categorical_kl(post, prior)
    assert kl.shape == (2,)
    assert kl[0] > 0.0
    assert kl[1] == pytest.approx(0.0, abs=1e-6)


def test_free_bits_clamps_below_floor_only():
    assert float(dr.free_bits(0.2, nats=1.0)) == pytest.approx(1.0)   # clamped up
    assert float(dr.free_bits(3.0, nats=1.0)) == pytest.approx(3.0)   # untouched


def test_kl_balance_weights_and_free_bits():
    # both KLs below the free-bits floor -> both clamp to 1 nat, then scaled
    total, dyn, rep = dr.kl_balance(0.1, 0.1, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1)
    assert dyn == pytest.approx(1.0)
    assert rep == pytest.approx(0.1)
    assert total == pytest.approx(1.1)


def test_kl_balance_uses_large_kls_unclamped():
    total, dyn, rep = dr.kl_balance(4.0, 2.0, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1)
    assert dyn == pytest.approx(4.0)
    assert rep == pytest.approx(0.2)
    assert total == pytest.approx(4.2)
