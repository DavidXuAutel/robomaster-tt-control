"""DreamerV3 recipe primitives — pure-numpy reference math (V1 world model).

The design doc (§1.5) transfers the DreamerV3 *recipe* to the aerial WM: symlog
prediction targets, two-hot regression heads, categorical-latent KL with free
bits, and KL balancing across fixed loss scales (βpred=1, βdyn=1, βrep=0.1). The
torch RSSM trainer (``dynamics_torch.py``, Phase 2b, H100-only) cannot run on
this torch-free host, so these functions are the **single source of truth** the
torch losses must reproduce exactly — the torch head builds the same two-hot
target, applies the same symlog, and its unit tests assert equality against
these references on tiny CPU tensors.

Everything here is stateless, exact, and depends only on numpy so the whole
``rl`` package keeps importing (and its tests keep running) without a GPU.

References:
  * symlog / two-hot regression: Hafner et al., "Mastering Diverse Domains
    through World Models" (DreamerV3), §"Symlog predictions".
  * categorical-latent KL + free bits + KL balancing: same, §"World model
    learning" (L_dyn = KL(sg(post)‖prior), L_rep = KL(post‖sg(prior))).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# DreamerV3 defaults. The reward/progress head regresses a two-hot distribution
# over ``DEFAULT_NUM_BINS`` bins spanning symlog space [-20, 20]; free bits clip
# each KL term at 1 nat.
DEFAULT_NUM_BINS = 255
DEFAULT_BIN_LO = -20.0
DEFAULT_BIN_HI = 20.0
DEFAULT_FREE_NATS = 1.0
# Loss scales (design doc §2.1 / §1.5). Kept here so the torch trainer and any
# numpy reference share one definition.
BETA_PRED = 1.0
BETA_DYN = 1.0
BETA_REP = 0.1

_EPS = 1e-8


# -- symlog / symexp ---------------------------------------------------------
def symlog(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log(1 + |x|)`` — DreamerV3's magnitude-compressing transform.

    Squashes wide-range targets (returns, progress in metres) into a stable
    range while staying invertible via :func:`symexp`.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`symlog`: ``sign(y) * (exp(|y|) - 1)``."""
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y))


# -- two-hot regression ------------------------------------------------------
def make_bins(
    num_bins: int = DEFAULT_NUM_BINS,
    lo: float = DEFAULT_BIN_LO,
    hi: float = DEFAULT_BIN_HI,
) -> np.ndarray:
    """Evenly spaced bin centres ``[num_bins]`` (ascending) for two-hot heads.

    The default grid lives in *symlog space*: pair with :func:`twohot_symlog_encode`
    / :func:`twohot_symexp_decode` so the head regresses symlog(target).
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    return np.linspace(float(lo), float(hi), int(num_bins), dtype=np.float64)


def two_hot_encode(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Two-hot target over ``bins`` — weight mass on the two bracketing bins so
    the distribution's expectation equals ``x`` exactly (values outside the grid
    clamp to the boundary bin).

    Returns an array shaped ``x.shape + (len(bins),)`` summing to 1 on the last
    axis. Inverse of :func:`two_hot_decode` for any ``x`` within ``[bins[0],
    bins[-1]]``.
    """
    x = np.asarray(x, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    if bins.ndim != 1 or bins.shape[0] < 2:
        raise ValueError("bins must be 1-D with length >= 2")
    k = bins.shape[0]
    xc = np.clip(x, bins[0], bins[-1])
    # lower = largest index with bins[lower] <= xc, clamped so upper=lower+1 valid.
    lower = np.clip(np.searchsorted(bins, xc, side="right") - 1, 0, k - 2)
    upper = lower + 1
    span = bins[upper] - bins[lower]
    w_upper = np.where(span > 0, (xc - bins[lower]) / np.where(span > 0, span, 1.0), 0.0)

    flat_n = int(xc.size)
    lo_f = np.asarray(lower).reshape(flat_n)
    up_f = np.asarray(upper).reshape(flat_n)
    wu_f = np.asarray(w_upper, dtype=np.float64).reshape(flat_n)
    out = np.zeros((flat_n, k), dtype=np.float64)
    rows = np.arange(flat_n)
    out[rows, lo_f] += 1.0 - wu_f
    out[rows, up_f] += wu_f
    return out.reshape(x.shape + (k,))


def two_hot_decode(probs: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Scalar readout = expected bin value ``sum(probs * bins)`` on the last axis."""
    probs = np.asarray(probs, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    if probs.shape[-1] != bins.shape[0]:
        raise ValueError(
            f"probs last axis ({probs.shape[-1]}) must match bins ({bins.shape[0]})"
        )
    return np.sum(probs * bins, axis=-1)


def twohot_symlog_encode(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """DreamerV3 regression target: two-hot over ``symlog(x)`` (bins in symlog space)."""
    return two_hot_encode(symlog(x), bins)


def twohot_symexp_decode(probs: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Readout of a symlog two-hot head: ``symexp(E[bins])`` back to raw units."""
    return symexp(two_hot_decode(probs, bins))


# -- categorical-latent KL / free bits / KL balancing ------------------------
def categorical_kl(post: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """KL(post ‖ prior) for categorical distributions given probabilities.

    Summed over the last (class) axis. Used for the RSSM discrete-latent KL; the
    torch trainer computes the same quantity from logits with stop-gradient on
    one side (see :func:`kl_balance`).
    """
    post = np.asarray(post, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if post.shape != prior.shape:
        raise ValueError(f"post {post.shape} and prior {prior.shape} must match")
    return np.sum(post * (np.log(post + _EPS) - np.log(prior + _EPS)), axis=-1)


def free_bits(kl: np.ndarray, nats: float = DEFAULT_FREE_NATS) -> np.ndarray:
    """Clip KL below ``nats`` (DreamerV3 free bits): ``max(kl, nats)``.

    Stops the KL term from being driven to zero (posterior collapse) once it is
    already small enough, so the model keeps encoding information.
    """
    return np.maximum(np.asarray(kl, dtype=np.float64), float(nats))


def kl_balance(
    kl_dyn: np.ndarray,
    kl_rep: np.ndarray,
    *,
    free_nats: float = DEFAULT_FREE_NATS,
    beta_dyn: float = BETA_DYN,
    beta_rep: float = BETA_REP,
) -> Tuple[float, float, float]:
    """DreamerV3 balanced KL loss from the two free-bits-clamped KL terms.

    ``kl_dyn`` = KL(sg(post) ‖ prior) trains the prior toward the posterior;
    ``kl_rep`` = KL(post ‖ sg(prior)) regularises the posterior. Stop-gradients
    are applied *before* these KLs in the torch trainer; here the inputs are the
    already-computed scalar KLs. Returns ``(total, dyn_term, rep_term)`` where
    ``total = beta_dyn * fb(kl_dyn) + beta_rep * fb(kl_rep)``.
    """
    dyn = float(beta_dyn) * float(free_bits(kl_dyn, free_nats))
    rep = float(beta_rep) * float(free_bits(kl_rep, free_nats))
    return dyn + rep, dyn, rep
