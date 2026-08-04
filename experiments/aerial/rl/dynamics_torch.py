"""Torch DreamerV3 RSSM latent world model (design doc §2.1/§2.2) — Phase 2b.

RUNS ON THE H100 ONLY (torch ``2.7.1+cu128``). ``torch`` is imported at module
top, so ``train_rl.py`` imports this module **lazily** (only when
``dynamics.kind='torch'``), keeping the stub/mock path torch-free and this whole
``rl`` package importable on the GPU-less dev host.

The loss math mirrors the pure-numpy reference in :mod:`dreamer_recipe` exactly —
symlog / two-hot / categorical-KL / free-bits / KL-balancing with the fixed loss
scales βpred=1, βdyn=1, βrep=0.1 (§1.5). That module is the single source of
truth; the torch primitives here (``_symlog``, ``_twohot_targets``,
``_categorical_kl``) are unit-tested for equality against it on tiny CPU tensors,
so the recipe is pinned without a GPU.

Follows FastWAM's module conventions (``src/fastwam/models/wan22/fastwam.py``):
``nn.Module`` subclass · explicit typed ``__init__`` kwargs · ``@classmethod
from_config(dict)`` factory · ``training_loss(sample) -> (loss, loss_dict)`` with
``forward`` delegating to it · ``save_checkpoint``/``load_checkpoint(path,
optimizer=, step=)`` · ``self.device``/``self.torch_dtype`` · ``get_logger``.
Optimizer lives in-model (AdamW betas=(0.9, 0.95) + ``clip_grad_norm_``) because
the RL corrector is a single in-loop process — deliberately NOT the
accelerate/Hydra ``Wan22Trainer`` (that orchestrates the pixel model).

§1.5 boundary: the DreamerV3 *recipe* transfers, its *tuning* does not. Do NOT
port T=16, reward-threshold 50.0, or gaze-emergence — this is a 4-D kinematic
SEARCH regime, not CTBR racing. ``MAX_IMAGINATION_HORIZON`` stays 15.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.aerial.rl.dreamer_recipe import (
    BETA_DYN,
    BETA_PRED,
    BETA_REP,
    DEFAULT_BIN_HI,
    DEFAULT_BIN_LO,
    DEFAULT_FREE_NATS,
    DEFAULT_NUM_BINS,
)
from experiments.aerial.rl.dynamics import DynamicsOutput, LatentDynamics
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.wm_data import windows_to_arrays

try:  # FastWAM's distributed-safe logger when available; std logging otherwise.
    from fastwam.utils.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fastwam not importable off-H100
    logger = logging.getLogger(__name__)

_EPS = 1e-8


# ============================================================================
# Torch mirrors of the dreamer_recipe numpy primitives (unit-tested for equality)
# ============================================================================
def _symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + |x|)`` — must match :func:`dreamer_recipe.symlog`."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _symexp(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_symlog` — matches :func:`dreamer_recipe.symexp`."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def _twohot_targets(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot target distribution over ``bins`` for scalars ``x``.

    Mirrors :func:`dreamer_recipe.two_hot_encode`: mass on the two bracketing
    bins so the distribution's expectation equals ``x`` (clamped to the grid).
    Returns ``x.shape + (len(bins),)`` summing to 1 on the last axis.
    """
    k = bins.shape[0]
    xc = x.clamp(min=float(bins[0]), max=float(bins[-1]))
    # largest index with bins[lower] <= xc, clamped so upper = lower + 1 is valid.
    lower = (torch.searchsorted(bins, xc.contiguous(), right=True) - 1).clamp(0, k - 2)
    upper = lower + 1
    span = (bins[upper] - bins[lower]).clamp_min(_EPS)
    w_upper = (xc - bins[lower]) / span
    target = torch.zeros(*x.shape, k, device=x.device, dtype=x.dtype)
    target.scatter_(-1, lower.unsqueeze(-1), (1.0 - w_upper).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))
    return target


def _twohot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected bin value ``sum(probs * bins)`` — matches ``two_hot_decode``."""
    return (probs * bins).sum(dim=-1)


def _categorical_kl(post_probs: torch.Tensor, prior_probs: torch.Tensor) -> torch.Tensor:
    """KL(post ‖ prior) summed over the last (class) axis.

    Matches :func:`dreamer_recipe.categorical_kl` (same ``+_EPS`` inside the
    logs) so the torch and numpy references agree elementwise.
    """
    return (
        post_probs * (torch.log(post_probs + _EPS) - torch.log(prior_probs + _EPS))
    ).sum(dim=-1)


# ============================================================================
# Encoder / decoder / RSSM submodules
# ============================================================================
class _RGBEncoder(nn.Module):
    """Stride-2 conv stack (DreamerV3-style, depth-doubling) → flat embedding.

    RGB-only per §1.2. The flattened conv dim is measured with a dry forward so
    the module adapts to ``image_size`` without a hand-computed shape.
    """

    def __init__(self, image_size: int, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        chans = [in_ch, base, base * 2, base * 4, base * 8]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),
                       nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, image_size, image_size)
            self.flat_dim = int(self.conv(dummy).flatten(1).shape[1])

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:  # rgb [N, C, H, W] in [0,1]
        return self.conv(rgb).flatten(1)


class _RGBDecoder(nn.Module):
    """Transposed-conv decoder — TRAIN-ONLY reconstruction supervision (§2.3).

    Never stepped in the online loop (the fast latent WM does not render pixels);
    it only shapes the representation and feeds the posterior-collapse detector.
    """

    def __init__(self, feature_dim: int, flat_dim: int, spatial: int,
                 out_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self._spatial = spatial
        self._c0 = base * 8
        self.fc = nn.Linear(feature_dim, self._c0 * spatial * spatial)
        chans = [base * 8, base * 4, base * 2, base, out_ch]
        layers = []
        for i, (cin, cout) in enumerate(zip(chans[:-1], chans[1:])):
            layers.append(nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1))
            if i < len(chans) - 2:
                layers.append(nn.SiLU())
        self.deconv = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:  # -> [N, C, H, W]
        x = self.fc(feature).view(-1, self._c0, self._spatial, self._spatial)
        return self.deconv(x)


class _RSSM(nn.Module):
    """Recurrent State-Space Model: deterministic ``h`` (GRU) + discrete ``z``.

    ``h`` is the real skeleton change §2.2 flagged over the stub's flat latent.
    ``z`` is a ``stoch_dim × stoch_classes`` categorical sampled straight-through
    with a small uniform mix (``unimix``) for stability (DreamerV3).
    """

    def __init__(self, embed_dim: int, action_dim: int, recurrent_dim: int,
                 stoch_dim: int, stoch_classes: int, hidden: int, unimix: float) -> None:
        super().__init__()
        self.recurrent_dim = recurrent_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.unimix = float(unimix)
        z_flat = stoch_dim * stoch_classes

        self.gru = nn.GRUCell(hidden, recurrent_dim)
        self.in_proj = nn.Sequential(nn.Linear(z_flat + action_dim, hidden), nn.SiLU())
        self.prior_net = nn.Sequential(nn.Linear(recurrent_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, z_flat))
        self.post_net = nn.Sequential(nn.Linear(recurrent_dim + embed_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, z_flat))

    # -- categorical helpers --------------------------------------------------
    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.view(*logits.shape[:-1], self.stoch_dim, self.stoch_classes)
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0.0:  # mix in a uniform for numerical stability (DreamerV3)
            uniform = torch.ones_like(probs) / self.stoch_classes
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        return probs

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot sample; returns flattened ``[.., z_flat]``."""
        flat = probs.reshape(-1, self.stoch_classes)
        idx = torch.multinomial(flat, 1).squeeze(-1)
        onehot = F.one_hot(idx, self.stoch_classes).to(probs.dtype).view_as(probs)
        sample = onehot + probs - probs.detach()  # straight-through gradient
        return sample.reshape(*probs.shape[:-2], self.stoch_dim * self.stoch_classes)

    def initial_h(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.recurrent_dim, device=device, dtype=dtype)

    def prior_probs(self, h: torch.Tensor) -> torch.Tensor:
        return self._logits_to_probs(self.prior_net(h))

    def post_probs(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        return self._logits_to_probs(self.post_net(torch.cat([h, embed], dim=-1)))

    def advance_h(self, h: torch.Tensor, z_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.gru(self.in_proj(torch.cat([z_flat, action], dim=-1)), h)


# ============================================================================
# The world model
# ============================================================================
class TorchRSSMDynamics(LatentDynamics, nn.Module):
    """DreamerV3 latent world model implementing the ``LatentDynamics`` contract.

    V1 exercises exactly two entry points: :meth:`update` (AdamW steps on replay
    windows) and :meth:`training_loss`. :meth:`encode` / :meth:`step` satisfy the
    imagination interface and run a genuine RSSM prior rollout, but online
    imagination (goal-conditioning, ``h``-threading through ``imagine()``) is a
    V4 concern — do not wire it into the corrector before that milestone.

    ``encode`` packs the latent as a single flat vector ``[h ‖ z]`` of width
    ``latent_dim = recurrent_dim + stoch_dim*stoch_classes``, so ``step(z, a)``
    carries the recurrent state without changing the numpy ``imagine()`` signature.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        rgb_channels: int = 3,
        proprio_dim: int = 4,
        action_dim: int = 4,
        recurrent_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 256,
        num_bins: int = DEFAULT_NUM_BINS,
        bin_lo: float = DEFAULT_BIN_LO,
        bin_hi: float = DEFAULT_BIN_HI,
        free_nats: float = DEFAULT_FREE_NATS,
        beta_pred: float = BETA_PRED,
        beta_dyn: float = BETA_DYN,
        beta_rep: float = BETA_REP,
        unimix: float = 0.01,
        lr: float = 1e-4,
        grad_clip: float = 1000.0,
        train_steps_per_update: int = 1,
        decoder_train_only: bool = True,
        collapse_entropy_frac: float = 0.10,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        if int(image_size) < 16 or int(image_size) % 16 != 0:
            raise ValueError(
                f"image_size must be a multiple of 16 and >= 16 (got {image_size}): "
                "the encoder does 4 stride-2 downsamples and the decoder reconstructs "
                "via image_size//16. 224 (=14*16) is the working resolution."
            )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.recurrent_dim = int(recurrent_dim)
        self.stoch_dim = int(stoch_dim)
        self.stoch_classes = int(stoch_classes)
        self.z_flat = self.stoch_dim * self.stoch_classes
        self.latent_dim = self.recurrent_dim + self.z_flat  # LatentDynamics attr
        self.free_nats = float(free_nats)
        self.beta_pred = float(beta_pred)
        self.beta_dyn = float(beta_dyn)
        self.beta_rep = float(beta_rep)
        self.grad_clip = float(grad_clip)
        self.train_steps_per_update = int(train_steps_per_update)
        self.decoder_train_only = bool(decoder_train_only)
        # Below this fraction of the max categorical entropy the posterior has
        # collapsed toward one-hot (§2.3); update() logs a warning if seen.
        self.collapse_entropy_frac = float(collapse_entropy_frac)

        self.encoder = _RGBEncoder(image_size, in_ch=rgb_channels)
        self.proprio_mlp = nn.Sequential(nn.Linear(self.proprio_dim, hidden_dim), nn.SiLU())
        embed_dim = self.encoder.flat_dim + hidden_dim
        self.rssm = _RSSM(embed_dim, self.action_dim, self.recurrent_dim,
                          self.stoch_dim, self.stoch_classes, hidden_dim, unimix)

        feature_dim = self.recurrent_dim + self.z_flat
        # Spatial size after the encoder's stride-2 stack (for the decoder base).
        spatial = image_size
        for _ in range(4):
            spatial //= 2
        self.decoder = _RGBDecoder(feature_dim, self.encoder.flat_dim, spatial,
                                   out_ch=rgb_channels)
        self.reward_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, int(num_bins)))
        self.continue_head = nn.Linear(feature_dim, 1)
        self.coll_head = nn.Linear(feature_dim, 1)

        self.register_buffer(
            "bins", torch.linspace(float(bin_lo), float(bin_hi), int(num_bins)))

        self.to(device=self.device, dtype=self.torch_dtype)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(lr), betas=(0.9, 0.95))

    # -- factory (FastWAM from_config idiom) ---------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TorchRSSMDynamics":
        """Build from the ``world_model:`` YAML block (see ``configs/aerial_rl.yaml``)."""
        def g(key: str, default: Any) -> Any:
            return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

        scales = g("loss_scales", {}) or {}
        dec = g("decoder", {}) or {}
        return cls(
            image_size=int(g("image_size", 224)),
            recurrent_dim=int(g("recurrent_dim", 512)),
            stoch_dim=int(g("stoch_dim", 32)),
            stoch_classes=int(g("stoch_classes", 32)),
            num_bins=int(g("num_bins", DEFAULT_NUM_BINS)),
            bin_lo=float(g("bin_lo", DEFAULT_BIN_LO)),
            bin_hi=float(g("bin_hi", DEFAULT_BIN_HI)),
            free_nats=float(g("free_bits", DEFAULT_FREE_NATS)),
            beta_pred=float(scales.get("pred", BETA_PRED)) if isinstance(scales, dict) else BETA_PRED,
            beta_dyn=float(scales.get("dyn", BETA_DYN)) if isinstance(scales, dict) else BETA_DYN,
            beta_rep=float(scales.get("rep", BETA_REP)) if isinstance(scales, dict) else BETA_REP,
            lr=float(g("lr", 1e-4)),
            grad_clip=float(g("grad_clip", 1000.0)),
            train_steps_per_update=int(g("train_steps_per_update", 1)),
            decoder_train_only=bool(dec.get("train_only", True)) if isinstance(dec, dict) else True,
            device=str(g("device", "cuda")),
        )

    # -- embedding -----------------------------------------------------------
    def _embed(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """RGB (uint8 [N,H,W,3] or float [N,C,H,W]) + proprio → embedding."""
        if rgb.dim() == 4 and rgb.shape[-1] in (1, 3):     # NHWC uint8 -> NCHW [0,1]
            rgb = rgb.permute(0, 3, 1, 2).to(self.torch_dtype) / 255.0
        return torch.cat([self.encoder(rgb), self.proprio_mlp(proprio)], dim=-1)

    # -- training loss (FastWAM signature) -----------------------------------
    def training_loss(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One-batch DreamerV3 world-model loss over a ``[B, L, ...]`` window.

        ``sample`` keys mirror :func:`wm_data.windows_to_arrays` (torch tensors on
        ``self.device``): ``rgb`` NHWC uint8 or normalized NCHW, ``proprio`` [B,L,4],
        ``action`` [B,L,4], ``reward`` [B,L], ``done`` [B,L], ``collided`` [B,L].

        Loss = βpred·(recon + reward + continue + collision) + βdyn·fb(KL(sg(post)‖prior))
        + βrep·fb(KL(post‖sg(prior))), with free-bits and KL-balancing matching
        :func:`dreamer_recipe.kl_balance`.
        """
        rgb, proprio, action = sample["rgb"], sample["proprio"], sample["action"]
        reward, done, collided = sample["reward"], sample["done"], sample["collided"]
        B, L = proprio.shape[0], proprio.shape[1]

        h = self.rssm.initial_h(B, self.device, self.torch_dtype)
        z = torch.zeros(B, self.z_flat, device=self.device, dtype=self.torch_dtype)
        feats, post_probs_seq, prior_probs_seq = [], [], []
        for t in range(L):
            embed_t = self._embed(rgb[:, t], proprio[:, t])
            prior_p = self.rssm.prior_probs(h)
            post_p = self.rssm.post_probs(h, embed_t)
            z = self.rssm._sample(post_p)
            feats.append(torch.cat([h, z], dim=-1))
            post_probs_seq.append(post_p)
            prior_probs_seq.append(prior_p)
            h = self.rssm.advance_h(h, z, action[:, t].to(self.torch_dtype))

        feature = torch.stack(feats, dim=1)                 # [B, L, feature_dim]
        post_p = torch.stack(post_probs_seq, dim=1)         # [B, L, groups, classes]
        prior_p = torch.stack(prior_probs_seq, dim=1)

        # -- prediction heads (βpred) ---------------------------------------
        recon = self.decoder(feature.reshape(B * L, -1))
        rgb_target = rgb
        if rgb_target.dim() == 5 and rgb_target.shape[-1] in (1, 3):
            rgb_target = rgb_target.permute(0, 1, 4, 2, 3).to(self.torch_dtype) / 255.0
        recon_target = rgb_target.reshape(B * L, *recon.shape[1:])
        loss_recon = F.mse_loss(recon, recon_target)

        reward_logits = self.reward_head(feature)           # [B, L, num_bins]
        reward_target = _twohot_targets(_symlog(reward.to(self.torch_dtype)), self.bins)
        loss_reward = -(reward_target * F.log_softmax(reward_logits, dim=-1)).sum(-1).mean()

        cont_target = (~done.bool()).to(self.torch_dtype)   # continue = 1 - done
        loss_cont = F.binary_cross_entropy_with_logits(
            self.continue_head(feature).squeeze(-1), cont_target)
        loss_coll = F.binary_cross_entropy_with_logits(
            self.coll_head(feature).squeeze(-1), collided.to(self.torch_dtype))

        loss_pred = loss_recon + loss_reward + loss_cont + loss_coll

        # -- dynamics / representation KL (βdyn / βrep) ---------------------
        kl_dyn = _categorical_kl(post_p.detach(), prior_p).sum(-1)   # KL(sg(post)‖prior)
        kl_rep = _categorical_kl(post_p, prior_p.detach()).sum(-1)   # KL(post‖sg(prior))
        kl_dyn = kl_dyn.mean().clamp_min(self.free_nats)             # free bits
        kl_rep = kl_rep.mean().clamp_min(self.free_nats)
        loss_dyn = self.beta_dyn * kl_dyn
        loss_rep = self.beta_rep * kl_rep

        loss_total = self.beta_pred * loss_pred + loss_dyn + loss_rep

        # posterior-collapse telemetry (§2.3): mean categorical entropy vs. max.
        with torch.no_grad():
            ent = -(post_p * torch.log(post_p + _EPS)).sum(-1).mean()
            max_ent = float(np.log(self.stoch_classes))
        loss_dict = {
            "loss": float(loss_total.detach().item()),
            "loss_pred": float((self.beta_pred * loss_pred).detach().item()),
            "loss_dyn": float(loss_dyn.detach().item()),
            "loss_rep": float(loss_rep.detach().item()),
            "loss_recon": float(loss_recon.detach().item()),
            "loss_reward": float(loss_reward.detach().item()),
            "recon_err": float(loss_recon.detach().item()),
            "post_entropy": float(ent.item()),
            "post_entropy_frac": float(ent.item() / max_ent) if max_ent > 0 else 0.0,
        }
        return loss_total, loss_dict

    def forward(self, *args, **kwargs):  # FastWAM aliases forward -> training_loss
        return self.training_loss(*args, **kwargs)

    # -- V1 gate: real world-model update ------------------------------------
    def update(self, windows: Any) -> Dict[str, Any]:
        """Run ``train_steps_per_update`` AdamW steps on replay ``windows``.

        Returns the last step's loss breakdown with NO ``skipped`` key, so
        ``corrector._update_world_model`` reports status ``"updated"`` (a real
        training step happened). Raises nothing on well-formed windows; the
        corrector guards empty/insufficient buffers before calling.
        """
        arrays = windows_to_arrays(windows)
        sample = self._arrays_to_tensors(arrays)
        self.train()
        last: Dict[str, float] = {}
        grad_norm = 0.0
        for _ in range(self.train_steps_per_update):
            loss, last = self.training_loss(sample)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip))
            self.optimizer.step()
        if last.get("post_entropy_frac", 1.0) < self.collapse_entropy_frac:
            logger.warning(
                "posterior-collapse watch: entropy at %.1f%% of max (< %.0f%%); "
                "the discrete latent is near one-hot (§2.3)",
                100.0 * last.get("post_entropy_frac", 0.0),
                100.0 * self.collapse_entropy_frac,
            )
        return {"grad_norm": grad_norm, **last}

    def _arrays_to_tensors(self, arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in arrays.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("proprio", "action", "reward", "depth"):
                t = t.to(self.torch_dtype)
            out[k] = t.to(self.device)
        return out

    # -- imagination interface (V4; see class docstring) --------------------
    @torch.no_grad()
    def encode(self, obs: Observation) -> np.ndarray:
        """Single observation → packed latent ``[h ‖ z]`` (numpy, width latent_dim).

        No history is available for a lone observation, so ``h`` starts at zero and
        ``z`` is the posterior sample given that ``h`` — the standard RSSM initial
        state. Packing ``h`` inside the returned vector lets :meth:`step` advance
        the recurrent state without changing the numpy ``imagine()`` signature.
        """
        self.eval()
        rgb = torch.from_numpy(np.ascontiguousarray(obs.rgb)).unsqueeze(0).to(self.device)
        proprio = torch.from_numpy(
            np.ascontiguousarray(obs.proprio4())).to(self.torch_dtype).unsqueeze(0).to(self.device)
        h = self.rssm.initial_h(1, self.device, self.torch_dtype)
        embed = self._embed(rgb, proprio)
        z = self.rssm._sample(self.rssm.post_probs(h, embed))
        return torch.cat([h, z], dim=-1).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def step(self, z: np.ndarray, action: np.ndarray) -> DynamicsOutput:
        """One imagined RSSM prior transition from packed latent ``[h ‖ z]``.

        V4 note: ``progress`` here is the reward head's symexp readout (the WM's
        learned per-step reward), and goal-conditioned progress / arrival are
        finalized at V4 — for V1 the imagination path is unused (gated OFF).
        """
        self.eval()
        packed = torch.from_numpy(np.ascontiguousarray(z)).to(
            self.device, self.torch_dtype).reshape(1, self.latent_dim)
        h = packed[:, : self.recurrent_dim]
        z_flat = packed[:, self.recurrent_dim :]
        a = torch.from_numpy(np.ascontiguousarray(action)).to(
            self.device, self.torch_dtype).reshape(1, self.action_dim)

        h_next = self.rssm.advance_h(h, z_flat, a)
        z_next = self.rssm._sample(self.rssm.prior_probs(h_next))
        feature = torch.cat([h_next, z_next], dim=-1)

        reward_probs = F.softmax(self.reward_head(feature), dim=-1)
        progress = float(_symexp(_twohot_decode(reward_probs, self.bins)).item())
        p_coll = float(torch.sigmoid(self.coll_head(feature)).item())
        cont = float(torch.sigmoid(self.continue_head(feature)).item())
        done = bool(cont < 0.5 or p_coll >= 1.0)
        packed_next = torch.cat([h_next, z_next], dim=-1).squeeze(0).float().cpu().numpy()
        return DynamicsOutput(
            z_next=packed_next, p_coll=p_coll, progress=progress, done=done, arrived=False,
        )

    # -- checkpoint I/O (FastWAM idiom: path + optional optimizer/step) ------
    def save_checkpoint(self, path: str, optimizer: Optional[Any] = None,
                        step: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {
            "model": self.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer: Optional[Any] = None) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        if "model" in payload:
            self.load_state_dict(payload["model"], strict=False)
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is not None and "optimizer" in payload:
            opt.load_state_dict(payload["optimizer"])
        return payload
