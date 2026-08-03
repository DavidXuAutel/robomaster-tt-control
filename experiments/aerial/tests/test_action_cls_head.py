from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fastwam.models.wan22.action_dit import ActionDiT


def _tiny_dit(**kwargs):
    cfg = dict(
        hidden_dim=32,
        action_dim=4,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=1,
        enable_action_cls=True,
    )
    cfg.update(kwargs)
    return ActionDiT(**cfg)


def test_classify_from_tokens_shape_and_requires_t():
    dit = _tiny_dit()
    B, T = 2, 5
    noisy = torch.randn(B, T, 4)
    timestep = torch.rand(B)
    context = torch.randn(B, 3, 16)
    pre = dit.pre_dit(noisy, timestep, context)
    tokens = torch.randn(B, T, dit.hidden_dim)
    logits = dit.classify_from_tokens(tokens, pre)
    assert logits.shape == (B, 10)
    bad = dict(pre)
    bad.pop("t")
    with pytest.raises(ValueError):
        dit.classify_from_tokens(tokens, bad)


def test_head_cls_disabled_by_default():
    dit = _tiny_dit(enable_action_cls=False)
    assert dit.head_cls is None
