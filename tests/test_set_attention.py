"""Tests for src.models.set_attention."""
from __future__ import annotations

import torch
import pytest

from src.models.set_attention import SetAttention


def test_set_attention_forward_shapes():
    B, P, D = 4, 10, 32
    model = SetAttention(embed_dim=D, num_heads=4, dropout=0.0)
    x = torch.randn(B, P, D)
    out, w = model(x)
    assert out.shape == (B, D)
    assert w.shape == (B, P)


def test_set_attention_with_mask():
    B, P, D = 2, 8, 16
    model = SetAttention(embed_dim=D, num_heads=2, dropout=0.0)
    x = torch.randn(B, P, D)
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, 6:] = True  # last 2 padded per row
    out, w = model(x, key_padding_mask=mask)
    assert out.shape == (B, D)
    assert w.shape == (B, P)


def test_set_attention_with_minutes():
    B, P, D = 2, 5, 16
    model = SetAttention(embed_dim=D, num_heads=2, dropout=0.0, minutes_bias_weight=0.3)
    x = torch.randn(B, P, D)
    minutes = torch.rand(B, P).clamp(min=0.1)
    out, w = model(x, minutes=minutes)
    assert out.shape == (B, D)
    assert w.shape == (B, P)
    assert torch.allclose(w.sum(dim=1), torch.ones(B, device=w.device), atol=1e-5)


def test_set_attention_minutes_bias_zero_disabled():
    B, P, D = 2, 5, 16
    model = SetAttention(embed_dim=D, num_heads=2, dropout=0.0, minutes_bias_weight=0.0)
    x = torch.randn(B, P, D)
    minutes = torch.rand(B, P)
    out, w = model(x, minutes=minutes)
    assert w.shape == (B, P)
    assert torch.allclose(w.sum(dim=1), torch.ones(B, device=w.device), atol=1e-5)


def test_set_attention_fallback_strategies():
    """Both fallback strategies produce valid normalized weights."""
    B, P, D = 2, 8, 16
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, 6:] = True
    minutes = torch.rand(B, P).clamp(min=0.01)
    x = torch.randn(B, P, D)
    for strategy in ("minutes", "uniform"):
        model = SetAttention(
            embed_dim=D, num_heads=2, dropout=0.0,
            fallback_strategy=strategy,
        )
        out, w = model(x, key_padding_mask=mask, minutes=minutes)
        assert out.shape == (B, D)
        assert w.shape == (B, P)
        assert torch.allclose(w.sum(dim=1), torch.ones(B, device=w.device), atol=1e-5)
        # Masked positions should be 0
        assert (w[mask] == 0.0).all()
