"""Tests for ListMLE loss, including position-aware (position_discount) option."""
from __future__ import annotations

import torch
import pytest

from src.models.listmle_loss import listmle_loss


def test_listmle_backward_compat():
    """position_discount=None equals default (no second arg)."""
    scores = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0], [0.0, 1.0, 0.5, -0.5, 0.0]])
    rel = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0], [3.0, 5.0, 4.0, 2.0, 1.0]])
    out_default = listmle_loss(scores, rel)
    out_none = listmle_loss(scores, rel, position_discount=None)
    assert torch.isclose(out_default, out_none)
    assert out_default.ndim == 0 and out_default.isfinite()


def test_listmle_position_discount_none_same_as_default():
    """position_discount='none' matches unweighted sum."""
    scores = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
    rel = torch.tensor([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    out_default = listmle_loss(scores, rel)
    out_none = listmle_loss(scores, rel, position_discount="none")
    assert torch.isclose(out_default, out_none)


def test_listmle_log2_different_from_unweighted():
    """With imperfect order, log2-weighted loss differs from unweighted."""
    # Scores that give a different order than rel (so NLL terms vary by position)
    scores = torch.tensor([[0.0, 1.0, 0.5], [0.5, 0.0, 1.0]])  # not aligned with rel
    rel = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    out_unweighted = listmle_loss(scores, rel, position_discount="none")
    out_log2 = listmle_loss(scores, rel, position_discount="log2")
    assert out_log2.ndim == 0 and out_log2.isfinite()
    assert not torch.isclose(out_unweighted, out_log2)


def test_listmle_linear_scalar_finite():
    """position_discount='linear' returns finite scalar."""
    scores = torch.tensor([[1.0, 0.0, -0.5], [0.0, 0.5, 0.0]])
    rel = torch.tensor([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    out = listmle_loss(scores, rel, position_discount="linear")
    assert out.ndim == 0 and out.isfinite()
