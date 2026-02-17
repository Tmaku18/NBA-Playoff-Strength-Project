"""Differentiable surrogate losses for Spearman and rank RMSE.

Use training.loss_type: "spearman_surrogate" or "rank_rmse_surrogate" to train
Model A toward correlation or rank error instead of ListMLE.
"""

from __future__ import annotations

import torch


def _soft_ranks(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable soft ranks (1-based, 1 = best = highest score).

    scores: (B, L). Returns (B, L) with soft_rank in [1, L].
    Uses sigmoid((s_j - s_i)/tau) to approximate "how many j beat i".
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_ones(B, L)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
    scores = scores.clamp(-50.0, 50.0)
    # (B, L, L): [b, i, j] = scores[b,j] - scores[b,i]
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    # sum over j: count (soft) how many j have score >= s_i; subtract self (sigmoid(0)=0.5)
    sum_sigmoid = torch.sigmoid(diff / max(tau, 1e-6)).sum(dim=2)
    soft_rank = 1.0 + (sum_sigmoid - 0.5)
    return soft_rank


def _actual_ranks_from_rel(rel: torch.Tensor) -> torch.Tensor:
    """Convert relevance (higher = better) to 1-based ranks (1 = best).

    rel: (B, L). Returns (B, L) float, no grad.
    """
    B, L = rel.shape
    order = rel.argsort(dim=1, descending=True)
    ranks = torch.zeros(B, L, device=rel.device, dtype=torch.float32)
    ranks.scatter_(
        1,
        order,
        torch.arange(1, L + 1, device=rel.device, dtype=torch.float32).unsqueeze(0).expand(B, L),
    )
    return ranks


def spearman_surrogate_loss(scores: torch.Tensor, rel: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable surrogate for Spearman rank correlation (maximize correlation -> minimize loss).

    Loss = 1 - Pearson(soft_rank_pred, actual_rank) per row, then mean over batch.
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(1).mean()
    soft_rank = _soft_ranks(scores, tau)
    actual = _actual_ranks_from_rel(rel)
    # Center and normalize per row
    sp_mean = soft_rank.mean(dim=1, keepdim=True)
    ap_mean = actual.mean(dim=1, keepdim=True)
    sp_centered = soft_rank - sp_mean
    ap_centered = actual - ap_mean
    cov = (sp_centered * ap_centered).mean(dim=1)
    std_p = (sp_centered.pow(2).mean(dim=1) + 1e-8).sqrt()
    std_a = (ap_centered.pow(2).mean(dim=1) + 1e-8).sqrt()
    r = cov / (std_p * std_a)
    loss = (1.0 - r).mean()
    return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)


def rank_rmse_surrogate_loss(scores: torch.Tensor, rel: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable surrogate for rank RMSE (minimize MSE between soft ranks and actual ranks)."""
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(1).mean()
    soft_rank = _soft_ranks(scores, tau)
    actual = _actual_ranks_from_rel(rel)
    mse = (soft_rank - actual).pow(2).mean(dim=1).mean()
    return torch.nan_to_num(mse, nan=0.0, posinf=1e4, neginf=1e4)
