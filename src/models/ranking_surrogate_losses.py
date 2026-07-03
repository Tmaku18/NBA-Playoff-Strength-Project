"""Differentiable surrogate losses for Spearman and rank RMSE.

Use training.loss_type: "spearman_surrogate", "weighted_spearman_surrogate", or
"rank_rmse_surrogate" to train Model A toward correlation or rank error instead
of ListMLE. weighted_spearman_surrogate up-weights the top of the list
(weights ∝ 1/rank^training.loss_top_weight_power) for championship questions.
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
    # NOTE: unsqueeze(1) puts s_j along dim 2, unsqueeze(2) puts s_i along dim 1,
    # so s_j - s_i = scores.unsqueeze(1) - scores.unsqueeze(2). The previous version
    # had this backwards, which inverted Model A's score orientation (best team got
    # the lowest score and the stacker had to flip it).
    diff = scores.unsqueeze(1) - scores.unsqueeze(2)
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


def weighted_spearman_surrogate_loss(
    scores: torch.Tensor,
    rel: torch.Tensor,
    tau: float = 1.0,
    top_weight_power: float = 1.0,
) -> torch.Tensor:
    """Top-weighted Spearman surrogate: weights ∝ 1/actual_rank^power (rank 1 = best).

    Emphasizes agreement at the top of the list (championship contenders) instead of
    treating all 30 rank positions equally. top_weight_power=0 reduces to the unweighted
    version; 1.0 gives weights 1, 1/2, 1/3, ...; >1 concentrates further on the top.
    Loss = 1 - weighted Pearson(soft_rank_pred, actual_rank) per row, mean over batch.
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(1).mean()
    soft_rank = _soft_ranks(scores, tau)
    actual = _actual_ranks_from_rel(rel)
    p = max(float(top_weight_power), 0.0)
    w = actual.pow(-p) if p > 0 else torch.ones_like(actual)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    sp_mean = (w * soft_rank).sum(dim=1, keepdim=True)
    ap_mean = (w * actual).sum(dim=1, keepdim=True)
    sp_centered = soft_rank - sp_mean
    ap_centered = actual - ap_mean
    cov = (w * sp_centered * ap_centered).sum(dim=1)
    std_p = ((w * sp_centered.pow(2)).sum(dim=1) + 1e-8).sqrt()
    std_a = ((w * ap_centered.pow(2)).sum(dim=1) + 1e-8).sqrt()
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
