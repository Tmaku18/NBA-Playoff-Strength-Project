"""Numerically stable ListMLE using torch.logsumexp / logcumsumexp."""

from __future__ import annotations

from typing import Literal

import torch

PositionDiscount = Literal[None, "none", "log2", "linear"]


def listmle_loss(
    scores: torch.Tensor,
    rel: torch.Tensor,
    position_discount: str | None = None,
) -> torch.Tensor:
    """
    scores: (B, L) model scores. rel: (B, L) relevance/rank (higher = better). List is ordered by rel descending.
    Loss = -log P(permutation | scores). Stable: use logsumexp. P = prod_i exp(s_i) / sum_{j in remain} exp(s_j).
    -log P = sum_i [ log(sum_{j>=i} exp(s_j)) - s_i ].
    For stability: subtract max_s; then logsumexp = log(sum exp(s-m)) + m. Guards against NaN/inf.

    If position_discount is "log2" or "linear", per-position NLL terms are weighted so top positions
    contribute more (NDCG-style when "log2"). None or "none" keeps unweighted sum (default behavior).
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(B).mean()

    # Guard against NaN/inf so logsumexp and gradients stay finite
    scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
    scores = scores.clamp(-50.0, 50.0)

    # Order indices by rel descending (best first)
    _, order = rel.sort(dim=1, descending=True)
    s = torch.gather(scores, 1, order)  # (B, L)

    # Per-row max for numerical stability (logsumexp)
    max_s = s.max(dim=1, keepdim=True)[0]
    s_stable = s - max_s
    log_denom = torch.stack(
        [torch.logsumexp(s_stable[:, i:], dim=1) for i in range(L)],
        dim=1,
    )
    terms = log_denom - s  # (B, L) per-position NLL terms

    if position_discount in (None, "none"):
        nll = terms.sum(dim=1)
    else:
        # Build position weights on same device/dtype as s; position 0 = top.
        device, dtype = s.device, s.dtype
        i = torch.arange(L, device=device, dtype=dtype)
        if position_discount == "log2":
            weight = 1.0 / (torch.log2(i + 2.0).clamp(min=1e-8))
        elif position_discount == "linear":
            weight = 1.0 / (i + 1.0).clamp(min=1e-8)
        else:
            weight = torch.ones(L, device=device, dtype=dtype)
        # Normalize so weights sum to L to keep loss scale comparable
        weight = weight * (L / weight.sum())
        nll = (weight.unsqueeze(0) * terms).sum(dim=1)
    out = nll.mean()
    return torch.nan_to_num(out, nan=0.0, posinf=1e2, neginf=1e2)
