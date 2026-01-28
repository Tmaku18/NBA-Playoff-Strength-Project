"""Numerically stable ListMLE using torch.logsumexp / logcumsumexp."""

from __future__ import annotations

import torch


def listmle_loss(scores: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """
    scores: (B, L) model scores. rel: (B, L) relevance/rank (higher = better). List is ordered by rel descending.
    Loss = -log P(permutation | scores). Stable: use logsumexp. P = prod_i exp(s_i) / sum_{j in remain} exp(s_j).
    -log P = sum_i [ log(sum_{j>=i} exp(s_j)) - s_i ].
    For stability: subtract max_s; then logsumexp = log(sum exp(s-m)) + m.
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(B).mean()

    # Order indices by rel descending (best first)
    _, order = rel.sort(dim=1, descending=True)
    # gather scores into that order: s_ordered[b,l] = scores[b, order[b,l]]
    s = torch.gather(scores, 1, order)  # (B, L)

    # For each position i: log(sum_{j>=i} exp(s_j)) - s_i.
    # cumsum from right: for i, tail = s[i:]. logsumexp(tail) - s[i].
    # logcumsumexp from right: L-1, L-2, ... . We need for i=0..L-1: lse(s[i:]) - s[i].
    # lse(s[i:]) = log(sum(exp(s[i:]))) = we can do: reverse, cumsum-like, reverse.
    # PyTorch doesn't have logcumsumexp. Do: f(i) = log(sum(exp(s[i:L]))). 
    # f(i) = log(exp(s[i]) + exp(s[i+1]) + ...) = logaddexp(f(i+1), s[i]) if we go from right? No.
    # f(L-1) = s[L-1]. f(L-2) = log(exp(s[L-2])+exp(s[L-1])) = logaddexp(s[L-2], s[L-1]). f(i) = logaddexp(s[i], f(i+1)).
    # So we need a right-to-left scan: init last = s[:,-1]. for i from L-2 to 0: last = logaddexp(s[:,i], last). Then log_denom[i] = last at step i. Actually we want log_denom[i] = lse(s[:,i:]). 
    # Simpler: for each i, log_denom_i = torch.logsumexp(s[:, i:], dim=1). That's (B,) for each i. We need (B,L). 
    # mask = (arange(L) >= arange(L).unsqueeze(1))  # (L,L) [i,j]=i<=j. Then for each b: logsumexp(s[b] + mask) doesn't work directly.
    # Direct: log_denom = torch.stack([torch.logsumexp(s[:, i:], dim=1) for i in range(L)], dim=1)  # (B,L)
    max_s = s.max(dim=1, keepdim=True)[0]
    s_stable = s - max_s
    log_denom = torch.stack(
        [torch.logsumexp(s_stable[:, i:], dim=1) for i in range(L)],
        dim=1,
    )
    # log P = sum_i (s_i - log_denom_i) = sum_i (s_i - (logsumexp(s[i:]) = log(sum exp(s[i:])))) 
    # -log P = sum_i (log_denom_i - s_i). 
    nll = (log_denom - s).sum(dim=1)
    return nll.mean()
