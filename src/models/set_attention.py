"""Set attention: MultiheadAttention with batch_first=True and key_padding_mask. Minutes-weighting.

Fallback when attention collapses (all or nearly zero on valid positions):
- "minutes": use normalized minutes on valid positions as weights.
  Pros: Domain-aligned (playing time reflects importance); smooth gradients from minutes.
  Cons: Can reinforce existing imbalance if the model already under-weights low-minute players.
- "uniform": use 1/n_valid on valid positions, 0 on masked.
  Pros: Maximally neutral; equal gradient to all valid players; simple and robust.
  Cons: Ignores playing time; may slow convergence when ranking by impact.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

# Sum of attention on valid positions below this => treat as collapsed and use fallback
COLLAPSE_THRESHOLD = 1e-6


class SetAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        *,
        minutes_bias_weight: float = 0.3,
        minutes_sum_min: float = 1e-6,
        fallback_strategy: Literal["minutes", "uniform"] = "minutes",
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.minutes_bias_weight = float(minutes_bias_weight)
        self.minutes_sum_min = float(minutes_sum_min)
        self.fallback_strategy = fallback_strategy

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        minutes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, P, D). key_padding_mask: (B, P) bool, True = ignore.
        minutes: (B, P) optional weights. Returns (out, attn_weights).
        """
        # query from a single learned vector or mean; use masked mean of x as query for set-pooling
        if key_padding_mask is not None and key_padding_mask.shape[:2] == x.shape[:2]:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            q = (x * valid).sum(dim=1, keepdim=True) / denom
        else:
            q = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        out, w = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        # out (B, 1, D), w (B, 1, P)
        if minutes is not None and w.shape[-1] == minutes.shape[-1]:
            mins = minutes
            if key_padding_mask is not None and key_padding_mask.shape == minutes.shape:
                mins = mins.masked_fill(key_padding_mask, 0.0)
            mins = mins.clamp(min=0.0)
            mins_sum = mins.sum(dim=-1, keepdim=True)
            bias_weight = max(0.0, min(1.0, float(self.minutes_bias_weight)))
            if bias_weight > 0:
                valid_m = mins_sum > max(self.minutes_sum_min, 0.0)
                if valid_m.any():
                    mins_norm = mins / mins_sum.clamp(min=1e-8)
                    bias = mins_norm.unsqueeze(1)
                    w = torch.where(
                        valid_m.unsqueeze(-1),
                        (1.0 - bias_weight) * w + bias_weight * bias,
                        w,
                    )
                    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        # Fallback when attention on valid positions is (nearly) zero or NaN/inf so gradients can flow
        if key_padding_mask is not None and key_padding_mask.shape[:2] == w.shape[:2]:
            valid_mask = ~key_padding_mask  # (B, P)
            w_masked = w.masked_fill(key_padding_mask.unsqueeze(1), 0.0)  # (B, 1, P)
            sum_valid = w_masked.sum(dim=-1).squeeze(1)  # (B,)
            has_valid = valid_mask.any(dim=-1)  # (B,) at least one valid position
            collapsed = ((sum_valid < COLLAPSE_THRESHOLD) | ~torch.isfinite(sum_valid)) & has_valid
            if collapsed.any():
                fallback = self._fallback_weights(
                    key_padding_mask, minutes, w.shape, x.device
                )
                w = torch.where(collapsed.unsqueeze(1).unsqueeze(2), fallback, w)
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        return out.squeeze(1), w.squeeze(1)

    def _fallback_weights(
        self,
        key_padding_mask: torch.Tensor,
        minutes: torch.Tensor | None,
        w_shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Return (B, 1, P) fallback weights; valid positions sum to 1, masked are 0."""
        B, _, P = w_shape
        valid = ~key_padding_mask  # (B, P)
        n_valid = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)

        if self.fallback_strategy == "uniform":
            fallback = valid.float().unsqueeze(1) / n_valid.unsqueeze(-1)
        else:
            # minutes-based
            if minutes is not None and minutes.shape == key_padding_mask.shape:
                mins = minutes.masked_fill(key_padding_mask, 0.0).clamp(min=0.0)
                mins_sum = mins.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                fallback = (mins / mins_sum).unsqueeze(1)
            else:
                fallback = valid.float().unsqueeze(1) / n_valid.unsqueeze(-1)
        return fallback.to(device)
