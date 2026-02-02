"""Set attention: MultiheadAttention with batch_first=True and key_padding_mask. Minutes-weighting."""

from __future__ import annotations

import torch
import torch.nn as nn


class SetAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        *,
        minutes_bias_weight: float = 0.3,
        minutes_sum_min: float = 1e-6,
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
        # Per-row sum of raw attention for sanity: only apply minutes blend when raw attention is meaningful
        w_sum = w.sum(dim=-1, keepdim=True)
        w_finite = torch.isfinite(w).all(dim=-1, keepdim=True)
        raw_attention_ok = (w_sum > 1e-8) & w_finite

        if minutes is not None and w.shape[-1] == minutes.shape[-1]:
            mins = minutes
            if key_padding_mask is not None and key_padding_mask.shape == minutes.shape:
                mins = mins.masked_fill(key_padding_mask, 0.0)
            mins = mins.clamp(min=0.0)
            mins_sum = mins.sum(dim=-1, keepdim=True)
            # Only apply minutes reweighting when minutes are meaningful (above threshold)
            mins_meaningful = mins_sum > max(self.minutes_sum_min, 1e-6)
            bias_weight = max(0.0, min(1.0, float(self.minutes_bias_weight)))
            # Blend only when both raw attention is ok and minutes are meaningful
            if bias_weight > 0:
                valid = (mins_meaningful & raw_attention_ok.squeeze(-1)).unsqueeze(-1)
                if valid.any():
                    mins_norm = mins / mins_sum.clamp(min=1e-8)
                    bias = mins_norm.unsqueeze(1)
                    w = torch.where(
                        valid,
                        (1.0 - bias_weight) * w + bias_weight * bias,
                        w,
                    )
        # Normalize only rows with positive sum; for zero/non-finite attention use uniform over non-masked
        w_sum = w.sum(dim=-1, keepdim=True)
        need_fallback = ((w_sum <= 0) | ~torch.isfinite(w_sum)).expand_as(w)
        if need_fallback.any():
            # Uniform over valid (unmasked) positions so gradient can flow
            if key_padding_mask is not None and key_padding_mask.shape[:2] == (x.shape[0], x.shape[1]):
                valid_mask = (~key_padding_mask).float().unsqueeze(1)
                denom = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                uniform = valid_mask / denom
            else:
                uniform = torch.ones_like(w, device=w.device, dtype=w.dtype) / w.shape[-1]
            w = torch.where(need_fallback, uniform, w / (w_sum.clamp(min=1e-8)))
        else:
            w = w / (w_sum.clamp(min=1e-8))
        return out.squeeze(1), w.squeeze(1)
