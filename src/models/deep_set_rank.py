"""DeepSet ranker: forward returns (score, Z, attn_weights)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .player_embedding import PlayerEmbedding
from .player_encoder import PlayerEncoder
from .set_attention import SetAttention


class DeepSetRank(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        stat_dim: int,
        encoder_hidden: list[int],
        attention_heads: int,
        dropout: float = 0.2,
        *,
        minutes_bias_weight: float = 0.3,
        minutes_sum_min: float = 1e-6,
        fallback_strategy: str = "minutes",
        attention_temperature: float = 1.0,
        attention_input_dropout: float = 0.0,
        attention_use_pre_norm: bool = True,
        attention_use_residual: bool = True,
    ):
        super().__init__()
        self.emb = PlayerEmbedding(num_embeddings, embedding_dim)
        self.enc = PlayerEncoder(stat_dim + embedding_dim, encoder_hidden, dropout)
        self.attn = SetAttention(
            self.enc.output_dim,
            attention_heads,
            dropout,
            minutes_bias_weight=minutes_bias_weight,
            minutes_sum_min=minutes_sum_min,
            fallback_strategy=fallback_strategy,
            attention_temperature=attention_temperature,
        )
        self.scorer = nn.Linear(self.enc.output_dim, 1)

    def forward(
        self,
        embedding_indices: torch.Tensor,
        player_stats: torch.Tensor,
        minutes: torch.Tensor,
        key_padding_mask: torch.Tensor,
        temperature_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        embedding_indices: (B, P) int. player_stats: (B, P, S). minutes: (B, P). key_padding_mask: (B, P) bool, True=ignore.
        temperature_override: optional; when set (e.g. multi-temp inference), passed to attention softmax.
        Returns (score, Z, attn_weights): score (B,), Z (B, D), attn_weights (B, P).
        """
        B, P, _ = player_stats.shape
        e = self.emb(embedding_indices)  # (B, P, E)
        x = torch.cat([e, player_stats], dim=-1)  # (B, P, E+S)
        z = self.enc(x)  # (B, P, D)
        pooled, attn_w = self.attn(
            z,
            key_padding_mask=key_padding_mask,
            minutes=minutes,
            temperature_override=temperature_override,
        )
        Z = pooled  # (B, D)
        score = self.scorer(Z).squeeze(-1)  # (B,)
        return score, Z, attn_w


class DeepSetRankEnsemble(nn.Module):
    """Seed-averaging ensemble: forward returns the mean score/Z/attention across member models.

    Drop-in replacement for a single DeepSetRank in inference paths (same forward signature),
    so multi-temp aggregation, confidence, and IG attribution work unchanged.
    """

    def __init__(self, models: list[DeepSetRank]):
        super().__init__()
        if not models:
            raise ValueError("DeepSetRankEnsemble requires at least one model")
        self.models = nn.ModuleList(models)

    def forward(
        self,
        embedding_indices: torch.Tensor,
        player_stats: torch.Tensor,
        minutes: torch.Tensor,
        key_padding_mask: torch.Tensor,
        temperature_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, zs, attns = [], [], []
        for m in self.models:
            s, z, a = m(embedding_indices, player_stats, minutes, key_padding_mask, temperature_override=temperature_override)
            scores.append(s)
            zs.append(z)
            attns.append(a)
        return (
            torch.stack(scores, dim=0).mean(dim=0),
            torch.stack(zs, dim=0).mean(dim=0),
            torch.stack(attns, dim=0).mean(dim=0),
        )
