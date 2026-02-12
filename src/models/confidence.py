"""Per-instance confidence for ensemble weighting.

Model A: from attention entropy (high = diffuse = high confidence) and max weight
(high max = star-dependent = high risk = low confidence).
XGB: from tree-level prediction variance (high variance = low confidence).
"""

from __future__ import annotations

import numpy as np


def confidence_from_attention(
    attn_weights_1d: np.ndarray,
    *,
    entropy_weight: float = 0.5,
    max_weight_weight: float = 0.5,
) -> float:
    """Compute Model A confidence from 1D attention weights over players.

    High entropy = diffuse attention = high confidence (no single player dominates).
    Low entropy = concentrated = star-dependent = high risk = low confidence.
    High max weight = more decisive focus on one player = lower confidence.

    Formula: c_A = entropy_weight * (H / H_max) + max_weight_weight * (1 - max_w),
    clipped to [0, 1]. If n_players == 0, return 0.5 (neutral).

    Args:
        attn_weights_1d: Non-negative attention weights (can be unnormalized).
        entropy_weight: Weight for normalized entropy term.
        max_weight_weight: Weight for (1 - max_weight) term.

    Returns:
        Confidence in [0, 1].
    """
    w = np.asarray(attn_weights_1d).ravel()
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    n = w.size
    if n == 0:
        return 0.5
    total = float(np.sum(w))
    if total <= 0:
        return 0.5
    p = w / total
    # Entropy: -sum(p * log(p)), 0*log(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.where(p > 0, p, 1.0))
    entropy = -float(np.sum(np.where(p > 0, p * log_p, 0.0)))
    H_max = np.log(max(n, 1))
    norm_entropy = (entropy / H_max) if H_max > 0 else 0.0
    max_w = float(np.max(p))
    c = entropy_weight * norm_entropy + max_weight_weight * (1.0 - max_w)
    return float(np.clip(c, 0.0, 1.0))
