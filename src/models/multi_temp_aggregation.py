"""Multi-temperature aggregation for Model A scores.

Used when `model_a.attention.multi_temp_enabled` is true: run Model A at multiple
attention temperatures and combine per-team scores into a single score plus a
per-team confidence signal derived from agreement across temperatures.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Return ranks 1..K where higher score => better (rank 1)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    k = s.size
    # argsort descending; stable sort for reproducibility
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(k, dtype=np.float64)
    ranks[order] = np.arange(1, k + 1, dtype=np.float64)
    return ranks


def aggregate_multi_temp_scores(
    scores_by_temp: Mapping[int, np.ndarray],
    base_weights: Mapping[int, float] | None = None,
    starter_availability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate Model A scores across multiple attention temperatures.

    Args:
        scores_by_temp: temp -> per-team scores array (shape (K,)). Higher score = better.
        base_weights: temp -> base weight (e.g. {1:0.85, 5:1.0, 10:0.7}). Missing temps default to 1.0.
        starter_availability: optional per-team availability scalar in [0,1] (shape (K,)).
            If provided, modulates weights to emphasize temp=5 when starters are available
            and emphasize high temps (depth) when starters are unavailable.

    Returns:
        (final_scores, conf_a) where:
        - final_scores: aggregated per-team score (shape (K,))
        - conf_a: per-team confidence in [0,1] from agreement across temps (shape (K,))
    """
    if not scores_by_temp:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    temps = sorted(int(t) for t in scores_by_temp.keys())
    scores_list = [np.asarray(scores_by_temp[int(t)], dtype=np.float64).ravel() for t in temps]
    k = int(scores_list[0].size)
    if any(int(s.size) != k for s in scores_list):
        raise ValueError("aggregate_multi_temp_scores: all score arrays must have same length.")

    # Agreement across temperatures: per-team std of ranks -> agreement in (0,1]
    ranks_by_temp = np.stack([_ranks_desc(s) for s in scores_list], axis=0)  # (T, K)
    std_ranks = np.std(ranks_by_temp, axis=0, ddof=0)
    conf_a = 1.0 / (1.0 + std_ranks)
    conf_a = np.clip(conf_a, 0.0, 1.0)

    # Build per-temp weights with agreement modulation
    bw = dict(base_weights or {})
    w_agree = 0.5 + 0.5 * conf_a  # (K,)

    avail = None
    if starter_availability is not None:
        avail = np.asarray(starter_availability, dtype=np.float64).ravel()
        if avail.size != k:
            avail = None
        else:
            avail = np.clip(avail, 0.0, 1.0)

    weights = []
    for t in temps:
        w0 = float(bw.get(int(t), 1.0))
        w = w0 * w_agree
        # Availability modulation (simple, monotone):
        # - When starters are available (avail high), boost temp 5.
        # - When starters are not available (avail low), boost depth temps (8/10).
        if avail is not None:
            if int(t) == 5:
                w = w * (0.5 + 0.5 * avail)
            elif int(t) in (8, 10):
                w = w * (0.5 + 0.5 * (1.0 - avail))
        weights.append(w)

    W = np.stack(weights, axis=0)  # (T, K)
    S = np.stack(scores_list, axis=0)  # (T, K)
    denom = np.sum(W, axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    final_scores = np.sum(W * S, axis=0) / denom
    return final_scores.astype(np.float64), conf_a.astype(np.float64)

