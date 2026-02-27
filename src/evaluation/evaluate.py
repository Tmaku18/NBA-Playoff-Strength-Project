"""Evaluate ranking, upset ROC-AUC, walk-forward runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from .baselines import rank_by_srs, rank_by_net_rating
from .metrics import ndcg_score, spearman, kendall_tau, mrr, pearson, precision_at_k, roc_auc_upset


def evaluate_ranking(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int | None = 10,
) -> dict[str, float]:
    return {
        "ndcg": ndcg_score(y_true, y_score, k=k),
        "spearman": spearman(y_true, y_score),
        "kendall_tau": kendall_tau(y_true, y_score),
        "pearson": pearson(y_true, y_score),
        "precision_at_4": precision_at_k(y_true, y_score, 4),
        "precision_at_8": precision_at_k(y_true, y_score, 8),
        "mrr_top2": mrr(y_true, y_score, top_n_teams=2),
        "mrr_top4": mrr(y_true, y_score, top_n_teams=4),
    }


def evaluate_upset(y_binary: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {"roc_auc_upset": roc_auc_upset(y_binary, y_score)}


def run_walk_forward(
    seasons: list[str],
    *,
    train_val_test: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Stub: train/val/test by season blocks. Returns metrics per season + aggregate."""
    return {"seasons": seasons, "metrics": {}, "aggregate": {}}
