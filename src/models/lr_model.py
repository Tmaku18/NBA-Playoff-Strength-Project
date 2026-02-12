"""Logistic Regression model for Model C (diagnostics only; not in ensemble). No net_rating in features.

Uses binarized target (top-half = 1) so predict_proba(X)[:, 1] gives a strength score for ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def build_lr(config: dict | None = None) -> Any:
    cfg = config or {}
    return LogisticRegression(
        C=float(cfg.get("C", 1.0)),
        max_iter=int(cfg.get("max_iter", 1000)),
        solver=cfg.get("solver", "lbfgs"),
        random_state=cfg.get("random_state", 42),
    )


def fit_lr(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train on binarized target: y_bin = 1 if y <= median(y), else 0. Model learns P(top-half)."""
    y = np.asarray(y_train).ravel()
    median = np.median(y)
    y_bin = (y <= median).astype(int)
    model.fit(X_train, y_bin)
    return model
