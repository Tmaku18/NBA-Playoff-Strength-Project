"""Platt scaling calibration: (1) on meta-learner output, (2) on raw model outputs then combine."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def platt_scale_meta(
    scores: np.ndarray,
    y_binary: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> LogisticRegression:
    """
    Fit Platt scaling on meta-learner (stacker) output. scores: (N,) raw ensemble scores; y_binary: (N,) 0/1.
    Returns fitted LogisticRegression; use .predict_proba(scores.reshape(-1, 1))[:, 1] for calibrated probs.
    """
    X = np.asarray(scores).reshape(-1, 1).astype(np.float64)
    y = np.asarray(y_binary).ravel().astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    platt = LogisticRegression(max_iter=500, fit_intercept=fit_intercept)
    platt.fit(X, y)
    return platt


def platt_scale_per_model(
    X_oof: np.ndarray,
    y_binary: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> list[LogisticRegression]:
    """
    Fit Platt scaling per model (columns of X_oof). X_oof: (N, 3) raw OOF from model_a, xgb, rf; y_binary: (N,) 0/1.
    Returns list of 3 LogisticRegression; combine e.g. mean(platt[i].predict_proba(X[:, i:i+1])[:, 1]).
    """
    X = np.asarray(X_oof, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y_binary).ravel().astype(np.int32)
    fitters = []
    for j in range(X.shape[1]):
        platt = LogisticRegression(max_iter=500, fit_intercept=fit_intercept)
        platt.fit(X[:, j : j + 1], y)
        fitters.append(platt)
    return fitters


def calibrated_probs_meta(
    platt: LogisticRegression,
    scores: np.ndarray,
) -> np.ndarray:
    """Apply fitted Platt (meta) to new scores. Returns (N,) calibrated probabilities."""
    X = np.asarray(scores).reshape(-1, 1).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return platt.predict_proba(X)[:, 1]


def calibrated_probs_per_model(
    platt_list: list[LogisticRegression],
    X: np.ndarray,
    *,
    combine: str = "mean",
) -> np.ndarray:
    """Apply per-model Platt to (N, 3) and combine. combine: 'mean' or 'max'. Returns (N,) calibrated probs."""
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.column_stack([
        platt_list[j].predict_proba(X[:, j : j + 1])[:, 1]
        for j in range(min(len(platt_list), X.shape[1]))
    ])
    if combine == "mean":
        return probs.mean(axis=1)
    if combine == "max":
        return probs.max(axis=1)
    return probs.mean(axis=1)
