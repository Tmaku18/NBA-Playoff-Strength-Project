"""BayesianRidge model for team-stats rank prediction (predictive mean + std)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


@dataclass
class BayesianRidgeWithScaler:
    scaler: StandardScaler
    model: Any

    def predict(self, X: np.ndarray, *, return_std: bool = False) -> Any:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        if return_std:
            mean, std = self.model.predict(Xs, return_std=True)
            mean = np.asarray(mean).ravel().astype(np.float32)
            std = np.asarray(std).ravel().astype(np.float32)
            std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return mean, np.maximum(std, 1e-6)
        mean = np.asarray(self.model.predict(Xs)).ravel().astype(np.float32)
        return mean


def fit_bayesian_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    random_state: int = 42,
) -> BayesianRidgeWithScaler:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = BayesianRidge(
        alpha_1=float(alpha_1),
        alpha_2=float(alpha_2),
        lambda_1=float(lambda_1),
        lambda_2=float(lambda_2),
        compute_score=False,
    )
    # BayesianRidge doesn't accept random_state in older sklearn versions; keep deterministic via data split + seed elsewhere
    _ = random_state
    model.fit(Xs, y_train)
    return BayesianRidgeWithScaler(scaler=scaler, model=model)

