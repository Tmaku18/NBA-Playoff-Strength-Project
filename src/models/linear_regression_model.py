"""Linear regression model for team-stats rank prediction (with uncertainty).

This is distinct from src/models/lr_model.py (logistic regression diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearRegressionWithUncertainty:
    scaler: StandardScaler
    model: Any
    resid_std_: float

    def predict(self, X: np.ndarray, *, return_std: bool = False) -> Any:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        mean = np.asarray(self.model.predict(Xs)).ravel().astype(np.float32)
        if not return_std:
            return mean
        std = np.full(mean.shape, float(self.resid_std_), dtype=np.float32)
        return mean, std


def fit_linear_regression_with_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> LinearRegressionWithUncertainty:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(Xs, y_train)
    pred = np.asarray(model.predict(Xs)).ravel()
    resid = y_train - pred
    resid_std = float(np.sqrt(np.mean(resid**2))) if resid.size else 0.0
    resid_std = float(np.nan_to_num(resid_std, nan=0.0, posinf=0.0, neginf=0.0))
    return LinearRegressionWithUncertainty(scaler=scaler, model=model, resid_std_=max(resid_std, 1e-6))

