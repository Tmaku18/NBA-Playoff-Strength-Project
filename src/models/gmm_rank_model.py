"""Multi-Gaussian supervised rank model built on GaussianMixture.

Approach:
- Fit a GaussianMixture on X (team-context features).
- For each component k, estimate the distribution of y using responsibilities as weights:
    mean_k = sum(r_ik * y_i) / sum(r_ik)
    var_k  = sum(r_ik * (y_i - mean_k)^2) / sum(r_ik)
- For a new x, compute responsibilities r_k(x) and return predictive mixture over y:
    mean = sum(r_k * mean_k)
    var  = sum(r_k * (var_k + mean_k^2)) - mean^2
This yields a mean/std and preserves multi-Gaussian structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _weighted_mean_var(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.asarray(w, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    s = float(np.sum(w))
    if s <= 0:
        return 0.0, 0.0
    m = float(np.sum(w * y) / s)
    v = float(np.sum(w * (y - m) ** 2) / s)
    v = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    return m, max(v, 1e-6)


@dataclass
class GmmSupervisedRankRegressor:
    scaler: StandardScaler
    gmm: Any
    y_means_: np.ndarray  # (K,)
    y_vars_: np.ndarray  # (K,)

    def predict(self, X: np.ndarray, *, return_std: bool = False) -> Any:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        resp = np.asarray(self.gmm.predict_proba(Xs), dtype=np.float64)  # (n, K)
        means = np.asarray(self.y_means_, dtype=np.float64).reshape(1, -1)
        vars_ = np.asarray(self.y_vars_, dtype=np.float64).reshape(1, -1)
        mean = np.sum(resp * means, axis=1)
        second_moment = np.sum(resp * (vars_ + means**2), axis=1)
        var = np.maximum(second_moment - mean**2, 1e-6)
        mean = mean.astype(np.float32)
        if not return_std:
            return mean
        std = np.sqrt(var).astype(np.float32)
        std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return mean, np.maximum(std, 1e-6)


def fit_gmm_supervised_rank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_components_grid: list[int] | None = None,
    covariance_type_grid: list[str] | None = None,
    random_state: int = 42,
) -> GmmSupervisedRankRegressor:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    n_components_grid = n_components_grid or [1, 2, 3, 4, 5]
    covariance_type_grid = covariance_type_grid or ["full", "diag"]

    best_gmm: GaussianMixture | None = None
    best_bic: float | None = None
    for cov in covariance_type_grid:
        for k in n_components_grid:
            gmm = GaussianMixture(n_components=int(k), covariance_type=str(cov), random_state=int(random_state))
            gmm.fit(Xs)
            bic = float(gmm.bic(Xs))
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_gmm = gmm
    assert best_gmm is not None

    resp = np.asarray(best_gmm.predict_proba(Xs), dtype=np.float64)  # (n, K)
    K = resp.shape[1]
    means = np.zeros(K, dtype=np.float32)
    vars_ = np.zeros(K, dtype=np.float32)
    for j in range(K):
        m, v = _weighted_mean_var(y_train, resp[:, j])
        means[j] = float(m)
        vars_[j] = float(v)
    return GmmSupervisedRankRegressor(scaler=scaler, gmm=best_gmm, y_means_=means, y_vars_=vars_)

