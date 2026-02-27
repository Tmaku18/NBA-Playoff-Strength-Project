"""Gaussian Process regression for team-stats rank prediction (mean + std)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.preprocessing import StandardScaler

KernelName = Literal["rbf", "matern", "rational_quadratic"]


def _build_kernel(name: KernelName, cfg: dict) -> Any:
    if name == "rbf":
        rbf = cfg.get("rbf", {}) if isinstance(cfg, dict) else {}
        ls = float(rbf.get("length_scale", 1.0))
        bounds = rbf.get("length_scale_bounds", (1e-2, 1e2))
        return 1.0 * RBF(length_scale=ls, length_scale_bounds=tuple(bounds)) + WhiteKernel(noise_level=1e-3)
    if name == "matern":
        mt = cfg.get("matern", {}) if isinstance(cfg, dict) else {}
        ls = float(mt.get("length_scale", 1.0))
        bounds = mt.get("length_scale_bounds", (1e-2, 1e2))
        nu = float(mt.get("nu", 1.5))
        return 1.0 * Matern(length_scale=ls, length_scale_bounds=tuple(bounds), nu=nu) + WhiteKernel(noise_level=1e-3)
    rq = cfg.get("rational_quadratic", {}) if isinstance(cfg, dict) else {}
    ls = float(rq.get("length_scale", 1.0))
    alpha = float(rq.get("alpha", 1.0))
    return 1.0 * RationalQuadratic(length_scale=ls, alpha=alpha) + WhiteKernel(noise_level=1e-3)


@dataclass
class GprWithScaler:
    scaler: StandardScaler
    model: Any
    kernel_name: str

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


def fit_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kernel_name: KernelName,
    cfg: dict,
    random_state: int = 42,
) -> GprWithScaler:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    kernel = _build_kernel(kernel_name, cfg)
    gpr_cfg = cfg if isinstance(cfg, dict) else {}
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(gpr_cfg.get("alpha", 1e-6)),
        normalize_y=bool(gpr_cfg.get("normalize_y", True)),
        n_restarts_optimizer=int(gpr_cfg.get("n_restarts_optimizer", 1)),
        random_state=int(random_state),
    )
    model.fit(Xs, y_train)
    return GprWithScaler(scaler=scaler, model=model, kernel_name=str(kernel_name))

