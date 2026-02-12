"""XGBoost ranker/regressor with early stopping. No net_rating in features."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def build_xgb(config: dict | None = None) -> Any:
    cfg = config or {}
    if not _HAS_XGB:
        raise ImportError("xgboost is required")
    kwargs = {
        "n_estimators": cfg.get("n_estimators", 500),
        "max_depth": cfg.get("max_depth", 6),
        "learning_rate": cfg.get("learning_rate", 0.05),
        "random_state": cfg.get("random_state", 42),
    }
    if cfg.get("subsample") is not None:
        kwargs["subsample"] = float(cfg["subsample"])
    if cfg.get("colsample_bytree") is not None:
        kwargs["colsample_bytree"] = float(cfg["colsample_bytree"])
    return xgb.XGBRegressor(**kwargs)


def fit_xgb(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    eval_set: list[tuple] | None = None,
    early_stopping_rounds: int = 20,
) -> Any:
    if eval_set is None and X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    kwargs: dict = {"verbose": False}
    if eval_set:
        kwargs["eval_set"] = eval_set
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    try:
        model.fit(X_train, y_train, **kwargs)
    except TypeError:
        kwargs.pop("early_stopping_rounds", None)
        model.fit(X_train, y_train, **kwargs)
    return model


def predict_with_uncertainty(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, per-sample std across trees) for XGB confidence.

    Uses get_booster() and iteration_range to get per-tree predictions, then
    std across trees per sample. Higher std -> lower confidence.

    Returns:
        pred: shape (n_samples,) same as model.predict(X).
        std: shape (n_samples,) std across trees; use e.g. c_X = 1/(1+std).
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is required")
    X = np.asarray(X, dtype=np.float32)
    pred = model.predict(X)
    try:
        booster = model.get_booster()
        n_trees = booster.num_boosted_rounds()
        if n_trees == 0:
            return pred, np.zeros_like(pred)
        dmat = xgb.DMatrix(X)
        tree_preds = []
        for i in range(n_trees):
            p = booster.predict(dmat, iteration_range=(i, i + 1))
            tree_preds.append(np.asarray(p).ravel())
        tree_preds = np.stack(tree_preds, axis=0)
        std = np.std(tree_preds, axis=0)
        std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(pred).ravel(), std
    except Exception:
        return np.asarray(pred).ravel(), np.zeros(np.asarray(pred).size)
