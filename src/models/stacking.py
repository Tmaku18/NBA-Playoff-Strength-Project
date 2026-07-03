"""OOF stacking: build OOF for A + XGB (+ optional confidence and standings columns).

Column order convention (must match inference in src/inference/predict.py):
- 2 cols: [oof_a, oof_xgb]
- 3 cols: [oof_a, oof_xgb, standings]
- 4 cols: [oof_a, oof_xgb, conf_a, conf_xgb]
- 5 cols: [oof_a, oof_xgb, conf_a, conf_xgb, standings]
standings = win rate to date (0-1, higher = better), anchoring the meta on the
standings baseline (which beats the raw ensemble on rank MAE).
Fit RidgeCV on pooled OOF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV


def build_oof(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    y: np.ndarray,
    conf_a: np.ndarray | None = None,
    conf_xgb: np.ndarray | None = None,
    standings: np.ndarray | None = None,
) -> np.ndarray:
    """Stack OOF for meta: (N, 2), (N, 3), (N, 4) or (N, 5) depending on provided columns."""
    a = np.asarray(oof_deep_set).ravel()
    x = np.asarray(oof_xgb).ravel()
    cols = [a, x]
    if conf_a is not None and conf_xgb is not None:
        ca = np.asarray(conf_a).ravel()
        cx = np.asarray(conf_xgb).ravel()
        if ca.size == a.size and cx.size == a.size:
            cols.extend([ca, cx])
    if standings is not None:
        st = np.asarray(standings).ravel()
        if st.size == a.size:
            cols.append(st)
    return np.column_stack(cols)


def fit_ridgecv_on_oof(
    X_oof: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
    cv: int = 5,
) -> RidgeCV:
    """X_oof: (N, 2..5) from build_oof. y: (N,). Imputes NaN in X with 0 so RidgeCV can fit."""
    X = np.asarray(X_oof, dtype=np.float64)
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    meta = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    meta.fit(X, np.asarray(y).ravel())
    return meta


def save_oof(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    y: np.ndarray,
    path: Path | str,
    conf_a: np.ndarray | None = None,
    conf_xgb: np.ndarray | None = None,
    standings: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "oof_deep_set": np.asarray(oof_deep_set).ravel(),
        "oof_xgb": np.asarray(oof_xgb).ravel(),
        "y": np.asarray(y).ravel(),
    }
    if conf_a is not None and conf_xgb is not None:
        d["conf_a"] = np.asarray(conf_a).ravel()
        d["conf_xgb"] = np.asarray(conf_xgb).ravel()
    if standings is not None:
        d["standings"] = np.asarray(standings).ravel()
    pd.DataFrame(d).to_parquet(path, index=False)


def load_oof(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (oof_deep_set, oof_xgb, y, conf_a, conf_xgb, standings). Optional columns are None if missing."""
    df = pd.read_parquet(path)
    oof_a = df["oof_deep_set"].values
    oof_x = df["oof_xgb"].values
    y = df["y"].values
    conf_a = df["conf_a"].values if "conf_a" in df.columns else None
    conf_xgb = df["conf_xgb"].values if "conf_xgb" in df.columns else None
    standings = df["standings"].values if "standings" in df.columns else None
    return oof_a, oof_x, y, conf_a, conf_xgb, standings
