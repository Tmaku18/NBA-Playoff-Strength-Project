"""Train stacking: OOF for A + XGB (2 or 4 cols with confidence), fit RidgeCV, persist oof_pooled.parquet and meta."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.stacking import build_oof, fit_ridgecv_on_oof, save_oof


def train_stacking(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    y: np.ndarray,
    config: dict,
    output_dir: str | Path,
    conf_a: np.ndarray | None = None,
    conf_xgb: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = build_oof(oof_deep_set, oof_xgb, y, conf_a=conf_a, conf_xgb=conf_xgb)
    y_arr = np.asarray(y).ravel()
    meta = fit_ridgecv_on_oof(X, y_arr, cv=config.get("training", {}).get("n_folds", 5))

    oof_path = output_dir / "oof_pooled.parquet"
    save_oof(oof_deep_set, oof_xgb, y_arr, oof_path, conf_a=conf_a, conf_xgb=conf_xgb)

    meta_path = output_dir / "ridgecv_meta.joblib"
    joblib.dump(meta, meta_path)
    return meta_path
