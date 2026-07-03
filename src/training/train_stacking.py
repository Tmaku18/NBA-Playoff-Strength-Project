"""Train stacking: OOF for A + XGB (+ optional confidence and standings cols), fit RidgeCV.

Persists oof_pooled.parquet and ridgecv_meta.joblib. When a conference array is
provided, also fits true per-conference metas (ridgecv_meta_E.joblib /
ridgecv_meta_W.joblib) on conference-only OOF rows so East and West get their
own blend weights (the East consistently underperforms with a global meta).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.stacking import build_oof, fit_ridgecv_on_oof, save_oof

MIN_CONFERENCE_ROWS = 50


def train_stacking(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    y: np.ndarray,
    config: dict,
    output_dir: str | Path,
    conf_a: np.ndarray | None = None,
    conf_xgb: np.ndarray | None = None,
    standings: np.ndarray | None = None,
    conference: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = config.get("training", {}).get("n_folds", 5)
    X = build_oof(oof_deep_set, oof_xgb, y, conf_a=conf_a, conf_xgb=conf_xgb, standings=standings)
    y_arr = np.asarray(y).ravel()
    meta = fit_ridgecv_on_oof(X, y_arr, cv=cv)

    oof_path = output_dir / "oof_pooled.parquet"
    save_oof(oof_deep_set, oof_xgb, y_arr, oof_path, conf_a=conf_a, conf_xgb=conf_xgb, standings=standings)

    meta_path = output_dir / "ridgecv_meta.joblib"
    joblib.dump(meta, meta_path)

    # Per-conference metas fit on conference-only OOF rows (not copies of the global meta).
    if conference is not None:
        conf_arr = np.asarray(conference).ravel()
        if conf_arr.size == y_arr.size:
            for c in ("E", "W"):
                mask = conf_arr == c
                if int(mask.sum()) >= MIN_CONFERENCE_ROWS:
                    meta_c = fit_ridgecv_on_oof(X[mask], y_arr[mask], cv=cv)
                    joblib.dump(meta_c, output_dir / f"ridgecv_meta_{c}.joblib")
                    print(
                        f"Per-conference meta {c}: {int(mask.sum())} rows, "
                        f"coef={np.asarray(meta_c.coef_).ravel().round(3).tolist()}",
                        flush=True,
                    )

    return meta_path
