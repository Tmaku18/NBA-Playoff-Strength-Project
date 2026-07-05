"""Script 4: Train Model B (XGBoost) and optionally Model C (Random Forest).

What this does:
- Loads team-context features (ELO, rolling stats, SOS/SRS, etc.) from DB.
- Uses the same train/test split as script 3 (from split_info.json).
- Trains XGBoost (Model B). Optionally trains RF (Model C) when training.train_model_c is true.
- Ensemble uses A + B only; Model C is not in ensemble (analytics comparison only when present).
- Produces K-fold OOF predictions for stacking (script 4b): oof_xgb only.
- Saves oof_model_b.parquet (oof_xgb), xgb_model.joblib, and rf_model.joblib only if train_model_c.

Run after script 3. Required before stacking (4b) and inference (6)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.db_loader import load_training_data
from src.features.feature_cache import (
    compute_feature_cache_key,
    get_feature_cache_dir,
    load_feature_cache,
    save_feature_cache,
)
from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates, get_team_context_feature_cols
from src.training.build_lists import build_lists
from src.training.train_model_b import train_model_b
from src.utils.split import load_split_info

from src.models.xgb_model import build_xgb, fit_xgb, predict_with_uncertainty
from src.models.rf_model import build_rf, fit_rf

from src.models.bayesian_ridge_model import fit_bayesian_ridge
from src.models.gmm_rank_model import fit_gmm_supervised_rank
from src.models.gpr_model import fit_gpr
from src.models.linear_regression_model import fit_linear_regression_with_uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/defaults.yaml)")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    # Same list structure as script 3: one row per (team, date) with target y (e.g. win rate or playoff rank).
    lists = build_lists(tgl, games, teams)
    if not lists:
        print("No lists from build_lists (empty games/tgl?). Exiting.", file=sys.stderr)
        sys.exit(1)
    rows = []
    for lst in lists:
        conf = lst.get("conference", "E")
        for tid, wr in zip(lst["team_ids"], lst["win_rates"]):
            rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"], "y": float(wr), "conference": conf})
    flat = pd.DataFrame(rows)
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    team_dates_hash = hashlib.sha256(json.dumps(sorted(team_dates), sort_keys=True).encode()).hexdigest()[:16]

    # Optional feature cache: reuse build_team_context output when config/DB/team_dates unchanged
    cache_dir = get_feature_cache_dir(config, ROOT)
    cache_key = compute_feature_cache_key(config, db_path, team_dates_hash) if cache_dir else None
    feat_df = load_feature_cache(cache_dir, cache_key) if cache_dir and cache_key else None
    if feat_df is not None:
        print(f"Feature cache hit: {cache_key}.parquet", flush=True)
    if feat_df is None:
        feat_df = build_team_context_as_of_dates(
            tgl, games, team_dates,
            config=config, teams=teams, pgl=pgl,
        )
        if cache_dir and cache_key:
            save_feature_cache(cache_dir, cache_key, feat_df)
            if not feat_df.empty:
                print(f"Feature cache saved: {cache_key}.parquet", flush=True)
    df = flat.merge(feat_df, on=["team_id", "as_of_date"], how="inner")

    # Optional: add Model A OOF score as a feature for team-stats models (leak-safe via OOF join).
    tsm_cfg = config.get("team_stats_models", {}) or {}
    if bool(tsm_cfg.get("include_model_a_score", False)):
        oof_a_path = Path(config["paths"]["outputs"])
        if not oof_a_path.is_absolute():
            oof_a_path = ROOT / oof_a_path
        oof_a_path = oof_a_path / "oof_model_a.parquet"
        if oof_a_path.exists():
            try:
                oof_a = pd.read_parquet(oof_a_path)
                if {"team_id", "as_of_date", "oof_a"} <= set(oof_a.columns):
                    oof_a = oof_a[["team_id", "as_of_date", "oof_a"]].copy()
                    oof_a["team_id"] = oof_a["team_id"].astype(int)
                    oof_a["as_of_date"] = oof_a["as_of_date"].astype(str)
                    df = df.merge(oof_a, on=["team_id", "as_of_date"], how="left")
                    df["model_a_score"] = df["oof_a"].fillna(0.0).astype(np.float32)
            except Exception:
                pass
    all_feat_cols = get_team_context_feature_cols(config)
    feat_cols = [c for c in all_feat_cols if c in df.columns]
    if not feat_cols:
        print("No feature columns. Exiting.", file=sys.stderr)
        sys.exit(1)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    # Use only dates that script 3 marked as train (so OOF and final models match the same split).
    split_info = load_split_info(out)
    train_dates_set = set(split_info.get("train_dates", []))
    if not train_dates_set:
        print("split_info.json has no train_dates. Exiting.", file=sys.stderr)
        sys.exit(1)
    df = df[df["as_of_date"].isin(train_dates_set)].copy()
    flat = flat[flat["as_of_date"].isin(train_dates_set)].copy()
    print(f"Models B & C: using {len(df)} rows on {len(train_dates_set)} train dates", flush=True)

    n_folds = config.get("training", {}).get("n_folds", 5)
    dates_sorted = sorted(df["as_of_date"].unique())
    n_folds = min(n_folds, len(dates_sorted))
    if n_folds < 2:
        X = df[feat_cols].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)
        p1, p2 = train_model_b(X, y, None, None, config, feat_cols, out)
        print(f"Saved {p1}" + (f", {p2}" if p2 else " (Model C skipped)") + " (too few dates for OOF)")
        return

    # Assign each date to a fold so validation is time-based (same idea as script 3).
    fold_size = (len(dates_sorted) + n_folds - 1) // n_folds
    date_to_fold = {}
    for fold in range(n_folds):
        start = fold * fold_size
        end = min((fold + 1) * fold_size, len(dates_sorted))
        for i in range(start, end):
            date_to_fold[dates_sorted[i]] = fold
    df["_fold"] = df["as_of_date"].map(date_to_fold)

    mb = config.get("model_b", {})
    xgb_cfg = mb.get("xgb", {})
    rf_cfg = mb.get("rf", {})
    es = xgb_cfg.get("early_stopping_rounds", 20)
    train_model_c = config.get("training", {}).get("train_model_c", False)
    train_lin = bool(config.get("training", {}).get("train_team_stats_linear", False))
    train_bayes = bool(config.get("training", {}).get("train_team_stats_bayesian_ridge", False))
    train_gpr = bool(config.get("training", {}).get("train_team_stats_gpr", False))
    train_gmm = bool(config.get("training", {}).get("train_team_stats_gmm", False))
    tsm_cfg = config.get("team_stats_models", {}) or {}
    use_model_a_score = bool(tsm_cfg.get("include_model_a_score", False)) and ("model_a_score" in df.columns)
    feat_cols_ts = list(feat_cols) + (["model_a_score"] if use_model_a_score else [])
    gpr_cfg = tsm_cfg.get("gpr", {}) if isinstance(tsm_cfg.get("gpr", {}), dict) else {}
    gpr_kernels = gpr_cfg.get("kernels", ["rbf", "matern", "rational_quadratic"])
    gmm_cfg = tsm_cfg.get("gmm", {}) if isinstance(tsm_cfg.get("gmm", {}), dict) else {}
    gmm_n_grid = gmm_cfg.get("n_components_grid", [1, 2, 3, 4, 5])
    gmm_cov_grid = gmm_cfg.get("covariance_type_grid", ["full", "diag"])
    lin_cfg = tsm_cfg.get("linear", {}) if isinstance(tsm_cfg.get("linear", {}), dict) else {}
    br_cfg = tsm_cfg.get("bayesian_ridge", {}) if isinstance(tsm_cfg.get("bayesian_ridge", {}), dict) else {}

    oof_rows = []
    # Causal expanding-window OOF (same as script 3): validate fold f training only on
    # earlier folds; the old scheme trained on future folds when validating early ones.
    for fold in range(1, n_folds):
        train_mask = df["_fold"] < fold
        val_mask = df["_fold"] == fold
        X_train = df.loc[train_mask, feat_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, "y"].values.astype(np.float32)
        X_val = df.loc[val_mask, feat_cols].values.astype(np.float32)
        y_val = df.loc[val_mask, "y"].values.astype(np.float32)
        if X_train.size == 0 or X_val.size == 0:
            continue
        xgb_m = build_xgb(xgb_cfg)
        fit_xgb(xgb_m, X_train, y_train, X_val, y_val, early_stopping_rounds=es)
        if train_model_c:
            rf_m = build_rf(rf_cfg)
            fit_rf(rf_m, X_train, y_train)
        try:
            oof_xgb, tree_std = predict_with_uncertainty(xgb_m, X_val)
            oof_xgb = oof_xgb.astype(np.float32)
            conf_xgb = (1.0 / (1.0 + np.clip(tree_std, 0, None))).astype(np.float32)
        except Exception:
            oof_xgb = xgb_m.predict(X_val).astype(np.float32)
            conf_xgb = np.full(oof_xgb.shape, 0.5, dtype=np.float32)
        val_cols = ["team_id", "as_of_date", "y"]
        if "conference" in df.columns:
            val_cols.append("conference")
        val_df = df.loc[val_mask, val_cols].copy()
        val_df["oof_xgb"] = oof_xgb
        val_df["conf_xgb"] = conf_xgb

        # Additional team-stats models (optional)
        if any([train_lin, train_bayes, train_gpr, train_gmm]):
            X_train_ts = df.loc[train_mask, feat_cols_ts].values.astype(np.float32)
            X_val_ts = df.loc[val_mask, feat_cols_ts].values.astype(np.float32)

            if train_lin:
                try:
                    lr_m = fit_linear_regression_with_uncertainty(
                        X_train_ts, y_train, fit_intercept=bool(lin_cfg.get("fit_intercept", True))
                    )
                    p_mean, p_std = lr_m.predict(X_val_ts, return_std=True)
                    val_df["oof_linreg"] = p_mean
                    val_df["std_linreg"] = p_std
                except Exception:
                    pass

            if train_bayes:
                try:
                    br_m = fit_bayesian_ridge(
                        X_train_ts,
                        y_train,
                        alpha_1=float(br_cfg.get("alpha_1", 1e-6)),
                        alpha_2=float(br_cfg.get("alpha_2", 1e-6)),
                        lambda_1=float(br_cfg.get("lambda_1", 1e-6)),
                        lambda_2=float(br_cfg.get("lambda_2", 1e-6)),
                    )
                    p_mean, p_std = br_m.predict(X_val_ts, return_std=True)
                    val_df["oof_bayes_ridge"] = p_mean
                    val_df["std_bayes_ridge"] = p_std
                except Exception:
                    pass

            if train_gmm:
                try:
                    gmm_m = fit_gmm_supervised_rank(
                        X_train_ts,
                        y_train,
                        n_components_grid=list(gmm_n_grid),
                        covariance_type_grid=list(gmm_cov_grid),
                        random_state=int(gmm_cfg.get("random_state", 42)),
                    )
                    p_mean, p_std = gmm_m.predict(X_val_ts, return_std=True)
                    val_df["oof_gmm"] = p_mean
                    val_df["std_gmm"] = p_std
                except Exception:
                    pass

            if train_gpr:
                best_kernel = None
                best_mse = None
                best_mean = None
                best_std = None
                for kname in gpr_kernels:
                    try:
                        gpr_m = fit_gpr(
                            X_train_ts,
                            y_train,
                            kernel_name=str(kname),
                            cfg=gpr_cfg,
                            random_state=int(gpr_cfg.get("random_state", 42)),
                        )
                        p_mean, p_std = gpr_m.predict(X_val_ts, return_std=True)
                        mse = float(np.mean((p_mean - y_val) ** 2)) if len(y_val) else float("inf")
                        if best_mse is None or mse < best_mse:
                            best_mse = mse
                            best_kernel = str(kname)
                            best_mean = p_mean
                            best_std = p_std
                    except Exception:
                        continue
                if best_mean is not None and best_std is not None:
                    val_df["oof_gpr"] = best_mean
                    val_df["std_gpr"] = best_std
                    val_df["gpr_kernel"] = best_kernel

        oof_rows.append(val_df)
        print(f"Fold {fold+1}/{n_folds} OOF collected {len(val_df)} rows")

    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_path = out / "oof_model_b.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_df)} rows)")
    else:
        print(
            "No OOF rows collected (need at least 2 folds with non-empty train/val).",
            file=sys.stderr,
        )

    # Train final XGB and LR on all train data (with a small validation holdout for early stopping if configured).
    X = df[feat_cols].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)
    dates_sorted_full = sorted(df["as_of_date"].unique())
    n_val = max(1, int(0.2 * len(dates_sorted_full)))
    val_dates = set(dates_sorted_full[-n_val:])
    val_mask = df["as_of_date"].isin(val_dates)
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask] if val_mask.any() else None
    y_val = y[val_mask] if val_mask.any() else None
    p1, p2 = train_model_b(X_train, y_train, X_val, y_val, config, feat_cols, out)
    print(f"Saved {p1}" + (f", {p2}" if p2 else " (Model C skipped)"))

    # Train and persist extra team-stats models on full train split (optional)
    if any([train_lin, train_bayes, train_gpr, train_gmm]):
        X_train_ts = df.loc[~val_mask, feat_cols_ts].values.astype(np.float32)
        y_train_ts = df.loc[~val_mask, "y"].values.astype(np.float32)
        X_val_ts = df.loc[val_mask, feat_cols_ts].values.astype(np.float32) if val_mask.any() else None
        y_val_ts = df.loc[val_mask, "y"].values.astype(np.float32) if val_mask.any() else None

        if train_lin:
            try:
                lr_m = fit_linear_regression_with_uncertainty(
                    X_train_ts, y_train_ts, fit_intercept=bool(lin_cfg.get("fit_intercept", True))
                )
                joblib.dump({"model": lr_m, "feature_cols": feat_cols_ts}, out / "linreg_model.joblib")
            except Exception:
                pass

        if train_bayes:
            try:
                br_m = fit_bayesian_ridge(
                    X_train_ts,
                    y_train_ts,
                    alpha_1=float(br_cfg.get("alpha_1", 1e-6)),
                    alpha_2=float(br_cfg.get("alpha_2", 1e-6)),
                    lambda_1=float(br_cfg.get("lambda_1", 1e-6)),
                    lambda_2=float(br_cfg.get("lambda_2", 1e-6)),
                )
                joblib.dump({"model": br_m, "feature_cols": feat_cols_ts}, out / "bayesian_ridge_model.joblib")
            except Exception:
                pass

        if train_gmm:
            try:
                gmm_m = fit_gmm_supervised_rank(
                    X_train_ts,
                    y_train_ts,
                    n_components_grid=list(gmm_n_grid),
                    covariance_type_grid=list(gmm_cov_grid),
                    random_state=int(gmm_cfg.get("random_state", 42)),
                )
                joblib.dump({"model": gmm_m, "feature_cols": feat_cols_ts}, out / "gmm_rank_model.joblib")
            except Exception:
                pass

        if train_gpr:
            best_kernel = None
            best_mse = None
            best_model = None
            if X_val_ts is not None and y_val_ts is not None and len(y_val_ts):
                for kname in gpr_kernels:
                    try:
                        gpr_m = fit_gpr(
                            X_train_ts,
                            y_train_ts,
                            kernel_name=str(kname),
                            cfg=gpr_cfg,
                            random_state=int(gpr_cfg.get("random_state", 42)),
                        )
                        p_mean = gpr_m.predict(X_val_ts, return_std=False)
                        mse = float(np.mean((p_mean - y_val_ts) ** 2))
                        if best_mse is None or mse < best_mse:
                            best_mse = mse
                            best_kernel = str(kname)
                            best_model = gpr_m
                    except Exception:
                        continue
            if best_model is None:
                # Fallback: just train the first kernel
                try:
                    best_kernel = str(gpr_kernels[0])
                    best_model = fit_gpr(
                        X_train_ts,
                        y_train_ts,
                        kernel_name=str(best_kernel),
                        cfg=gpr_cfg,
                        random_state=int(gpr_cfg.get("random_state", 42)),
                    )
                except Exception:
                    best_model = None
            if best_model is not None:
                joblib.dump(
                    {"model": best_model, "feature_cols": feat_cols_ts, "kernel": best_kernel},
                    out / "gpr_model.joblib",
                )


if __name__ == "__main__":
    main()
