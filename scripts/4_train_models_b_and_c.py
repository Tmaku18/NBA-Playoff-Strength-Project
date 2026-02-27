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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.db_loader import load_training_data
from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates, get_team_context_feature_cols
from src.training.build_lists import build_lists
from src.training.train_model_b import train_model_b
from src.utils.split import load_split_info

from src.models.xgb_model import build_xgb, fit_xgb, predict_with_uncertainty
from src.models.rf_model import build_rf, fit_rf


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
    # Build team-context features (rolling stats, ELO, etc.) for each (team_id, as_of_date).
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(
        tgl, games, team_dates,
        config=config, teams=teams, pgl=pgl,
    )
    df = flat.merge(feat_df, on=["team_id", "as_of_date"], how="inner")
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

    oof_rows = []
    for fold in range(n_folds):
        train_mask = df["_fold"] != fold
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


if __name__ == "__main__":
    main()
