"""Script 4c: Train optional playoff-classifier (binary: playoff team yes/no).

What this does:
- Trains XGBoost to classify teams as playoff (top-15) vs non-playoff.
- Uses train 2015-22, val 2023, holdout 2024.
- Evaluates with AUC-ROC and Brier score.
- Optional; not part of main ranking pipeline. Use clone_classifier config."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to clone config (default: config/clone_classifier.yaml)")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "clone_classifier.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        clone_cfg = yaml.safe_load(f)
    # Merge clone config (train/val/holdout seasons, XGB params) into main config.
    for k, v in clone_cfg.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k].update(v)
        else:
            config[k] = v

    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1 and 2 first.", file=sys.stderr)
        sys.exit(1)

    from src.data.db_loader import load_training_data
    from src.evaluation.playoffs import compute_eos_playoff_standings
    from src.features.team_context import build_team_context_as_of_dates, get_team_context_feature_cols
    from src.training.build_lists import build_lists
    from src.utils.split import date_to_season

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        print("DB has no games/tgl.", file=sys.stderr)
        sys.exit(1)

    lists = build_lists(tgl, games, teams, config=config)
    if not lists:
        print("No lists.", file=sys.stderr)
        sys.exit(1)

    seasons_cfg = config.get("seasons") or {}
    train_seasons = set(config.get("training", {}).get("train_seasons", []))
    val_seasons = set(config.get("training", {}).get("validation_seasons", ["2023-24"]))
    holdout_seasons = set(config.get("training", {}).get("holdout_seasons", ["2024-25"]))

    rows = []
    for lst in lists:
        season = date_to_season(lst.get("as_of_date", ""), seasons_cfg)
        if not season:
            continue
        standings = compute_eos_playoff_standings(
            games, tgl, season,
            season_start=seasons_cfg.get(season, {}).get("start"),
            season_end=seasons_cfg.get(season, {}).get("end"),
        )
        # Binary target: 1 if team made playoffs (top 15 by end-of-season standings), else 0.
        for tid, wr in zip(lst["team_ids"], lst["win_rates"]):
            tid = int(tid)
            rank = standings.get(tid, 30)
            y = 1 if rank <= 15 else 0
            rows.append({"team_id": tid, "as_of_date": lst["as_of_date"], "y": y, "season": season})
    flat = pd.DataFrame(rows)
    if flat.empty:
        print("No rows.", file=sys.stderr)
        sys.exit(1)

    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates, config=config, teams=teams, pgl=pgl)
    df = flat.merge(feat_df, on=["team_id", "as_of_date"], how="inner")
    feat_cols = [c for c in get_team_context_feature_cols(config) if c in df.columns]
    if not feat_cols:
        print("No feature columns.", file=sys.stderr)
        sys.exit(1)

    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"].isin(val_seasons)]
    holdout_df = df[df["season"].isin(holdout_seasons)]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["y"].values.astype(np.int32)
    X_val = val_df[feat_cols].values.astype(np.float32) if not val_df.empty else None
    y_val = val_df["y"].values.astype(np.int32) if not val_df.empty else None
    X_hold = holdout_df[feat_cols].values.astype(np.float32) if not holdout_df.empty else None
    y_hold = holdout_df["y"].values.astype(np.int32) if not holdout_df.empty else None

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost required.", file=sys.stderr)
        sys.exit(1)

    cc = config.get("clone_classifier", {})
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=cc.get("eval_metric", ["auc", "logloss"]),
        n_estimators=cc.get("n_estimators", 250),
        max_depth=cc.get("max_depth", 4),
        learning_rate=cc.get("learning_rate", 0.08),
        random_state=config.get("repro", {}).get("seed", 42),
    )
    eval_set = [(X_train, y_train)]
    if X_val is not None and len(X_val) > 0:
        eval_set.append((X_val, y_val))
    clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    from sklearn.metrics import roc_auc_score, brier_score_loss

    report = {"config": str(config_path), "feat_cols": feat_cols}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("holdout", X_hold, y_hold)]:
        if X is None or len(X) == 0:
            continue
        proba = clf.predict_proba(X)[:, 1]
        report[f"{name}_auc"] = float(roc_auc_score(y, proba))
        report[f"{name}_brier"] = float(brier_score_loss(y, proba))
        report[f"{name}_n"] = int(len(y))

    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "clone_xgb_classifier.joblib"
    import joblib
    joblib.dump({"model": clf, "feat_cols": feat_cols, "config": config}, model_path)
    report_path = out / "clone_classifier_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {model_path}")
    print(f"Report: {report_path}")
    for k, v in report.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
