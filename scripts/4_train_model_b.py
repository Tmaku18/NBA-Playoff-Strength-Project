"""Train Model B (XGB + RF) on team-context features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.db_loader import load_training_data
from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
from src.training.build_lists import build_lists
from src.training.train_model_b import train_model_b


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, _ = load_training_data(db_path)
    lists = build_lists(tgl, games, teams)
    if not lists:
        print("No lists from build_lists (empty games/tgl?). Exiting.", file=sys.stderr)
        sys.exit(1)
    max_lists = int(config.get("training", {}).get("max_lists_oof", 0) or 0)
    if max_lists and len(lists) > max_lists:
        rng = np.random.default_rng(int(config.get("repro", {}).get("seed", 42)))
        idx = rng.choice(len(lists), size=max_lists, replace=False)
        lists = [lists[i] for i in sorted(idx)]
    rows = []
    for lst in lists:
        for tid, wr in zip(lst["team_ids"], lst["win_rates"]):
            rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"], "y": float(wr)})
    flat = pd.DataFrame(rows)
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
    df = flat.merge(feat_df, on=["team_id", "as_of_date"], how="inner")
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in df.columns]
    if not feat_cols:
        print("No feature columns. Exiting.", file=sys.stderr)
        sys.exit(1)

    X = df[feat_cols].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)
    dates_sorted = sorted(df["as_of_date"].unique())
    val_frac = float(config.get("model_a", {}).get("early_stopping_val_frac", 0.2))
    n_val = max(1, int(val_frac * len(dates_sorted)))
    val_dates = set(dates_sorted[-n_val:])
    val_mask = df["as_of_date"].isin(val_dates)
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask] if val_mask.any() else None
    y_val = y[val_mask] if val_mask.any() else None

    out = Path(config["paths"]["outputs"])
    p1, p2 = train_model_b(X_train, y_train, X_val, y_val, config, feat_cols, out)
    print(f"Saved {p1}, {p2}")


if __name__ == "__main__":
    main()
