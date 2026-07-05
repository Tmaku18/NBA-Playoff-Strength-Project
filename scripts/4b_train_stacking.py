"""Script 4b: Train stacking meta-learner.

What this does:
- Loads OOF predictions from Model A (oof_model_a.parquet) and Model B (oof_model_b.parquet).
- Trains a RidgeCV meta-learner to blend Model A + XGBoost into ensemble predictions.
  Optional extra column: standings win rate to date (stacking.use_standings) so the
  meta can anchor on the standings baseline. Inputs are per-date rank-transformed
  when stacking.rank_transform is on (must match inference).
- Fits per-conference metas (ridgecv_meta_E/W.joblib) on conference-only OOF rows.
- Saves ridgecv_meta.joblib for use during inference (script 6).

Target convention: y is always "higher = better" (playoff target uses 31 - playoff rank),
matching inference, which ranks teams by descending meta output.

Run after scripts 3 and 4. Required before inference (6)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_stacking import train_stacking


def _season_from_date(as_of_date: str, seasons_config: dict) -> str | None:
    """Return season key (e.g. '2023-24') if as_of_date falls in that season's range."""
    try:
        from datetime import datetime
        d = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    except Exception:
        return None
    for season, rng in (seasons_config or {}).items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return season
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    # Both scripts 3 and 4 must have produced OOF parquets so we can merge and train the meta-learner.
    path_a = out / "oof_model_a.parquet"
    path_b = out / "oof_model_b.parquet"
    if not path_a.exists() or not path_b.exists():
        print(
            "OOF parquets not found. Run scripts 3 and 4 with OOF output first "
            "(outputs/oof_model_a.parquet and outputs/oof_model_b.parquet).",
            file=sys.stderr,
        )
        sys.exit(1)
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    # Align Model A and Model B OOF by (team_id, as_of_date).
    merged = df_a.merge(
        df_b,
        on=["team_id", "as_of_date"],
        how="inner",
        suffixes=("", "_b"),
    )
    # Model B's y is win rate to date = standings signal; keep it as an optional stacking column.
    if "y_b" in merged.columns:
        merged = merged.rename(columns={"y_b": "standings_wr"})
    if merged.empty:
        print("No overlapping (team_id, as_of_date) between OOF files.", file=sys.stderr)
        sys.exit(1)

    # Optionally replace y with playoff-based rank (e.g. final playoff finish) when target_rank is "playoffs".
    target_rank = (config.get("training") or {}).get("target_rank", "standings")
    if target_rank == "playoffs":
        db_path = Path(config.get("paths", {}).get("db", "data/processed/nba_build_run.duckdb"))
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        if db_path.exists():
            try:
                from src.data.db_loader import load_playoff_data, load_training_data
                from src.evaluation.playoffs import compute_playoff_performance_rank

                games, tgl, teams, _ = load_training_data(db_path)
                pg, ptgl, _ = load_playoff_data(db_path)
                seasons_cfg = config.get("seasons") or {}
                playoff_rank_by_season: dict[str, dict[int, int]] = {}
                for season in seasons_cfg:
                    rng = seasons_cfg.get(season, {})
                    season_start = rng.get("start")
                    season_end = rng.get("end")
                    rank_map = compute_playoff_performance_rank(
                        pg, ptgl, games, tgl, season,
                        all_team_ids=teams["team_id"].astype(int).unique().tolist() if not teams.empty else None,
                        season_start=season_start,
                        season_end=season_end,
                        debug=False,
                    )
                    if rank_map:
                        playoff_rank_by_season[season] = rank_map
                # y convention: higher = better (matches inference, which ranks by descending
                # meta output, and matches Model A's rel target 31 - rank). Rows without a
                # playoff rank get NaN and are mean-imputed below, so we never mix scales.
                y_list = []
                n_missing = 0
                for _, row in merged.iterrows():
                    tid = int(row["team_id"])
                    season = _season_from_date(str(row["as_of_date"]), seasons_cfg)
                    if season and season in playoff_rank_by_season and tid in playoff_rank_by_season[season]:
                        y_list.append(31.0 - float(playoff_rank_by_season[season][tid]))
                    else:
                        y_list.append(np.nan)
                        n_missing += 1
                merged["y"] = y_list
                if n_missing:
                    print(
                        f"Playoff target: {n_missing}/{len(merged)} rows without playoff rank "
                        "(dropped below).",
                        flush=True,
                    )
            except Exception as e:
                print(f"Playoff target failed, using standings y: {e}", file=sys.stderr)

    # Drop rows without a target. Mean-imputing y=15.5 taught the meta that a
    # league-average finish is the default outcome regardless of the inputs.
    if merged["y"].isna().any():
        n_before = len(merged)
        merged = merged[merged["y"].notna()].reset_index(drop=True)
        print(f"Dropped {n_before - len(merged)}/{n_before} rows with missing target y.", flush=True)
    if merged.empty:
        print("No rows with a valid target after dropping missing y.", file=sys.stderr)
        sys.exit(1)

    # Impute any NaN in OOF feature columns so Ridge regression gets finite inputs.
    for col in ["oof_a", "oof_xgb", "standings_wr"]:
        if col in merged.columns and merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].mean())
    stacking_cfg = config.get("stacking", {}) or {}
    use_standings = bool(stacking_cfg.get("use_standings", True))
    # Per-date percentile-rank transform: makes the meta scale-invariant (OOF fold-model
    # scores and final-model inference scores have different scales) and Spearman-aligned.
    # Must match _meta_X in src/inference/predict.py (stacking.rank_transform).
    rank_transform = bool(stacking_cfg.get("rank_transform", True))
    if rank_transform:
        for col in ["oof_a", "oof_xgb", "standings_wr"]:
            if col in merged.columns:
                merged[col] = merged.groupby("as_of_date")[col].rank(pct=True)
    oof_a = merged["oof_a"].values.astype("float32")
    oof_xgb = merged["oof_xgb"].values.astype("float32")
    y = merged["y"].values.astype("float32")
    # Confidence columns are deliberately excluded from the meta: conf_a means different
    # things in OOF (attention entropy) vs inference (cross-temp agreement) and got
    # near-zero/negative coefficients.
    standings = (
        merged["standings_wr"].values.astype("float32")
        if use_standings and "standings_wr" in merged.columns
        else None
    )
    # Conference for per-conference metas: prefer a column with both E and W present
    # (older oof_model_a files carried a stale constant conference; Model B's is from build_lists).
    conference = None
    for col in ("conference", "conference_b"):
        if col in merged.columns:
            vals = merged[col].astype(str)
            if {"E", "W"} <= set(vals.unique()):
                conference = vals.values
                break
    path = train_stacking(
        oof_a, oof_xgb, y, config, out,
        standings=standings, conference=conference,
    )
    n_cols = 2 + (1 if standings is not None else 0)
    print(f"Meta columns: {n_cols} (rank_transform={'on' if rank_transform else 'off'}, standings={'on' if standings is not None else 'off'})")
    print(f"Saved {path}, {out / 'oof_pooled.parquet'}")


if __name__ == "__main__":
    main()
