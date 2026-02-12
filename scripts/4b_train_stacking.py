"""Script 4b: Train stacking meta-learner.

What this does:
- Loads OOF predictions from Model A (oof_model_a.parquet) and Model B (oof_model_b.parquet).
- Trains a RidgeCV meta-learner to blend Model A + XGBoost (2 columns) into ensemble predictions.
- Saves ridgecv_meta.joblib for use during inference (script 6).

Run after scripts 3 and 4. Required before inference (6)."""
import argparse
import sys
from pathlib import Path

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
    # Align Model A and Model B OOF by (team_id, as_of_date); drop duplicate y column from B.
    merged = df_a.merge(
        df_b,
        on=["team_id", "as_of_date"],
        how="inner",
        suffixes=("", "_b"),
    )
    if "y_b" in merged.columns:
        merged = merged.drop(columns=["y_b"])
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
                y_list = []
                for _, row in merged.iterrows():
                    tid = int(row["team_id"])
                    season = _season_from_date(str(row["as_of_date"]), seasons_cfg)
                    if season and season in playoff_rank_by_season and tid in playoff_rank_by_season[season]:
                        y_list.append(float(playoff_rank_by_season[season][tid]))
                    else:
                        y_list.append(float(row["y"]) if "y" in row and pd.notna(row["y"]) else 15.0)
                merged["y"] = y_list
            except Exception as e:
                print(f"Playoff target failed, using standings y: {e}", file=sys.stderr)

    # Impute any NaN in OOF or target so Ridge regression gets finite inputs.
    for col in ["oof_a", "oof_xgb", "y"]:
        if col in merged.columns and merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].mean())
    oof_a = merged["oof_a"].values.astype("float32")
    oof_xgb = merged["oof_xgb"].values.astype("float32")
    y = merged["y"].values.astype("float32")
    conf_a = merged["conf_a"].values.astype("float32") if "conf_a" in merged.columns and "conf_xgb" in merged.columns else None
    conf_xgb = merged["conf_xgb"].values.astype("float32") if "conf_a" in merged.columns and "conf_xgb" in merged.columns else None
    path = train_stacking(oof_a, oof_xgb, y, config, out, conf_a=conf_a, conf_xgb=conf_xgb)
    print(f"Saved {path}, {out / 'oof_pooled.parquet'}")


if __name__ == "__main__":
    main()
