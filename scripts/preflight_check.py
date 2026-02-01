"""Pre-flight check: verify DB, config, and model prerequisites."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
MIN_PLAYOFF_TEAMS = 16


def _load_config() -> dict:
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _latest_run_id(outputs_dir: Path) -> str | None:
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and pattern.match(p.name):
            if (p / "predictions.json").exists():
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers:
        return None
    return f"run_{max(numbers):03d}"


def _check_seasons(config: dict, errors: list[str]) -> tuple[str | None, dict | None]:
    seasons_cfg = config.get("seasons") or {}
    if not seasons_cfg:
        errors.append("No seasons configured in config/defaults.yaml.")
        return None, None
    latest = None
    latest_end = None
    for season, rng in seasons_cfg.items():
        start = rng.get("start")
        end = rng.get("end")
        if not start or not end:
            errors.append(f"Season {season} missing start/end.")
            continue
        try:
            start_dt = datetime.fromisoformat(str(start)).date()
            end_dt = datetime.fromisoformat(str(end)).date()
        except ValueError:
            errors.append(f"Season {season} has invalid date format: start={start}, end={end}.")
            continue
        if start_dt > end_dt:
            errors.append(f"Season {season} has start after end: {start_dt} > {end_dt}.")
            continue
        if latest_end is None or end_dt > latest_end:
            latest_end = end_dt
            latest = season
    return latest, seasons_cfg.get(latest) if latest else None


def _check_raw_data(raw_path: Path, errors: list[str], warnings: list[str]) -> None:
    if not raw_path.exists():
        errors.append(f"Raw data directory missing: {raw_path}")
        return
    has_files = any(
        p.is_file() and p.name != ".gitkeep"
        for p in raw_path.rglob("*")
    )
    if not has_files:
        warnings.append(f"Raw data directory appears empty: {raw_path}")


def _check_db(db_path: Path, latest_season_range: dict | None, errors: list[str], warnings: list[str]) -> None:
    if not db_path.exists():
        errors.append(f"Database missing: {db_path}")
        return
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        required = {"games", "team_game_logs", "player_game_logs", "teams", "players"}
        missing = required - tables
        if missing:
            errors.append(f"Database missing required tables: {sorted(missing)}")

        playoff_tables = {"playoff_games", "playoff_team_game_logs"}
        if not playoff_tables.issubset(tables):
            warnings.append("Playoff tables missing; playoff metrics/ranks will be skipped.")
        elif latest_season_range:
            start = latest_season_range.get("start")
            end = latest_season_range.get("end")
            if start and end:
                count_games = con.execute(
                    "SELECT COUNT(*) FROM playoff_games WHERE game_date BETWEEN ? AND ?",
                    [str(start), str(end)],
                ).fetchone()[0]
                if count_games == 0:
                    warnings.append("No playoff_games found for the latest season range.")
                count_teams = con.execute(
                    """
                    SELECT COUNT(DISTINCT team_id)
                    FROM playoff_team_game_logs
                    WHERE game_id IN (
                        SELECT game_id FROM playoff_games WHERE game_date BETWEEN ? AND ?
                    )
                    """,
                    [str(start), str(end)],
                ).fetchone()[0]
                if count_teams < MIN_PLAYOFF_TEAMS:
                    warnings.append(
                        f"Only {count_teams} playoff teams in latest season (min {MIN_PLAYOFF_TEAMS})."
                    )
    finally:
        con.close()


def _archive_models(outputs_dir: Path, model_paths: list[Path]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = outputs_dir / "archived_models" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in model_paths:
        if p.exists():
            shutil.move(str(p), str(archive_dir / p.name))
    return archive_dir


def _check_models(outputs_dir: Path, force_retrain: bool, warnings: list[str]) -> None:
    model_files = [
        outputs_dir / "best_deep_set.pt",
        outputs_dir / "xgb_model.joblib",
        outputs_dir / "rf_model.joblib",
        outputs_dir / "meta_model.joblib",
    ]
    existing = [p for p in model_files if p.exists()]
    if force_retrain and existing:
        archive_dir = _archive_models(outputs_dir, existing)
        print(f"Archived existing models to {archive_dir}")
        return
    missing = [p.name for p in model_files if not p.exists()]
    if missing:
        warnings.append(f"Missing trained model files: {missing}")


def _check_run_id(config: dict, outputs_dir: Path, warnings: list[str]) -> None:
    run_id = config.get("inference", {}).get("run_id")
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        run_id = _latest_run_id(outputs_dir)
    else:
        run_id = str(run_id).strip()
    if run_id:
        pred_path = outputs_dir / run_id / "predictions.json"
        if not pred_path.exists():
            warnings.append(f"Predictions missing for run_id={run_id} (expected {pred_path}).")
    eval_path = outputs_dir / "eval_report.json"
    if eval_path.exists() and run_id:
        pred_path = outputs_dir / run_id / "predictions.json"
        if pred_path.exists() and eval_path.stat().st_mtime < pred_path.stat().st_mtime:
            warnings.append("eval_report.json is older than latest predictions.json; rerun 5_evaluate.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight checks before full pipeline run.")
    parser.add_argument("--force-retrain", action="store_true", help="Archive existing model files before training.")
    args = parser.parse_args()

    config = _load_config()
    errors: list[str] = []
    warnings: list[str] = []

    out_dir = _resolve_path(config.get("paths", {}).get("outputs", "outputs"))
    raw_dir = _resolve_path(config.get("paths", {}).get("raw", "data/raw"))
    db_path = _resolve_path(config.get("paths", {}).get("db", "data/processed/nba_build_run.duckdb"))

    latest_season, latest_range = _check_seasons(config, errors)
    if latest_season and latest_range:
        print(f"Latest season in config: {latest_season} ({latest_range.get('start')} -> {latest_range.get('end')})")

    _check_raw_data(raw_dir, errors, warnings)
    _check_db(db_path, latest_range, errors, warnings)
    _check_models(out_dir, args.force_retrain, warnings)
    _check_run_id(config, out_dir, warnings)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("\nPre-flight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
