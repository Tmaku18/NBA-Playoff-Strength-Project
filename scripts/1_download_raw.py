"""Script 1: Download raw NBA game logs.

What this does:
- Fetches player and team game logs from nba_api (regular season and playoffs).
- Writes files to data/raw/ (e.g., player_logs_2023_2024.parquet).
- Only downloads missing files to save time.
- Merges raw hashes into data/manifest.json (preserves DB build metadata).
- Retries any missing playoff files once.

Run this first before building the database (script 2)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(cfg.get("seasons", {}).keys())

    from src.data.manifest_utils import compute_raw_hashes, merge_manifest_raw
    from src.data.nba_api_client import fetch_season_logs

    # Regular season: only fetch when file is missing
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for kind, ext in [("T", "parquet"), ("P", "parquet")]:
            stem = "team_logs" if kind == "T" else "player_logs"
            path = raw_dir / f"{stem}_{y1}_{y2}.{ext}"
            if path.exists():
                continue
            try:
                fetch_season_logs(season, raw_dir, kind=kind, use_cache=False, cache_fmt=ext)
            except Exception as e:
                print(f"Skip {season} {kind}: {e}")

    # Playoffs: same as above but for playoff game logs; then retry any still-missing files.
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for kind, ext in [("T", "parquet"), ("P", "parquet")]:
            stem = "playoffs_team_logs" if kind == "T" else "playoffs_player_logs"
            path = raw_dir / f"{stem}_{y1}_{y2}.{ext}"
            if path.exists():
                continue
            try:
                fetch_season_logs(
                    season, raw_dir, kind=kind, use_cache=False, cache_fmt=ext,
                    season_type="Playoffs",
                )
            except Exception as e:
                print(f"Skip playoffs {season} {kind}: {e}")

    missing_playoff = []
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for stem in ("playoffs_team_logs", "playoffs_player_logs"):
            path = raw_dir / f"{stem}_{y1}_{y2}.parquet"
            if not path.exists():
                missing_playoff.append((season, stem))
    if missing_playoff:
        print("Retrying missing playoff files...", flush=True)
        for season, stem in missing_playoff:
            kind = "T" if "team" in stem else "P"
            y1, y2 = season.split("-")[0], season.split("-")[1]
            path = raw_dir / f"{stem}_{y1}_{y2}.parquet"
            try:
                fetch_season_logs(
                    season, raw_dir, kind=kind, use_cache=False, cache_fmt="parquet",
                    season_type="Playoffs",
                )
                if path.exists():
                    print(f"  Downloaded {path.name}")
            except Exception as e:
                print(f"  Failed {path.name}: {e}")
        still_missing = []
        for s, st in missing_playoff:
            y1, y2 = s.split("-")[0], s.split("-")[1]
            if not (raw_dir / f"{st}_{y1}_{y2}.parquet").exists():
                still_missing.append(f"{st}_{y1}_{y2}.parquet")
        if still_missing:
            print(
                f"Warning: {len(still_missing)} playoff file(s) still missing "
                f"(API may not have data for that season): {still_missing}",
                flush=True,
            )

    manifest_path = ROOT / "data" / "manifest.json"
    raw_hashes = compute_raw_hashes(raw_dir, seasons)
    merge_manifest_raw(manifest_path, raw_hashes, download_timestamp=str(time.time()))
    print(f"Updated {manifest_path} (raw hashes merged; DB metadata preserved)")


if __name__ == "__main__":
    main()
