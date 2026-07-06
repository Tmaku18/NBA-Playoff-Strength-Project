"""Script 2: Build DuckDB database from raw game logs.

What this does:
- Loads raw parquet/csv files from data/raw/ into a DuckDB database.
- Includes both regular season and playoff data.
- Skips work when raw file hashes are unchanged and DB already exists.
- Incrementally updates only seasons whose raw files changed (no full wipe).
- Updates data/manifest.json with processed DB hash and raw hashes.

Run after script 1. Required before training (scripts 3, 4, etc.)."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT))

    from src.data.db import get_connection
    from src.data.db_loader import db_has_regular_games, load_playoff_into_db, load_raw_into_db
    from src.data.manifest_utils import (
        compute_raw_hashes,
        load_manifest,
        seasons_with_changed_raw,
        update_manifest_after_build,
    )

    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw"])
    db_path = Path(cfg["paths"]["db"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    seasons = list(cfg.get("seasons", {}).keys())
    force_full = bool(cfg.get("build_db", {}).get("force_full_rebuild", False))

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = load_manifest(manifest_path)
    stored_raw = manifest.get("raw") or {}
    current_raw = compute_raw_hashes(raw_dir, seasons)

    changed_seasons = seasons_with_changed_raw(current_raw, stored_raw, seasons)
    db_exists = db_path.exists()

    if db_exists and not force_full and not changed_seasons:
        con = get_connection(db_path, read_only=True)
        try:
            has_data = db_has_regular_games(con)
        finally:
            con.close()
        if has_data:
            print("Raw files unchanged; skipping DB rebuild.", flush=True)
            update_manifest_after_build(
                manifest_path,
                raw_hashes=current_raw,
                db_path=db_path,
                root=ROOT,
            )
            return
        print("DB file exists but is empty; rebuilding all seasons.", flush=True)
        changed_seasons = seasons
    elif db_exists and not force_full and changed_seasons:
        print(
            f"Incremental DB update for {len(changed_seasons)} season(s): {', '.join(changed_seasons)}",
            flush=True,
        )
        load_raw_into_db(raw_dir, db_path, seasons=changed_seasons, incremental=True)
        load_playoff_into_db(raw_dir, db_path, seasons=changed_seasons, incremental=True)
        update_manifest_after_build(
            manifest_path,
            raw_hashes=current_raw,
            db_path=db_path,
            root=ROOT,
        )
        print(f"Updated {db_path}, refreshed {manifest_path}", flush=True)
        return
    elif not db_exists:
        print(f"Building new DB for {len(seasons)} season(s)...", flush=True)
    else:
        print("force_full_rebuild=true; rebuilding all seasons from scratch.", flush=True)

    load_raw_into_db(raw_dir, db_path, seasons=seasons, incremental=False)
    load_playoff_into_db(raw_dir, db_path, seasons=seasons, incremental=False)
    update_manifest_after_build(
        manifest_path,
        raw_hashes=current_raw,
        db_path=db_path,
        root=ROOT,
    )
    print(f"Built {db_path}, updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
