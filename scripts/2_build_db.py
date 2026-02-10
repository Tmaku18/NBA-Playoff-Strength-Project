"""Script 2: Build DuckDB database from raw game logs.

What this does:
- Loads raw parquet/csv files from data/raw/ into a DuckDB database.
- Includes both regular season and playoff data.
- Skips rebuild if raw file hashes are unchanged and DB already exists.
- Updates data/manifest.json with processed DB hash and raw hashes.

Run after script 1. Required before training (scripts 3, 4, etc.)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _hash_if_exists(path: Path) -> str | None:
    """Return SHA256 hex digest of file if it exists, else None."""
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _current_raw_hashes(raw_dir: Path, seasons: list[str]) -> dict[str, str]:
    """Compute hashes for all raw files (regular + playoff) to detect changes."""
    out: dict[str, str] = {}
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for stem, ext in [
            ("team_logs", "parquet"), ("player_logs", "parquet"),
            ("playoffs_team_logs", "parquet"), ("playoffs_player_logs", "parquet"),
        ]:
            for suffix in (".parquet", ".csv"):
                path = raw_dir / f"{stem}_{y1}_{y2}{suffix}"
                h = _hash_if_exists(path)
                if h is not None:
                    out[path.name] = h
    return out


def main():
    import sys
    sys.path.insert(0, str(ROOT))

    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw"])
    db_path = Path(cfg["paths"]["db"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    seasons = list(cfg.get("seasons", {}).keys())
    skip_if_exists = cfg.get("build_db", {}).get("skip_if_exists", False)

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Skip rebuild if raw files unchanged and DB exists
    current_raw = _current_raw_hashes(raw_dir, seasons)
    stored_raw = manifest.get("raw") or {}
    raw_unchanged = current_raw == stored_raw and len(current_raw) > 0
    if raw_unchanged and db_path.exists():
        print("Raw files unchanged; skipping DB rebuild.")
        return

    from src.data.db_loader import load_playoff_into_db, load_raw_into_db

    if skip_if_exists and db_path.exists():
        print(f"Skipping main build (DB exists and build_db.skip_if_exists=true): {db_path}")
        load_playoff_into_db(raw_dir, db_path, seasons=seasons)
    else:
        load_raw_into_db(raw_dir, db_path, seasons=seasons)
        load_playoff_into_db(raw_dir, db_path, seasons=seasons)

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    # Update manifest: store hash of the DB file and preserve or recompute raw file hashes.
    if db_path.exists():
        manifest["processed"] = hashlib.sha256(db_path.read_bytes()).hexdigest()
    # If script 1 was not run, we can still hash whatever raw files exist for the manifest.
    if "raw" not in manifest or not manifest["raw"]:
        manifest["raw"] = {}
        for p in (raw_dir).glob("*.parquet"):
            manifest["raw"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (raw_dir).glob("*.csv"):
            manifest["raw"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        manifest["db_path"] = str(db_path.relative_to(ROOT))
    except ValueError:
        manifest["db_path"] = str(db_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Built {db_path}, updated {manifest_path}")


if __name__ == "__main__":
    main()
