"""Read/write data/manifest.json without clobbering build-db metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str | None:
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def raw_files_for_season(season: str) -> list[str]:
    y1, y2 = season.split("-")[0], season.split("-")[1]
    return [
        f"team_logs_{y1}_{y2}.parquet",
        f"player_logs_{y1}_{y2}.parquet",
        f"playoffs_team_logs_{y1}_{y2}.parquet",
        f"playoffs_player_logs_{y1}_{y2}.parquet",
    ]


def compute_raw_hashes(raw_dir: Path, seasons: list[str]) -> dict[str, str]:
    """SHA256 for each existing raw parquet/csv (regular + playoff) in season list."""
    out: dict[str, str] = {}
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for stem in ("team_logs", "player_logs", "playoffs_team_logs", "playoffs_player_logs"):
            for suffix in (".parquet", ".csv"):
                path = raw_dir / f"{stem}_{y1}_{y2}{suffix}"
                h = hash_file(path)
                if h is not None:
                    out[path.name] = h
    return out


def seasons_with_changed_raw(
    current_raw: dict[str, str],
    stored_raw: dict[str, str],
    seasons: list[str],
) -> list[str]:
    """Return season keys whose raw file hashes differ from manifest (or are new)."""
    changed: list[str] = []
    for season in seasons:
        for fname in raw_files_for_season(season):
            cur = current_raw.get(fname)
            if cur is None:
                continue
            if stored_raw.get(fname) != cur:
                changed.append(season)
                break
    return changed


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def merge_manifest_raw(
    manifest_path: Path,
    raw_hashes: dict[str, str],
    *,
    download_timestamp: str | None = None,
) -> dict:
    """Update raw hashes in manifest; preserve processed / db_path from build step."""
    manifest = load_manifest(manifest_path)
    manifest["raw"] = raw_hashes
    ts = manifest.setdefault("timestamps", {})
    if download_timestamp is not None:
        ts["download"] = download_timestamp
    save_manifest(manifest_path, manifest)
    return manifest


def update_manifest_after_build(
    manifest_path: Path,
    *,
    raw_hashes: dict[str, str],
    db_path: Path,
    root: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    manifest["raw"] = raw_hashes
    if db_path.exists():
        manifest["processed"] = hashlib.sha256(db_path.read_bytes()).hexdigest()
    try:
        manifest["db_path"] = str(db_path.relative_to(root))
    except ValueError:
        manifest["db_path"] = str(db_path)
    save_manifest(manifest_path, manifest)
