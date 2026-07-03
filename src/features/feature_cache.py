"""Shared feature cache for build_team_context_as_of_dates output.

Used by script 4, 5b, 4c, plot scripts, and inference to avoid recomputing
team-context features when config, DB, and team_dates are unchanged.
Enable via config paths.feature_cache (e.g. data/processed/feature_cache).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def get_feature_cache_dir(config: dict, root: Path) -> Path | None:
    """Return resolved feature cache directory, or None if disabled."""
    raw = config.get("paths", {}).get("feature_cache")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("null", "")):
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return p


def compute_feature_cache_key(config: dict, db_path: Path, team_dates_hash: str) -> str:
    """Compute cache key from config + DB + team_dates hash (same logic as script 4)."""
    model_b = config.get("model_b", {})
    key_data = {
        "include_features": tuple(model_b.get("include_features") or []),
        "exclude_features": tuple(model_b.get("exclude_features") or []),
        "elo": bool(config.get("elo", {}).get("enabled", False)),
        "massey": bool(config.get("massey", {}).get("enabled", False)),
        "team_rolling": bool(config.get("team_rolling", {}).get("enabled", True)),
        "sos_srs": bool(config.get("sos_srs", {}).get("enabled", False)),
        "motivation": bool(config.get("motivation", {}).get("enabled", False)),
        "injury": bool(config.get("injury", {}).get("enabled", False)),
        "team_dates_hash": team_dates_hash,
        "db": str(db_path.resolve()),
    }
    if db_path.exists():
        st = db_path.stat()
        key_data["db_mtime"] = st.st_mtime
        key_data["db_size"] = st.st_size
    js = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(js.encode()).hexdigest()[:20]


def load_feature_cache(cache_dir: Path, cache_key: str) -> pd.DataFrame | None:
    """Load cached team-context DataFrame if present and valid."""
    path = cache_dir / f"{cache_key}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "team_id" in df.columns and "as_of_date" in df.columns:
            df["team_id"] = df["team_id"].astype(int)
            df["as_of_date"] = df["as_of_date"].astype(str)
            return df
    except Exception:
        pass
    return None


def save_feature_cache(cache_dir: Path, cache_key: str, df: pd.DataFrame) -> None:
    """Save team-context DataFrame to cache."""
    if df.empty:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / f"{cache_key}.parquet", index=False)
    except Exception:
        pass
