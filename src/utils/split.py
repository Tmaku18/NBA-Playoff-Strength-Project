"""Train/test split logic: seasons, cutoff, or date-based fallback. Used by script 3, 4, 6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _date_to_season(as_of_date: str, seasons_cfg: dict[str, Any]) -> str | None:
    """Map as_of_date (YYYY-MM-DD) to season key if within any season range."""
    try:
        d = pd.to_datetime(as_of_date).date()
    except Exception:
        return None
    for skey, rng in (seasons_cfg or {}).items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return skey
    return None


def compute_split(
    lists: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Split lists into train_lists and test_lists using config rules.
    Returns (train_lists, test_lists, split_info).
    Precedence: train_seasons/test_seasons > train_test_cutoff > train_frac fallback.
    """
    training_cfg = config.get("training") or {}
    seasons_cfg = config.get("seasons") or {}
    train_seasons = training_cfg.get("train_seasons")
    test_seasons = training_cfg.get("test_seasons")
    cutoff = training_cfg.get("train_test_cutoff")
    train_frac = float(training_cfg.get("train_frac", 0.75))

    train_lists: list[dict[str, Any]] = []
    test_lists: list[dict[str, Any]] = []
    split_mode: str

    if train_seasons and test_seasons:
        train_set = set(train_seasons)
        test_set = set(test_seasons)
        if train_set & test_set:
            raise ValueError(
                "train_seasons and test_seasons must not overlap. "
                f"train_seasons={train_seasons}, test_seasons={test_seasons}"
            )
        for lst in lists:
            season = _date_to_season(lst.get("as_of_date", ""), seasons_cfg)
            if season in train_set:
                train_lists.append(lst)
            elif season in test_set:
                test_lists.append(lst)
        split_mode = "seasons"
    elif cutoff:
        try:
            cutoff_d = pd.to_datetime(cutoff).date()
        except Exception:
            raise ValueError(f"train_test_cutoff must be a valid date string, got: {cutoff!r}")
        for lst in lists:
            try:
                d = pd.to_datetime(lst.get("as_of_date", "")).date()
                if d <= cutoff_d:
                    train_lists.append(lst)
                else:
                    test_lists.append(lst)
            except Exception:
                continue
        split_mode = "cutoff"
    else:
        unique_dates = sorted(set(lst.get("as_of_date", "") for lst in lists if lst.get("as_of_date")))
        if not unique_dates:
            return [], [], {"split_mode": "fallback", "n_train_dates": 0, "n_test_dates": 0}
        n = len(unique_dates)
        train_n = max(1, int(train_frac * n))
        if train_n >= n:
            train_n = max(1, n - 1)
        train_dates_set = set(unique_dates[:train_n])
        test_dates_set = set(unique_dates[train_n:])
        for lst in lists:
            if lst.get("as_of_date", "") in train_dates_set:
                train_lists.append(lst)
            elif lst.get("as_of_date", "") in test_dates_set:
                test_lists.append(lst)
        split_mode = "fallback"

    train_dates = sorted(set(lst.get("as_of_date", "") for lst in train_lists if lst.get("as_of_date")))
    test_dates = sorted(set(lst.get("as_of_date", "") for lst in test_lists if lst.get("as_of_date")))
    split_info: dict[str, Any] = {
        "split_mode": split_mode,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "train_frac": train_frac,
        "n_train_lists": len(train_lists),
        "n_test_lists": len(test_lists),
        "n_train_dates": len(train_dates),
        "n_test_dates": len(test_dates),
    }
    return train_lists, test_lists, split_info


def write_split_info(split_info: dict[str, Any], output_dir: Path) -> Path:
    """Write split_info.json to output_dir. Returns path."""
    path = Path(output_dir) / "split_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    return path


def load_split_info(output_dir: Path) -> dict[str, Any]:
    """Load split_info.json from output_dir. Raises FileNotFoundError if missing."""
    path = Path(output_dir) / "split_info.json"
    if not path.exists():
        raise FileNotFoundError(
            f"split_info.json not found at {path}. Run script 3 (train Model A) first to create the 75/25 split."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
