"""Tests for src.evaluation.playoffs."""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.playoffs import (
    MIN_PLAYOFF_TEAMS,
    _filter_by_date,
    _to_date,
    get_playoff_finish_label,
    get_playoff_wins,
)


def test_to_date_none():
    assert _to_date(None) is None


def test_to_date_string():
    d = _to_date("2024-04-15")
    assert d is not None
    assert d.year == 2024 and d.month == 4 and d.day == 15


def test_filter_by_date_empty_df():
    df = pd.DataFrame(columns=["game_date"])
    out = _filter_by_date(df, date_col="game_date", season_start="2024-01-01", season_end="2024-06-01")
    assert out.empty


def test_filter_by_date_missing_col():
    df = pd.DataFrame({"other": [1, 2, 3]})
    out = _filter_by_date(df, date_col="game_date", season_start="2024-01-01", season_end="2024-06-01")
    assert len(out) == 3 and "game_date" not in out.columns


def test_filter_by_date_in_range():
    df = pd.DataFrame({
        "game_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
    })
    out = _filter_by_date(
        df,
        date_col="game_date",
        season_start="2024-01-01",
        season_end="2024-04-15",
    )
    assert len(out) == 2
    assert "2024-07-01" not in out["game_date"].astype(str).values


def test_get_playoff_wins_empty():
    pg = pd.DataFrame(columns=["game_id", "game_date"])
    pt = pd.DataFrame(columns=["game_id", "team_id", "wl"])
    out = get_playoff_wins(pg, pt, "2023-24")
    assert out == {}


def test_get_playoff_wins_counts_wins():
    pg = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": ["2024-04-20", "2024-04-22"],
    })
    pt = pd.DataFrame({
        "game_id": ["g1", "g1", "g2"],
        "team_id": [1, 2, 1],
        "wl": ["W", "L", "W"],
    })
    out = get_playoff_wins(
        pg, pt, "2023-24",
        season_start="2024-04-01",
        season_end="2024-06-01",
    )
    assert out[1] == 2
    assert out[2] == 0


def test_get_playoff_finish_label():
    wins = {1: 16, 2: 12, 3: 0}
    assert "Champion" in get_playoff_finish_label(wins, 1)
    assert "Runner-Up" in get_playoff_finish_label(wins, 2) or "Finals" in get_playoff_finish_label(wins, 2)
    assert "Did not qualify" in get_playoff_finish_label(wins, 3) or "qualify" in get_playoff_finish_label(wins, 3).lower()


def test_min_playoff_teams_constant():
    assert MIN_PLAYOFF_TEAMS == 16
