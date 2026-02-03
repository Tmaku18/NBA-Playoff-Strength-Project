"""Tests for src.features.build_roster_set."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_roster_set import (
    build_roster_set,
    get_roster_as_of_date,
    hash_trick_index,
    latest_team_map_as_of,
)


def test_hash_trick_index_in_range():
    for pid in [1, "101", 99999]:
        idx = hash_trick_index(pid, 500)
        assert 0 <= idx < 500


def test_hash_trick_index_deterministic():
    assert hash_trick_index(42, 100) == hash_trick_index(42, 100)


def test_get_roster_as_of_date_empty_pgl():
    pgl = pd.DataFrame(columns=["game_date", "player_id", "team_id", "min"])
    out = get_roster_as_of_date(pgl, 1, "2024-06-01")
    assert out.empty
    assert list(out.columns) == ["player_id", "total_min", "rank"]


def test_get_roster_as_of_date_filters_by_team_and_date(sample_pgl):
    out = get_roster_as_of_date(sample_pgl, 1, "2024-01-12")
    assert not out.empty
    assert "player_id" in out.columns and "total_min" in out.columns and "rank" in out.columns
    # Only games before 2024-01-12 for team 1: player 101 has 30+28+25 min, 102 has 25
    assert set(out["player_id"].tolist()) <= {101, 102}


def test_latest_team_map_as_of_empty():
    pgl = pd.DataFrame(columns=["game_date", "player_id", "team_id"])
    out = latest_team_map_as_of(pgl, "2024-06-01")
    assert out == {}


def test_latest_team_map_as_of_keeps_last_team(sample_pgl):
    out = latest_team_map_as_of(sample_pgl, "2024-01-20")
    assert 101 in out and 102 in out
    assert out[101] == 1
    assert out[102] == 2


def test_build_roster_set_shape(sample_roster_df, sample_player_stats):
    from src.features.rolling import PLAYER_STAT_COLS_WITH_ON_OFF

    emb, rows, minutes, mask = build_roster_set(
        sample_roster_df,
        sample_player_stats,
        n_pad=15,
        stat_cols=PLAYER_STAT_COLS_WITH_ON_OFF,
        num_embeddings=500,
        team_continuity_scalar=0.75,
    )
    assert len(emb) == 15
    assert len(rows) == 15
    assert len(minutes) == 15
    assert len(mask) == 15
    assert sum(1 for m in mask if not m) == 2  # 2 valid players
    assert sum(1 for m in mask if m) == 13  # padding
    # Each row has 18 base (L10+L30 incl ts_pct, usage) + 2 on_off + team continuity + 3 usage features = 24
    assert len(rows[0]) == 24


def test_build_roster_set_padding_index(sample_roster_df, sample_player_stats):
    emb, _, _, mask = build_roster_set(
        sample_roster_df,
        sample_player_stats,
        n_pad=15,
        num_embeddings=500,
    )
    # Padding uses num_embeddings as index (distinct from hash range)
    for i in range(2, 15):
        assert emb[i] == 500
        assert mask[i] is True
