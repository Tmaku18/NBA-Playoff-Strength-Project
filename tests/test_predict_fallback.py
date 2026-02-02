"""Test inference fallback: when attention is all-zero, primary_contributors empty and contributors_are_fallback true."""
from __future__ import annotations

import pytest

from src.inference.predict import predict_teams


def test_all_zero_attention_empty_contributors_and_fallback_true():
    """When attention is all zero, output has empty primary_contributors and contributors_are_fallback true."""
    team_ids = [1, 2, 3]
    team_names = ["Team A", "Team B", "Team C"]
    attention_by_team = {
        1: [("Player X", 0.5), ("Player Y", 0.3)],
        2: [],
        3: [],
    }
    attention_fallback_by_team = {
        2: True,
        3: True,
    }
    out = predict_teams(
        team_ids,
        team_names,
        model_a_scores=[1.0, 0.5, 0.2],
        attention_by_team=attention_by_team,
        attention_fallback_by_team=attention_fallback_by_team,
    )
    assert len(out) == 3
    by_tid = {r["team_id"]: r for r in out}
    assert by_tid[1]["roster_dependence"]["primary_contributors"] == [
        {"player": "Player X", "attention_weight": 0.5},
        {"player": "Player Y", "attention_weight": 0.3},
    ]
    assert by_tid[1]["roster_dependence"]["contributors_are_fallback"] is False
    assert by_tid[2]["roster_dependence"]["primary_contributors"] == []
    assert by_tid[2]["roster_dependence"]["contributors_are_fallback"] is True
    assert by_tid[3]["roster_dependence"]["primary_contributors"] == []
    assert by_tid[3]["roster_dependence"]["contributors_are_fallback"] is True
