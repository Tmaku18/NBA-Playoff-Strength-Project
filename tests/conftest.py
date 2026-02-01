"""Pytest fixtures and path setup so `src` is importable when run from project root."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_pgl():
    """Minimal player_game_logs-style DataFrame for roster tests."""
    import pandas as pd

    return pd.DataFrame({
        "game_date": ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"],
        "player_id": [101, 101, 102, 102],
        "team_id": [1, 1, 1, 2],
        "min": [30, 28, 25, 32],
    })


@pytest.fixture
def sample_roster_df():
    """Minimal roster DataFrame (as returned by get_roster_as_of_date)."""
    import pandas as pd

    return pd.DataFrame({
        "player_id": [101, 102],
        "total_min": [100.0, 80.0],
        "rank": [0, 1],
    })


@pytest.fixture
def sample_player_stats():
    """Minimal player_stats DataFrame for build_roster_set."""
    import pandas as pd

    from src.features.rolling import PLAYER_STAT_COLS_WITH_ON_OFF

    cols = [c for c in PLAYER_STAT_COLS_WITH_ON_OFF if c != "player_id"]
    return pd.DataFrame({
        "player_id": [101, 102],
        **{c: [10.0, 8.0] for c in cols},
    })
