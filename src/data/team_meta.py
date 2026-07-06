"""Canonical NBA franchise metadata keyed by stable NBA ``team_id``.

The raw game logs store the *historical* abbreviation/name for each franchise
(e.g. the oldest season in the dataset), so a team that relocated or rebranded
(Seattle SuperSonics -> Oklahoma City Thunder, Vancouver -> Memphis, New Jersey
Nets -> Brooklyn, etc.) ends up with an out-of-date abbreviation and a ``NULL``
conference. That breaks conference-based list building, which falls back to a
modern-abbreviation lookup and silently drops the renamed franchises.

This module provides a single source of truth (modern abbreviation, display
name, conference) keyed by the immutable ``team_id`` so the ``teams`` table is
always complete and correct for all 30 franchises.
"""
from __future__ import annotations

# team_id -> (modern abbreviation, display name, conference)
TEAM_META: dict[int, tuple[str, str, str]] = {
    1610612737: ("ATL", "Atlanta Hawks", "E"),
    1610612738: ("BOS", "Boston Celtics", "E"),
    1610612739: ("CLE", "Cleveland Cavaliers", "E"),
    1610612740: ("NOP", "New Orleans Pelicans", "W"),
    1610612741: ("CHI", "Chicago Bulls", "E"),
    1610612742: ("DAL", "Dallas Mavericks", "W"),
    1610612743: ("DEN", "Denver Nuggets", "W"),
    1610612744: ("GSW", "Golden State Warriors", "W"),
    1610612745: ("HOU", "Houston Rockets", "W"),
    1610612746: ("LAC", "LA Clippers", "W"),
    1610612747: ("LAL", "Los Angeles Lakers", "W"),
    1610612748: ("MIA", "Miami Heat", "E"),
    1610612749: ("MIL", "Milwaukee Bucks", "E"),
    1610612750: ("MIN", "Minnesota Timberwolves", "W"),
    1610612751: ("BKN", "Brooklyn Nets", "E"),
    1610612752: ("NYK", "New York Knicks", "E"),
    1610612753: ("ORL", "Orlando Magic", "E"),
    1610612754: ("IND", "Indiana Pacers", "E"),
    1610612755: ("PHI", "Philadelphia 76ers", "E"),
    1610612756: ("PHX", "Phoenix Suns", "W"),
    1610612757: ("POR", "Portland Trail Blazers", "W"),
    1610612758: ("SAC", "Sacramento Kings", "W"),
    1610612759: ("SAS", "San Antonio Spurs", "W"),
    1610612760: ("OKC", "Oklahoma City Thunder", "W"),
    1610612761: ("TOR", "Toronto Raptors", "E"),
    1610612762: ("UTA", "Utah Jazz", "W"),
    1610612763: ("MEM", "Memphis Grizzlies", "W"),
    1610612764: ("WAS", "Washington Wizards", "E"),
    1610612765: ("DET", "Detroit Pistons", "E"),
    1610612766: ("CHA", "Charlotte Hornets", "E"),
}


def team_conference(team_id: int) -> str | None:
    meta = TEAM_META.get(int(team_id))
    return meta[2] if meta else None


def team_abbreviation(team_id: int, fallback: str | None = None) -> str | None:
    meta = TEAM_META.get(int(team_id))
    return meta[0] if meta else fallback


def team_name(team_id: int, fallback: str | None = None) -> str | None:
    meta = TEAM_META.get(int(team_id))
    return meta[1] if meta else fallback
