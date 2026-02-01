"""Monte Carlo championship simulation: game/series/season simulator."""
from __future__ import annotations

import random
from typing import Any

import numpy as np


def prob_win_game(rating_a: float, rating_b: float, hca: float, rating_scale: float = 400.0) -> float:
    """P(A beats B) = 1 / (1 + 10^(-(Rating_A - Rating_B + HCA)/400))."""
    exponent = -(rating_a - rating_b + hca) / rating_scale
    return 1.0 / (1.0 + 10**exponent)


def simulate_series(
    rating_high: float,
    rating_low: float,
    hca: float,
    *,
    n_sims: int = 1000,
    rating_scale: float = 400.0,
) -> float:
    """
    Simulate 2-2-1-1-1 format. Higher seed has home for games 1,2,6,7; away 3,4,5.
    Returns P(higher seed advances).
    """
    home_order = [1, 1, 0, 0, 1, 0, 1]  # 1 = higher seed home
    wins = 0
    for _ in range(n_sims):
        h_wins, l_wins = 0, 0
        for i, higher_home in enumerate(home_order):
            if h_wins == 4 or l_wins == 4:
                break
            if higher_home:
                p = prob_win_game(rating_high, rating_low, hca, rating_scale)
            else:
                p = prob_win_game(rating_high, rating_low, -hca, rating_scale)
            if random.random() < p:
                h_wins += 1
            else:
                l_wins += 1
        if h_wins == 4:
            wins += 1
    return wins / n_sims


def simulate_championship(
    team_ratings: dict[int, float],
    team_to_seed: dict[int, int],
    team_to_conference: dict[int, str],
    *,
    hca: float = 100.0,
    rating_scale: float = 400.0,
    n_seasons: int = 10000,
    n_series: int = 1000,
) -> dict[int, float]:
    """
    Simulate full bracket. East 8 seeds, West 8 seeds, finals.
    team_ratings: team_id -> rating (SRS or ensemble score).
    team_to_seed: team_id -> 1-8 within conference.
    team_to_conference: team_id -> 'E' or 'W'.
    Returns team_id -> championship probability.
    """
    east = [tid for tid, c in team_to_conference.items() if c == "E"]
    west = [tid for tid, c in team_to_conference.items() if c == "W"]
    east = sorted(east, key=lambda t: team_to_seed.get(t, 9))
    west = sorted(west, key=lambda t: team_to_seed.get(t, 9))
    if len(east) < 8 or len(west) < 8:
        return {tid: 0.0 for tid in team_ratings}

    def _advance(bracket: list[int]) -> int:
        """Single elimination: [1,8,4,5,2,7,3,6] -> winner."""
        if len(bracket) == 1:
            return bracket[0]
        next_round = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            ra = team_ratings.get(a, 0.0)
            rb = team_ratings.get(b, 0.0)
            higher, lower = (a, b) if ra >= rb else (b, a)
            p = simulate_series(
                max(ra, rb), min(ra, rb),
                hca, n_sims=n_series, rating_scale=rating_scale,
            )
            winner = higher if random.random() < p else lower
            next_round.append(winner)
        return _advance(next_round)

    # NBA bracket: 1v8, 4v5, 2v7, 3v6
    east_bracket = [east[0], east[7], east[3], east[4], east[1], east[6], east[2], east[5]]
    west_bracket = [west[0], west[7], west[3], west[4], west[1], west[6], west[2], west[5]]

    counts: dict[int, int] = {tid: 0 for tid in team_ratings}
    for _ in range(n_seasons):
        east_winner = _advance(east_bracket)
        west_winner = _advance(west_bracket)
        ra = team_ratings.get(east_winner, 0.0)
        rb = team_ratings.get(west_winner, 0.0)
        p = prob_win_game(ra, rb, hca, rating_scale)
        champ = east_winner if random.random() < p else west_winner
        counts[champ] = counts.get(champ, 0) + 1

    return {tid: c / n_seasons for tid, c in counts.items()}


def championship_probabilities_from_scores(
    team_ids: list[int],
    scores: np.ndarray,
    team_id_to_conference: dict[int, str],
    team_id_to_seed: dict[int, int] | None = None,
    *,
    hca: float = 100.0,
    rating_scale: float = 400.0,
    n_seasons: int = 10000,
) -> np.ndarray:
    """
    Convert ensemble scores to championship probabilities via Monte Carlo.
    team_id_to_seed: optional; if None, derive from score rank within conference.
    Returns array of probabilities aligned with team_ids.
    """
    scores = np.asarray(scores)
    if len(scores) != len(team_ids):
        return np.ones(len(team_ids)) / len(team_ids)
    team_ratings = {tid: float(s) for tid, s in zip(team_ids, scores)}
    east = [t for t in team_ids if team_id_to_conference.get(t) == "E"]
    west = [t for t in team_ids if team_id_to_conference.get(t) == "W"]
    if not east or not west:
        return np.ones(len(team_ids)) / len(team_ids)
    east_sorted = sorted(east, key=lambda t: -team_ratings.get(t, 0))
    west_sorted = sorted(west, key=lambda t: -team_ratings.get(t, 0))
    team_to_seed = {}
    for i, t in enumerate(east_sorted[:8]):
        team_to_seed[t] = i + 1
    for i, t in enumerate(west_sorted[:8]):
        team_to_seed[t] = i + 1
    if team_id_to_seed:
        team_to_seed.update(team_id_to_seed)
    team_to_conference = {t: team_id_to_conference.get(t, "E") for t in team_ids}
    probs = simulate_championship(
        team_ratings, team_to_seed, team_to_conference,
        hca=hca, rating_scale=rating_scale, n_seasons=n_seasons,
    )
    return np.array([probs.get(tid, 0.0) for tid in team_ids])
