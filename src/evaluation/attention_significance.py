"""Bootstrap significance for per-player attention weights.

Resamples teams with replacement to get a distribution of mean attention per player;
reports 95% CI and p-value (proportion of bootstrap means <= null_threshold) so
player importance is interpretable.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def attention_bootstrap_over_teams(
    teams: list[dict[str, Any]],
    *,
    B: int = 2000,
    seed: int = 42,
    null_threshold: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Bootstrap over teams to get per-player attention CI and p-value.

    Each team has roster_dependence.primary_contributors = [{"player": str, "attention_weight": float}, ...].
    We resample teams with replacement B times; each time we compute per-player mean attention
    (over all teams in the sample where that player appears). Then for each player we have
    B bootstrap means → 95% CI and p_value = proportion of bootstrap means <= null_threshold.

    Returns dict[player_id_or_name, {mean, ci_low, ci_high, p_value, n_teams_appeared}].
    """
    rng = np.random.default_rng(seed)
    n_teams = len(teams)
    if n_teams < 2:
        return {}

    # Collect (team_idx, player, weight) triples
    rows: list[tuple[int, str, float]] = []
    for i, t in enumerate(teams):
        rd = t.get("roster_dependence") or {}
        contrib = rd.get("primary_contributors") or []
        for c in contrib:
            player = c.get("player")
            w = c.get("attention_weight")
            if player is not None and isinstance(w, (int, float)) and math.isfinite(w):
                rows.append((i, str(player), float(w)))

    if not rows:
        return {}

    # For each bootstrap b, resample team indices, then from rows take (i, player, w) where i in resampled.
    # For each player, collect weights from that bootstrap sample, then mean. Store boot_means[player] = list of B means.
    boot_means: dict[str, list[float]] = {}
    for b in range(B):
        idx = rng.integers(0, n_teams, size=n_teams)
        team_set = set(idx.tolist())
        by_player: dict[str, list[float]] = {}
        for i, player, w in rows:
            if i not in team_set:
                continue
            by_player.setdefault(player, []).append(w)
        for player, weights in by_player.items():
            boot_means.setdefault(player, []).append(float(np.mean(weights)))

    out: dict[str, dict[str, float]] = {}
    for player, means in boot_means.items():
        if len(means) < 2:
            continue
        arr = np.array(means, dtype=np.float64)
        mean_val = float(np.mean(arr))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        p_value = float(np.mean(arr <= null_threshold))
        # n_teams_appeared: count teams that have this player in primary_contributors
        n_app = sum(1 for i, t in enumerate(teams) for c in (t.get("roster_dependence") or {}).get("primary_contributors") or [] if c.get("player") == player)
        out[player] = {
            "mean_attention": mean_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "n_teams_appeared": n_app,
        }
    return out
