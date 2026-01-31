"""Build Model A training batches from DuckDB data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.features.build_roster_set import build_roster_set, get_roster_as_of_date
from src.features.rolling import compute_rolling_stats
from src.training.build_lists import build_lists


@dataclass
class ModelABatchConfig:
    roster_size: int
    num_embeddings: int
    rolling_windows: list[int]
    stat_base: list[str]


def _stat_cols(cfg: ModelABatchConfig) -> list[str]:
    cols: list[str] = []
    for w in cfg.rolling_windows:
        cols.extend([f"{c}_L{w}" for c in cfg.stat_base])
        cols.append(f"availability_L{w}")
    return cols


def build_player_stats_as_of(
    rolled_stats: pd.DataFrame,
    as_of_date: str,
    cfg: ModelABatchConfig,
) -> pd.DataFrame:
    if rolled_stats.empty:
        return pd.DataFrame(columns=["player_id"] + _stat_cols(cfg))
    ad = pd.to_datetime(as_of_date)
    stats = rolled_stats[rolled_stats["game_date"] < ad].copy()
    if stats.empty:
        return pd.DataFrame(columns=["player_id"] + _stat_cols(cfg))
    stats = stats.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
    cols = ["player_id"] + [c for c in _stat_cols(cfg) if c in stats.columns]
    return stats[cols].copy()


def build_batch_for_list(
    lst: dict[str, Any],
    pgl_source: pd.DataFrame,
    rolled_stats: pd.DataFrame,
    cfg: ModelABatchConfig,
    cache: dict[str, pd.DataFrame],
) -> dict[str, torch.Tensor]:
    as_of_date = lst["as_of_date"]
    if as_of_date not in cache:
        cache[as_of_date] = build_player_stats_as_of(rolled_stats, as_of_date, cfg)
    player_stats = cache[as_of_date]

    team_ids = lst["team_ids"]
    rel = lst["win_rates"]
    K = len(team_ids)
    if K < 2:
        raise ValueError("List must have at least 2 teams")

    emb_list = []
    stats_list = []
    minutes_list = []
    mask_list = []
    for team_id in team_ids:
        roster_df = get_roster_as_of_date(
            pgl_source,
            int(team_id),
            as_of_date,
            n=cfg.roster_size,
        )
        emb, stats, minutes, mask = build_roster_set(
            roster_df,
            player_stats,
            n_pad=cfg.roster_size,
            stat_cols=_stat_cols(cfg),
            num_embeddings=cfg.num_embeddings,
        )
        emb_list.append(emb)
        stats_list.append(stats)
        minutes_list.append(minutes)
        mask_list.append(mask)

    batch = {
        "embedding_indices": torch.tensor([emb_list], dtype=torch.long),
        "player_stats": torch.tensor([stats_list], dtype=torch.float32),
        "minutes": torch.tensor([minutes_list], dtype=torch.float32),
        "key_padding_mask": torch.tensor([mask_list], dtype=torch.bool),
        "rel": torch.tensor([rel], dtype=torch.float32),
    }
    return batch


def _subsample_lists(lists: list[dict[str, Any]], max_lists: int, seed: int) -> list[dict[str, Any]]:
    if max_lists <= 0 or len(lists) <= max_lists:
        return lists
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lists), size=max_lists, replace=False)
    return [lists[i] for i in sorted(idx)]


def build_model_a_batches(
    config: dict,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    teams: pd.DataFrame,
    pgl: pd.DataFrame,
    *,
    max_lists: int | None = None,
    val_frac: float | None = None,
) -> tuple[list[dict], list[dict]]:
    lists = build_lists(tgl, games, teams)
    if not lists:
        return [], []

    seed = int(config.get("repro", {}).get("seed", 42))
    if max_lists is None:
        max_lists = int(config.get("training", {}).get("max_lists_oof", 30))
    lists = _subsample_lists(lists, max_lists, seed)

    # Focus on a single season (latest date) to keep runtime bounded.
    games_tmp = games.copy()
    games_tmp["game_date"] = pd.to_datetime(games_tmp["game_date"]).dt.date
    season_map = dict(zip(games_tmp["game_date"], games_tmp["season"]))
    max_date = max(pd.to_datetime(x["as_of_date"]).date() for x in lists)
    target_season = season_map.get(max_date)
    if target_season:
        lists = [
            x for x in lists
            if season_map.get(pd.to_datetime(x["as_of_date"]).date()) == target_season
        ]
    if not lists:
        return [], []

    dates_sorted = sorted({x["as_of_date"] for x in lists})
    if val_frac is None:
        val_frac = float(config.get("model_a", {}).get("early_stopping_val_frac", 0.1))
    if len(dates_sorted) < 2:
        # fallback: split by list index
        val_dates = {lists[-1]["as_of_date"]} if len(lists) > 1 else set()
    else:
        n_val = max(1, int(len(dates_sorted) * val_frac))
        n_val = min(n_val, max(1, len(dates_sorted) - 1))
        val_dates = set(dates_sorted[-n_val:])

    cfg = ModelABatchConfig(
        roster_size=int(config.get("training", {}).get("roster_size", 15)),
        num_embeddings=int(config.get("model_a", {}).get("num_embeddings", 500)),
        rolling_windows=list(config.get("training", {}).get("rolling_windows", [10, 30])),
        stat_base=["pts", "reb", "ast", "stl", "blk", "tov"],
    )

    max_date = pd.to_datetime(max(dates_sorted))
    team_ids = {int(tid) for lst in lists for tid in lst["team_ids"]}
    pgl_sub = pgl.copy()
    pgl_sub["game_date"] = pd.to_datetime(pgl_sub["game_date"])
    pgl_sub = pgl_sub[pgl_sub["game_date"] < max_date]
    if team_ids:
        pgl_sub = pgl_sub[pgl_sub["team_id"].isin(team_ids)]
    history_days = int(config.get("training", {}).get("model_a_history_days", 0) or 0)
    if history_days > 0:
        min_date = max_date - pd.Timedelta(days=history_days)
        pgl_sub = pgl_sub[pgl_sub["game_date"] >= min_date]

    rolled = compute_rolling_stats(
        pgl_sub,
        windows=cfg.rolling_windows,
        stat_cols=cfg.stat_base,
        as_of_date=None,
        min_col="min",
    )
    cache: dict[str, pd.DataFrame] = {}
    train_batches: list[dict] = []
    val_batches: list[dict] = []
    for lst in lists:
        batch = build_batch_for_list(lst, pgl_sub, rolled, cfg, cache)
        if lst["as_of_date"] in val_dates:
            val_batches.append(batch)
        else:
            train_batches.append(batch)

    if not val_batches and train_batches:
        # fallback: reuse last train batch for validation
        val_batches = [train_batches[-1]]

    return train_batches, val_batches
