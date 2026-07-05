"""Script 3: Train Model A (DeepSet + ListMLE).

What this does:
- Builds lists of team rankings from DB (by date, conference).
- Trains a DeepSet neural network with ListMLE loss to rank teams.
- Uses K-fold out-of-fold (OOF) predictions for downstream stacking.
- Optionally uses batch cache to avoid rebuilding batches across sweeps.
- Saves best_deep_set.pt and oof_model_a.parquet to outputs/run_NNN/.

Run after scripts 1 and 2. Required before Model B (script 4)."""
import os

# Set thread count for PyTorch/numpy before importing torch (speeds up CPU ops)
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "14"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "14"

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

try:
    import torch
except ImportError:
    torch = None


def _compute_batch_cache_key(config: dict, db_path: Path) -> str:
    """Build a cache key from config + DB so we reuse batches when config/DB unchanged."""
    training = config.get("training", {})
    model_a = config.get("model_a", {})
    rolling_windows = training.get("rolling_windows") or [10, 30]
    train_seasons = training.get("train_seasons") or []
    key_data = {
        "sampler": "stratified_v2",  # bumped when list subsampling changed (E/W-balanced); invalidates pre-fix caches
        "lists_version": "season_scoped_v3",  # build_lists win rates now season-to-date, not all-history
        "listmle_target": training.get("listmle_target"),
        "rolling_windows": tuple(rolling_windows),
        "train_seasons": tuple(sorted(train_seasons)),
        "max_lists_oof": training.get("max_lists_oof", 30),
        "max_final_batches": training.get("max_final_batches", 50),
        "n_folds": training.get("n_folds", 5),
        "roster_size": training.get("roster_size", 15),
        "use_prior_season_baseline": training.get("use_prior_season_baseline", False),
        "prior_season_lookback_days": training.get("prior_season_lookback_days", 365),
        "stat_dim": model_a.get("stat_dim"),
        "num_embeddings": model_a.get("num_embeddings", 500),
        "use_team_stats": model_a.get("use_team_stats", False),
        "team_stats_cols": tuple(model_a.get("team_stats_cols") or []),
        "db_path": str(db_path.resolve()),
    }
    stat = db_path.stat() if db_path.exists() else None
    if stat:
        key_data["db_mtime"] = stat.st_mtime
        key_data["db_size"] = stat.st_size
    js = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(js.encode()).hexdigest()[:16]


def _subsample_lists_stratified(lists: list[dict], max_n: int) -> list[dict]:
    """Evenly subsample lists by date while keeping every conference for each kept date.

    The old `sorted[::step]` walk over lists sorted by (date, conference) picked a single
    conference whenever step was even (lists alternate E/W per date), so OOF and the
    final model trained on East-only lists. Stratifying by date keeps E/W balanced.
    """
    if len(lists) <= max_n:
        return lists
    order = sorted(range(len(lists)), key=lambda i: (lists[i]["as_of_date"], lists[i].get("conference", "")))
    by_date: dict[str, list[int]] = {}
    for i in order:
        by_date.setdefault(str(lists[i]["as_of_date"]), []).append(i)
    dates = sorted(by_date.keys())
    per_date = max(1, round(len(lists) / len(dates)))
    n_dates = max(1, max_n // per_date)
    step = max(1, math.ceil(len(dates) / n_dates))
    picked: list[int] = []
    picked_dates = set()
    for d in dates[::step]:
        if len(picked) + len(by_date[d]) > max_n:
            break
        picked.extend(by_date[d])
        picked_dates.add(d)
    # Fill remaining budget with evenly spread unpicked dates (whole dates only, keeps E/W together)
    remaining = [d for d in dates if d not in picked_dates]
    for d in remaining[:: max(1, step)] + remaining:
        if d in picked_dates:
            continue
        if len(picked) + len(by_date[d]) > max_n:
            continue
        picked.extend(by_date[d])
        picked_dates.add(d)
    return [lists[i] for i in sorted(picked)]


def _resolve_batch_cache_dir(config: dict) -> Path | None:
    """Return the batch cache directory (e.g. data/processed/batch_cache)."""
    p = config.get("paths", {}).get("batch_cache")
    if p is None or (isinstance(p, str) and p.strip().lower() in ("null", "")):
        p = ROOT / "data" / "processed" / "batch_cache"
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _move_batches_to_device(batches: list, device) -> None:
    """Move tensor values in batch dicts to device (in place)."""
    if torch is None:
        return
    for b in batches:
        for k, v in list(b.items()):
            if isinstance(v, torch.Tensor):
                b[k] = v.to(device)


def _copy_batches_to_cpu(batches: list) -> list:
    """Return shallow copy of batches with tensors moved to CPU (for cache save)."""
    if torch is None:
        return batches
    out = []
    for b in batches:
        nb = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                nb[k] = v.cpu().clone()
            else:
                nb[k] = v
        out.append(nb)
    return out


def _next_run_id(outputs_dir: Path, run_id_base: int | None = None) -> str:
    """Same logic as script 6: next run_NNN; if no run_* and base set, return run_{base:03d}."""
    outputs_dir = Path(outputs_dir)
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    if outputs_dir.exists():
        for p in outputs_dir.iterdir():
            if p.is_dir() and pattern.match(p.name):
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers and run_id_base is not None:
        return f"run_{run_id_base:03d}"
    next_n = max(numbers, default=0) + 1
    return f"run_{next_n:03d}"


def _reserve_run_id(outputs_dir: Path, config: dict) -> None:
    """Reserve the next run_id for this pipeline run so script 6 uses the same folder.
    When inference.run_id is explicitly set (e.g. run_024 for phase1), use it directly."""
    inf = config.get("inference") or {}
    run_id = inf.get("run_id")
    if run_id and isinstance(run_id, str) and re.match(r"^run_\d+$", run_id.strip(), re.I):
        run_id = run_id.strip()
    else:
        run_id_base = inf.get("run_id_base")
        run_id = _next_run_id(outputs_dir, run_id_base=run_id_base)
    path = outputs_dir / ".current_run"
    path.write_text(run_id.strip(), encoding="utf-8")

from src.data.db_loader import load_playoff_data, load_training_data
from src.training.data_model_a import build_batches_from_db, build_batches_from_lists
from src.training.train_model_a import (
    get_model_a_seeds,
    predict_batches,
    predict_batches_with_attention,
    train_model_a,
    train_model_a_on_batches,
)
from src.models.confidence import confidence_from_attention
from src.training.build_lists import build_lists
from src.utils.repro import set_seeds
from src.utils.split import compute_split, get_train_seasons_ordered, group_lists_by_season, write_split_info


def _score_batches_like_inference(model, batches, device, config):
    """Score val batches exactly the way inference scores teams.

    When multi-temp attention is enabled, inference aggregates scores across attention
    temperatures (aggregate_multi_temp_scores) and derives confidence from cross-temp
    agreement; the old OOF path used a single temperature + attention-entropy confidence,
    so the meta-learner was trained on scores with a different distribution than it saw
    at inference. Returns (scores_per_batch, conf_per_batch) as lists of np arrays (K,).
    """
    import numpy as np

    attn_cfg = (config.get("model_a") or {}).get("attention", {})
    multi_temp = bool(attn_cfg.get("multi_temp_enabled", False))
    temps = attn_cfg.get("temperatures", [1, 5, 10])
    base_weights = attn_cfg.get("multi_temp_base_weights", {1: 0.85, 5: 1.0, 10: 0.7})
    if multi_temp and temps:
        from src.models.multi_temp_aggregation import aggregate_multi_temp_scores
        per_temp = []
        for t in temps:
            sl, _ = predict_batches_with_attention(model, batches, device, attention_temperature_override=float(t))
            per_temp.append(sl)
        scores_out, conf_out = [], []
        for i in range(len(batches)):
            scores_by_temp = {int(t): per_temp[j][i][0].numpy() for j, t in enumerate(temps)}
            k = len(next(iter(scores_by_temp.values())))
            s_final, c_a = aggregate_multi_temp_scores(scores_by_temp, base_weights, np.ones(k))
            scores_out.append(np.asarray(s_final, dtype=float))
            conf_out.append(np.asarray(c_a, dtype=float))
        return scores_out, conf_out
    scores_list, attn_list = predict_batches_with_attention(model, batches, device)
    conf_cfg = (config.get("model_a") or {}).get("confidence", {})
    ent_w = float(conf_cfg.get("entropy_weight", 0.5))
    max_w = float(conf_cfg.get("max_weight_weight", 0.5))
    scores_out, conf_out = [], []
    for score_tensor, attn_tensor in zip(scores_list, attn_list):
        k = score_tensor.shape[1]
        scores_out.append(score_tensor[0].numpy().astype(float))
        conf_out.append(np.array([
            confidence_from_attention(attn_tensor[0, ki, :].numpy(), entropy_weight=ent_w, max_weight_weight=max_w)
            for ki in range(k)
        ], dtype=float))
    return scores_out, conf_out


def _train_and_score_seed_avg(config, train_batches, val_batches, device, epochs, loss_log_path=None):
    """Train one model per seed, score val batches like inference, average across seeds.

    Returns (scores_per_batch, conf_per_batch, models) where models holds one trained
    model per seed (for optional reuse).
    """
    import numpy as np

    seeds = get_model_a_seeds(config)
    sum_scores = None
    sum_conf = None
    models = []
    for si, seed in enumerate(seeds):
        model = train_model_a_on_batches(
            config, train_batches, device,
            max_epochs=epochs, val_batches=val_batches or None,
            loss_log_path=loss_log_path if si == 0 else None,
            seed=seed,
        )
        models.append(model)
        if not val_batches:
            continue
        scores, conf = _score_batches_like_inference(model, val_batches, device, config)
        if sum_scores is None:
            sum_scores = [np.array(s, dtype=float) for s in scores]
            sum_conf = [np.array(c, dtype=float) for c in conf]
        else:
            for i in range(len(scores)):
                sum_scores[i] += scores[i]
                sum_conf[i] += conf[i]
    if sum_scores is None:
        return None, None, models
    n = float(len(seeds))
    return [s / n for s in sum_scores], [c / n for c in sum_conf], models


def _save_final_model_seed_avg(config, out, all_batches, device) -> Path:
    """Train the final model once per seed and save a checkpoint that carries all seed
    state dicts (model_states); inference averages their scores via DeepSetRankEnsemble."""
    import torch

    seeds = get_model_a_seeds(config)
    if len(seeds) == 1:
        return train_model_a(config, out, batches=all_batches, seed=seeds[0])
    states = []
    path = None
    for seed in seeds:
        print(f"Final model: training seed {seed} ({len(states)+1}/{len(seeds)})", flush=True)
        path = train_model_a(config, out, batches=all_batches, seed=seed)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        states.append(ck["model_state"])
    torch.save({"model_state": states[0], "model_states": states, "config": config}, path)
    print(f"Saved {path} with {len(states)} seed state dicts (seed averaging)", flush=True)
    return path


def _run_walk_forward(config, train_lists, games, tgl, teams, pgl, out, root, playoff_games=None, playoff_tgl=None):
    """Walk-forward training: train on seasons 1..k, validate on k+1; last step trains on all and saves."""
    import torch

    seasons_cfg = config.get("seasons") or {}
    train_seasons_ordered = get_train_seasons_ordered(config)
    if not train_seasons_ordered or not seasons_cfg:
        print("Walk-forward: no train_seasons in config. Falling back to pooled training.", flush=True)
        return
    grouped = group_lists_by_season(train_lists, seasons_cfg)
    if not grouped:
        print("Walk-forward: no lists grouped by season. Falling back to pooled.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_final = int(config.get("training", {}).get("max_final_batches", 50))
    epochs = int((config.get("model_a") or {}).get("epochs", 20))
    oof_rows = []
    n_steps = len(train_seasons_ordered)

    for step, k in enumerate(range(1, n_steps + 1), 1):
        train_season_set = set(train_seasons_ordered[:k])
        val_season = train_seasons_ordered[k] if k < n_steps else None
        step_train_lists = [lst for s in train_season_set for lst in grouped.get(s, [])]
        step_val_lists = list(grouped.get(val_season, [])) if val_season else []

        if not step_train_lists:
            print(f"Walk-forward step {step}/{n_steps}: no train lists, skip", flush=True)
            continue

        # Subsample if needed (date-stratified so both conferences are kept)
        if len(step_train_lists) > max_final:
            step_train_lists = _subsample_lists_stratified(step_train_lists, max_final)

        train_batches, _ = build_batches_from_lists(
            step_train_lists, games, tgl, teams, pgl, config, device=device,
        )
        if not train_batches:
            train_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games,
                playoff_tgl=playoff_tgl,
            )

        val_batches = None
        val_metas = []
        if step_val_lists:
            val_batches, val_metas = build_batches_from_lists(
                step_val_lists, games, tgl, teams, pgl, config, device=device,
            )

        loss_log_path = out / "training_loss.csv" if k == n_steps else None
        scores_avg, conf_avg, models = _train_and_score_seed_avg(
            config, train_batches, val_batches or None, device, epochs, loss_log_path=loss_log_path,
        )
        if val_batches and val_metas and scores_avg is not None:
            for scores, conf, meta in zip(scores_avg, conf_avg, val_metas):
                for ki in range(len(meta["team_ids"])):
                    oof_rows.append({
                        "team_id": meta["team_ids"][ki],
                        "as_of_date": meta["as_of_date"],
                        "conference": meta.get("conference", ""),
                        "oof_a": float(scores[ki]),
                        "conf_a": float(conf[ki]),
                        # rel_values matches the training target (31 - final_rank when rank
                        # targets are on); win_rates was a different scale than k-fold OOF y.
                        "y": meta.get("rel_values", meta["win_rates"])[ki],
                    })
            print(
                f"Walk-forward step {step}/{n_steps}: trained on seasons "
                f"{train_seasons_ordered[0]}..{train_seasons_ordered[k-1]}, validated on {val_season}, "
                f"OOF {len(val_batches)} lists",
                flush=True,
            )
        else:
            print(
                f"Walk-forward step {step}/{n_steps}: trained on seasons "
                f"{train_seasons_ordered[0]}..{train_seasons_ordered[k-1]} (final, no next season)",
                flush=True,
            )

        # Last step: save final model (trained on all train seasons); carry all seed states.
        if k == n_steps:
            path = out / "best_deep_set.pt"
            states = [{k2: v.detach().cpu().clone() for k2, v in m.state_dict().items()} for m in models]
            payload = {"model_state": states[0], "config": config}
            if len(states) > 1:
                payload["model_states"] = states
            torch.save(payload, path)
            print(f"Saved {path} (final model from walk-forward, {len(states)} seed(s))", flush=True)

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = out / "oof_model_a.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/defaults.yaml)")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    # Load config and database; exit if DB is missing.
    print("Script 3: loading config and DB...", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Seed the whole OOF path (list building, subsampling, batch construction);
    # per-fold/per-seed training re-seeds in train_model_a_on_batches.
    set_seeds(int((config.get("repro") or {}).get("seed", 42)))
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    # If we are training to predict playoff outcome, load playoff data for list labels.
    listmle_target = (config.get("training") or {}).get("listmle_target")
    playoff_games, playoff_tgl = None, None
    if listmle_target == "playoff_outcome":
        try:
            pg, ptgl, _ = load_playoff_data(db_path)
            if not pg.empty and not ptgl.empty:
                playoff_games, playoff_tgl = pg, ptgl
                print("Loaded playoff data for listmle_target=playoff_outcome", flush=True)
            else:
                print("Warning: playoff data empty; falling back to standings for ListMLE.", flush=True)
        except Exception as e:
            print(f"Warning: could not load playoff data ({e}); falling back to standings.", flush=True)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    # Reserve run_id so script 6 (inference) writes to the same run folder when running the full pipeline.
    _reserve_run_id(out, config)

    # Build ranking lists (e.g. one per snapshot date per conference) from standings or playoff outcome.
    lists = build_lists(
        tgl, games, teams,
        config=config,
        playoff_games=playoff_games,
        playoff_tgl=playoff_tgl,
    )
    print(f"build_lists: {len(lists)} lists", flush=True)
    if not lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no lists for OOF)")
        return

    valid_lists = [lst for lst in lists if len(lst["team_ids"]) >= 2]
    if not valid_lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no valid lists for OOF)")
        return

    # Split lists into train vs test (e.g. 75/25 by time); only train lists are used for OOF and final model.
    train_lists, test_lists, split_info = compute_split(valid_lists, config)
    write_split_info(split_info, out)
    print(f"Split: {split_info['split_mode']} — train {split_info['n_train_lists']} lists, test {split_info['n_test_lists']} lists", flush=True)
    if not train_lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no train lists after split)")
        return

    walk_forward = bool(config.get("training", {}).get("walk_forward", False))
    if walk_forward:
        _run_walk_forward(config, train_lists, games, tgl, teams, pgl, out, ROOT, playoff_games, playoff_tgl)
        return

    n_folds = config.get("training", {}).get("n_folds", 5)
    n_folds = min(n_folds, len(train_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(train_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    # Subsample lists for OOF to keep runtime manageable (time-stratified, within train only)
    max_lists_oof = config.get("training", {}).get("max_lists_oof", 30)
    oof_lists = train_lists
    if len(train_lists) > max_lists_oof:
        oof_lists = _subsample_lists_stratified(train_lists, max_lists_oof)
        confs = {lst.get("conference", "") for lst in oof_lists}
        print(f"OOF: using {len(oof_lists)} lists (subsampled from {len(train_lists)} train, conferences={sorted(confs)})", flush=True)
    n_folds = min(n_folds, len(oof_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(oof_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = None
    list_metas = None
    all_batches = None

    cache_dir = _resolve_batch_cache_dir(config)
    cache_key = _compute_batch_cache_key(config, db_path) if cache_dir else None
    cache_file = (cache_dir / f"{cache_key}.pt") if cache_dir and cache_key else None

    if cache_file and cache_file.exists():
        print(f"Batch cache hit: {cache_file.name}", flush=True)
        payload = torch.load(cache_file, map_location="cpu")
        batches = payload.get("oof_batches", [])
        list_metas = payload.get("list_metas", [])
        all_batches = payload.get("all_batches", [])
        _move_batches_to_device(batches, device)
        _move_batches_to_device(all_batches, device)

    if batches is None or list_metas is None:
        print("Building batches for OOF...", flush=True)
        batches, list_metas = build_batches_from_lists(oof_lists, games, tgl, teams, pgl, config, device=device)

    if not batches or not list_metas:
        print(
            "No batches from build_batches_from_lists (player_game_logs required). Skipping OOF; training final model only.",
            file=sys.stderr,
        )
        all_batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=all_batches)
        print(f"Saved {path} (no oof_model_a.parquet)")
        return

    # On cache miss: build all_batches and save to cache
    if all_batches is None:
        all_lists_for_cache = train_lists
        max_final = config.get("training", {}).get("max_final_batches", 50)
        if len(all_lists_for_cache) > max_final:
            all_lists_for_cache = _subsample_lists_stratified(all_lists_for_cache, max_final)
        all_batches, _ = build_batches_from_lists(
            all_lists_for_cache, games, tgl, teams, pgl, config, device=device
        )
        if not all_batches:
            all_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games, playoff_tgl=playoff_tgl,
            )
        if cache_dir and cache_key and all_batches:
            cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "split_info": split_info,
                "oof_lists": oof_lists,
                "all_lists": all_lists_for_cache,
                "oof_batches": _copy_batches_to_cpu(batches),
                "list_metas": list_metas,
                "all_batches": _copy_batches_to_cpu(all_batches),
            }
            tmp = cache_dir / f".{cache_key}.tmp"
            torch.save(payload, tmp)
            tmp.rename(cache_dir / f"{cache_key}.pt")
            print(f"Batch cache saved: {cache_key}.pt", flush=True)

    # Causal expanding-window OOF: sort lists by time, cut into n_folds contiguous blocks,
    # validate block f training ONLY on earlier blocks. The old k-fold scheme trained on
    # future blocks when validating early ones (future leakage into OOF scores).
    # Index over list_metas (aligned with batches): build_batches_from_lists can skip lists
    # with empty rosters, so oof_lists indices would be misaligned with batches.
    sorted_indices = sorted(range(len(batches)), key=lambda i: (list_metas[i]["as_of_date"], list_metas[i].get("conference", "")))
    n_folds = min(n_folds, len(batches))
    fold_size = (len(sorted_indices) + n_folds - 1) // n_folds
    oof_rows = []
    epochs = int((config.get("model_a") or {}).get("epochs", 20))
    for fold in range(1, n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, len(sorted_indices))
        val_idx = sorted_indices[val_start:val_end]
        train_idx = sorted_indices[:val_start]
        train_batches = [batches[i] for i in train_idx]
        val_batches = [batches[i] for i in val_idx]
        val_metas = [list_metas[i] for i in val_idx]
        if not train_batches or not val_batches:
            continue
        scores_avg, conf_avg, _ = _train_and_score_seed_avg(
            config, train_batches, val_batches, device, epochs,
        )
        if scores_avg is None:
            continue
        for scores, conf, meta in zip(scores_avg, conf_avg, val_metas):
            for k in range(len(meta["team_ids"])):
                y_val = meta.get("rel_values", meta["win_rates"])[k]
                oof_rows.append({
                    "team_id": meta["team_ids"][k],
                    "as_of_date": meta["as_of_date"],
                    "conference": meta.get("conference", ""),
                    "oof_a": float(scores[k]),
                    "conf_a": float(conf[k]),
                    "y": y_val,
                })
        print(f"Fold {fold+1}/{n_folds} OOF collected {len(val_batches)} lists (expanding-window, trained on {len(train_batches)})")

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = out / "oof_model_a.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_rows)} rows)")
    else:
        print("No OOF rows collected (every fold had empty train or val batches).", file=sys.stderr)

    # Train the final model on all train lists (or cached batches); this is what inference will load.
    if all_batches is None:
        all_lists = train_lists
        max_final = config.get("training", {}).get("max_final_batches", 50)
        if len(all_lists) > max_final:
            all_lists = _subsample_lists_stratified(all_lists, max_final)
            print(f"Final model: training on {len(all_lists)} lists (subsampled from {len(train_lists)} train)", flush=True)
        print("Building final batches...", flush=True)
        all_batches, _ = build_batches_from_lists(all_lists, games, tgl, teams, pgl, config, device=device)
        if not all_batches:
            all_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games, playoff_tgl=playoff_tgl,
            )
    path = _save_final_model_seed_avg(config, out, all_batches, device)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
