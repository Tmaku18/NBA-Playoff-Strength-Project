"""Sweep Model A epochs and Model B grids on real DB data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.db_loader import load_training_data
from src.evaluation.metrics import ndcg_score, spearman, mrr
from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
from src.models.deep_set_rank import DeepSetRank
from src.training.build_lists import build_lists
from src.training.model_a_data import build_model_a_batches
from src.training.train_model_a import train_model_a
from src.training.train_model_b import train_model_b


def _parse_list(value: str, cast: type) -> list[Any]:
    return [cast(v) for v in value.split(",") if v != ""]


def _parse_grid(spec: str) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    if not spec:
        return grid
    for part in spec.split(";"):
        if not part.strip():
            continue
        key, raw = part.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if "," in raw:
            vals = raw.split(",")
        else:
            vals = [raw]
        parsed: list[Any] = []
        for v in vals:
            v = v.strip()
            if v.lower() in {"true", "false"}:
                parsed.append(v.lower() == "true")
            else:
                try:
                    parsed.append(int(v))
                except ValueError:
                    try:
                        parsed.append(float(v))
                    except ValueError:
                        parsed.append(v)
        grid[key] = parsed
    return grid


def _cartesian_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = [{}]
    for k in keys:
        next_combos = []
        for base in combos:
            for v in grid[k]:
                c = dict(base)
                c[k] = v
                next_combos.append(c)
        combos = next_combos
    return combos


def _load_model_a(config: dict, path: Path) -> DeepSetRank:
    import torch

    ma = config.get("model_a", {})
    model = DeepSetRank(
        ma.get("num_embeddings", 500),
        ma.get("embedding_dim", 32),
        7,
        ma.get("encoder_hidden", [128, 64]),
        ma.get("attention_heads", 4),
        ma.get("dropout", 0.2),
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _evaluate_model_a(model: DeepSetRank, batches: list[dict]) -> dict[str, float]:
    import torch

    ndcgs = []
    spears = []
    mrrs = []
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = (
                batch["embedding_indices"].shape[0],
                batch["embedding_indices"].shape[1],
                batch["embedding_indices"].shape[2],
                batch["player_stats"].shape[-1],
            )
            embs = batch["embedding_indices"].reshape(B * K, P)
            stats = batch["player_stats"].reshape(B * K, P, S)
            minutes = batch["minutes"].reshape(B * K, P)
            mask = batch["key_padding_mask"].reshape(B * K, P)
            rel = batch["rel"].reshape(B, K)

            score, _, _ = model(embs, stats, minutes, mask)
            score = score.reshape(B, K)
            for i in range(B):
                y_true = rel[i].cpu().numpy()
                y_score = score[i].cpu().numpy()
                ndcgs.append(ndcg_score(y_true, y_score, k=10))
                spears.append(spearman(y_true, y_score))
                mrrs.append(mrr(y_true, y_score, top_k=1))
    if not ndcgs:
        return {"ndcg": 0.0, "spearman": 0.0, "mrr": 0.0}
    return {
        "ndcg": float(np.mean(ndcgs)),
        "spearman": float(np.mean(spears)),
        "mrr": float(np.mean(mrrs)),
    }


def _attention_stats(model: DeepSetRank, batches: list[dict]) -> dict[str, float]:
    import torch

    weights = []
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = (
                batch["embedding_indices"].shape[0],
                batch["embedding_indices"].shape[1],
                batch["embedding_indices"].shape[2],
                batch["player_stats"].shape[-1],
            )
            embs = batch["embedding_indices"].reshape(B * K, P)
            stats = batch["player_stats"].reshape(B * K, P, S)
            minutes = batch["minutes"].reshape(B * K, P)
            mask = batch["key_padding_mask"].reshape(B * K, P)

            _, _, attn_w = model(embs, stats, minutes, mask)
            weights.append(attn_w.detach().cpu().reshape(-1))

    if not weights:
        return {
            "attn_min": 0.0,
            "attn_mean": 0.0,
            "attn_max": 0.0,
            "attn_zero_pct": 0.0,
            "attn_nan_pct": 0.0,
        }

    w = torch.cat(weights)
    finite = torch.isfinite(w)
    if not finite.any():
        return {
            "attn_min": 0.0,
            "attn_mean": 0.0,
            "attn_max": 0.0,
            "attn_zero_pct": 0.0,
            "attn_nan_pct": 1.0,
        }

    w_f = w[finite]
    return {
        "attn_min": float(w_f.min().item()),
        "attn_mean": float(w_f.mean().item()),
        "attn_max": float(w_f.max().item()),
        "attn_zero_pct": float((w_f == 0).float().mean().item()),
        "attn_nan_pct": float((~finite).float().mean().item()),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="epochs_plus_model_b", choices=["epochs_only", "model_b_grid", "epochs_plus_model_b"])
    parser.add_argument("--batch-id", default="batch_002")
    parser.add_argument("--epochs", default="5,10,15,20,25")
    parser.add_argument("--xgb-grid", default="max_depth=4,6;learning_rate=0.05,0.1;n_estimators=300;subsample=0.8;colsample_bytree=0.8")
    parser.add_argument("--rf-grid", default="n_estimators=200,400;max_depth=12;min_samples_leaf=5")
    parser.add_argument("--val-frac", type=float, default=None)
    args = parser.parse_args()

    root = ROOT
    with open(root / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = root / config["paths"]["db"]
    if not db_path.exists():
        raise SystemExit("Database not found. Run scripts 1_download_raw and 2_build_db first.")

    games, tgl, teams, pgl = load_training_data(db_path)
    out_root = Path(config["paths"]["outputs"]) / "sweeps" / args.batch_id
    out_root.mkdir(parents=True, exist_ok=True)

    results = {"epochs": [], "model_b": []}
    val_frac = args.val_frac
    if val_frac is None:
        val_frac = float(config.get("model_a", {}).get("early_stopping_val_frac", 0.2))

    if args.batch in {"epochs_only", "epochs_plus_model_b"}:
        epochs = _parse_list(args.epochs, int)
        print("Building Model A batches...")
        train_batches, val_batches = build_model_a_batches(
            config,
            games,
            tgl,
            teams,
            pgl,
            val_frac=val_frac,
        )
        if not train_batches or not val_batches:
            raise SystemExit("Insufficient Model A batches from DB.")

        # disable early stopping to test full epoch counts
        config["model_a"]["early_stopping_patience"] = 0

        for e in epochs:
            config["model_a"]["epochs"] = e
            epoch_dir = out_root / f"model_a_epoch_{e}"
            ckpt = train_model_a(config, epoch_dir, train_batches=train_batches, val_batches=val_batches)
            model = _load_model_a(config, ckpt)
            metrics = _evaluate_model_a(model, val_batches)
            metrics.update(_attention_stats(model, val_batches))
            results["epochs"].append({"epochs": e, **metrics})
            print(f"Epochs={e} -> {metrics}")

    if args.batch in {"model_b_grid", "epochs_plus_model_b"}:
        print("Building Model B features...")
        lists = build_lists(tgl, games, teams)
        if not lists:
            raise SystemExit("No lists from build_lists (empty games/tgl?).")
        max_lists = int(config.get("training", {}).get("max_lists_oof", 0) or 0)
        if max_lists and len(lists) > max_lists:
            rng = np.random.default_rng(int(config.get("repro", {}).get("seed", 42)))
            idx = rng.choice(len(lists), size=max_lists, replace=False)
            lists = [lists[i] for i in sorted(idx)]
        rows = []
        for lst in lists:
            for tid, wr in zip(lst["team_ids"], lst["win_rates"]):
                rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"], "y": float(wr)})
        import pandas as pd
        flat_df = pd.DataFrame(rows)
        team_dates = [(int(a), str(b)) for a, b in flat_df[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
        feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
        df = flat_df.merge(feat_df, on=["team_id", "as_of_date"], how="inner")
        feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in df.columns]
        if not feat_cols:
            raise SystemExit("No feature columns for Model B.")

        X = df[feat_cols].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)
        dates_sorted = sorted(df["as_of_date"].unique())
        n_val = max(1, int(val_frac * len(dates_sorted)))
        val_dates = set(dates_sorted[-n_val:])
        val_mask = df["as_of_date"].isin(val_dates)
        X_train = X[~val_mask]
        y_train = y[~val_mask]
        X_val = X[val_mask] if val_mask.any() else None
        y_val = y[val_mask] if val_mask.any() else None

        xgb_grid = _cartesian_product(_parse_grid(args.xgb_grid))
        rf_grid = _cartesian_product(_parse_grid(args.rf_grid))
        idx = 0
        for xgb_params in xgb_grid:
            for rf_params in rf_grid:
                idx += 1
                cfg = dict(config)
                cfg["model_b"] = dict(config.get("model_b", {}))
                cfg["model_b"]["xgb"] = dict(cfg["model_b"].get("xgb", {}))
                cfg["model_b"]["rf"] = dict(cfg["model_b"].get("rf", {}))
                cfg["model_b"]["xgb"].update(xgb_params)
                cfg["model_b"]["rf"].update(rf_params)

                combo_dir = out_root / "model_b" / f"combo_{idx:03d}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                p1, p2 = train_model_b(X_train, y_train, X_val, y_val, cfg, feat_cols, combo_dir)

                xgb_preds = None
                rf_preds = None
                if y_val is not None and X_val is not None:
                    import joblib
                    xgb_m = joblib.load(p1)
                    rf_m = joblib.load(p2)
                    xgb_preds = xgb_m.predict(X_val).astype(np.float32)
                    rf_preds = rf_m.predict(X_val).astype(np.float32)

                if xgb_preds is not None and rf_preds is not None:
                    mean_preds = (xgb_preds + rf_preds) / 2.0
                    metrics = {
                        "rmse_xgb": _rmse(y_val, xgb_preds),
                        "rmse_rf": _rmse(y_val, rf_preds),
                        "rmse_mean": _rmse(y_val, mean_preds),
                        "spearman_xgb": spearman(y_val, xgb_preds),
                        "spearman_rf": spearman(y_val, rf_preds),
                        "spearman_mean": spearman(y_val, mean_preds),
                    }
                else:
                    metrics = {}
                results["model_b"].append({"xgb": xgb_params, "rf": rf_params, **metrics})
                print(f"Model B combo {idx}: {metrics}")

    out_json = out_root / "sweep_results.json"
    out_csv = out_root / "sweep_results.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # flatten for CSV
    rows = []
    for r in results["epochs"]:
        rows.append({"type": "epochs", **r})
    for r in results["model_b"]:
        row = {"type": "model_b", **r}
        rows.append(row)
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
