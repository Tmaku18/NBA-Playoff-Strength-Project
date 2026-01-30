"""Inference: load A/B/stacker, produce per-team JSON (predicted_rank, true_strength, delta, ensemble, contributors)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_models(
    model_a_path: str | Path | None = None,
    xgb_path: str | Path | None = None,
    rf_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    config: dict | None = None,
):
    """Load Model A, XGB, RF, RidgeCV meta. Returns (model_a, xgb, rf, meta) or Nones."""
    from src.models.deep_set_rank import DeepSetRank

    model_a, xgb, rf, meta = None, None, None, None
    cfg = config or {}
    ma = cfg.get("model_a", {})

    if model_a_path and Path(model_a_path).exists():
        ck = torch.load(model_a_path, map_location="cpu", weights_only=False)
        model_a = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            7,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
        )
        if "model_state" in ck:
            model_a.load_state_dict(ck["model_state"])
        model_a.eval()

    if xgb_path and Path(xgb_path).exists():
        import joblib
        xgb = joblib.load(xgb_path)
    if rf_path and Path(rf_path).exists():
        import joblib
        rf = joblib.load(rf_path)
    if meta_path and Path(meta_path).exists():
        import joblib
        meta = joblib.load(meta_path)

    return model_a, xgb, rf, meta


def predict_teams(
    team_ids: list[int],
    team_names: list[str],
    model_a_scores: np.ndarray | None = None,
    xgb_scores: np.ndarray | None = None,
    rf_scores: np.ndarray | None = None,
    meta_model: Any = None,
    actual_ranks: dict[int, int] | None = None,
    attention_by_team: dict[int, list[tuple[str, float]]] | None = None,
    *,
    true_strength_scale: str = "percentile",
) -> list[dict]:
    """
    Combine base scores, run meta if present. For each team output:
    predicted_rank, true_strength_score, delta (actual - predicted), classification, ensemble_diagnostics, primary_contributors.
    """
    n = len(team_ids)
    if model_a_scores is not None and len(model_a_scores) == n:
        sa = np.asarray(model_a_scores).ravel()
    else:
        sa = np.zeros(n)
    if xgb_scores is not None and len(xgb_scores) == n:
        sx = np.asarray(xgb_scores).ravel()
    else:
        sx = np.zeros(n)
    if rf_scores is not None and len(rf_scores) == n:
        sr = np.asarray(rf_scores).ravel()
    else:
        sr = np.zeros(n)
    sa = np.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)
    sx = np.nan_to_num(sx, nan=0.0, posinf=0.0, neginf=0.0)
    sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.column_stack([sa, sx, sr])
    if meta_model is not None:
        ens = meta_model.predict(X).ravel()
    else:
        ens = (sa + sx + sr) / 3.0

    pred_rank = np.argsort(np.argsort(-ens)) + 1
    if true_strength_scale == "percentile":
        tss = (np.argsort(np.argsort(ens)) + 1).astype(float) / (n + 1)
    else:
        tss = (ens - ens.min()) / (ens.max() - ens.min() + 1e-12)

    actual_ranks = actual_ranks or {}
    attention_by_team = attention_by_team or {}

    out = []
    for i, (tid, tname) in enumerate(zip(team_ids, team_names)):
        act = actual_ranks.get(tid)
        delta = (act - pred_rank[i]) if act is not None else None
        if delta is not None:
            if delta > 0:
                classification = f"Sleeper (Under-ranked by {delta} slots)"
            elif delta < 0:
                classification = f"Paper Tiger (Over-ranked by {-delta} slots)"
            else:
                classification = "Aligned"
        else:
            classification = "Unknown"

        r_a = np.argsort(np.argsort(-sa))[i] + 1 if len(sa) == n else None
        r_x = np.argsort(np.argsort(-sx))[i] + 1 if len(sx) == n else None
        r_r = np.argsort(np.argsort(-sr))[i] + 1 if len(sr) == n else None
        spread = max(r or 0 for r in [r_a, r_x, r_r]) - min(r or 0 for r in [r_a, r_x, r_r]) if any([r_a, r_x, r_r]) else 0
        agreement = "High" if spread <= 2 else "Low"

        contrib = attention_by_team.get(tid, [])

        out.append({
            "team_id": int(tid),
            "team_name": tname,
            "prediction": {"predicted_rank": int(pred_rank[i]), "true_strength_score": float(tss[i])},
            "analysis": {"actual_rank": int(act) if act is not None else None, "classification": classification},
            "ensemble_diagnostics": {"model_agreement": agreement, "deep_set_rank": int(r_a) if r_a is not None else None, "xgboost_rank": int(r_x) if r_x is not None else None, "random_forest_rank": int(r_r) if r_r is not None else None},
            "roster_dependence": {"primary_contributors": [{"player": str(p), "attention_weight": float(w)} for p, w in contrib]},
        })
    return out


def run_inference_from_db(
    output_dir: str | Path,
    config: dict,
    db_path: str | Path,
    run_id: str | None = None,
) -> Path:
    """Run inference using real DB: load data, build lists for target date, run Model A/B, write predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.db_loader import load_training_data
    from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
    from src.training.build_lists import build_lists
    from src.training.data_model_a import build_batches_from_lists
    from src.training.train_model_a import predict_batches

    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Load models from the outputs directory (same as script 3/4/4b)
    outputs_path = Path(output_dir).resolve()
    model_a, xgb, rf, meta = load_models(
        model_a_path=outputs_path / "best_deep_set.pt",
        xgb_path=outputs_path / "xgb_model.joblib",
        rf_path=outputs_path / "rf_model.joblib",
        meta_path=outputs_path / "ridgecv_meta.joblib",
        config=config,
    )

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        raise ValueError("DB has no games/tgl. Run 2_build_db with raw data first.")
    lists = build_lists(tgl, games, teams)
    if not lists:
        raise ValueError("No lists from build_lists.")
    # Target: latest date in DB (or last list date)
    dates_sorted = sorted(set(lst["as_of_date"] for lst in lists))
    target_date = dates_sorted[-1] if dates_sorted else None
    target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
    if not target_lists:
        target_lists = lists[-2:] if len(lists) >= 2 else lists
    # Flatten to one list of (team_id, as_of_date) across target lists; keep unique team_id for naming/rank
    team_id_to_as_of: dict[int, str] = {}
    team_id_to_actual_rank: dict[int, int] = {}
    team_id_to_win_rate: dict[int, float] = {}
    for lst in target_lists:
        for r, tid in enumerate(lst["team_ids"], start=1):
            tid = int(tid)
            team_id_to_as_of[tid] = lst["as_of_date"]
            team_id_to_actual_rank[tid] = r
            team_id_to_win_rate[tid] = lst["win_rates"][lst["team_ids"].index(tid)] if tid in lst["team_ids"] else 0.0
    unique_team_ids = list(dict.fromkeys(tid for lst in target_lists for tid in lst["team_ids"]))
    unique_team_ids = [int(t) for t in unique_team_ids]
    if not unique_team_ids:
        raise ValueError("No teams in target lists.")
    team_dates = [(tid, team_id_to_as_of.get(tid, target_date or "")) for tid in unique_team_ids]
    as_of_date = target_date or team_dates[0][1]

    device = torch.device("cpu")  # Match load_models map_location="cpu"
    tid_to_score_a: dict[int, float] = {}
    if model_a is not None:
        batches_a, _ = build_batches_from_lists(target_lists, games, tgl, teams, pgl, config, device=device)
        if batches_a:
            scores_list = predict_batches(model_a, batches_a, device)
            for lst, score_tensor in zip(target_lists, scores_list):
                for k, tid in enumerate(lst["team_ids"]):
                    tid_to_score_a[int(tid)] = float(score_tensor[0, k].item())
    sa = np.array([tid_to_score_a.get(tid, 0.0) for tid in unique_team_ids], dtype=np.float32)

    sx = np.zeros(len(unique_team_ids), dtype=np.float32)
    sr = np.zeros(len(unique_team_ids), dtype=np.float32)
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
    if not feat_df.empty and xgb is not None and rf is not None:
        feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in feat_df.columns]
        if feat_cols:
            for i, tid in enumerate(unique_team_ids):
                row = feat_df[(feat_df["team_id"] == tid) & (feat_df["as_of_date"] == team_id_to_as_of.get(tid, as_of_date))]
                if not row.empty:
                    X_row = row[feat_cols].values.astype(np.float32)
                    if xgb is not None:
                        sx[i] = float(xgb.predict(X_row)[0])
                    if rf is not None:
                        sr[i] = float(rf.predict(X_row)[0])

    actual_ranks = {tid: team_id_to_actual_rank.get(tid) for tid in unique_team_ids}
    team_names = []
    for tid in unique_team_ids:
        r = teams[teams["team_id"] == tid]
        name = r["name"].iloc[0] if not r.empty and "name" in r.columns else f"Team_{tid}"
        team_names.append(str(name))

    preds = predict_teams(
        unique_team_ids,
        team_names,
        model_a_scores=sa,
        xgb_scores=sx,
        rf_scores=sr,
        meta_model=meta,
        actual_ranks=actual_ranks,
        true_strength_scale=config.get("output", {}).get("true_strength_scale", "percentile"),
    )
    pj = out / "predictions.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump({"teams": preds}, f, indent=2)

    fig, ax = plt.subplots()
    pr = [t["prediction"]["predicted_rank"] for t in preds]
    ar = [t["analysis"]["actual_rank"] for t in preds]
    ar = [a if a is not None else 0 for a in ar]
    ax.scatter(ar, pr, label="teams")
    max_r = max(max(ar or [1]), max(pr or [1]), 1) + 1
    ax.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
    ax.set_xlabel("Actual rank")
    ax.set_ylabel("Predicted rank")
    ax.legend()
    ax.set_title("Predicted vs actual rank")
    fig.savefig(out / "pred_vs_actual.png", bbox_inches="tight")
    plt.close()
    return pj


def run_inference(output_dir: str | Path, config: dict, run_id: str | None = None) -> Path:
    """Run inference: from DB if present and has data, else exit with message (real run only)."""
    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    paths_cfg = config.get("paths", {})
    db_path = Path(paths_cfg.get("db", "data/processed/nba_build.duckdb"))
    if not db_path.is_absolute():
        from pathlib import Path as P
        root = P(__file__).resolve().parents[2]
        db_path = root / db_path
    if db_path.exists():
        try:
            return run_inference_from_db(output_dir, config, db_path, run_id=run_id)
        except Exception as e:
            raise RuntimeError(f"Inference from DB failed: {e}") from e
    raise FileNotFoundError(
        f"Database not found at {db_path}. Run scripts 1_download_raw and 2_build_db first."
    )
