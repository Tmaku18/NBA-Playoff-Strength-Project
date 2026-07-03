"""Inference: load whichever models are present (A, B, C, meta, extra), produce per-team JSON.

Runs on any subset of models: Model A only, XGB only, RF only, team-stats (linreg, bayesian_ridge, gpr, gmm),
or any combination. Ensemble is meta when both A and B exist; otherwise mean of available scores (normalized)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.confidence import confidence_from_attention

# Letter labels for extra (team-stats) models; matches docs/MODELS.md (Model D–G).
EXTRA_MODEL_LETTER: dict[str, str] = {
    "linreg": "d",
    "bayesian_ridge": "e",
    "gpr": "f",
    "gmm": "g",
}


def _state_dict_for_load(state: dict | None) -> dict:
    """Strip torch.compile _orig_mod. prefix so checkpoint saved from compiled model loads into plain model."""
    if not state:
        return state or {}
    return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}


def load_models(
    model_a_path: str | Path | None = None,
    xgb_path: str | Path | None = None,
    lr_path: str | Path | None = None,
    rf_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    config: dict | None = None,
):
    """Load Model A, XGB, RF (Model C), RidgeCV meta. Returns (model_a, xgb, rf, meta). Ensemble uses A + XGB only; RF for diagnostics."""
    from src.models.deep_set_rank import DeepSetRank

    model_a, xgb, rf, meta = None, None, None, None
    cfg = config or {}
    ma = cfg.get("model_a", {})

    if model_a_path and Path(model_a_path).exists():
        ck = torch.load(model_a_path, map_location="cpu", weights_only=False)
        attn_cfg = ma.get("attention", {})
        embed_dim = int(ma.get("embedding_dim", 32))
        # Infer stat_dim from checkpoint to match trained model (avoids config/checkpoint mismatch)
        state = _state_dict_for_load(ck.get("model_state", ck) if isinstance(ck, dict) else {})
        if "enc.mlp.0.weight" in state:
            enc_in = state["enc.mlp.0.weight"].shape[1]
            stat_dim = int(enc_in) - embed_dim
        else:
            stat_dim = int(ma.get("stat_dim", ma.get("expected_stat_dim", 14)))
        model_a = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            stat_dim,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
            minutes_bias_weight=float(ma.get("minutes_bias_weight", 0.3)),
            minutes_sum_min=float(ma.get("minutes_sum_min", 1e-6)),
            fallback_strategy=str(ma.get("attention_fallback_strategy", "minutes")),
            attention_temperature=float(attn_cfg.get("temperature", 1.0)),
            attention_input_dropout=float(attn_cfg.get("input_dropout", 0.0)),
            attention_use_pre_norm=bool(attn_cfg.get("use_pre_norm", True)),
            attention_use_residual=bool(attn_cfg.get("use_residual", True)),
        )
        if "model_state" in ck:
            model_a.load_state_dict(_state_dict_for_load(ck["model_state"]), strict=True)
        model_a.eval()

    if xgb_path and Path(xgb_path).exists():
        import joblib
        xgb = joblib.load(xgb_path)
        if isinstance(xgb, dict) or not callable(getattr(xgb, "predict", None)):
            xgb = None
    path_rf = rf_path or lr_path
    if path_rf and Path(path_rf).exists():
        import joblib
        rf = joblib.load(path_rf)
        if isinstance(rf, dict) or not callable(getattr(rf, "predict", None)):
            rf = None
    if meta_path and Path(meta_path).exists():
        import joblib
        meta = joblib.load(meta_path)
        if isinstance(meta, dict):
            meta = meta.get("model") or meta.get("meta")
        if meta is None or not callable(getattr(meta, "predict", None)):
            meta = None

    return model_a, xgb, rf, meta


def predict_teams(
    team_ids: list[int],
    team_names: list[str],
    model_a_scores: np.ndarray | None = None,
    xgb_scores: np.ndarray | None = None,
    rf_scores: np.ndarray | None = None,
    model_a_score_std: np.ndarray | None = None,
    xgb_score_std: np.ndarray | None = None,
    rf_score_std: np.ndarray | None = None,
    extra_model_scores: dict[str, np.ndarray] | None = None,
    extra_model_score_std: dict[str, np.ndarray] | None = None,
    meta_model: Any = None,
    conf_a: np.ndarray | None = None,
    conf_xgb: np.ndarray | None = None,
    standings_win_rate: np.ndarray | None = None,
    actual_ranks: dict[int, int] | None = None,
    actual_global_ranks: dict[int, int] | None = None,
    attention_by_team: dict[int, list[tuple[str, float]]] | None = None,
    attention_by_temp_by_team: dict[int, dict[int, list[tuple[str, float]]]] | None = None,
    attention_fallback_by_team: dict[int, bool] | None = None,
    team_id_to_conference: dict[int, str] | None = None,
    playoff_rank: dict[int, int] | None = None,
    eos_playoff_standings: dict[int, int] | None = None,
    model_presence: dict[str, bool] | None = None,
    *,
    true_strength_scale: str = "percentile",
    odds_temperature: float = 1.0,
    championship_odds_method: str = "softmax",
    monte_carlo_config: dict | None = None,
    xgb_weight: float | None = None,
    meta_by_conference: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Combine base scores, run meta if present. For each team output:
    conference_rank (1-15), predicted_strength (global rank 1-30, used internally for eval), ensemble_score,
    championship_odds, delta, classification, analysis.historic_conference_rank, post_playoff_rank and rank_delta_playoffs when available.
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
    # lr_scores (or rf_scores for backward compat): Model C diagnostics only, not in ensemble
    if rf_scores is not None and len(rf_scores) == n:
        sr = np.asarray(rf_scores).ravel()
    else:
        sr = np.zeros(n)
    sa = np.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)
    sx = np.nan_to_num(sx, nan=0.0, posinf=0.0, neginf=0.0)
    sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
    extra_model_scores = extra_model_scores or {}
    extra_model_score_std = extra_model_score_std or {}

    # Build ensemble from whatever models are available (A, B, C, extra). Run on any subset.
    # Use meta / xgb_weight only when both A and B are present (model_presence); else use mean of available scores.
    team_id_to_conf = team_id_to_conference or {}
    pm = model_presence or {}
    has_a = pm.get("a", True) and len(sa) == n and np.any(sa != 0)
    has_xgb = pm.get("xgb", True) and len(sx) == n and np.any(sx != 0)
    has_both_ab = (pm.get("a", True) and pm.get("xgb", True)) and (len(sa) == n and len(sx) == n)
    has_extra = bool(extra_model_scores and any(np.any(np.asarray(v) != 0) for v in extra_model_scores.values() if v is not None and len(v) == n))

    # Shared meta-input columns (must match build_oof in src/models/stacking.py):
    # 2 cols [a, x]; 3 cols [a, x, standings]; 4 cols [a, x, conf_a, conf_xgb];
    # 5 cols [a, x, conf_a, conf_xgb, standings].
    _c_a = np.asarray(conf_a).ravel() if conf_a is not None and len(conf_a) == n else np.full(n, 0.5, dtype=np.float64)
    _c_x = np.asarray(conf_xgb).ravel() if conf_xgb is not None and len(conf_xgb) == n else np.full(n, 0.5, dtype=np.float64)
    _c_a = np.nan_to_num(_c_a, nan=0.5, posinf=0.5, neginf=0.5)
    _c_x = np.nan_to_num(_c_x, nan=0.5, posinf=0.5, neginf=0.5)
    _st = (
        np.asarray(standings_win_rate, dtype=np.float64).ravel()
        if standings_win_rate is not None and len(standings_win_rate) == n
        else np.full(n, 0.5, dtype=np.float64)
    )
    _st = np.nan_to_num(_st, nan=0.5, posinf=0.5, neginf=0.5)

    def _meta_n_cols(m: Any) -> int:
        coefs = getattr(m, "coef_", None)
        return len(np.asarray(coefs).ravel()) if coefs is not None else 2

    def _meta_X(m: Any, a_col: np.ndarray, x_col: np.ndarray) -> np.ndarray:
        n_cols = _meta_n_cols(m)
        if n_cols == 3:
            return np.column_stack([a_col, x_col, _st])
        if n_cols == 4:
            return np.column_stack([a_col, x_col, _c_a, _c_x])
        if n_cols == 5:
            return np.column_stack([a_col, x_col, _c_a, _c_x, _st])
        return np.column_stack([a_col, x_col])

    if xgb_weight is not None and 0.0 < xgb_weight < 1.0 and has_both_ab:
        ens = (1.0 - float(xgb_weight)) * sa + float(xgb_weight) * sx
    elif meta_by_conference and team_id_to_conf and has_both_ab:
        # Option 3A: per-conference meta (E/W) when we have A and/or B
        ens = np.zeros(n, dtype=np.float64)
        X_by_meta_cols: dict[int, np.ndarray] = {}
        for i in range(n):
            tid = team_ids[i]
            conf = team_id_to_conf.get(tid, "E")
            meta_c = meta_by_conference.get(conf) or meta_by_conference.get("E") or meta_model
            if meta_c is not None and not isinstance(meta_c, dict) and callable(getattr(meta_c, "predict", None)):
                X_all = X_by_meta_cols.setdefault(_meta_n_cols(meta_c), _meta_X(meta_c, sa, sx))
                ens[i] = float(meta_c.predict(X_all[i : i + 1]).ravel()[0])
            else:
                ens[i] = (sa[i] + sx[i]) / 2.0
    elif meta_model is not None and not isinstance(meta_model, dict) and callable(getattr(meta_model, "predict", None)) and has_both_ab:
        # Ensemble: meta (2-5 cols) when A and/or B present
        ens = meta_model.predict(_meta_X(meta_model, sa, sx)).ravel()
    else:
        # Run on any model(s): use mean of available scores (A, B, C, extra), each normalized to [0,1]
        parts: list[np.ndarray] = []
        def _norm(s: np.ndarray) -> np.ndarray:
            s = np.asarray(s, dtype=np.float64).ravel()
            if len(s) == n and (np.ptp(s) > 1e-12):
                s = (s - s.min()) / (s.max() - s.min() + 1e-12)
            return s
        if len(sa) == n and np.any(sa != 0):
            parts.append(_norm(sa))
        if len(sx) == n and np.any(sx != 0):
            parts.append(_norm(sx))
        if len(sr) == n and np.any(sr != 0):
            parts.append(_norm(sr))
        for _name, arr in (extra_model_scores or {}).items():
            if arr is not None and len(arr) == n and np.any(np.asarray(arr) != 0):
                parts.append(_norm(arr))
        if parts:
            ens = np.mean(parts, axis=0)
        else:
            ens = (sa + sx) / 2.0  # fallback (all zeros → tied rank)
    ens = np.nan_to_num(ens, nan=0.0, posinf=0.0, neginf=0.0)

    pred_rank = np.argsort(np.argsort(-ens)) + 1  # global rank 1-30
    if true_strength_scale == "percentile":
        rank_order = (np.argsort(np.argsort(ens)) + 1).astype(float)
        tss = (rank_order - 1.0) / (n - 1) if n > 1 else np.zeros(n)
    else:
        tss = (ens - ens.min()) / (ens.max() - ens.min() + 1e-12)

    # Championship odds: softmax on strength scale so odds differentiate (stacker raw output can be narrow → uniform odds)
    # Use percentile/scale (tss) which is spread over [0,1]; higher = stronger → higher odds.
    T = max(odds_temperature, 1e-6)
    strength_for_odds = tss if true_strength_scale == "percentile" else (ens - ens.min()) / (ens.max() - ens.min() + 1e-12)
    exp_s = np.exp(np.clip(strength_for_odds / T, -50, 50))
    odds = exp_s / exp_s.sum()

    # Conference rank (1-15 within E/W)
    team_id_to_conf = team_id_to_conference or {}
    conf_rank: dict[int, int] = {}
    for conf in ("E", "W"):
        idx = [i for i in range(n) if team_id_to_conf.get(team_ids[i], "E") == conf]
        if not idx:
            continue
        sub_ens = ens[idx]
        sub_rank = np.argsort(np.argsort(-sub_ens)) + 1
        for k, i in enumerate(idx):
            conf_rank[team_ids[i]] = int(sub_rank[k])

    actual_ranks = actual_ranks or {}
    actual_global_ranks = actual_global_ranks or {}
    attention_by_team = attention_by_team or {}
    attention_by_temp_by_team = attention_by_temp_by_team or {}
    attention_fallback_by_team = attention_fallback_by_team or {}
    playoff_rank = playoff_rank or {}
    eos_playoff_standings = eos_playoff_standings or {}
    model_presence = model_presence or {"a": True, "xgb": True, "rf": True}

    # Uncertainty: rank intervals via MC sampling from score distributions
    unc_cfg = monte_carlo_config or {}
    unc_enabled = bool(unc_cfg.get("enabled", True))
    mc_n = int(unc_cfg.get("mc_samples", 200))
    alpha = float(unc_cfg.get("alpha", 0.1))
    score_std_floor = float(unc_cfg.get("score_std_floor", 1e-6))
    conf_to_std_scale = float(unc_cfg.get("conf_to_std_scale", 0.25))

    def _as_std(arr_std: np.ndarray | None, conf: np.ndarray | None) -> np.ndarray:
        if arr_std is not None and len(arr_std) == n:
            s = np.asarray(arr_std, dtype=np.float64).ravel()
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(s, score_std_floor)
        if conf is not None and len(conf) == n:
            c = np.asarray(conf, dtype=np.float64).ravel()
            c = np.nan_to_num(c, nan=0.5, posinf=0.5, neginf=0.5)
            return np.maximum((1.0 - c) * conf_to_std_scale, score_std_floor)
        return np.full(n, score_std_floor, dtype=np.float64)

    std_a = _as_std(model_a_score_std, conf_a)
    std_x = _as_std(xgb_score_std, conf_xgb)
    std_r = _as_std(rf_score_std, None)

    def _mc_rank_bounds(means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray(means, dtype=np.float64).ravel()
        stds = np.asarray(stds, dtype=np.float64).ravel()
        if not unc_enabled or mc_n < 50:
            r = (np.argsort(np.argsort(-means)) + 1).astype(int)
            return r, r
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=means.reshape(1, -1), scale=stds.reshape(1, -1), size=(mc_n, n))
        ranks = np.argsort(np.argsort(-samples, axis=1), axis=1) + 1  # (mc_n, n)
        lo_q = alpha / 2.0
        hi_q = 1.0 - alpha / 2.0
        lo = np.quantile(ranks, lo_q, axis=0, method="nearest").astype(int)
        hi = np.quantile(ranks, hi_q, axis=0, method="nearest").astype(int)
        return lo, hi

    # Per-model global rank intervals
    a_lo, a_hi = _mc_rank_bounds(sa, std_a)
    x_lo, x_hi = _mc_rank_bounds(sx, std_x)
    r_lo, r_hi = _mc_rank_bounds(sr, std_r)

    extra_rank_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    extra_point_ranks: dict[str, np.ndarray] = {}
    for name, mean_arr in extra_model_scores.items():
        std_arr = extra_model_score_std.get(name)
        if std_arr is None or len(std_arr) != n:
            continue
        lo, hi = _mc_rank_bounds(mean_arr, np.asarray(std_arr))
        extra_rank_bounds[str(name)] = (lo, hi)
        extra_point_ranks[str(name)] = np.argsort(np.argsort(-np.asarray(mean_arr).ravel())) + 1

    # Ensemble global rank interval: sample (sa, sx) then recompute ensemble each draw
    ens_lo = pred_rank.astype(int)
    ens_hi = pred_rank.astype(int)
    if unc_enabled and mc_n >= 50:
        rng = np.random.default_rng(42)
        sa_s = rng.normal(loc=np.asarray(sa, dtype=np.float64), scale=std_a, size=(mc_n, n))
        sx_s = rng.normal(loc=np.asarray(sx, dtype=np.float64), scale=std_x, size=(mc_n, n))
        if xgb_weight is not None and 0.0 < float(xgb_weight) < 1.0:
            ens_s = (1.0 - float(xgb_weight)) * sa_s + float(xgb_weight) * sx_s
        else:
            # Use the same meta (2-5 cols) as deterministic path (conf/standings treated as fixed).
            if meta_model is not None and not isinstance(meta_model, dict) and callable(getattr(meta_model, "predict", None)):
                ens_s = np.zeros((mc_n, n), dtype=np.float64)
                for t in range(mc_n):
                    ens_s[t] = np.asarray(meta_model.predict(_meta_X(meta_model, sa_s[t], sx_s[t]))).ravel()
            else:
                ens_s = (sa_s + sx_s) / 2.0
        ens_ranks = np.argsort(np.argsort(-ens_s, axis=1), axis=1) + 1
        ens_lo = np.quantile(ens_ranks, alpha / 2.0, axis=0, method="nearest").astype(int)
        ens_hi = np.quantile(ens_ranks, 1.0 - alpha / 2.0, axis=0, method="nearest").astype(int)

    out = []
    for i, (tid, tname) in enumerate(zip(team_ids, team_names)):
        act = actual_ranks.get(tid)
        act_global = actual_global_ranks.get(tid)
        act_for_class = act_global if act_global is not None else act
        delta = (act_for_class - pred_rank[i]) if act_for_class is not None else None
        if delta is not None:
            if delta > 0:
                classification = f"Over-ranked by {delta} slots"
            elif delta < 0:
                classification = f"Under-ranked by {-delta} slots"
            else:
                classification = "Aligned"
        else:
            classification = "Unknown"

        # model_a_rank, model_b_rank, model_c_rank: global rank (1-30) by each model's score. Aligned with eval report names.
        r_a = np.argsort(np.argsort(-sa))[i] + 1 if model_presence.get("a", True) and len(sa) == n else None
        r_x = np.argsort(np.argsort(-sx))[i] + 1 if model_presence.get("xgb", True) and len(sx) == n else None
        r_r = np.argsort(np.argsort(-sr))[i] + 1 if model_presence.get("rf", True) and len(sr) == n else None
        ranks_present = [r for r in (r_a, r_x, r_r) if r is not None]
        if len(ranks_present) >= 2:
            spread = max(ranks_present) - min(ranks_present)
            threshold_high = max(2, n // 10)
            threshold_med = max(5, n // 5)
            if spread <= threshold_high:
                agreement = "High"
            elif spread <= threshold_med:
                agreement = "Medium"
            else:
                agreement = "Low"
        elif len(ranks_present) == 1:
            agreement = "Single"
        else:
            agreement = "Unknown"

        contrib = attention_by_team.get(tid, [])
        attn_by_temp = attention_by_temp_by_team.get(tid, {})
        p_rank = playoff_rank.get(tid)
        rank_delta_playoffs = (p_rank - pred_rank[i]) if p_rank is not None else None

        pred_dict: dict[str, Any] = {
            "predicted_strength": int(pred_rank[i]),
            "predicted_strength_low": int(ens_lo[i]),
            "predicted_strength_high": int(ens_hi[i]),
            "predicted_strength_minus": int(max(0, int(pred_rank[i]) - int(ens_lo[i]))),
            "predicted_strength_plus": int(max(0, int(ens_hi[i]) - int(pred_rank[i]))),
            "ensemble_score": float(tss[i]),
            "ensemble_score_100": round(float(tss[i]) * 100.0, 1),
            "conference_rank": conf_rank.get(tid),
            "championship_odds": f"{float(odds[i]) * 100:.1f}%",
        }
        eos_standings = eos_playoff_standings.get(tid) if eos_playoff_standings else None
        analysis_dict: dict[str, Any] = {
            "historic_conference_rank": int(act) if act is not None else None,
            "EOS_global_rank": int(act_global) if act_global is not None else None,
            "EOS_playoff_standings": int(eos_standings) if eos_standings is not None else None,
            "classification": classification,
            "post_playoff_rank": int(p_rank) if p_rank is not None else None,
            "rank_delta_playoffs": int(rank_delta_playoffs) if rank_delta_playoffs is not None else None,
        }

        conf = team_id_to_conference.get(tid) if team_id_to_conference else None
        diag: dict[str, Any] = {
            "model_agreement": agreement,
            "model_a_rank": int(r_a) if r_a is not None else None,
            "model_a_rank_low": int(a_lo[i]) if a_lo is not None else None,
            "model_a_rank_high": int(a_hi[i]) if a_hi is not None else None,
            "model_b_rank": int(r_x) if r_x is not None else None,
            "model_b_rank_low": int(x_lo[i]) if x_lo is not None else None,
            "model_b_rank_high": int(x_hi[i]) if x_hi is not None else None,
            "model_c_rank": int(r_r) if r_r is not None else None,
            "model_c_rank_low": int(r_lo[i]) if r_lo is not None else None,
            "model_c_rank_high": int(r_hi[i]) if r_hi is not None else None,
        }
        for name, bounds in extra_rank_bounds.items():
            letter = EXTRA_MODEL_LETTER.get(name)
            if letter:
                r_point = int(extra_point_ranks[name][i]) if name in extra_point_ranks else None
                diag[f"model_{letter}_rank"] = r_point
                diag[f"model_{letter}_rank_low"] = int(bounds[0][i])
                diag[f"model_{letter}_rank_high"] = int(bounds[1][i])
        diag["extra_model_ranks"] = {
            name: {"rank_low": int(bounds[0][i]), "rank_high": int(bounds[1][i])}
            for name, bounds in extra_rank_bounds.items()
        } if extra_rank_bounds else None
        out.append({
            "team_id": int(tid),
            "team_name": tname,
            "conference": conf,
            "prediction": pred_dict,
            "analysis": analysis_dict,
            "ensemble_diagnostics": diag,
            "roster_dependence": {
                "primary_contributors": [
                    {"player": str(p), "attention_weight": float(w)}
                    for p, w in contrib if np.isfinite(w)
                ],
                "attention_by_temp": {
                    str(temp): [{"player": str(p), "attention_weight": float(w)} for p, w in lst if np.isfinite(w)]
                    for temp, lst in sorted(attn_by_temp.items())
                } if attn_by_temp else None,
                "contributors_are_fallback": bool(attention_fallback_by_team.get(int(tid), False)),
            },
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

    from src.data.db import get_connection
    from src.data.db_loader import load_training_data
    from src.features.team_context import build_team_context_as_of_dates
    from src.models.multi_temp_aggregation import aggregate_multi_temp_scores
    from src.training.build_lists import TEAM_CONFERENCE, build_lists
    from src.training.data_model_a import build_batches_from_lists
    from src.training.train_model_a import predict_batches_with_attention
    from src.utils.split import date_to_season, load_split_info

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
    # Optional additional team-stats models (trained by script 4)
    extra_models: dict[str, dict[str, Any]] = {}
    try:
        import joblib
        for name, fname in (
            ("linreg", "linreg_model.joblib"),
            ("bayesian_ridge", "bayesian_ridge_model.joblib"),
            ("gpr", "gpr_model.joblib"),
            ("gmm", "gmm_rank_model.joblib"),
        ):
            p = outputs_path / fname
            if p.exists():
                obj = joblib.load(p)
                if isinstance(obj, dict) and ("model" in obj) and ("feature_cols" in obj):
                    extra_models[str(name)] = obj
    except Exception:
        extra_models = {}
    # Option 3A: load per-conference metas when present
    meta_by_conference: dict[str, Any] = {}
    for conf in ("E", "W"):
        p = outputs_path / f"ridgecv_meta_{conf}.joblib"
        if p.exists():
            import joblib
            m = joblib.load(p)
            if isinstance(m, dict):
                m = m.get("model") or m.get("meta")
            if m is not None and callable(getattr(m, "predict", None)):
                meta_by_conference[conf] = m
    if not meta_by_conference:
        meta_by_conference = None
    if model_a is None and xgb is None and rf is None and not extra_models:
        print(
            "Warning: No models loaded (no best_deep_set.pt, xgb_model.joblib, rf_model.joblib, or team-stats artifacts). "
            "Predictions will be tied. Point config paths.outputs to a directory that has at least one model.",
            file=sys.stderr,
        )
    xgb_weight = config.get("stacking", {}).get("xgb_weight")
    if xgb_weight is not None and (not isinstance(xgb_weight, (int, float)) or xgb_weight <= 0 or xgb_weight >= 1):
        xgb_weight = None

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        raise ValueError("DB has no games/tgl. Run 2_build_db with raw data first.")
    # Load player_id -> player_name for attention primary_contributors
    player_id_to_name: dict[int, str] = {}
    try:
        con = get_connection(Path(db_path), read_only=True)
        players_df = con.execute("SELECT player_id, player_name FROM players").df()
        con.close()
        if not players_df.empty:
            player_id_to_name = dict(
                zip(players_df["player_id"].astype(int), players_df["player_name"].astype(str))
            )
    except Exception:
        pass
    lists = build_lists(tgl, games, teams)
    if not lists:
        raise ValueError("No lists from build_lists.")
    dates_sorted = sorted(set(lst["as_of_date"] for lst in lists))
    # Use split_info: primary = last test date; optional second run = last train date
    try:
        split_info = load_split_info(Path(output_dir))
        test_dates = split_info.get("test_dates", [])
        train_dates = split_info.get("train_dates", [])
        test_seasons = split_info.get("test_seasons")
    except FileNotFoundError:
        split_info = {}
        test_dates = []
        train_dates = []
        test_seasons = None
    if test_seasons is None:
        test_seasons = config.get("training", {}).get("test_seasons") or []
    seasons_cfg = config.get("seasons") or {}

    run_specs: list[tuple[str | None, list, str, str | None]] = []
    if test_seasons and seasons_cfg and test_dates:
        for season in test_seasons:
            season_dates = [d for d in test_dates if date_to_season(d, seasons_cfg) == season]
            if not season_dates:
                continue
            target_date = sorted(season_dates)[-1]
            target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
            if not target_lists:
                target_lists = [lst for lst in lists if lst["as_of_date"] == season_dates[-1]]
            if target_lists:
                run_specs.append((target_date, target_lists, f"predictions_{season}.json", season))
    if not run_specs:
        target_date = test_dates[-1] if test_dates else (dates_sorted[-1] if dates_sorted else None)
        target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
        if not target_lists:
            target_lists = [lists[-1]] if lists else []
        run_specs = [(target_date, target_lists, "predictions.json", None)]
    inf_cfg = config.get("inference", {}) or {}
    also_train = bool(inf_cfg.get("also_train_predictions", False))
    also_validation = bool(inf_cfg.get("also_validation_predictions", False))
    if also_train and train_dates:
        train_date = train_dates[-1]
        train_target_lists = [lst for lst in lists if lst["as_of_date"] == train_date]
        if train_target_lists:
            run_specs.append((train_date, train_target_lists, "train_predictions.json", None))
    if also_validation and train_dates:
        n_val = max(1, int(0.2 * len(train_dates)))
        val_dates = train_dates[-n_val:]
        val_date = val_dates[-1]
        val_target_lists = [lst for lst in lists if lst["as_of_date"] == val_date]
        if val_target_lists:
            run_specs.append((val_date, val_target_lists, "val_predictions.json", None))

    test_specs = [s for s in run_specs if s[2] not in ("train_predictions.json", "val_predictions.json")]
    train_specs = [s for s in run_specs if s[2] == "train_predictions.json"]
    val_specs = [s for s in run_specs if s[2] == "val_predictions.json"]
    if not test_specs:
        test_specs = [(dates_sorted[-1] if dates_sorted else None, lists[-1:] if lists else [], "predictions.json", None)]

    # Collect all (team_id, as_of_date) from all specs and build features once (or load from cache)
    def _inference_feature_cache_key(cfg: dict, path: Path, team_dates_hash: str) -> str | None:
        cache_dir = cfg.get("paths", {}).get("feature_cache")
        if not cache_dir or (isinstance(cache_dir, str) and cache_dir.strip().lower() in ("null", "")):
            return None
        model_b = cfg.get("model_b", {})
        key_data = {
            "include_features": tuple(model_b.get("include_features") or []),
            "exclude_features": tuple(model_b.get("exclude_features") or []),
            "elo": bool(cfg.get("elo", {}).get("enabled", False)),
            "massey": bool(cfg.get("massey", {}).get("enabled", False)),
            "team_rolling": bool(cfg.get("team_rolling", {}).get("enabled", True)),
            "team_dates_hash": team_dates_hash,
            "db": str(path.resolve()),
        }
        if path.exists():
            st = path.stat()
            key_data["db_mtime"], key_data["db_size"] = st.st_mtime, st.st_size
        return hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()[:20]

    all_specs = test_specs + train_specs + val_specs
    # Collect ALL (team_id, as_of_date) pairs across specs. Each season/spec has its own
    # as_of_date per team; keeping only the first-seen date starved later specs of features
    # (empty inner join -> XGB never ran -> Model B scores all zero for those seasons).
    team_dates_all_set: set[tuple[int, str]] = set()
    for _date, target_lists, _file, _season in all_specs:
        for lst in target_lists:
            ao = str(lst.get("as_of_date", _date or ""))
            for tid in lst.get("team_ids", []):
                team_dates_all_set.add((int(tid), ao))
    team_dates_all = sorted(team_dates_all_set)
    team_dates_hash = hashlib.sha256(json.dumps(team_dates_all, sort_keys=True).encode()).hexdigest()[:16]

    feat_df_all: pd.DataFrame | None = None
    cache_dir_raw = config.get("paths", {}).get("feature_cache")
    cache_dir = None
    if cache_dir_raw and isinstance(cache_dir_raw, str) and cache_dir_raw.strip().lower() not in ("null", ""):
        cache_dir = Path(cache_dir_raw)
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parents[2] / cache_dir
    cache_key = _inference_feature_cache_key(config, db_path, team_dates_hash) if cache_dir else None
    cache_file = (cache_dir / f"inf_{cache_key}.parquet") if cache_dir and cache_key else None
    if cache_file and cache_file.exists():
        try:
            feat_df_all = pd.read_parquet(cache_file)
            feat_df_all["team_id"] = feat_df_all["team_id"].astype(int)
            feat_df_all["as_of_date"] = feat_df_all["as_of_date"].astype(str)
        except Exception:
            feat_df_all = None
    if feat_df_all is None and team_dates_all:
        feat_df_all = build_team_context_as_of_dates(
            tgl, games, team_dates_all, config=config, teams=teams, pgl=pgl,
        )
        if cache_dir and cache_key and not feat_df_all.empty:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                feat_df_all.to_parquet(cache_dir / f"inf_{cache_key}.parquet", index=False)
            except Exception:
                pass

    def _run_inference_for_spec(target_date: str | None, target_lists: list, output_file: str, season: str | None, *, draw_figures: bool = True, feat_df_all: pd.DataFrame | None = None) -> Path:
        pj = out / output_file
        fig_suffix = f"_{season}" if season else ""

            # Flatten to one list of (team_id, as_of_date) across target lists; keep unique team_id for naming/rank
        team_id_to_as_of: dict[int, str] = {}
        team_id_to_actual_rank: dict[int, int] = {}
        team_id_to_win_rate: dict[int, float] = {}
        for lst in target_lists:
            win_rates = lst.get("win_rates", [])
            for idx, tid in enumerate(lst["team_ids"]):
                tid = int(tid)
                if tid not in team_id_to_as_of:
                    team_id_to_as_of[tid] = lst["as_of_date"]
                if tid not in team_id_to_actual_rank:
                    team_id_to_actual_rank[tid] = idx + 1
                if tid not in team_id_to_win_rate:
                    team_id_to_win_rate[tid] = float(win_rates[idx]) if idx < len(win_rates) else 0.0
        unique_team_ids = list(dict.fromkeys(tid for lst in target_lists for tid in lst["team_ids"]))
        unique_team_ids = [int(t) for t in unique_team_ids]
        if not unique_team_ids:
            raise ValueError("No teams in target lists.")
        team_dates = [(tid, team_id_to_as_of.get(tid, target_date or "")) for tid in unique_team_ids]
        as_of_date = target_date or team_dates[0][1]
        win_rate_map = {tid: float(team_id_to_win_rate.get(tid, 0.0)) for tid in unique_team_ids}
        sorted_global = sorted(win_rate_map.items(), key=lambda x: (-x[1], x[0]))
        actual_global_rank = {tid: i + 1 for i, (tid, _) in enumerate(sorted_global)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tid_to_score_a: dict[int, float] = {}
        tid_to_conf_a: dict[int, float] = {}
        attention_by_team: dict[int, list[tuple[str, float]]] = {}  # team_id -> [(player_name, weight), ...]
        attention_by_temp_by_team: dict[int, dict[int, list[tuple[str, float]]]] = {}  # team_id -> temp -> [(player_name, weight), ...]
        attention_fallback_by_team: dict[int, bool] = {}
        team_id_to_batch: dict[int, tuple[int, int]] = {}
        team_id_to_player_ids: dict[int, list[int | None]] = {}
        batches_a: list[dict[str, Any]] = []
        attn_debug = {"teams": 0, "empty_roster": 0, "all_zero": 0, "attn_sum": [], "attn_max": []}
        model_a_dev = model_a.to(device) if model_a is not None else None
        if model_a_dev is not None:
            batches_a, list_metas = build_batches_from_lists(target_lists, games, tgl, teams, pgl, config, device=device)
            if batches_a:
                attn_cfg = (config.get("model_a") or {}).get("attention", {})
                multi_temp = bool(attn_cfg.get("multi_temp_enabled", False))
                temps = attn_cfg.get("temperatures", [1, 5, 10])
                base_weights = attn_cfg.get("multi_temp_base_weights", {1: 0.85, 5: 1.0, 10: 0.7})
                use_avail_agg = bool(attn_cfg.get("use_availability_in_aggregation", True))
                if multi_temp and temps:
                    scores_list = []
                    attn_list = []
                    for t in temps:
                        sl, al = predict_batches_with_attention(model_a_dev, batches_a, device, attention_temperature_override=float(t))
                        scores_list.append(sl)
                        attn_list.append(al)
                    attn_list_for_display = attn_list[temps.index(5)] if 5 in temps else attn_list[0]
                    temps_for_attn = temps
                else:
                    scores_list, attn_list = predict_batches_with_attention(model_a_dev, batches_a, device)
                    attn_list_for_display = attn_list
                    default_temp = int(attn_cfg.get("temperature", 3))
                    temps_for_attn = [default_temp]
                    attn_list = [attn_list]
                for i, meta in enumerate(list_metas):
                    if i >= len(attn_list_for_display):
                        break
                    attn_tensor = attn_list_for_display[i]
                    s_final_batch = None
                    c_A_batch = None
                    if multi_temp and temps:
                        scores_by_temp = {int(t): scores_list[j][i][0].numpy() for j, t in enumerate(temps)}
                        starter_avail = np.array(meta.get("starter_availability", [1.0] * len(meta["team_ids"])))
                        if not use_avail_agg or len(starter_avail) == 0:
                            starter_avail = None
                        s_final_batch, c_A_batch = aggregate_multi_temp_scores(scores_by_temp, base_weights, starter_avail)
                    for k, tid in enumerate(meta["team_ids"]):
                        tid = int(tid)
                        if s_final_batch is not None:
                            tid_to_score_a[tid] = float(s_final_batch[k])
                            tid_to_conf_a[tid] = float(c_A_batch[k])
                        else:
                            tid_to_score_a[tid] = float(scores_list[i][0, k].item())
                        player_ids = meta.get("player_ids_per_team", [[]])[k] if k < len(meta.get("player_ids_per_team", [])) else []
                        if tid not in team_id_to_batch:
                            team_id_to_batch[tid] = (i, k)
                            team_id_to_player_ids[tid] = [int(pid) if pid is not None else None for pid in player_ids]
                        attn_weights = attn_tensor[0, k].numpy() if attn_tensor.dim() >= 2 else attn_tensor[k].numpy()
                        attn_weights = np.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                        max_len = min(len(player_ids), len(attn_weights))
                        attn_weights = attn_weights[:max_len]
                        player_ids = player_ids[:max_len]
                        if max_len == 0:
                            attn_debug["empty_roster"] += 1
                            continue
                        attn_debug["teams"] += 1
                        if s_final_batch is None:
                            conf_cfg = (config.get("model_a") or {}).get("confidence", {})
                            ent_w = float(conf_cfg.get("entropy_weight", 0.5))
                            max_w = float(conf_cfg.get("max_weight_weight", 0.5))
                            tid_to_conf_a[tid] = confidence_from_attention(
                                attn_weights, entropy_weight=ent_w, max_weight_weight=max_w
                            )
                        attn_sum = float(np.sum(attn_weights))
                        attn_max = float(np.max(attn_weights)) if max_len else 0.0
                        attn_debug["attn_sum"].append(attn_sum)
                        attn_debug["attn_max"].append(attn_max)
                        if attn_sum <= 0:
                            attn_debug["all_zero"] += 1
                        order = np.argsort(-attn_weights)
                        contrib: list[tuple[str, float]] = []
                        for idx in order[:10]:
                            if idx >= len(player_ids) or player_ids[idx] is None:
                                continue
                            w = float(attn_weights[idx])
                            if not np.isfinite(w) or w <= 0:
                                continue
                            pid = player_ids[idx]
                            name = player_id_to_name.get(int(pid), f"Player_{pid}")
                            contrib.append((name, w))
                        fallback_used = False
                        if not contrib and max_len > 0:
                            # Fallback 1: take top-k by raw weight even if <= 0
                            fallback_used = True
                            for idx in order[:10]:
                                if idx >= len(player_ids) or player_ids[idx] is None:
                                    continue
                                w = float(attn_weights[idx])
                                if not np.isfinite(w) or w <= 0:
                                    continue
                                pid = player_ids[idx]
                                name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                contrib.append((name, w))
                        if not contrib and max_len > 0 and i < len(batches_a):
                            # Fallback 2: top-3 by minutes when attention yields nothing
                            b = batches_a[i]
                            if "minutes" in b:
                                min_t = b["minutes"]
                                if min_t.dim() >= 3 and k < min_t.shape[1]:
                                    minutes_row = min_t[0, k].cpu().numpy()
                                    order_min = np.argsort(-minutes_row)
                                    for idx in order_min[:3]:
                                        if idx < len(player_ids) and player_ids[idx] is not None:
                                            pid = player_ids[idx]
                                            name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                            contrib.append((name, float(minutes_row[idx])))
                            fallback_used = True
                        if contrib:
                            attention_by_team[tid] = contrib
                        if fallback_used:
                            attention_fallback_by_team[tid] = True
                        for j, temp_val in enumerate(temps_for_attn):
                            if j >= len(attn_list) or i >= len(attn_list[j]):
                                continue
                            attn_t = attn_list[j][i]
                            aw = attn_t[0, k].numpy() if attn_t.dim() >= 2 else attn_t[k].numpy()
                            aw = np.nan_to_num(aw, nan=0.0, posinf=0.0, neginf=0.0)
                            ml = min(len(player_ids), len(aw))
                            aw = aw[:ml]
                            pids = player_ids[:ml]
                            order_j = np.argsort(-aw)
                            contrib_temp: list[tuple[str, float]] = []
                            for idx in order_j:
                                if idx >= len(pids) or pids[idx] is None:
                                    continue
                                w = float(aw[idx])
                                if not np.isfinite(w):
                                    continue
                                pid = pids[idx]
                                name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                contrib_temp.append((name, w))
                            if tid not in attention_by_temp_by_team:
                                attention_by_temp_by_team[tid] = {}
                            attention_by_temp_by_team[tid][temp_val] = contrib_temp
        if attn_debug["teams"] > 0:
            mean_sum = float(np.mean(attn_debug["attn_sum"])) if attn_debug["attn_sum"] else 0.0
            mean_max = float(np.mean(attn_debug["attn_max"])) if attn_debug["attn_max"] else 0.0
            print(
                "Attention debug:",
                f"teams={attn_debug['teams']}",
                f"empty_roster={attn_debug['empty_roster']}",
                f"all_zero={attn_debug['all_zero']}",
                f"attn_sum_mean={mean_sum:.4f}",
                f"attn_max_mean={mean_max:.4f}",
                flush=True,
            )
        sa = np.array([tid_to_score_a.get(tid, 0.0) for tid in unique_team_ids], dtype=np.float32)
        conf_a_arr = np.array([tid_to_conf_a.get(tid, 0.5) for tid in unique_team_ids], dtype=np.float32)

        sx = np.zeros(len(unique_team_ids), dtype=np.float32)
        sr = np.zeros(len(unique_team_ids), dtype=np.float32)
        std_xgb_arr = np.zeros(len(unique_team_ids), dtype=np.float32)
        std_rf_arr = np.zeros(len(unique_team_ids), dtype=np.float32)
        conf_xgb_arr = np.full(len(unique_team_ids), 0.5, dtype=np.float32)
        extra_scores: dict[str, np.ndarray] = {k: np.zeros(len(unique_team_ids), dtype=np.float32) for k in extra_models.keys()}
        extra_stds: dict[str, np.ndarray] = {k: np.zeros(len(unique_team_ids), dtype=np.float32) for k in extra_models.keys()}
        if feat_df_all is not None and not feat_df_all.empty:
            td_df = pd.DataFrame(team_dates, columns=["team_id", "as_of_date"])
            td_df["team_id"] = td_df["team_id"].astype(int)
            td_df["as_of_date"] = td_df["as_of_date"].astype(str)
            feat_df = feat_df_all.merge(td_df, on=["team_id", "as_of_date"], how="inner")
        else:
            feat_df = build_team_context_as_of_dates(
                tgl, games, team_dates,
                config=config, teams=teams, pgl=pgl,
            )
        # Optional: include Model A score as an extra feature for team-stats models.
        tsm_cfg = config.get("team_stats_models", {}) or {}
        if bool(tsm_cfg.get("include_model_a_score", False)) and not feat_df.empty:
            try:
                score_rows = []
                for tid, ao in team_dates:
                    score_rows.append(
                        {"team_id": int(tid), "as_of_date": str(ao), "model_a_score": float(tid_to_score_a.get(int(tid), 0.0))}
                    )
                score_df = pd.DataFrame(score_rows)
                feat_df = feat_df.merge(score_df, on=["team_id", "as_of_date"], how="left")
                feat_df["model_a_score"] = feat_df["model_a_score"].fillna(0.0)
            except Exception:
                pass
        if not feat_df.empty and (xgb is not None or rf is not None or extra_models):
            from src.features.team_context import get_team_context_feature_cols
            all_feat = get_team_context_feature_cols(config)
            feat_cols = [c for c in all_feat if c in feat_df.columns]
            if feat_cols:
                X_rows = []
                for i, tid in enumerate(unique_team_ids):
                    row = feat_df[(feat_df["team_id"] == tid) & (feat_df["as_of_date"] == team_id_to_as_of.get(tid, as_of_date))]
                    if not row.empty:
                        X_rows.append((i, row[feat_cols].values.astype(np.float32)))
                if X_rows:
                    idx_order = [r[0] for r in X_rows]
                    X_full = np.vstack([r[1] for r in X_rows])
                    if xgb is not None:
                        try:
                            from src.models.xgb_model import predict_with_uncertainty
                            pred_xgb, tree_std = predict_with_uncertainty(xgb, X_full)
                            for j, i in enumerate(idx_order):
                                sx[i] = float(pred_xgb[j])
                                conf_xgb_arr[i] = float(1.0 / (1.0 + min(tree_std[j], 1e6)))
                                std_xgb_arr[i] = float(max(tree_std[j], 1e-6))
                        except Exception:
                            for j, i in enumerate(idx_order):
                                sx[i] = float(xgb.predict(X_full[j : j + 1])[0])
                    for j, i in enumerate(idx_order):
                        if rf is not None and hasattr(rf, "predict"):
                            sr[i] = float(rf.predict(X_full[j : j + 1])[0])

                    # RF per-tree std (if available)
                    if rf is not None and hasattr(rf, "estimators_"):
                        try:
                            preds = np.stack([t.predict(X_full) for t in rf.estimators_], axis=0)
                            rf_std = np.std(preds, axis=0)
                            rf_std = np.nan_to_num(rf_std, nan=0.0, posinf=0.0, neginf=0.0)
                            for j, i in enumerate(idx_order):
                                std_rf_arr[i] = float(max(rf_std[j], 1e-6))
                        except Exception:
                            pass

                    # Extra team-stats models (each may use a different feature set)
                    if extra_models:
                        for name, pack in extra_models.items():
                            try:
                                cols = [c for c in pack.get("feature_cols", []) if c in feat_df.columns]
                                if not cols:
                                    continue
                                X_rows2 = []
                                for i, tid in enumerate(unique_team_ids):
                                    row = feat_df[(feat_df["team_id"] == tid) & (feat_df["as_of_date"] == team_id_to_as_of.get(tid, as_of_date))]
                                    if not row.empty:
                                        X_rows2.append((i, row[cols].values.astype(np.float32)))
                                if not X_rows2:
                                    continue
                                idx2 = [r[0] for r in X_rows2]
                                X2 = np.vstack([r[1] for r in X_rows2])
                                m = pack.get("model")
                                if m is None or not callable(getattr(m, "predict", None)):
                                    continue
                                mean, std = m.predict(X2, return_std=True)
                                for j, i in enumerate(idx2):
                                    extra_scores[name][i] = float(mean[j])
                                    extra_stds[name][i] = float(std[j])
                            except Exception:
                                continue

        actual_ranks = {tid: team_id_to_actual_rank.get(tid) for tid in unique_team_ids}
        team_names = []
        for tid in unique_team_ids:
            r = teams[teams["team_id"] == tid]
            name = r["name"].iloc[0] if not r.empty and "name" in r.columns else f"Team_{tid}"
            team_names.append(str(name))

        # Conference map for conference_rank and plot
        team_id_to_conf: dict[int, str] = {}
        abbr_col = "abbreviation" if "abbreviation" in teams.columns else "ABBREVIATION"
        conf_col = "conference" if "conference" in teams.columns else "CONFERENCE"
        for _, row in teams.iterrows():
            tid = int(row["team_id"])
            c = row.get(conf_col)
            if c is not None and str(c).strip():
                c = str(c).strip().upper()
                team_id_to_conf[tid] = "E" if c in ("E", "EAST") else "W" if c in ("W", "WEST") else c[0]
            else:
                abbr = row.get(abbr_col)
                team_id_to_conf[tid] = TEAM_CONFERENCE.get(str(abbr).strip() if abbr is not None else "", "E")

        # Playoff rank, EOS final rank (Option B), and EOS playoff standings for target season
        playoff_rank_map: dict[int, int] = {}
        eos_final_rank_map: dict[int, int] = {}
        eos_playoff_standings_map: dict[int, int] = {}
        eos_rank_source = "standings"
        seasons_cfg = config.get("seasons") or {}
        target_season = None
        season_start = None
        season_end = None
        as_of_d = pd.to_datetime(as_of_date).date() if as_of_date else None
        for season, rng in seasons_cfg.items():
            start = pd.to_datetime(rng.get("start")).date()
            end = pd.to_datetime(rng.get("end")).date()
            if as_of_d and start <= as_of_d <= end:
                target_season = season
                season_start = rng.get("start")
                season_end = rng.get("end")
                break
        if target_season and season_start and season_end:
            try:
                from src.data.db_loader import load_playoff_data
                from src.evaluation.playoffs import (
                    _filtered_playoff_tgl,
                    compute_eos_final_rank,
                    compute_eos_playoff_standings,
                    compute_playoff_performance_rank,
                )
                season_end_d = pd.to_datetime(season_end).date()
                reg_season_complete = as_of_d and as_of_d >= season_end_d
                pg, ptgl, _ = load_playoff_data(db_path)
                if pg is not None and ptgl is not None and not pg.empty and not ptgl.empty:
                    pt_check = _filtered_playoff_tgl(pg, ptgl, target_season)
                    tid_col = "team_id" if "team_id" in pt_check.columns else "TEAM_ID"
                    if not pt_check.empty and len(pt_check[tid_col].unique()) >= 16:
                        reg_season_complete = True
                if reg_season_complete:
                    eos_playoff_standings_map = compute_eos_playoff_standings(
                        games, tgl, target_season,
                        season_start=season_start,
                        season_end=season_end,
                        all_team_ids=unique_team_ids,
                    )
                if not pg.empty and not ptgl.empty:
                    playoff_debug = bool((config.get("logging") or {}).get("playoff_debug", False))
                    playoff_rank_map = compute_playoff_performance_rank(
                        pg, ptgl, games, tgl, target_season,
                        all_team_ids=unique_team_ids,
                        season_start=season_start,
                        season_end=season_end,
                        debug=playoff_debug,
                    )
                    eos_final_rank_map = compute_eos_final_rank(
                        pg, ptgl, games, tgl, target_season,
                        all_team_ids=unique_team_ids,
                        season_start=season_start,
                        season_end=season_end,
                        debug=playoff_debug,
                    )
                    if eos_final_rank_map and len(eos_final_rank_map) >= 16:
                        actual_global_rank = {int(tid): int(r) for tid, r in eos_final_rank_map.items()}
                        eos_rank_source = "eos_final_rank"
            except Exception as e:
                print(
                    f"EOS/playoff rank failed (falling back to standings): {e}",
                    file=sys.stderr,
                )
        if not playoff_rank_map and target_season:
            print(
                f"Warning: No playoff_rank for season {target_season}; predictions will lack post_playoff_rank. "
                "eval_report.json will not include playoff_metrics (sweep --objective playoff_spearman will be -inf). "
                "Ensure DB has playoff_games and playoff_team_game_logs for this season.",
                file=sys.stderr,
            )

        if config.get("inference", {}).get("require_eos_final_rank", False) and eos_rank_source != "eos_final_rank":
            print(
                "Inference requires eos_final_rank (playoff-based EOS) but DB returned standings. "
                "Ensure DB has playoff_games and playoff_team_game_logs populated (run 2_build_db with playoff raw data).",
                file=sys.stderr,
            )
            sys.exit(1)

        # Model A score std from confidence (heuristic)
        unc_cfg = config.get("uncertainty", {}) or {}
        scale = float(unc_cfg.get("conf_to_std_scale", 0.25))
        std_floor = float(unc_cfg.get("score_std_floor", 1e-6))
        std_a_arr = np.maximum((1.0 - conf_a_arr) * scale, std_floor).astype(np.float32)

        preds = predict_teams(
            unique_team_ids,
            team_names,
            model_a_scores=sa,
            xgb_scores=sx,
            rf_scores=sr,
            model_a_score_std=std_a_arr,
            xgb_score_std=std_xgb_arr,
            rf_score_std=std_rf_arr,
            extra_model_scores=extra_scores,
            extra_model_score_std=extra_stds,
            meta_model=meta,
            conf_a=conf_a_arr,
            conf_xgb=conf_xgb_arr,
            standings_win_rate=np.array([win_rate_map.get(tid, 0.0) for tid in unique_team_ids], dtype=np.float64),
            actual_ranks=actual_ranks,
            actual_global_ranks=actual_global_rank,
            attention_by_team=attention_by_team if attention_by_team else None,
            attention_by_temp_by_team=attention_by_temp_by_team if attention_by_temp_by_team else None,
            attention_fallback_by_team=attention_fallback_by_team if attention_fallback_by_team else None,
            team_id_to_conference=team_id_to_conf,
            playoff_rank=playoff_rank_map if playoff_rank_map else None,
            eos_playoff_standings=eos_playoff_standings_map if eos_playoff_standings_map else None,
            model_presence={"a": model_a is not None, "xgb": xgb is not None, "rf": rf is not None},
            true_strength_scale=config.get("output", {}).get("true_strength_scale", "percentile"),
            odds_temperature=float(config.get("output", {}).get("odds_temperature", 1.0)),
            championship_odds_method=config.get("output", {}).get("championship_odds_method", "softmax"),
            monte_carlo_config=config.get("uncertainty"),
            xgb_weight=xgb_weight,
            meta_by_conference=meta_by_conference,
        )

        # Integrated Gradients summary in predictions.json (optional, top-K per conference)
        ig_by_team: dict[int, list[dict[str, Any]]] = {}
        ig_top_k = int(config.get("output", {}).get("ig_inference_top_k", 1))
        if ig_top_k > 0 and model_a_dev is not None and batches_a and team_id_to_batch:
            try:
                from src.viz.integrated_gradients import ig_attr, _HAS_CAPTUM
                if _HAS_CAPTUM:
                    ig_steps = int(config.get("output", {}).get("ig_inference_steps", 50))
                    for conf in ("E", "W"):
                        conf_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == conf]
                        conf_preds = sorted(
                            conf_preds,
                            key=lambda t: t["prediction"].get("conference_rank") or t["prediction"]["predicted_strength"],
                        )
                        for t in conf_preds[:ig_top_k]:
                            tid = int(t["team_id"])
                            if tid not in team_id_to_batch:
                                continue
                            b_idx, k = team_id_to_batch[tid]
                            if b_idx >= len(batches_a):
                                continue
                            batch = batches_a[b_idx]
                            emb = batch["embedding_indices"][:, k, :]
                            stats = batch["player_stats"][:, k, :, :]
                            minu = batch["minutes"][:, k, :]
                            msk = batch["key_padding_mask"][:, k, :]
                            with torch.no_grad():
                                s_check, _, _ = model_a_dev(emb, stats, minu, msk)
                            if not torch.isfinite(s_check).all():
                                continue
                            attr, _ = ig_attr(model_a_dev, emb, stats, minu, msk, n_steps=ig_steps)
                            if attr is None or attr.numel() == 0:
                                continue
                            attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
                            if not torch.isfinite(attr).all():
                                continue
                            norms = torch.norm(attr[0].float(), dim=1)
                            if norms.numel() == 0:
                                continue
                            topk = min(5, norms.shape[0])
                            vals, idxs = norms.topk(topk, largest=True)
                            player_ids = team_id_to_player_ids.get(tid, [])
                            contrib: list[dict[str, Any]] = []
                            for v, idx in zip(vals.tolist(), idxs.tolist()):
                                if idx >= len(player_ids) or player_ids[idx] is None:
                                    continue
                                if not np.isfinite(v):
                                    continue
                                pid = player_ids[idx]
                                name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                contrib.append({"player": name, "attribution_norm": float(v)})
                            if contrib:
                                ig_by_team[tid] = contrib
                else:
                    print("Integrated Gradients skipped (captum not installed).", file=sys.stderr)
            except Exception as e:
                print(f"Integrated Gradients inference failed: {e}", file=sys.stderr)

        if ig_by_team:
            for t in preds:
                tid = int(t["team_id"])
                if tid in ig_by_team:
                    rd = t.get("roster_dependence") or {}
                    rd["ig_contributors"] = ig_by_team[tid]
                    t["roster_dependence"] = rd

        pred_payload: dict[str, Any] = {"teams": preds}
        pred_payload["eos_rank_source"] = eos_rank_source
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(pred_payload, f, indent=2, allow_nan=False)

        if draw_figures:
            east_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == "E"]
            west_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "W") == "W"]

            fig, (ax_east, ax_west) = plt.subplots(1, 2, figsize=(14, 6))

            def _draw_panel(ax, pred_list, title, marker="o"):
                """East = circle (o), West = diamond (D); color by tab20."""
                if not pred_list:
                    ax.text(0.5, 0.5, f"No {title} teams", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    ax.set_xlabel("Conference specific Outcome Rank")
                    ax.set_ylabel("Predicted conference rank")
                    ax.grid(True, linestyle="--", alpha=0.7)
                    return
                points = []
                for t in pred_list:
                    pr = t["prediction"].get("conference_rank")
                    ar = t["analysis"].get("historic_conference_rank") or t["analysis"].get("actual_conference_rank") or t["analysis"].get("EOS_conference_rank")
                    if pr is None or ar is None:
                        continue
                    points.append((ar, pr, t["team_name"]))
                if not points:
                    ax.text(0.5, 0.5, f"No valid {title} ranks", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    ax.set_xlabel("Conference specific Outcome Rank")
                    ax.set_ylabel("Predicted conference rank")
                    ax.grid(True, linestyle="--", alpha=0.7)
                    return
                ar, pr, names = zip(*points)
                max_r = max(max(ar or [1]), max(pr or [1]), 1) + 1
                ax.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                cmap = plt.get_cmap("tab20")
                for i, (a, p) in enumerate(zip(ar, pr)):
                    color = cmap(i % 20)
                    ax.scatter(a, p, c=[color], label=names[i], s=60, marker=marker, edgecolors="k", linewidths=0.5)
                ax.set_xlabel("Conference specific Outcome Rank")
                ax.set_ylabel("Predicted conference rank")
                ax.set_title(title)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(loc="best", fontsize=7, ncol=2)
                ax.set_xlim(-0.5, max_r)
                ax.set_ylim(-0.5, max_r)

            _draw_panel(ax_east, east_preds, "East", marker="o")
            _draw_panel(ax_west, west_preds, "West", marker="D")
            fig.suptitle("Predicted vs Conference specific Outcome Rank (1-15)", fontsize=12)
            fig.tight_layout()
            fig.savefig(out / f"pred_vs_actual_conference_rank{fig_suffix}.png", bbox_inches="tight")
            plt.close()

            # pred_vs_playoff_outcome_rank: x = playoff outcome rank (1-30), y = predicted rank (1-30). Dots = teams.
            # Legend outside so all points visible; East = circle (o), West = diamond (D); color coordinated
            if playoff_rank_map:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                pts = [(t["analysis"].get("post_playoff_rank"), t["prediction"].get("global_rank") or t["prediction"]["predicted_strength"], t["team_name"], team_id_to_conf.get(t["team_id"], "E")) for t in preds if t["analysis"].get("post_playoff_rank") is not None]
                if not pts:
                    ax2.text(0.5, 0.5, "No playoff outcome ranks available", ha="center", va="center", transform=ax2.transAxes)
                    ax2.set_xlabel("Playoff outcome rank (1-30)")
                    ax2.set_ylabel("Predicted rank (1-30)")
                    ax2.set_title("Predicted vs playoff outcome rank")
                    ax2.grid(True, linestyle="--", alpha=0.7)
                    fig2.savefig(out / f"pred_vs_playoff_outcome_rank{fig_suffix}.png", bbox_inches="tight")
                    plt.close(fig2)
                else:
                    east_pts = [(p_rank, g_rank, name) for (p_rank, g_rank, name, conf) in pts if conf == "E"]
                    west_pts = [(p_rank, g_rank, name) for (p_rank, g_rank, name, conf) in pts if conf == "W"]
                    max_r = max(max(g for _, g, _, _ in pts), max(p for p, _, _, _ in pts), 1) + 1
                    ax2.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                    cmap = plt.get_cmap("tab20")
                    for i, (p_rank, g_rank, name) in enumerate(east_pts):
                        ax2.scatter(p_rank, g_rank, c=[cmap(i % 20)], label=name, s=50, marker="o", edgecolors="k", linewidths=0.5)
                    for i, (p_rank, g_rank, name) in enumerate(west_pts):
                        ax2.scatter(p_rank, g_rank, c=[cmap((len(east_pts) + i) % 20)], label=name, s=50, marker="D", edgecolors="k", linewidths=0.5)
                    ax2.set_xlabel("Playoff outcome rank (1-30)")
                    ax2.set_ylabel("Predicted rank (1-30)")
                    ax2.set_title("Predicted vs playoff outcome rank")
                    ax2.grid(True, linestyle="--", alpha=0.7)
                    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6, ncol=1)
                    ax2.set_xlim(-0.5, max_r)
                    ax2.set_ylim(-0.5, max_r)
                    fig2.tight_layout(rect=[0, 0, 0.72, 1])
                    fig2.savefig(out / f"pred_vs_playoff_outcome_rank{fig_suffix}.png", bbox_inches="tight")
                    plt.close(fig2)

            # Championship odds top-10 bar chart
            sorted_preds = sorted(preds, key=lambda t: float(t["prediction"]["championship_odds"].rstrip("%")), reverse=True)[:10]
            if sorted_preds:
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                names10 = [t["team_name"] for t in sorted_preds]
                odds10 = [float(t["prediction"]["championship_odds"].rstrip("%")) for t in sorted_preds]
                ax3.barh(range(len(names10)), odds10, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names10))))
                ax3.set_yticks(range(len(names10)))
                ax3.set_yticklabels(names10, fontsize=9)
                ax3.set_xlabel("Championship odds (%)")
                ax3.set_title("Top 10 championship odds")
                ax3.grid(True, axis="x", linestyle="--", alpha=0.7)
                fig3.tight_layout()
                fig3.savefig(out / f"odds_top10{fig_suffix}.png", bbox_inches="tight")
                plt.close(fig3)

            # Title contender scatter: championship odds vs regular-season wins (win rate * games proxy)
            # East = circle (o), West = diamond (D); color coordinated (tab20)
            team_id_to_wins: dict[int, float] = team_id_to_win_rate
            n_games = 82.0
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            east_preds_4 = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == "E"]
            west_preds_4 = [t for t in preds if team_id_to_conf.get(t["team_id"], "W") == "W"]
            cmap4 = plt.get_cmap("tab20")
            for i, t in enumerate(east_preds_4):
                ax4.scatter(team_id_to_wins.get(t["team_id"], 0.0) * n_games, float(t["prediction"]["championship_odds"].rstrip("%")), s=80, label=t["team_name"], alpha=0.8, c=[cmap4(i % 20)], marker="o", edgecolors="k", linewidths=0.5)
            for i, t in enumerate(west_preds_4):
                ax4.scatter(team_id_to_wins.get(t["team_id"], 0.0) * n_games, float(t["prediction"]["championship_odds"].rstrip("%")), s=80, label=t["team_name"], alpha=0.8, c=[cmap4((len(east_preds_4) + i) % 20)], marker="D", edgecolors="k", linewidths=0.5)
            ax4.set_xlabel("Regular-season wins (proxy from standings-to-date win rate × 82)")
            ax4.set_ylabel("Championship odds (%)")
            ax4.set_title("Title contender: odds vs wins (top-left = sleeper)")
            ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
            ax4.grid(True, linestyle="--", alpha=0.7)
            fig4.tight_layout()
            fig4.savefig(out / f"title_contender_scatter{fig_suffix}.png", bbox_inches="tight")
            plt.close(fig4)

            # EOS standings (x) vs playoff outcome rank (y). Dots = teams. East = circle (o), West = diamond (D).
            if eos_playoff_standings_map and eos_final_rank_map:
                fig5, ax5 = plt.subplots(figsize=(8, 6))
                pts = []
                for t in preds:
                    standings = t["analysis"].get("EOS_playoff_standings")
                    eos = t["analysis"].get("EOS_global_rank")
                    if standings is not None and eos is not None:
                        conf = team_id_to_conf.get(t["team_id"], "E")
                        pts.append((standings, eos, t["team_name"], conf))
                if pts:
                    east_pts5 = [(x, y, name) for (x, y, name, conf) in pts if conf == "E"]
                    west_pts5 = [(x, y, name) for (x, y, name, conf) in pts if conf == "W"]
                    all_x = [p[0] for p in pts]
                    all_y = [p[1] for p in pts]
                    max_r = max(max(all_x or [1]), max(all_y or [1]), 1) + 1
                    ax5.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                    cmap = plt.get_cmap("tab20")
                    for i, (x, y, name) in enumerate(east_pts5):
                        ax5.scatter(x, y, c=[cmap(i % 20)], label=name, s=50, marker="o", edgecolors="k", linewidths=0.5)
                    for i, (x, y, name) in enumerate(west_pts5):
                        ax5.scatter(x, y, c=[cmap((len(east_pts5) + i) % 20)], label=name, s=50, marker="D", edgecolors="k", linewidths=0.5)
                    ax5.set_xlabel("EOS standings (1-30)")
                    ax5.set_ylabel("Playoff outcome rank (1-30)")
                    ax5.set_title("EOS standings vs playoff outcome rank (identity = agreement)")
                    ax5.grid(True, linestyle="--", alpha=0.7)
                    ax5.legend(loc="best", fontsize=6, ncol=2)
                    ax5.set_xlim(-0.5, max_r)
                    ax5.set_ylim(-0.5, max_r)
                fig5.savefig(out / f"eos_standings_vs_playoff_outcome_rank{fig_suffix}.png", bbox_inches="tight")
                plt.close(fig5)

        return pj

    last_pj = None
    for target_date, target_lists, output_file, season in test_specs:
        last_pj = _run_inference_for_spec(target_date, target_lists, output_file, season, draw_figures=True, feat_df_all=feat_df_all)
    if last_pj is not None and last_pj.name != "predictions.json":
        import shutil
        shutil.copy(last_pj, out / "predictions.json")
    for target_date, target_lists, output_file, _ in train_specs:
        _run_inference_for_spec(target_date, target_lists, output_file, None, draw_figures=False, feat_df_all=feat_df_all)
    for target_date, target_lists, output_file, _ in val_specs:
        _run_inference_for_spec(target_date, target_lists, output_file, None, draw_figures=False, feat_df_all=feat_df_all)

    return last_pj if last_pj is not None else out / "predictions.json"



def run_inference(output_dir: str | Path, config: dict, run_id: str | None = None) -> Path:
    """Run inference: from DB if present and has data, else exit with message (real run only)."""
    import os

    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    paths_cfg = config.get("paths", {})
    db_path = Path(paths_cfg.get("db", "data/processed/nba_build.duckdb"))
    # Use canonical DB with playoff data when NBA_DB_PATH is set (e.g. from worktree)
    db_override = os.environ.get("NBA_DB_PATH")
    if db_override and str(db_override).strip():
        db_path = Path(db_override.strip())
    elif not db_path.is_absolute():
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
