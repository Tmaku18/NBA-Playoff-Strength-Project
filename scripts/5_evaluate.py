"""Script 5: Evaluate predictions and write metrics.

What this does:
- Loads predictions.json from the latest run (or specified run_id).
- Computes NDCG, Spearman, rank MAE/RMSE, MRR, Brier, etc. vs actual playoff outcome.
- Writes eval_report.json and optionally ANALYSIS_NN.md to the run folder.
- Compares ensemble, Model A, XGB, and RF performance.

Run after script 6 (inference). Can re-run on existing predictions to update metrics."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_ranking, evaluate_upset
from src.evaluation.metrics import brier_champion, ndcg_at_4, ndcg_at_30, ndcg_score, rank_mae, rank_rmse, spearman
from src.utils.split import load_split_info


def _next_analysis_number(run_dir: Path) -> int:
    """Return next ANALYSIS number (01, 02, ...) for run_dir. Scans existing ANALYSIS_*.md."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return 1
    numbers = []
    for p in run_dir.glob("ANALYSIS_*.md"):
        m = re.match(r"^ANALYSIS_(\d+)\.md$", p.name, re.I)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers, default=0) + 1


def _latest_run_id(outputs_dir: Path) -> str | None:
    """Return the latest run_NNN (highest number) present in outputs_dir, or None."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return None
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and pattern.match(p.name):
            if (p / "predictions.json").exists() or list(p.glob("predictions_*.json")):
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers:
        return None
    return f"run_{max(numbers):03d}"


def _teams_to_arrays(teams: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, str]]:
    """From the teams list in predictions.json, extract actual ranks, predicted scores/ranks, team IDs, and conference map."""
    actual_ranks = []
    pred_scores = []
    pred_ranks = []
    team_ids = []
    team_id_to_conf: dict[int, str] = {}
    for t in teams:
        analysis = t.get("analysis", {})
        act = analysis.get("EOS_global_rank")
        pred_rank = t.get("prediction", {}).get("predicted_strength")
        tss = t.get("prediction", {}).get("ensemble_score")
        conf = t.get("conference")
        if act is None:
            continue
        actual_ranks.append(float(act))
        pred_ranks.append(pred_rank if pred_rank is not None else 0)
        pred_scores.append(tss if tss is not None else 0.0)
        team_ids.append(int(t["team_id"]))
        if conf is not None:
            team_id_to_conf[int(t["team_id"])] = str(conf)
    y_actual = np.array(actual_ranks, dtype=np.float32)
    y_score = np.array(pred_scores, dtype=np.float32)
    pred_ranks_arr = np.array(pred_ranks, dtype=np.float32)
    return y_actual, y_score, pred_ranks_arr, team_ids, team_id_to_conf


def _teams_to_arrays_by_model(
    teams: list,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[int], dict[int, str]]:
    """Extract y_actual, y_score, pred_ranks for ensemble, model_a, xgb, rf (Model C: LR diagnostics)."""
    actual_ranks = []
    team_ids = []
    team_id_to_conf: dict[int, str] = {}
    scores_by_model: dict[str, list[float]] = {"ensemble": [], "model_a": [], "xgb": [], "rf": []}
    pred_ranks_by_model: dict[str, list[float]] = {"ensemble": [], "model_a": [], "xgb": [], "rf": []}
    for t in teams:
        analysis = t.get("analysis", {})
        act = analysis.get("EOS_global_rank")
        if act is None:
            continue
        pred = t.get("prediction", {})
        diag = t.get("ensemble_diagnostics", {})
        actual_ranks.append(float(act))
        team_ids.append(int(t["team_id"]))
        conf = t.get("conference")
        if conf is not None:
            team_id_to_conf[int(t["team_id"])] = str(conf)
        tss = pred.get("ensemble_score")
        pr = pred.get("predicted_strength")
        scores_by_model["ensemble"].append(float(tss) if tss is not None else 0.0)
        pred_ranks_by_model["ensemble"].append(float(pr) if pr is not None else 0.0)
        r_a = diag.get("deep_set_rank")
        r_x = diag.get("xgboost_rank")
        r_r = diag.get("model_c_rank") or diag.get("random_forest_rank")
        scores_by_model["model_a"].append(31.0 - float(r_a) if r_a is not None else 0.0)
        scores_by_model["xgb"].append(31.0 - float(r_x) if r_x is not None else 0.0)
        scores_by_model["rf"].append(31.0 - float(r_r) if r_r is not None else 0.0)
        pred_ranks_by_model["model_a"].append(float(r_a) if r_a is not None else 0.0)
        pred_ranks_by_model["xgb"].append(float(r_x) if r_x is not None else 0.0)
        pred_ranks_by_model["rf"].append(float(r_r) if r_r is not None else 0.0)
    y_actual = np.array(actual_ranks, dtype=np.float32)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in scores_by_model:
        if scores_by_model[name]:
            out[name] = (
                y_actual,
                np.array(scores_by_model[name], dtype=np.float32),
                np.array(pred_ranks_by_model[name], dtype=np.float32),
            )
    return out, team_ids, team_id_to_conf


def _compute_metrics_from_arrays(
    y_actual: np.ndarray,
    y_score: np.ndarray,
    pred_ranks_arr: np.ndarray,
    *,
    k: int = 30,
) -> dict:
    """Compute ranking metrics (NDCG, Spearman, MRR, ROC-AUC for upset prediction) from aligned arrays."""
    n = len(y_actual)
    if n < 2:
        return {
            "ndcg": 0.0, "ndcg_at_4": 0.0, "ndcg_at_12": 0.0, "ndcg_at_16": 0.0, "ndcg_at_20": 0.0, "ndcg_at_30": 0.0,
            "spearman": 0.0, "mrr_top2": 0.0, "mrr_top4": 0.0, "roc_auc_upset": 0.5,
            "rank_mae_pred_vs_playoff_outcome_rank": float("nan"), "rank_rmse_pred_vs_playoff_outcome_rank": float("nan"),
        }
    max_rank = int(np.max(y_actual)) if n else 0
    y_true_relevance = (max_rank - y_actual + 1).clip(1, max_rank if max_rank > 0 else 1)
    m = evaluate_ranking(y_true_relevance, y_score, k=min(k, n))
    m["ndcg_at_4"] = float(ndcg_score(y_true_relevance, y_score, k=min(4, n)))
    m["ndcg_at_12"] = float(ndcg_score(y_true_relevance, y_score, k=min(12, n)))
    m["ndcg_at_16"] = float(ndcg_score(y_true_relevance, y_score, k=min(16, n)))
    m["ndcg_at_20"] = float(ndcg_score(y_true_relevance, y_score, k=min(20, n)))
    m["ndcg_at_30"] = float(ndcg_score(y_true_relevance, y_score, k=min(30, n)))
    # MAE and RMSE of predicted rank vs actual outcome rank (same outcome as standings are scored against)
    m["rank_mae_pred_vs_playoff_outcome_rank"] = float(rank_mae(pred_ranks_arr, y_actual))
    m["rank_rmse_pred_vs_playoff_outcome_rank"] = float(rank_rmse(pred_ranks_arr, y_actual))
    delta = y_actual - pred_ranks_arr
    y_bin = (delta > 0).astype(np.float32)
    if np.unique(y_bin).size >= 2:
        m2 = evaluate_upset(y_bin, y_score)
        m.update(m2)
    else:
        m["roc_auc_upset"] = 0.5
    return m


def _compute_metrics(teams: list, *, k: int = 30) -> dict:
    """Compute ndcg, spearman, mrr, roc_auc_upset and optionally playoff_metrics from teams list."""
    y_actual, y_score, pred_ranks_arr, _, _ = _teams_to_arrays(teams)
    m = _compute_metrics_from_arrays(y_actual, y_score, pred_ranks_arr, k=k)
    # Rank-distance metrics: predicted rank vs Playoff Outcome Rank (lower is better)
    m["rank_mae_pred_vs_playoff_outcome_rank"] = float(rank_mae(pred_ranks_arr, y_actual))
    m["rank_rmse_pred_vs_playoff_outcome_rank"] = float(rank_rmse(pred_ranks_arr, y_actual))
    # W/L record standings vs Playoff Outcome Rank (baseline: how far reg-season rank was from playoff outcome)
    standings_list = []
    actual_list = []
    for t in teams:
        act = t.get("analysis", {}).get("EOS_global_rank")
        stand = t.get("analysis", {}).get("EOS_playoff_standings")
        if act is not None and stand is not None:
            actual_list.append(float(act))
            standings_list.append(float(stand))
    if len(actual_list) >= 16:
        m["rank_mae_wl_record_standings_vs_playoff_outcome_rank"] = float(rank_mae(np.array(standings_list), np.array(actual_list)))
        m["rank_rmse_wl_record_standings_vs_playoff_outcome_rank"] = float(rank_rmse(np.array(standings_list), np.array(actual_list)))
    # Standings-based metrics: pred vs W/L record standings (EOS_playoff_standings)
    standings_arr = []
    pred_scores_for_standings = []
    pred_ranks_for_standings = []
    for t in teams:
        stand = t.get("analysis", {}).get("EOS_playoff_standings")
        if stand is None:
            continue
        tss = t.get("prediction", {}).get("ensemble_score")
        pr = t.get("prediction", {}).get("predicted_strength")
        standings_arr.append(float(stand))
        pred_scores_for_standings.append(float(tss) if tss is not None else 0.0)
        pred_ranks_for_standings.append(float(pr) if pr is not None else 0.0)
    if len(standings_arr) >= 2:
        standings_arr = np.array(standings_arr, dtype=np.float32)
        pred_scores_for_standings = np.array(pred_scores_for_standings, dtype=np.float32)
        pred_ranks_for_standings = np.array(pred_ranks_for_standings, dtype=np.float32)
        max_rank_s = int(np.max(standings_arr))
        relevance_standings = (max_rank_s - standings_arr + 1).clip(1, max_rank_s if max_rank_s > 0 else 1)
        m["spearman_standings"] = float(spearman(standings_arr, pred_scores_for_standings))
        m["ndcg_at_4_standings"] = float(ndcg_score(relevance_standings, pred_scores_for_standings, k=min(4, len(standings_arr))))
        m["ndcg_at_16_standings"] = float(ndcg_score(relevance_standings, pred_scores_for_standings, k=min(16, len(standings_arr))))
        m["ndcg_at_30_standings"] = float(ndcg_score(relevance_standings, pred_scores_for_standings, k=min(30, len(standings_arr))))
        m["rank_rmse_standings"] = float(rank_rmse(pred_ranks_for_standings, standings_arr))
    playoff_rows = []
    for t in teams:
        p_rank = t.get("analysis", {}).get("post_playoff_rank")
        if p_rank is None:
            continue
        g_rank = t.get("prediction", {}).get("global_rank") or t.get("prediction", {}).get("predicted_strength") or 0
        odds_str = t.get("prediction", {}).get("championship_odds", "0%")
        playoff_rows.append((float(p_rank), float(g_rank), odds_str))
    if len(playoff_rows) >= 16:
        p_rank = np.array([r[0] for r in playoff_rows], dtype=np.float32)
        g_rank = np.array([r[1] for r in playoff_rows], dtype=np.float32)
        odds_pct = np.array([float(r[2].rstrip("%")) / 100.0 for r in playoff_rows], dtype=np.float32)
        champion_onehot = (p_rank == 1).astype(np.float32)
        m["playoff_metrics"] = {
            "spearman_pred_vs_playoff_outcome_rank": float(spearman(p_rank, g_rank)),
            "ndcg_at_4_final_four": float(ndcg_at_4(p_rank, -g_rank)),
            "ndcg_at_30_pred_vs_playoff_outcome_rank": float(ndcg_at_30(p_rank, -g_rank)),
            "brier_championship_odds": float(brier_champion(champion_onehot, odds_pct)),
            "rank_mae_pred_vs_playoff_outcome_rank": m["rank_mae_pred_vs_playoff_outcome_rank"],
            "rank_rmse_pred_vs_playoff_outcome_rank": m["rank_rmse_pred_vs_playoff_outcome_rank"],
            "rank_mae_wl_record_standings_vs_playoff_outcome_rank": m.get("rank_mae_wl_record_standings_vs_playoff_outcome_rank", float("nan")),
            "rank_rmse_wl_record_standings_vs_playoff_outcome_rank": m.get("rank_rmse_wl_record_standings_vs_playoff_outcome_rank", float("nan")),
        }
    return m


def _model_vs_standings_comparison(
    teams: list,
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    """Compare each model to regular-season W/L standings vs same outcome ranks. Includes MAE/RMSE per model,
    improvement over standings, and paired bootstrap significance (model better than standings?)."""
    y_actual_list = []
    standings_list = []
    pred_by_model: dict[str, list[float]] = {"ensemble": [], "model_a": [], "xgb": [], "rf": []}
    for t in teams:
        act = t.get("analysis", {}).get("EOS_global_rank")
        stand = t.get("analysis", {}).get("EOS_playoff_standings")
        if act is None or stand is None:
            continue
        y_actual_list.append(float(act))
        standings_list.append(float(stand))
        pred = t.get("prediction", {})
        diag = t.get("ensemble_diagnostics", {})
        pr = pred.get("predicted_strength")
        pred_by_model["ensemble"].append(float(pr) if pr is not None else 0.0)
        r_a = diag.get("deep_set_rank")
        r_x = diag.get("xgboost_rank")
        r_r = diag.get("model_c_rank") or diag.get("random_forest_rank")
        pred_by_model["model_a"].append(float(r_a) if r_a is not None else 0.0)
        pred_by_model["xgb"].append(float(r_x) if r_x is not None else 0.0)
        pred_by_model["rf"].append(float(r_r) if r_r is not None else 0.0)
    n = len(y_actual_list)
    if n < 10:
        return {}
    y_actual = np.array(y_actual_list, dtype=np.float64)
    standings = np.array(standings_list, dtype=np.float64)
    out: dict = {
        "n_teams": n,
        "standings_vs_outcome": {
            "rank_mae": float(rank_mae(standings, y_actual)),
            "rank_rmse": float(rank_rmse(standings, y_actual)),
        },
        "models": {},
        "significance": {},
    }
    rng = np.random.default_rng(seed)
    for name, pred_list in pred_by_model.items():
        pred_arr = np.array(pred_list, dtype=np.float64)
        mae = float(rank_mae(pred_arr, y_actual))
        rmse = float(rank_rmse(pred_arr, y_actual))
        stand_mae = out["standings_vs_outcome"]["rank_mae"]
        stand_rmse = out["standings_vs_outcome"]["rank_rmse"]
        out["models"][name] = {
            "rank_mae_pred_vs_outcome": mae,
            "rank_rmse_pred_vs_outcome": rmse,
            "improvement_mae_vs_standings": stand_mae - mae,
            "improvement_rmse_vs_standings": stand_rmse - rmse,
        }
        # Paired bootstrap: per-team absolute errors; diff_i = err_standings_i - err_model_i
        err_standings = np.abs(standings - y_actual)
        err_model = np.abs(pred_arr - y_actual)
        diffs = err_standings - err_model  # positive when model is better
        mean_improvement = float(np.mean(diffs))
        boot_means = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            boot_means.append(float(np.mean(diffs[idx])))
        boot_means = np.array(boot_means)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        # p-value: proportion of bootstrap samples where improvement <= 0 (null: no improvement)
        p_value = float(np.mean(boot_means <= 0))
        out["significance"][name] = {
            "mean_mae_improvement": mean_improvement,
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "p_value_model_better_than_standings": p_value,
            "method": "paired bootstrap over teams (resample teams, mean(standings_ae - model_ae)); H0: no improvement",
        }
    return out


def _metrics_by_conference(
    teams: list,
    *,
    use_conference_ranks: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute NDCG and Spearman per conference (E, W).
    When use_conference_ranks=True (default), relevance is defined **within conference** using
    EOS global rank: within each conference, teams are ranked 1..n by EOS_global_rank (ascending),
    so best team in conference gets rank 1. NDCG uses that relevance vs ensemble_score;
    Spearman uses derived Historic Conference Rank vs prediction.conference_rank.
    """
    out: dict[str, dict[str, float]] = {}
    for conf in ("E", "W"):
        conf_teams = [t for t in teams if t.get("conference") == conf]
        if len(conf_teams) < 2:
            continue
        if use_conference_ranks:
            # Derive actual within-conference rank from EOS_global_rank (same source as global metrics)
            rows = []
            for t in conf_teams:
                eos = t.get("analysis", {}).get("EOS_global_rank")
                pred_rank = t.get("prediction", {}).get("conference_rank")
                score = t.get("prediction", {}).get("ensemble_score")
                if eos is None:
                    continue
                rows.append((float(eos), float(pred_rank) if pred_rank is not None else None, float(score) if score is not None else 0.0))
            if len(rows) < 2:
                continue
            # Sort by EOS global rank ascending: best (lowest number) first → conference rank 1, 2, ...
            rows.sort(key=lambda x: (x[0], x[1] or 0))
            n = len(rows)
            derived_actual_rank = np.arange(1, n + 1, dtype=np.float32)  # 1=best in conf, n=worst
            relevance = (n - derived_actual_rank + 1).clip(1, n)  # higher = better
            pred_ranks = np.array([r[1] if r[1] is not None else np.nan for r in rows], dtype=np.float32)
            y_score = np.array([r[2] for r in rows], dtype=np.float32)
            # NDCG: relevance (EOS strength within conf) vs ensemble_score (higher = better)
            ndcg = ndcg_score(relevance, y_score, k=min(10, n))
            # Spearman: derived actual conference rank vs predicted conference rank (both lower = better)
            valid = np.isfinite(pred_ranks)
            if np.sum(valid) >= 2:
                sp = spearman(derived_actual_rank[valid], pred_ranks[valid])
            else:
                sp = 0.0
        else:
            y_actual, y_score, _, team_ids, team_id_to_conf = _teams_to_arrays(teams)
            idx = [i for i, tid in enumerate(team_ids) if team_id_to_conf.get(tid) == conf]
            if len(idx) < 2:
                continue
            ya = y_actual[idx]
            ys = y_score[idx]
            max_rank = int(np.max(ya))
            rel = (max_rank - ya + 1).clip(1, max_rank if max_rank > 0 else 1)
            ndcg = ndcg_score(rel, ys, k=min(10, len(rel)))
            sp = spearman(ya, ys)
        # W/L record standings vs Playoff Outcome Rank (baseline) per conference
        standings_list = []
        actual_list = []
        pred_ensemble = []
        pred_model_a = []
        pred_xgb = []
        pred_rf = []
        for t in conf_teams:
            act = t.get("analysis", {}).get("EOS_global_rank")
            stand = t.get("analysis", {}).get("EOS_playoff_standings")
            diag = t.get("ensemble_diagnostics", {})
            pred = t.get("prediction", {})
            if act is not None and stand is not None:
                actual_list.append(float(act))
                standings_list.append(float(stand))
                pred_ensemble.append(float(pred.get("predicted_strength") or 0))
                pred_model_a.append(float(diag.get("deep_set_rank") or 0))
                pred_xgb.append(float(diag.get("xgboost_rank") or diag.get("model_b_rank") or 0))
                pred_rf.append(float(diag.get("random_forest_rank") or diag.get("model_c_rank") or 0))
        conf_entry = {"ndcg": float(ndcg), "spearman": float(sp)}
        if len(standings_list) >= 2:
            actual_arr = np.array(actual_list, dtype=np.float32)
            conf_entry["rank_mae_wl_record_standings_vs_playoff_outcome_rank"] = float(
                rank_mae(np.array(standings_list), actual_arr)
            )
            conf_entry["rank_rmse_wl_record_standings_vs_playoff_outcome_rank"] = float(
                rank_rmse(np.array(standings_list), actual_arr)
            )
            # Model pred vs playoff outcome rank per conference
            conf_entry["rank_mae_ensemble_pred_vs_playoff_outcome_rank"] = float(
                rank_mae(np.array(pred_ensemble), actual_arr)
            )
            conf_entry["rank_rmse_ensemble_pred_vs_playoff_outcome_rank"] = float(
                rank_rmse(np.array(pred_ensemble), actual_arr)
            )
            conf_entry["rank_mae_model_a_pred_vs_playoff_outcome_rank"] = float(
                rank_mae(np.array(pred_model_a), actual_arr)
            )
            conf_entry["rank_rmse_model_a_pred_vs_playoff_outcome_rank"] = float(
                rank_rmse(np.array(pred_model_a), actual_arr)
            )
            conf_entry["rank_mae_xgb_pred_vs_playoff_outcome_rank"] = float(
                rank_mae(np.array(pred_xgb), actual_arr)
            )
            conf_entry["rank_rmse_xgb_pred_vs_playoff_outcome_rank"] = float(
                rank_rmse(np.array(pred_xgb), actual_arr)
            )
            conf_entry["rank_mae_rf_pred_vs_playoff_outcome_rank"] = float(
                rank_mae(np.array(pred_rf), actual_arr)
            )
            conf_entry["rank_rmse_rf_pred_vs_playoff_outcome_rank"] = float(
                rank_rmse(np.array(pred_rf), actual_arr)
            )
        out[conf] = conf_entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_dir = Path(config["paths"]["outputs"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_id = config.get("inference", {}).get("run_id")
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        run_id = _latest_run_id(out_dir) or "run_001"
    else:
        run_id = str(run_id).strip()
    run_dir = out_dir / run_id
    pred_path = run_dir / "predictions.json"
    per_season_preds = list(run_dir.glob("predictions_*.json"))
    if not pred_path.exists() and not per_season_preds:
        print("Predictions not found. Run inference (script 6) first.", file=sys.stderr)
        sys.exit(1)

    report: dict = {"notes": {}}

    # Attach split info from script 3 (train/test dates, split mode) if available.
    try:
        split_info = load_split_info(out_dir)
        report["split_info"] = {
            "train_frac": split_info.get("train_frac"),
            "n_train_dates": split_info.get("n_train_dates"),
            "n_test_dates": split_info.get("n_test_dates"),
            "split_mode": split_info.get("split_mode"),
        }
        report["notes"]["eval_on"] = "test+train" if (run_dir / "train_predictions.json").exists() else "test"
    except FileNotFoundError:
        report["split_info"] = {}
        report["notes"]["eval_on"] = "test"

    by_season: dict[str, dict] = {}

    # Evaluate per-season files (e.g. predictions_2024-25.json) and write eval_report_<season>.json.
    for pred_file in sorted(per_season_preds):
        season = pred_file.stem.replace("predictions_", "")
        with open(pred_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = data.get("teams", [])
        if not teams:
            continue
        eos_source = data.get("eos_rank_source", "standings")
        missing_actual = sum(1 for t in teams if t.get("analysis", {}).get("EOS_global_rank") is None)
        if missing_actual > 0:
            print(f"Warning: {season}: {missing_actual} teams missing EOS_global_rank; skipped in evaluation.")
        metrics_ensemble = _compute_metrics(teams)
        by_model, _, _ = _teams_to_arrays_by_model(teams)
        metrics_model_a = _compute_metrics_from_arrays(by_model["model_a"][0], by_model["model_a"][1], by_model["model_a"][2]) if "model_a" in by_model else {}
        metrics_xgb = _compute_metrics_from_arrays(by_model["xgb"][0], by_model["xgb"][1], by_model["xgb"][2]) if "xgb" in by_model else {}
        metrics_rf = _compute_metrics_from_arrays(by_model["rf"][0], by_model["rf"][1], by_model["rf"][2]) if "rf" in by_model else {}
        conf_metrics = _metrics_by_conference(teams)
        model_vs_standings = _model_vs_standings_comparison(teams)
        season_report: dict = {
            "test_metrics_ensemble": metrics_ensemble,
            "test_metrics_model_a": metrics_model_a,
            "test_metrics_model_b": metrics_xgb,
            "test_metrics_model_c": metrics_rf,
            "test_metrics_by_conference": conf_metrics,
            "model_vs_standings_comparison": model_vs_standings,
            "notes": {"eos_rank_source": eos_source},
        }
        by_season[season] = season_report
        season_path = run_dir / f"eval_report_{season}.json"
        with open(season_path, "w", encoding="utf-8") as f:
            json.dump(season_report, f, indent=2)
        print(f"Wrote {season_path}")

    if not by_season and pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        primary_teams = data.get("teams", [])
        if primary_teams:
            report["notes"]["eos_rank_source"] = data.get("eos_rank_source", "standings")
            metrics_ens = _compute_metrics(primary_teams)
            by_model, _, _ = _teams_to_arrays_by_model(primary_teams)
            report["test_metrics_ensemble"] = metrics_ens
            report["test_metrics_model_a"] = _compute_metrics_from_arrays(by_model["model_a"][0], by_model["model_a"][1], by_model["model_a"][2]) if "model_a" in by_model else {}
            report["test_metrics_model_b"] = _compute_metrics_from_arrays(by_model["xgb"][0], by_model["xgb"][1], by_model["xgb"][2]) if "xgb" in by_model else {}
            report["test_metrics_model_c"] = _compute_metrics_from_arrays(by_model["rf"][0], by_model["rf"][1], by_model["rf"][2]) if "rf" in by_model else {}
            report["test_metrics_by_conference"] = _metrics_by_conference(primary_teams)
            report["model_vs_standings_comparison"] = _model_vs_standings_comparison(primary_teams)
    elif by_season:
        last_season = sorted(by_season.keys())[-1]
        last_report = by_season[last_season]
        report["test_metrics_ensemble"] = last_report["test_metrics_ensemble"].copy()
        report["test_metrics_model_a"] = last_report["test_metrics_model_a"]
        report["test_metrics_model_b"] = last_report.get("test_metrics_model_b") or last_report.get("test_metrics_xgb", {})
        report["test_metrics_model_c"] = last_report.get("test_metrics_model_c") or last_report.get("test_metrics_rf", {})
        report["test_metrics_by_conference"] = last_report["test_metrics_by_conference"]
        report["notes"]["eos_rank_source"] = last_report["notes"].get("eos_rank_source", "standings")
        report["by_season"] = by_season
        # Ensure playoff_metrics appear in main report if any season has them (sweep reads test_metrics_ensemble.playoff_metrics)
        if not report["test_metrics_ensemble"].get("playoff_metrics"):
            for _s, sreport in by_season.items():
                pm = sreport.get("test_metrics_ensemble", {}).get("playoff_metrics")
                if isinstance(pm, dict) and pm:
                    report["test_metrics_ensemble"]["playoff_metrics"] = pm
                    break
        # Model vs standings comparison and significance (from last season's teams)
        last_season = sorted(by_season.keys())[-1]
        last_pred_file = run_dir / f"predictions_{last_season}.json"
        if last_pred_file.exists():
            with open(last_pred_file, "r", encoding="utf-8") as f:
                last_teams = json.load(f).get("teams", [])
            if last_teams:
                report["model_vs_standings_comparison"] = _model_vs_standings_comparison(last_teams)
                report["model_vs_standings_comparison"]["season"] = last_season

    report["notes"]["eos_rank_source_meaning"] = "standings = W/L record standings; eos_final_rank = Playoff Outcome Rank"
    report["notes"]["upset_definition"] = "sleeper = EOS_global_rank > predicted_strength"
    report["notes"]["mrr_top2"] = "1/rank of first team in top 2 (champion+runner-up) in predicted order."
    report["notes"]["mrr_top4"] = "1/rank of first team in top 4 (Conference Finals) in predicted order."
    report["notes"]["ndcg_cutoff_labels"] = "ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order."
    report["notes"]["per_conference_relevance"] = "Within E/W: actual rank derived from EOS_global_rank (1=best in conf); NDCG/Spearman use this relevance."

    # Train metrics (if train_predictions.json exists)
    train_pred_path = run_dir / "train_predictions.json"
    if train_pred_path.exists():
        with open(train_pred_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        train_teams = train_data.get("teams", [])
        if train_teams:
            report["train_metrics_ensemble"] = _compute_metrics(train_teams)
            by_model_t, _, _ = _teams_to_arrays_by_model(train_teams)
            report["train_metrics_model_a"] = _compute_metrics_from_arrays(by_model_t["model_a"][0], by_model_t["model_a"][1], by_model_t["model_a"][2]) if "model_a" in by_model_t else {}
            report["train_metrics_model_b"] = _compute_metrics_from_arrays(by_model_t["xgb"][0], by_model_t["xgb"][1], by_model_t["xgb"][2]) if "xgb" in by_model_t else {}
            report["train_metrics_model_c"] = _compute_metrics_from_arrays(by_model_t["rf"][0], by_model_t["rf"][1], by_model_t["rf"][2]) if "rf" in by_model_t else {}
            report["train_metrics_by_conference"] = _metrics_by_conference(train_teams)
            report["notes"]["train_eos_rank_source"] = train_data.get("eos_rank_source", "standings")

    if report.get("test_metrics_ensemble", {}).get("playoff_metrics"):
        report["notes"]["playoff_metrics"] = (
            "Spearman (pred global vs Playoff Outcome Rank), NDCG@4 (Conference Finals), Brier (champion vs odds). "
            "rank_mae/rank_rmse: pred vs Playoff Outcome Rank; W/L record standings vs Playoff Outcome Rank (baseline)."
        )
    if report.get("model_vs_standings_comparison"):
        report["notes"]["model_vs_standings"] = (
            "Regular-season W/L standings vs same final outcome ranks as models. Each model has MAE and RMSE vs outcome; "
            "improvement_mae/improvement_rmse = standings_error - model_error (positive = model better). "
            "Significance: paired bootstrap over teams (resample teams, mean(standings_ae - model_ae)); p_value = proportion of bootstrap <= 0."
        )

    # Write report into the run folder so each run keeps its own evaluation.
    run_report_path = run_dir / "eval_report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {run_report_path}")
    # Also write a copy at outputs root as the "latest" report for backward compatibility.
    out = out_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")

    # Numbered analysis in run folder (outputs3 convention)
    run_dir.mkdir(parents=True, exist_ok=True)
    nn = _next_analysis_number(run_dir)
    analysis_path = run_dir / f"ANALYSIS_{nn:02d}.md"
    ens = report.get("test_metrics_ensemble") or report.get("test_metrics", {}) or {}
    ndcg_note = report.get("notes", {}).get("ndcg_cutoff_labels", "")
    lines = [
        f"# Analysis {nn:02d} — Evaluation summary",
        "",
        f"**Run:** {run_id}",
        f"**EOS source:** {report.get('notes', {}).get('eos_rank_source', 'standings')}",
        "",
        "## Test metrics (ensemble)",
        "",
    ]
    # Short labels so NDCG numbers are clearly defined in reports
    NDCG_LABELS = {
        "ndcg_at_4": "Conference Finals (top 4)",
        "ndcg_at_12": "Clinch Playoff (top 12)",
        "ndcg_at_16": "One Play-In Tournament (top 16)",
        "ndcg_at_20": "Qualify for Playoffs (top 20)",
    }
    if ndcg_note:
        lines.append(f"*NDCG cutoffs: {ndcg_note}*")
        lines.append("")
    for k, v in ens.items():
        label = NDCG_LABELS.get(k, "")
        disp = f" ({label})" if label else ""
        if isinstance(v, (int, float)):
            lines.append(f"- {k}{disp}: {v:.4f}" if isinstance(v, float) else f"- {k}{disp}: {v}")
        elif isinstance(v, dict):
            lines.append(f"- {k}{disp}: " + ", ".join(f"{kk}={vv:.4f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in list(v.items())[:8]))
    # Model vs standings comparison and statistical significance
    mvs = report.get("model_vs_standings_comparison") or {}
    if mvs.get("standings_vs_outcome") and mvs.get("models"):
        lines.extend([
            "",
            "## Model vs regular-season standings (same outcome ranks)",
            "",
            "All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).",
            "",
            "| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |",
            "|--------|----------------|-----------------|--------------------|---------------------|",
        ])
        stand = mvs["standings_vs_outcome"]
        lines.append(f"| W/L standings (baseline) | {stand['rank_mae']:.3f} | {stand['rank_rmse']:.3f} | — | — |")
        for name, mod in mvs["models"].items():
            d_mae = mod["improvement_mae_vs_standings"]
            d_rmse = mod["improvement_rmse_vs_standings"]
            lines.append(
                f"| {name} | {mod['rank_mae_pred_vs_outcome']:.3f} | {mod['rank_rmse_pred_vs_outcome']:.3f} | "
                f"{d_mae:+.3f} | {d_rmse:+.3f} |"
            )
        lines.append("")
        sig = mvs.get("significance", {})
        if sig.get("ensemble"):
            s = sig["ensemble"]
            p = s.get("p_value_model_better_than_standings")
            ci_lo, ci_hi = s.get("bootstrap_95_ci_low"), s.get("bootstrap_95_ci_high")
            sig_ok = p < 0.05 if p is not None else False
            lines.extend([
                "### Statistical significance (ensemble vs standings)",
                "",
                f"- **Method:** Paired bootstrap over teams (resample teams with replacement; mean improvement in MAE per team).",
                f"- **Mean MAE improvement:** {s.get('mean_mae_improvement', 0):.4f} (positive = ensemble better).",
                f"- **95% CI for improvement:** [{ci_lo:.4f}, {ci_hi:.4f}].",
                f"- **p-value (H0: no improvement):** {p:.4f} → ensemble is **{'statistically significantly better' if sig_ok else 'not significantly better'}** than standings at α=0.05.",
                "",
            ])
    lines.extend(["", "See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).", ""])
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {analysis_path}")


if __name__ == "__main__":
    main()
