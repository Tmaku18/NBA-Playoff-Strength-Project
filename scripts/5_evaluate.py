"""Run evaluation on real predictions; write outputs/eval_report.json. Requires predictions.json from script 6."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_ranking, evaluate_upset
from src.evaluation.metrics import brier_champion, ndcg_at_4, ndcg_score, spearman
from src.utils.split import load_split_info


def _latest_run_id(outputs_dir: Path) -> str | None:
    """Return the latest run_NNN (highest number) present in outputs_dir, or None."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return None
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and pattern.match(p.name):
            if (p / "predictions.json").exists():
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers:
        return None
    return f"run_{max(numbers):03d}"


def _teams_to_arrays(teams: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, str]]:
    """Extract y_actual, y_score, pred_ranks, and team_id list; team_id -> conference for filtering."""
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


def _compute_metrics(teams: list, *, k: int = 10) -> dict:
    """Compute ndcg, spearman, mrr, roc_auc_upset and optionally playoff_metrics from teams list."""
    y_actual, y_score, pred_ranks_arr, _, _ = _teams_to_arrays(teams)
    n = len(y_actual)
    if n < 2:
        return {"ndcg": 0.0, "spearman": 0.0, "mrr": 0.0, "roc_auc_upset": 0.5}
    max_rank = int(np.max(y_actual)) if n else 0
    y_true_relevance = (max_rank - y_actual + 1).clip(1, max_rank if max_rank > 0 else 1)
    m = evaluate_ranking(y_true_relevance, y_score, k=min(k, n))
    delta = y_actual - pred_ranks_arr
    y_bin = (delta > 0).astype(np.float32)
    if np.unique(y_bin).size >= 2:
        m2 = evaluate_upset(y_bin, y_score)
        m.update(m2)
    else:
        m["roc_auc_upset"] = 0.5
    playoff_rows = []
    for t in teams:
        p_rank = t.get("analysis", {}).get("playoff_rank")
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
            "spearman_pred_vs_playoff_rank": float(spearman(p_rank, g_rank)),
            "ndcg_at_4_final_four": float(ndcg_at_4(p_rank, -g_rank)),
            "brier_championship_odds": float(brier_champion(champion_onehot, odds_pct)),
        }
    return m


def _metrics_by_conference(teams: list) -> dict[str, dict[str, float]]:
    """Compute NDCG and Spearman per conference (E, W)."""
    y_actual, y_score, _, team_ids, team_id_to_conf = _teams_to_arrays(teams)
    if len(y_actual) < 2:
        return {}
    max_rank = int(np.max(y_actual))
    y_true_relevance = (max_rank - y_actual + 1).clip(1, max_rank if max_rank > 0 else 1)
    out: dict[str, dict[str, float]] = {}
    for conf in ("E", "W"):
        idx = [i for i, tid in enumerate(team_ids) if team_id_to_conf.get(tid) == conf]
        if len(idx) < 2:
            continue
        ya = y_actual[idx]
        ys = y_score[idx]
        rel = (int(np.max(ya)) - ya + 1).clip(1, int(np.max(ya)))
        ndcg = ndcg_score(rel, ys, k=min(10, len(rel)))
        sp = spearman(ya, ys)
        out[conf] = {"ndcg": float(ndcg), "spearman": float(sp)}
    return out


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
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
    if not pred_path.exists():
        print("Predictions not found. Run inference (script 6) first.", file=sys.stderr)
        sys.exit(1)

    report: dict = {"notes": {}}

    # Split info (if script 3 has been run)
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

    # Test metrics (primary: predictions.json)
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    teams = data.get("teams", [])
    if not teams:
        print("No teams in predictions.json.", file=sys.stderr)
        sys.exit(1)
    missing_actual = sum(1 for t in teams if t.get("analysis", {}).get("EOS_global_rank") is None)
    if missing_actual > 0:
        print(f"Warning: {missing_actual} teams missing EOS_global_rank; skipped in evaluation.")
    report["test_metrics"] = _compute_metrics(teams)
    report["test_metrics_by_conference"] = _metrics_by_conference(teams)
    report["notes"]["upset_definition"] = "sleeper = EOS_global_rank > predicted_strength"
    report["notes"]["mrr"] = "top_k=2; 1/rank of first max-relevance item in predicted order (two conferences)."

    # Train metrics (if train_predictions.json exists)
    train_pred_path = run_dir / "train_predictions.json"
    if train_pred_path.exists():
        with open(train_pred_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        train_teams = train_data.get("teams", [])
        if train_teams:
            report["train_metrics"] = _compute_metrics(train_teams)
            report["train_metrics_by_conference"] = _metrics_by_conference(train_teams)

    if report["test_metrics"].get("playoff_metrics"):
        report["notes"]["playoff_metrics"] = "Spearman (pred global vs playoff rank), NDCG@4 (final four), Brier (champion vs odds)."

    out = out_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
