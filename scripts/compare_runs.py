"""Compare evaluation metrics across all run_XXX/predictions.json. One-off for analysis."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_ranking, evaluate_upset


def _normalize_teams(teams: list) -> list:
    """Normalize old (predicted_rank, actual_global_rank, true_strength_score) to current field names."""
    out = []
    for t in teams:
        analysis = t.get("analysis", {})
        pred = t.get("prediction", {})
        act = analysis.get("EOS_global_rank") or analysis.get("actual_global_rank")
        pred_rank = pred.get("predicted_strength") or pred.get("predicted_rank")
        score = pred.get("ensemble_score") or pred.get("true_strength_score")
        if act is None:
            continue
        out.append({
            "team_id": t["team_id"],
            "conference": t.get("conference"),
            "analysis": {"EOS_global_rank": act},
            "prediction": {"predicted_strength": pred_rank, "ensemble_score": score},
        })
    return out


def compute_metrics(teams: list, *, k: int = 10) -> dict:
    """Same logic as script 5: ndcg, spearman, mrr, roc_auc_upset."""
    if not teams:
        return {"ndcg": 0.0, "spearman": 0.0, "mrr_top2": 0.0, "mrr_top4": 0.0, "roc_auc_upset": 0.5}
    actual_ranks = [float(t["analysis"]["EOS_global_rank"]) for t in teams]
    pred_ranks = [float(t["prediction"].get("predicted_strength") or 0) for t in teams]
    pred_scores = [float(t["prediction"].get("ensemble_score") or 0.0) for t in teams]
    y_actual = np.array(actual_ranks, dtype=np.float32)
    y_score = np.array(pred_scores, dtype=np.float32)
    pred_ranks_arr = np.array(pred_ranks, dtype=np.float32)
    n = len(y_actual)
    max_rank = int(np.max(y_actual))
    y_true_relevance = (max_rank - y_actual + 1).clip(1, max_rank if max_rank > 0 else 1)
    m = evaluate_ranking(y_true_relevance, y_score, k=min(k, n))
    delta = y_actual - pred_ranks_arr
    y_bin = (delta > 0).astype(np.float32)
    if np.unique(y_bin).size >= 2:
        m2 = evaluate_upset(y_bin, y_score)
        m.update(m2)
    else:
        m["roc_auc_upset"] = 0.5
    return m


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_name = config.get("paths", {}).get("outputs", "outputs")
    out_dir = Path(out_name) if Path(out_name).is_absolute() else ROOT / out_name
    pattern = re.compile(r"^run_(\d+)$", re.I)
    run_dirs = []
    for p in out_dir.iterdir():
        if p.is_dir() and pattern.match(p.name):
            pred_path = p / "predictions.json"
            if pred_path.exists():
                run_dirs.append((int(pattern.match(p.name).group(1)), p))
    run_dirs.sort(key=lambda x: x[0])

    print("Run    NDCG@10   Spearman   MRR@2  MRR@4  ROC-AUC (upset)")
    print("----   -------   --------   ----   ----   ----------------")
    results = {}
    for num, path in run_dirs:
        with open(path / "predictions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = data.get("teams", [])
        normalized = _normalize_teams(teams)
        if len(normalized) < 2:
            continue
        m = compute_metrics(normalized)
        results[f"run_{num:03d}"] = m
        print(f"run_{num:03d}   {m['ndcg']:.4f}     {m['spearman']:.4f}     {m['mrr_top2']:.2f}   {m['mrr_top4']:.2f}   {m['roc_auc_upset']:.4f}")

    # Write comparison JSON for ANALYSIS.md
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'run_comparison.json'}")


if __name__ == "__main__":
    main()
