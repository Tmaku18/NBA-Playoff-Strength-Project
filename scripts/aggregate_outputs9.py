"""One-off: aggregate outputs9 eval_report.json and print summary."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
batch = ROOT / "output/outputs9/sweeps/outputs9_listmle_spearman"
results = []
for i in range(40):
    p = batch / f"combo_{i:04d}" / "outputs" / "eval_report.json"
    if not p.exists():
        continue
    with open(p) as f:
        d = json.load(f)
    ens = d.get("test_metrics_ensemble") or {}
    pm = ens.get("playoff_metrics") or {}
    row = {
        "combo": i,
        "spearman": ens.get("spearman"),
        "ndcg_at_4": ens.get("ndcg_at_4"),
        "ndcg_at_16": ens.get("ndcg_at_16"),
        "rank_mae": ens.get("rank_mae_pred_vs_playoff_outcome_rank"),
        "playoff_spearman": pm.get("spearman_pred_vs_playoff_outcome_rank"),
    }
    results.append(row)

valid = [r for r in results if r.get("spearman") is not None]
if not valid:
    print("No valid spearman")
    raise SystemExit(1)

best_sp = max(valid, key=lambda x: x["spearman"])
worst_sp = min(valid, key=lambda x: x["spearman"])
sp_vals = [r["spearman"] for r in valid]
print("=== Spearman (ensemble) ===")
print(f"  Best:  combo {best_sp['combo']} = {best_sp['spearman']:.4f}")
print(f"  Worst: combo {worst_sp['combo']} = {worst_sp['spearman']:.4f}")
print(f"  Mean:  {sum(sp_vals)/len(sp_vals):.4f}")
print(f"  Range: [{min(sp_vals):.4f}, {max(sp_vals):.4f}]")
print()
print("=== NDCG@4 ===")
best_ndcg4 = max(valid, key=lambda x: x.get("ndcg_at_4") or 0)
print(f"  Best:  combo {best_ndcg4['combo']} = {(best_ndcg4.get('ndcg_at_4') or 0):.4f}")
print()
print("=== Top 10 by Spearman ===")
for r in sorted(valid, key=lambda x: x["spearman"], reverse=True)[:10]:
    ndcg4 = r.get("ndcg_at_4") or 0
    print(f"  combo {r['combo']:2d}: spearman={r['spearman']:.4f}, ndcg@4={ndcg4:.4f}, playoff_spearman={r.get('playoff_spearman') or 0:.4f}")
