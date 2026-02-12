"""Script 5b: Explain model predictions (SHAP + attention/IG).

What this does:
- Runs SHAP on Model B (XGBoost) to show feature importance for team strength.
- Runs attention/Integrated Gradients on Model A (DeepSet) for player-level explanations.
- Writes shap_summary.png, attention plots, and IG outputs to run folder.
- Use --config to point at a sweep combo (e.g. outputs4/sweeps/.../combo_0018/config.yaml).

Run after inference (script 6). Optional; useful for interpreting which features/players matter."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _copy_to_run_dir(out: Path, src: Path, name: str, config: dict) -> None:
    """Copy a file (e.g. shap_summary.png) into the current run folder when run_id is set or in .current_run."""
    import re
    run_id = config.get("inference", {}).get("run_id")
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        current_run = out / ".current_run"
        if current_run.exists():
            run_id = current_run.read_text(encoding="utf-8").strip()
    if run_id and re.match(r"^run_\d+$", run_id, re.I):
        run_dir = out / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dst = run_dir / name
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print("Wrote", dst)


def main():
    parser = argparse.ArgumentParser(description="Explain Model B (SHAP) and Model A (attention/IG)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML; default: config/defaults.yaml")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    db_path = Path(config["paths"]["db"])
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)

    from src.data.db_loader import load_training_data
    from src.features.team_context import build_team_context_as_of_dates, get_team_context_feature_cols
    from src.training.build_lists import build_lists
    from src.training.data_model_a import build_batches_from_lists

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        print("DB has no games/tgl. Run 2_build_db with raw data first.", file=sys.stderr)
        sys.exit(1)

    lists = build_lists(tgl, games, teams)
    if not lists:
        print("No lists from build_lists.", file=sys.stderr)
        sys.exit(1)
    rows = []
    for lst in lists:
        for tid, _ in zip(lst["team_ids"], lst["win_rates"]):
            rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"]})
    flat = pd.DataFrame(rows)
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(
        tgl, games, team_dates,
        config=config, teams=teams, pgl=pgl,
    )
    feat_cols = [c for c in get_team_context_feature_cols(config) if c in feat_df.columns]
    if not feat_cols:
        print("No feature columns for SHAP.", file=sys.stderr)
        sys.exit(1)
    X_real = feat_df[feat_cols].fillna(0.0).values.astype(np.float32)
    # Cap sample size so SHAP runs in reasonable time.
    if X_real.shape[0] > 500:
        X_real = X_real[:500]

    # SHAP: explain Model B (XGB) feature importance on team-context features.
    xgb_path = out / "xgb_model.joblib"
    if not xgb_path.exists():
        print("XGB model not found. Run script 4 first.", file=sys.stderr)
        sys.exit(1)
    try:
        import joblib
        from src.viz.shap_summary import shap_summary
        xgb = joblib.load(xgb_path)
        shap_path = out / "shap_summary.png"
        shap_summary(xgb, X_real, feature_names=feat_cols, out_path=shap_path)
        print("Wrote", shap_path)
        _copy_to_run_dir(out, shap_path, "shap_summary.png", config)
    except Exception as e:
        print("SHAP failed:", e, file=sys.stderr)
        sys.exit(1)

    # Attention ablation: mask top-k attention weights in Model A and see how score changes.
    model_a_path = out / "best_deep_set.pt"
    if not model_a_path.exists():
        print("Model A not found. Run script 3 first.", file=sys.stderr)
        sys.exit(1)
    try:
        from src.models.deep_set_rank import DeepSetRank
        from src.viz.integrated_gradients import attention_ablation
        valid_lists = [lst for lst in lists if len(lst["team_ids"]) >= 2]
        if not valid_lists:
            print("No valid lists for attention ablation.", file=sys.stderr)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batches_a, _ = build_batches_from_lists(valid_lists[:1], games, tgl, teams, pgl, config, device=device)
            if not batches_a:
                print("No batches for attention ablation.", file=sys.stderr)
            else:
                ck = torch.load(model_a_path, map_location=device, weights_only=False)
                ma = config.get("model_a", {})
                # Use stat_dim from batch so model matches checkpoint (trained with same data)
                stat_dim = int(batches_a[0]["player_stats"].shape[-1])
                model = DeepSetRank(
                    ma.get("num_embeddings", 500),
                    ma.get("embedding_dim", 32),
                    stat_dim,
                    ma.get("encoder_hidden", [128, 64]),
                    ma.get("attention_heads", 4),
                    ma.get("dropout", 0.2),
                    minutes_bias_weight=float(ma.get("minutes_bias_weight", 0.3)),
                    minutes_sum_min=float(ma.get("minutes_sum_min", 1e-6)),
                )
                if "model_state" in ck:
                    from src.inference.predict import _state_dict_for_load
                    model.load_state_dict(_state_dict_for_load(ck["model_state"]), strict=True)
                model = model.to(device)
                model.eval()
                batch = batches_a[0]
                emb = batch["embedding_indices"].to(device).reshape(-1, batch["embedding_indices"].shape[2])
                stats = batch["player_stats"].to(device).reshape(-1, batch["player_stats"].shape[2], batch["player_stats"].shape[3])
                minu = batch["minutes"].to(device).reshape(-1, batch["minutes"].shape[2])
                msk = batch["key_padding_mask"].to(device).reshape(-1, batch["key_padding_mask"].shape[2])
                with torch.no_grad():
                    _, _, attn = model(emb, stats, minu, msk)
                top_k = min(2, attn.shape[-1])
                v = attention_ablation(model, emb, stats, minu, msk, attn, top_k=top_k)
                if not math.isfinite(v):
                    print("Attention ablation (top-%d masked) score mean: NaN (masked forward produced non-finite scores)" % top_k)
                else:
                    print("Attention ablation (top-%d masked) score mean: %s" % (top_k, v))

                # Integrated Gradients for Model A (one team, optional)
                try:
                    from src.viz.integrated_gradients import ig_attr, _HAS_CAPTUM
                    if not _HAS_CAPTUM:
                        print("Integrated Gradients skipped (captum not installed).")
                    else:
                        n_steps = 50
                        emb_1 = emb[0:1]
                        stats_1 = stats[0:1]
                        minu_1 = minu[0:1]
                        msk_1 = msk[0:1]
                        attr, delta = ig_attr(model, emb_1, stats_1, minu_1, msk_1, n_steps=n_steps)
                        if attr is not None and attr.numel() > 0:
                            attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
                            if not torch.isfinite(attr).all():
                                print("Integrated Gradients: non-finite attributions after sanitization.")
                                attr = None
                        if attr is not None and attr.numel() > 0:
                            # attr (1, P, S); L2 norm per player
                            norms = torch.norm(attr[0].float(), dim=1)
                            k = min(5, norms.shape[0])
                            _, top_idx = norms.topk(k, largest=True)
                            lines = ["Integrated Gradients (Model A) top-%d player indices by attribution L2 norm:" % k]
                            for i, idx in enumerate(top_idx.tolist(), 1):
                                lines.append("  %d. player_idx=%d norm=%.4f" % (i, idx, norms[idx].item()))
                            summary = "\n".join(lines)
                            print(summary)
                            ig_path = out / "ig_model_a_attributions.txt"
                            ig_path.write_text(summary, encoding="utf-8")
                            print("Wrote", ig_path)
                            _copy_to_run_dir(out, ig_path, "ig_model_a_attributions.txt", config)
                        else:
                            print("Integrated Gradients: no attributions (empty result).")
                except ImportError:
                    print("Integrated Gradients skipped (captum not installed).")
                except Exception as e:
                    print("Integrated Gradients failed:", e, file=sys.stderr)
    except Exception as e:
        print("Attention ablation failed:", e, file=sys.stderr)
        sys.exit(1)

    # Attention significance: bootstrap over teams for per-player CI and p-value
    import re
    run_id = config.get("inference", {}).get("run_id")
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        current_run = out / ".current_run"
        if current_run.exists():
            run_id = current_run.read_text(encoding="utf-8").strip()
    if run_id and re.match(r"^run_\d+$", run_id, re.I):
        run_dir = out / run_id
        pred_files = list(run_dir.glob("predictions_*.json")) or ([run_dir / "predictions.json"] if (run_dir / "predictions.json").exists() else [])
        if pred_files:
            import json
            all_teams = []
            for pf in sorted(pred_files)[:3]:  # cap to avoid huge load
                with open(pf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_teams.extend(data.get("teams", []))
            if all_teams:
                from src.evaluation.attention_significance import attention_bootstrap_over_teams
                sig = attention_bootstrap_over_teams(all_teams, B=1000, seed=42)
                if sig:
                    sig_path = run_dir / "attention_significance.json"
                    with open(sig_path, "w", encoding="utf-8") as f:
                        json.dump({"per_player": sig, "n_teams": len(all_teams), "B": 1000}, f, indent=2)
                    print("Wrote", sig_path)


if __name__ == "__main__":
    main()
