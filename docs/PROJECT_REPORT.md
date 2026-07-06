# NBA True Strength Prediction — Project Report

**Author:** Tanaka Makuvaza  
**Course:** Georgia State University — Advanced Machine Learning  
**Date:** January 28, 2026

---

## 1. Goal

Build a system that predicts **NBA team “true strength”** for **future outcomes** (e.g. playoff seed, championship odds) and distinguishes **Sleepers** (under-ranked by conventional metrics) from **Paper Tigers** (over-ranked), **without circular evaluation** — i.e. never using the same information both as input and as evaluation target.

---

## 2. Methodology

- **Multi-modal stacking ensemble:** Two base models feed a meta-learner.
  - **Model A (Deep Set):** Roster-based; uses player-level stats and set attention over players to produce a latent representation **Z** and a team score. Trained with **ListMLE** on ordered lists (by date/conference) so the model learns to rank teams.
  - **Model B (XGBoost):** Team-context features only (rolling stats, ELO, SOS/SRS, etc.); no roster. Trained by regression on the same target (e.g. win rate or rank) as used for ListMLE.
- **Ensemble:** Model A + Model B only. Level-2 **RidgeCV** meta-learner is fit on **out-of-fold (OOF)** predictions from both models (K-fold over time). The meta output is the **ensemble score**, which is then mapped to percentile (0–1 and 0–100) for interpretability and championship odds.
- **Model C (Random Forest):** Trained on the same team-context features as XGB when `training.train_model_c` is true; used **only for evaluation and diagnostics** (e.g. `model_c_rank` in predictions), not in the stack.
- **Evaluation:** NDCG (e.g. NDCG@10, NDCG@16), Spearman correlation, playoff-specific metrics (when playoff data exists), Brier score on championship odds, rank MAE/RMSE. See `eval_report.json` and sweep outputs (e.g. `output/4_listmle/sweeps/`, `docs/SWEEP_ANALYSIS.md`).

---

## 3. Design choices

- **Target:** Future W/L or **final playoff seed** — never efficiency or net rating, to avoid leakage and to align with “who will win” rather than “who played well in the past.”
- **No net_rating leakage:** `net_rating` is excluded from model inputs and from targets and evaluation metrics (allowed only in baselines if needed).
- **ListMLE:** Game-level lists per conference/date; numerical stability via `torch.logsumexp`, input clamping, gradient clipping. Set attention uses **σReparam** (Zhai et al.) to bound attention logits and reduce entropy collapse.
- **Two-model ensemble:** Only Model A and XGB are stacked; Model C is for comparison and diagnostics. This keeps the meta-learner simple (2 or 4 inputs) and interpretable.
- **Optional confidence-weighted meta:** When OOF includes per-instance confidence, the meta-learner is trained on **4 inputs** \((s_A, s_X, c_A, c_X)\) so that the **more confident** model can have higher effective weight per team (learned from data). See §4.

---

## 4. Confidence semantics and 4-input meta

To make the ensemble **confidence-aware** (higher weight to the model that is more sure on each prediction), we add:

- **Model A confidence \(c_A\):** Derived from **attention weights** over players.  
  - **High entropy** (diffuse attention) ⇒ no single player dominates ⇒ **high confidence**.  
  - **Low entropy** or **high max weight** ⇒ prediction is driven by one or few players (star-dependent) ⇒ **high risk** (e.g. injury) ⇒ **low confidence**, and we tend to rely less on that prediction when ranking.  
  - Formula: \(c_A = w_{\text{ent}} \cdot (H / H_{\max}) + w_{\max} \cdot (1 - \max_p w_p)\), with configurable weights (defaults 0.5, 0.5).
- **XGB confidence \(c_X\):** From **tree-level prediction variance** (std across trees per sample). Higher variance ⇒ **lower confidence**. We use \(c_X = 1/(1 + \text{std})\) so that high variance maps to low \(c_X\).

These are computed at **training time** (script 3 for \(c_A\), script 4 for \(c_X\)) and written into OOF parquets. Script 4b trains RidgeCV on \((s_A, s_X, c_A, c_X)\) when both confidence columns exist. At **inference**, we compute \(c_A\) from attention and \(c_X\) from tree variance and pass 4 columns to the meta when it has 4 coefficients; otherwise we keep the 2-column path for backward compatibility.

---

## 5. Pipeline

1. **Scripts 1–2:** Data download and DB build (DuckDB; optional playoff tables).
2. **Script 3:** Train Model A (Deep Set + ListMLE); produce OOF and optionally `conf_a`; save `best_deep_set.pt` and `oof_model_a.parquet`.
3. **Script 4:** Train Model B (XGB) and optionally Model C (RF); produce OOF and optionally `conf_xgb`; save `oof_model_b.parquet`, `xgb_model.joblib`, and `rf_model.joblib` when `train_model_c` is true.
4. **Script 4b:** Merge OOF from A and B; train RidgeCV meta (2 or 4 cols); save `ridgecv_meta.joblib` and `oof_pooled.parquet`.
5. **Script 5:** Evaluate on test dates/seasons (NDCG, Spearman, playoff metrics, etc.); write `eval_report.json`.
6. **Script 6:** Run inference (load models, build lists for target date, run A/B, stack with meta, output predictions JSON and figures).

Sweeps (e.g. Optuna) vary hyperparameters and objectives (spearman, ndcg4, ndcg16, playoff_spearman, rank_rmse); see `scripts/sweep_hparams.py` and `scripts/aggregate_sweep_results.py`. Analysis is in `output/4_listmle/sweeps/`, `docs/SWEEP_ANALYSIS.md`, and run-specific `ANALYSIS.md` where present.

---

## 6. References and docs

- **Attention and collapse:** `.cursor/plans/Attention_Report.md`
- **Confidence-weighted ensemble options:** `docs/CONFIDENCE_WEIGHTED_ENSEMBLE_OPTIONS.md`
- **Sweep and metrics:** `docs/SWEEP_ANALYSIS.md`, `output/4_listmle/sweeps/SWEEP_PHASE1_ANALYSIS.md`
- **Attention analysis:** `docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md`
- **Production best (Jul 2026):** `output/8_spearman_surrogate/improved_07-06/ANALYSIS_03.md`
- **2025-26 predictions:** `docs/PREDICTIONS_2025-26.md`

---

## 7. Latest results (production best, Jul 2026)

**Run:** `output/8_spearman_surrogate/improved_07-06/outputs/run_026_07-06_0937`  
**Config:** `config/8_spearman_improved.yaml` (Spearman-surrogate Model A + XGBoost Model B + RidgeCV stacking)

| Metric | Value |
|--------|-------|
| Pooled ensemble Spearman | **0.789** |
| NDCG@4 (Conference Finals) | **0.620** |
| Rank MAE vs playoff outcome | **3.73** |
| Test seasons | 2023-24, 2024-25, 2025-26 (6 checkpoints) |

**Champion identification (ensemble, final snapshots):** 2023-24 Boston ✓ · 2024-25 Oklahoma City ✓ · 2025-26 New York #8 (actual champion; model favored OKC/SA).

**Model roles:** Model B (XGBoost, Spearman 0.778) provides the strongest tabular signal; Model A (Deep Set, Spearman 0.643) adds roster context; RidgeCV stacking improves NDCG@4 over B alone (0.620 vs 0.522).

**Data fix:** Franchise metadata (`src/data/team_meta.py`) ensures all 30 teams appear in conference-date lists (prior runs dropped 11 renamed franchises).

This report summarizes methodology through the confidence-weighted meta-learner; detailed run and sweep results live in the outputs and analysis documents above.
