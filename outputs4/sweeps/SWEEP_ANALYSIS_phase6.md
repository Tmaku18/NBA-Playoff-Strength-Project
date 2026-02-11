# Phase 6 Sweep Results Analysis

**Analytics standard (project rule):** All sweep/analytics reports must include **East and West (by-conference)** NDCG and Spearman; see §3b below and `.cursor/rules/analytics-include-east-west.mdc`.

---

## 1. phase6_ndcg16_playoff_narrow (primary sweep)

**Setup**
- **Objective:** Maximize **NDCG@16** (playoff-style ranking over 16 teams).
- **Method:** Optuna, 20 trials, 4 parallel jobs.
- **Phase:** `phase2_playoff_narrow`; validation fraction 0.25.
- **Hyperparameters swept:** `rolling_windows`, `model_a_epochs`, `max_depth`, `learning_rate`, `n_estimators_xgb`, `n_estimators_rf`, `subsample`, `colsample_bytree`, `min_samples_leaf`.

**Best trial (Optuna / NDCG@16)**
- **Combo:** 16  
- **NDCG@16 (value):** **0.5430**
- **Params:**  
  `rolling_windows=[15,30]`, `model_a_epochs=28`, `max_depth=5`, `learning_rate≈0.0837`, `n_estimators_xgb=230`, `n_estimators_rf=190`, `subsample=0.8`, `colsample_bytree=0.7`, `min_samples_leaf=5`

**Best-by-metric summary**

| Metric | Best combo | Value |
|--------|------------|--------|
| **NDCG@16** (sweep objective) | 16 | **0.5430** |
| Spearman | 5 | 0.5314 |
| NDCG@4 | 6 | 0.5421 |
| NDCG@10 | 10 | 0.5329 |
| NDCG@12 | 6 | 0.5421 |
| NDCG@20 | 6 | 0.5463 |
| Rank MAE (pred vs playoff) | 5 | 6.53 |
| Rank RMSE (pred vs playoff) | 5 | 8.34 |
| Playoff Spearman | 5 | 0.5404 |

**Findings**
- **Combo 16** is best for NDCG@16 and is the Optuna-selected trial; it also has strong **NDCG@4** (0.5059) and **ROC AUC upset** (0.822).
- **Combo 5** is best for **Spearman**, **rank MAE/RMSE**, and **playoff Spearman** (0.5404), with slightly lower NDCG@16 (0.5314) but better rank error (6.53 MAE).
- **Combo 6** ties or leads on NDCG@4/12/20 and has best **Final Four NDCG@4** (0.5059); NDCG@16 = 0.5421.
- **Combo 10** is best on NDCG@10 (0.5315) and overall NDCG (0.5315).
- **Conference gap:** West metrics (NDCG ~0.77, Spearman ~0.64–0.72) are much higher than East (NDCG ~0.25, Spearman ~0.25–0.35), indicating the model ranks Western teams more reliably.

**Recommendation**
- For **ranking quality (NDCG@16):** use **combo 16** (artifacts in `combo_0016/`).
- For **playoff-order correlation and rank error:** use **combo 5** (best playoff Spearman and MAE/RMSE).

---

## 2. phase6_feature_subset (feature ablation)

**Setup**
- **Objective:** Likely NDCG@16 with feature flags (which features to include).
- **Best trial (combo 0):** NDCG@16 = **0.5382**.

**Comparison to phase6_ndcg16_playoff_narrow**
- Narrow sweep best NDCG@16: **0.5430** (combo 16).  
- Feature-subset best: **0.5382**.  
- So the **hyperparameter sweep (narrow) beats the best feature-subset run** by ~0.005 NDCG@16.
- Best feature-subset run still has strong rank MAE (6.53) and similar Brier (0.0322).

**Takeaway**
- Hyperparameter tuning in `phase6_ndcg16_playoff_narrow` is giving better NDCG@16 than the feature-subset sweep; for production ranking, prefer combo 16 from the narrow sweep.

---

## 3. Summary table (phase6_ndcg16_playoff_narrow)

| Combo | NDCG@16 | Spearman | Playoff Spearman | Rank MAE | NDCG@4 (Final Four) |
|-------|---------|----------|------------------|----------|----------------------|
| 5 | 0.5314 | **0.5359** | **0.5404** | **6.53** | 0.464 |
| 6 | 0.5421 | 0.5217 | 0.5244 | 6.60 | **0.5059** |
| 10 | 0.5329 | 0.5034 | 0.5266 | 6.60 | 0.464 |
| **16** | **0.5430** | 0.4816 | 0.5012 | 7.07 | 0.5059 |

Use **combo 16** for best NDCG@16; **combo 5** for best correlation and rank accuracy; **combo 6** if you care most about top-4 (Final Four) NDCG with strong NDCG@16.

---

## 3b. East and West (by-conference) metrics — phase6_ndcg16_playoff_narrow

All 20 combos; NDCG and Spearman are **within-conference** (8 teams each).

| Combo | East NDCG | East Spearman | West NDCG | West Spearman |
|-------|-----------|----------------|-----------|---------------|
| 0 | 0.247 | 0.254 | 0.774 | 0.764 |
| 1 | 0.259 | 0.279 | 0.763 | 0.729 |
| 2 | 0.243 | 0.171 | 0.752 | 0.639 |
| 3 | 0.268 | 0.289 | 0.752 | 0.679 |
| 4 | 0.260 | 0.232 | 0.773 | 0.707 |
| **5** | **0.279** | **0.346** | 0.763 | **0.718** |
| **6** | 0.255 | 0.268 | **0.784** | **0.743** |
| 7 | **0.298** | **0.300** | 0.752 | 0.650 |
| 8 | 0.246 | 0.221 | 0.772 | 0.693 |
| 9 | 0.258 | 0.204 | 0.750 | 0.639 |
| 10 | 0.255 | 0.246 | 0.773 | 0.689 |
| 11 | 0.255 | 0.218 | 0.772 | 0.657 |
| 12 | 0.270 | 0.296 | 0.774 | 0.721 |
| 13 | 0.264 | 0.221 | 0.772 | 0.679 |
| 14 | 0.258 | 0.204 | 0.774 | 0.668 |
| 15 | 0.268 | 0.296 | 0.750 | 0.639 |
| **16** | 0.246 | 0.254 | 0.772 | 0.636 |
| 17 | 0.252 | 0.214 | 0.729 | 0.629 |
| 18 | 0.258 | 0.211 | 0.774 | 0.668 |
| 19 | 0.260 | 0.279 | 0.728 | 0.621 |

**Ranges across combos:** East NDCG 0.24–0.30, East Spearman 0.17–0.35; West NDCG 0.73–0.78, West Spearman 0.62–0.74.

**Best by conference:** Combo **5** has best East (NDCG 0.279, Spearman 0.346) and strong West (0.72 Spearman). Combo **6** has best West (NDCG 0.784, Spearman 0.743). Combo **7** has highest East NDCG (0.298) but West is lower.

---

## 4. Comparison to previous runs — where did “~0.7 with reduced features” go?

The **0.7** you remember is **not** the same metric the sweep reports.

| What you remember | What it actually is | Where |
|-------------------|---------------------|--------|
| **~0.7** | **West conference only** Spearman = **0.7** (and West NDCG ≈ **0.75**) | `outputs4/eval_report.json` → `test_metrics_by_conference.W` |
| **Reduced input features** | phase6_feature_subset (feature flags); same runs still show West NDCG ~0.75, West Spearman ~0.73–0.74 | phase6_feature_subset combo_0009, combo_0008 eval_report.json |

**Important distinction**

- **Sweep “value” (e.g. 0.543)** = **ensemble NDCG@16 over all 16 playoff teams** (East + West). That’s what Optuna optimizes and what the summary table reports.
- **0.7 / 0.75** = **by_conference West only** (8 teams). When we evaluate ranking **within the Western Conference**, the model gets Spearman ≈ 0.7 and NDCG ≈ 0.75. East is much lower (~0.25), which pulls the **overall** 16-team metric down to ~0.48–0.54.

So we are **not** worse than before: we’re comparing

- **Overall (16 teams):** NDCG@16 ~0.54, Spearman ~0.48–0.54 ← sweep and summary.
- **West only (8 teams):** NDCG ~0.75, Spearman ~0.7 ← “0.7 with reduced features.”

**Takeaway:** The ~0.7 with reduced features is **West-only** performance. It’s still there in the same runs (see `test_metrics_by_conference.W` in any eval_report). The phase6 sweeps report **overall** NDCG@16 (~0.54), which is the right number for “rank all 16 playoff teams”; West-only stays strong (~0.75 NDCG, ~0.7 Spearman).

---

## 5. Best East-only runs (all previous sweeps)

Searched all `outputs4/sweeps/*/sweep_results.csv` for best **East** by-conference metrics (NDCG and Spearman). Main runs (`outputs4/eval_report.json`, `run_025`) have East ~0.25 NDCG / ~0.22–0.25 Spearman — below these sweep combos.

**Best East NDCG (all sweeps)**

| Sweep | Combo | East NDCG | East Spearman |
|-------|--------|-----------|----------------|
| **phase3_coarse_ndcg16_final_rank** | 3 | **0.3441** | 0.3321 |
| phase3_coarse_ndcg4_final_rank | 0 | **0.3441** | 0.3321 |
| phase2_coarse_spearman_final_rank | 8 | 0.3430 | 0.3321 |
| phase5_ndcg16_playoff_broad | 13 | 0.3429 | 0.2929 |
| phase2_coarse_spearman_final_rank | 10 | 0.3421 | 0.3000 |

**Best East Spearman (all sweeps)**

| Sweep | Combo | East Spearman | East NDCG |
|-------|--------|----------------|-----------|
| **phase1_rolling_spearman_final_rank** | 5 | **0.3821** | 0.2517 |
| phase3_coarse_ndcg16_final_rank | 8 | 0.3500 | 0.2981 |
| phase4_ndcg16_playoff_outcome | 7 | 0.3500 | 0.2557 |
| phase3_fine_ndcg16_final_rank | 11 | 0.3464 | 0.2644 |
| phase6_ndcg16_playoff_narrow | 5 | 0.3464 | 0.2786 |

**Summary**

- **Best East NDCG:** **phase3_coarse_ndcg16_final_rank** combo **3** (or phase3_coarse_ndcg4_final_rank combo 0) — East NDCG **0.344**, East Spearman 0.33.
- **Best East Spearman:** **phase1_rolling_spearman_final_rank** combo **5** — East Spearman **0.382** (East NDCG 0.25).
- Phase6 best East in this sweep is combo **5** (East Spearman 0.346, East NDCG 0.279); phase3/phase1 configs still lead for East-only.
- To reproduce: use config `outputs4/sweeps/<sweep>/combo_<N>/config.yaml` and run eval (e.g. pipeline from that config).
