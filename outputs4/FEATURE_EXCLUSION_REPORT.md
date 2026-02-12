# Model B feature exclusion report

**Generated from:** `scripts/export_feature_importances.py` with `--threshold 0.05` on Phase 5 combo_002 and Phase 3 combo_18.  
**Config in use:** `config/defaults_reduced_features.yaml` uses **threshold 0.04**: exclude only features whose avg importance is **below 0.04 in both** combos. Features with avg ≥ 0.04 in either combo are **reintroduced**.

---

## Excluded features (6)

Features whose **average importance** (XGB + RF) / 2 was **below 0.04** in **both** reference combos are excluded.

| Feature | Description | Combo_002 (XGB / RF) | Combo_18 (XGB / RF) | Reason excluded |
|---------|-------------|----------------------|---------------------|-----------------|
| **ORB_pct** | Offensive rebound % (Four Factors) | 0.024 / 0.030 | 0.019 / 0.030 | Avg &lt; 0.04 in both |
| **massey_rating** | Massey rating (config-enabled) | 0.047 / 0.017 | 0.045 / 0.017 | Avg &lt; 0.04 in both |
| **days_until_playoffs** | Motivation (config-enabled) | 0.001 / 0.001 | 0.001 / 0.001 | Near zero in both |
| **late_season** | Motivation (config-enabled) | 0 / 0 | 0 / 0 | Zero in both |
| **raptor_offense_sum_top5** | RAPTOR offense top-5 (config-enabled) | 0.035 / 0.008 | 0.023 / 0.008 | Avg &lt; 0.04 in both |
| **raptor_defense_sum_top5** | RAPTOR defense top-5 (config-enabled) | 0.023 / 0.010 | 0.022 / 0.010 | Avg &lt; 0.04 in both |

**Reintroduced (avg ≥ 0.04 in both combos):** **TOV_pct** (combo_002 avg 0.0445, combo_18 avg 0.0405), **FT_rate** (avg 0.0455 / 0.0435). Now **included** in the reduced-feature config.  
**Reintroduced (only low in one combo):** **elimination_status** — above threshold in combo_002 (XGB 0.107); below only in combo_18. Now **included** in the reduced-feature config.

---

## Included features (7 after exclusion, threshold 0.04)

| Feature | Description | Combo_002 avg importance | Combo_18 avg importance |
|---------|-------------|--------------------------|--------------------------|
| **eFG** | Effective FG% (Four Factors) | High (XGB 0.19, RF 0.61) | High (XGB 0.16, RF 0.61) |
| **pace** | Possessions per game | Moderate (XGB 0.05, RF 0.09) | Moderate (XGB 0.04, RF 0.09) |
| **elo** | Elo rating (config-enabled) | High (XGB 0.13, RF 0.14) | High (XGB 0.12, RF 0.14) |
| **eliminated_x_late_season** | Motivation (config-enabled) | Very high in XGB (0.31); low in RF | Very high in XGB (0.40); low in RF |
| **elimination_status** | Motivation (config-enabled) | Above threshold (XGB 0.107); **reintroduced** | Below in combo_18 only |
| **TOV_pct** | Turnover rate (Four Factors) | Avg 0.0445 (XGB 0.042, RF 0.047) | Avg 0.0405; **reintroduced** (≥ 0.04) |
| **FT_rate** | Free-throw rate (Four Factors) | Avg 0.0455; **reintroduced** (≥ 0.04) | Avg 0.0435 |

**Note:** With **threshold 0.04**, the config excludes only the 6 features in the table above. **TOV_pct** and **FT_rate** are reintroduced (avg ≥ 0.04 in both combos). The **effective included set** in `defaults_reduced_features.yaml` is: **eFG, pace, elo, eliminated_x_late_season, elimination_status, TOV_pct, FT_rate** (7 features).

---

## Methodology

- **Source models:** Phase 5 NDCG@16 combo_002 (playoff_outcome), Phase 3 fine NDCG@16 combo_18 (final_rank).  
- **Importance:** Tree `feature_importances_` from XGBoost and Random Forest; feature names from `get_team_context_feature_cols(config)`.  
- **Threshold:** 0.04. For each feature, average = (XGB_importance + RF_importance) / 2; if average &lt; 0.04 **in both** combos, the feature is excluded. Features with avg ≥ 0.04 (e.g. TOV_pct, FT_rate) are **reintroduced**. Features below threshold in only one combo (e.g. elimination_status) are also **reintroduced**.  
- **Config:** `config/defaults_reduced_features.yaml` sets `model_b.exclude_features` to the 6 features with avg &lt; 0.04 in both combos; TOV_pct, FT_rate, and elimination_status are included.

---

## Files

- **Exclusion list in config:** `config/defaults_reduced_features.yaml` → `model_b.exclude_features` (6 features; TOV_pct and FT_rate reintroduced with threshold 0.04)  
- **Full importance JSON (combo_002):** `outputs4/sweeps/phase5_ndcg16_playoff_broad/feature_importances_combo_002.json`  
- **Full importance JSON (combo_18):** `outputs4/sweeps/phase3_fine_ndcg16_final_rank/feature_importances_combo_18.json`  
- **Export script:** `python -m scripts.export_feature_importances --combo-dir <path> --threshold 0.05`
