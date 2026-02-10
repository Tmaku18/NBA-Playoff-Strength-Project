# Model B feature exclusion report

**Generated from:** `scripts/export_feature_importances.py` with `--threshold 0.05` on Phase 5 combo_002 and Phase 3 combo_18.  
**Config in use:** `config/defaults_reduced_features.yaml` sets `model_b.exclude_features` to the list below.

---

## Excluded features (9)

Features whose **average importance** (XGB + RF) / 2 was **below 0.05** in at least one of the two reference combos are excluded in the reduced-feature config.

| Feature | Description | Combo_002 (XGB / RF) | Combo_18 (XGB / RF) | Reason excluded |
|---------|-------------|----------------------|---------------------|-----------------|
| **TOV_pct** | Turnover rate (Four Factors) | 0.042 / 0.047 | 0.034 / 0.047 | Low in both; avg &lt; 0.05 |
| **FT_rate** | Free-throw rate (Four Factors) | 0.042 / 0.049 | 0.038 / 0.049 | Low in both; avg &lt; 0.05 |
| **ORB_pct** | Offensive rebound % (Four Factors) | 0.024 / 0.030 | 0.019 / 0.030 | Low in both; avg &lt; 0.05 |
| **massey_rating** | Massey rating (config-enabled) | 0.047 / 0.017 | 0.045 / 0.017 | Low in both; avg &lt; 0.05 |
| **days_until_playoffs** | Motivation (config-enabled) | 0.001 / 0.001 | 0.001 / 0.001 | Near zero in both |
| **elimination_status** | Motivation (config-enabled) | 0.107 / 0.000 | 0.088 / 0.000 | RF ~0; avg &lt; 0.05 in combo_18 |
| **late_season** | Motivation (config-enabled) | 0 / 0 | 0 / 0 | Zero in both |
| **raptor_offense_sum_top5** | RAPTOR offense top-5 (config-enabled) | 0.035 / 0.008 | 0.023 / 0.008 | Low in both; avg &lt; 0.05 |
| **raptor_defense_sum_top5** | RAPTOR defense top-5 (config-enabled) | 0.023 / 0.010 | 0.022 / 0.010 | Low in both; avg &lt; 0.05 |

---

## Included features (4–5 after exclusion)

| Feature | Description | Combo_002 avg importance | Combo_18 avg importance |
|---------|-------------|--------------------------|--------------------------|
| **eFG** | Effective FG% (Four Factors) | High (XGB 0.19, RF 0.61) | High (XGB 0.16, RF 0.61) |
| **pace** | Possessions per game | Moderate (XGB 0.05, RF 0.09) | Moderate (XGB 0.04, RF 0.09) |
| **elo** | Elo rating (config-enabled) | High (XGB 0.13, RF 0.14) | High (XGB 0.12, RF 0.14) |
| **eliminated_x_late_season** | Motivation (config-enabled) | Very high in XGB (0.31); low in RF | Very high in XGB (0.40); low in RF |
| **elimination_status** | Motivation (config-enabled) | Included in combo_002 only (above threshold there) | Excluded in combo_18 (below 0.05) |

**Note:** The applied config uses the **union** of suggested exclusions from both combos, so **elimination_status** is excluded (it was below threshold in combo_18). So the **effective included set** in `defaults_reduced_features.yaml` is: **eFG, pace, elo, eliminated_x_late_season** (4 features from the original 13).

---

## Methodology

- **Source models:** Phase 5 NDCG@16 combo_002 (playoff_outcome), Phase 3 fine NDCG@16 combo_18 (final_rank).  
- **Importance:** Tree `feature_importances_` from XGBoost and Random Forest; feature names from `get_team_context_feature_cols(config)`.  
- **Threshold:** 0.05. For each feature, average = (XGB_importance + RF_importance) / 2; if average &lt; 0.05, the feature was suggested for exclusion.  
- **Config:** `config/defaults_reduced_features.yaml` sets `model_b.exclude_features` to the union of suggested exclusions from both combos so one config can be used for either default.

---

## Files

- **Exclusion list in config:** `config/defaults_reduced_features.yaml` → `model_b.exclude_features`  
- **Full importance JSON (combo_002):** `outputs4/sweeps/phase5_ndcg16_playoff_broad/feature_importances_combo_002.json`  
- **Full importance JSON (combo_18):** `outputs4/sweeps/phase3_fine_ndcg16_final_rank/feature_importances_combo_18.json`  
- **Export script:** `python -m scripts.export_feature_importances --combo-dir <path> --threshold 0.05`
