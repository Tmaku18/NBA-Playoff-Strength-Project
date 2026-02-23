# Precision and Smoothing

## Context

Baseline W/L standings can outperform models on rank MAE/RMSE vs playoff outcome (e.g. standings MAE 3.1–3.7 vs ensemble 6.5–7.3). Models may be over-smoothed: predictions are less responsive to recent/actual strength than standings. This document lists smoothing sources and proposed experiments.

## 1. Identified Smoothing Sources

| Source | Location | Current | Effect |
|--------|----------|---------|--------|
| Rolling windows | `config/defaults.yaml` `training.rolling_windows` | [15, 30] | Longer windows = more smoothing of recent form |
| odds_temperature | `config/defaults.yaml` `output.odds_temperature` | 0.5 | Lower = sharper championship odds |
| attention_temperature | `model_a.attention.temperature` | 3 | Lower = sharper attention |
| Percentile mapping | `src/inference/predict.py` | `true_strength_scale: percentile` | Compresses score spread |
| RidgeCV meta | Script 4b | L2 regularization | Shrinks meta weights |
| minutes_bias_weight | `model_a.minutes_bias_weight` | 0.3 | Blends attention with minutes |
| Dropout | `model_a.dropout` | 0.2 | Regularization |
| Elo regression_to_mean | `elo.regression_to_mean` | 0.25 | Pulls Elo toward mean |

## 2. Proposed Experiments (Reduce Smoothing)

### 2a. Rolling Windows (highest impact)

- **Hypothesis:** Shorter windows capture recent form better.
- **Change:** Sweep `rolling_windows: [5,15]`, `[7,21]`, `[10,20]` vs `[15,30]`.
- **Config:** Add `config/outputs5_precision_rolling.yaml` or extend sweep phase.

### 2b. Odds Temperature

- **Hypothesis:** Lower temp sharpens championship odds.
- **Change:** `odds_temperature: 0.3` or 0.2 vs 0.5.
- **Risk:** Very low (e.g. 0.1) can cause numerical issues.

### 2c. Attention Temperature

- **Hypothesis:** Sharper attention concentrates on fewer players.
- **Change:** `model_a.attention.temperature: 0.7` or 0.5.
- **Risk:** Too low can cause attention collapse.

### 2d. Percentile vs Raw Scale

- **Hypothesis:** Percentile compresses score differences.
- **Change:** Try `true_strength_scale: minmax` or raw scaling.
- **Location:** `src/inference/predict.py`; `config` `output.true_strength_scale`.

### 2e. Minutes Bias Weight

- **Hypothesis:** Lower blend lets learned attention dominate.
- **Change:** `minutes_bias_weight: 0.1` or 0.0 vs 0.3.

## 3. Experiment Log (placeholder)

| Change | Config | rank_mae | rank_rmse | ndcg4 | spearman |
|--------|--------|----------|-----------|-------|----------|
| Baseline | defaults | — | — | — | — |
| (add rows as experiments run) | | | | | |

## 4. Recommendation Order

1. Rolling windows
2. odds_temperature
3. attention_temperature
4. Percentile vs minmax
5. minutes_bias_weight
