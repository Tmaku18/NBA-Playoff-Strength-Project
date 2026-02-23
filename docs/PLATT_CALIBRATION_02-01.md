# Platt scaling (both variants)

Two calibration variants are implemented for testing; compare ECE and Brier in [outputs3/](outputs3/) runs and document in numbered ANALYSIS in the run folder.

## 1. Platt on meta-learner (stacker) output

- **Code:** [src/models/calibration.py](../src/models/calibration.py) — `platt_scale_meta(scores, y_binary)` and `calibrated_probs_meta(platt, scores)`.
- **Flow:** Use RidgeCV (or current meta) to get raw ensemble scores; fit Logistic Regression on (scores, binary labels); at inference, pass raw meta-learner output through the fitted sigmoid.
- **Use:** When the final blended score is the single quantity to calibrate (e.g. championship probability).

## 2. Platt on raw model outputs (per-model then combine)

- **Code:** [src/models/calibration.py](../src/models/calibration.py) — `platt_scale_per_model(X_oof, y_binary)` and `calibrated_probs_per_model(platt_list, X, combine="mean")`.
- **Flow:** Fit one Platt (LogisticRegression) per column of OOF (model_a, xgb, rf); at inference, calibrate each model’s score then combine (e.g. mean or max of calibrated probs).
- **Use:** When you want per-model calibrated probabilities then aggregate.

## Comparing in outputs3

- Run evaluation with and without each variant; write ECE, Brier, NDCG, Spearman to `outputs3/<run_id>/ANALYSIS_XX.md` (or eval_report + ANALYSIS).
- Document which variant was used and the metrics in the run folder so both can be compared.
