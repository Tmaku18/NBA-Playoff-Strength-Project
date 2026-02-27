# Analysis 01 — Singular run (combo_004 high epochs)

**Run:** Single pipeline run with **combo_004** hyperparameters and **80 epochs** max (early_stopping_patience 10).  
**Config:** `config/outputs13_combo004_high_epochs.yaml`  
**Output dir:** `outputs13_rmse_surrogate/combo_004_high_epochs/`

---

## Test metrics (ensemble)

### By season

| Season   | Spearman | playoff_spearman | rank_mae | rank_rmse | NDCG@30 |
|----------|----------|------------------|----------|-----------|---------|
| 2023-24  | -0.21    | -0.38            | 10.73    | 13.48     | 0.318   |
| 2024-25  | -0.04    | -0.14            | 9.80     | 12.49     | 0.294   |

Root `eval_report.json` (aggregate / primary test): **Spearman -0.04**, **rank_mae 9.80**, **rank_rmse 12.49**, **playoff_spearman -0.14**, NDCG@30 0.294.

---

## Comparison: high-epoch vs 22-epoch combo_004 (sweep)

| Metric           | 22-epoch (sweep combo_004) | 80-epoch (this run) | Note                          |
|------------------|----------------------------|----------------------|-------------------------------|
| Spearman         | -0.16                      | -0.21 (23-24), -0.04 (24-25) | 23-24 worse; 24-25 better     |
| playoff_spearman | -0.30                      | -0.38 (23-24), -0.14 (24-25) | 23-24 worse; 24-25 better     |
| rank_mae         | 11.0                       | 10.73 (23-24), 9.80 (24-25)  | Slightly better with more epochs |
| rank_rmse        | 13.21                      | 13.48 (23-24), 12.49 (24-25) | 23-24 slightly worse; 24-25 better |

**Summary:** More epochs gave **mixed** results: **2023-24** is slightly worse (more negative correlation); **2024-25** is better (near-zero Spearman, better rank_mae/rank_rmse). The RMSE surrogate still does **not** achieve positive correlation on 2023-24 and remains far below the Spearman surrogate.

---

## Comparison: vs official best (outputs8 Spearman surrogate)

| Metric           | outputs8 best (combo_0033/0038) | This run (80-epoch RMSE) |
|------------------|----------------------------------|---------------------------|
| Spearman         | **0.777**                        | -0.04 to -0.21            |
| playoff_spearman | **0.854**                        | -0.14 to -0.38            |
| rank_mae         | **4.80**                         | 9.80–10.73                |
| rank_rmse        | **5.78**                         | 12.49–13.48               |
| NDCG@30          | **0.52**                         | 0.29–0.32                 |

**Conclusion:** The singular high-epoch RMSE run does **not** close the gap to the Spearman surrogate. **outputs8 remains the recommended production setup.**

---

## Training loss

- **training_loss.csv:** Not present in this output directory. If the run used a build that logs loss, it may be in another run dir or the final walk-forward step did not write it (e.g. different script version). For future runs, script 3 writes `training_loss.csv` in the config output dir when available.

---

## vs W/L standings

- **2024-25:** Standings rank_mae 3.13, rank_rmse 4.45; ensemble rank_mae 9.80, rank_rmse 12.49 — ensemble is **worse** than standings.
- **2023-24:** Standings rank_mae 3.73, rank_rmse 6.09; ensemble rank_mae 10.73, rank_rmse 13.48 — ensemble is **worse** than standings.

---

## Takeaway

- **More epochs (80 vs 22)** improved **2024-25** (correlation near zero, better rank error) but **2023-24** stayed or got slightly worse.
- **RMSE surrogate** with this setup still underperforms **Spearman surrogate** (outputs8) by a large margin and does not beat W/L standings. Prefer **Spearman surrogate** and **outputs8** for ranking vs playoff outcome.

See `run_025/eval_report_2023-24.json`, `run_025/eval_report_2024-25.json`, and `eval_report.json` for full metrics.
