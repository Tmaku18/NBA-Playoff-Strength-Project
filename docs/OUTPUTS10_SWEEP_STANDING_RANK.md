# outputs10 Sweep: Standing Rank as Input (Same as outputs8)

**Purpose:** Run the same Optuna sweep as outputs8 (Spearman-surrogate, playoff_outcome) but with the **new implementation** that includes **standing rank as an input feature** for Model A, B, and C. Compare results to outputs8 to test whether adding standings as input increases accuracy.

---

## Hypothesis

In prior work (e.g. outputs5 ListMLE outcome vs standings), **standings-trained** models matched or beat outcome-trained models when evaluated on playoff outcome. That suggests the model benefits from information correlated with standings. We **hypothesize** that providing **current regular-season standing as an explicit input** (standing_rank_norm) should **increase accuracy**: the model can use this signal directly in addition to roster and team-context features, rather than inferring it only indirectly. The outputs10 sweep tests this by comparing Optuna best values (Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG) to outputs8 with the same methodology and objectives.

---

## Setup

- **Branch:** `feature/listmle-position-aware` (standing rank + position-aware ListMLE in code).
- **Config:** `config/outputs10_sweep_standing_rank.yaml` — same as outputs8: `paths.outputs: "outputs10"`, `training.listmle_target: playoff_outcome`, `training.loss_type: spearman_surrogate`.
- **Output root:** `outputs10/`. Sweep writes to `outputs10/sweeps/<batch_id>/`.

---

## How to Run (WSL recommended)

From project root with `PYTHONPATH` set:

```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs10_sweep_standing_rank.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome
```

Optional: set a named batch for easier reference:

```bash
python -m scripts.sweep_hparams --config config/outputs10_sweep_standing_rank.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id standing_rank_spearman_40
```

Other objectives (same as outputs8):

- `--objective playoff_spearman`
- `--objective ndcg16`
- `--objective rank_rmse`

---

## Optuna Results (where to find them)

After the sweep finishes, under **`outputs10/sweeps/<batch_id>/`**:

| File | Contents |
|------|----------|
| **optuna_study.json** | Optuna study state (trials, best value, params). |
| **optuna_importances.json** | Parameter importances from Optuna (which HPs mattered most). |
| **sweep_results.csv** | One row per combo: combo index, metrics (spearman, playoff_spearman, rank_mae, rank_rmse, ndcg*, etc.), and trial params. |
| **sweep_results_summary.json** | Summary with **best_optuna_trial** (combo index, value, params), best_by_spearman, best_by_ndcg16, etc. |

Best config path for the chosen objective:  
`outputs10/sweeps/<batch_id>/combo_<NN>/config.yaml`  
where `combo_<NN>` is the combo index in `best_optuna_trial` (e.g. `combo_0033`).

---

## Comparison to outputs8

Once the outputs10 sweep has run, compare:

- **outputs8 best** (e.g. combo 33 Spearman 0.777, combo 38 playoff_spearman 0.854): [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md).
- **outputs10 best** from `sweep_results_summary.json` and `sweep_results.csv`.

If outputs10 best Spearman / playoff_spearman are higher (or rank_mae/rank_rmse lower) than outputs8, that supports the hypothesis that standing rank as input increases accuracy.
