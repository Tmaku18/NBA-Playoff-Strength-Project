## Branch archive: `feature/listmle-position-aware` (why we can delete it)

This document is a **permanent archive** of what we implemented and ran on branch `feature/listmle-position-aware`, the **inferences/results it produced**, and how those results compare to the **current best outputs8 sweep**. After this doc exists on `main`, the branch can be deleted without losing the story.

---

## Summary (one paragraph)

The branch added **position-aware ListMLE** (optional position-weighted listwise loss), introduced a **standing-rank input feature** (outputs10 plan), and fixed/standardized several **Model A inference + evaluation** components (multi-temp aggregation support, bias/variance summaries). The **key experiment** was an **outputs9 ListMLE sweep** and a follow-up **best_position_aware** run. Neither beat **outputs8** on the headline playoff-outcome correlation targets (Spearman / playoff Spearman) or rank error; outputs8 remains the best choice for that objective set.

---

## What changed on the branch (high-signal)

From `main...feature/listmle-position-aware`, the branch contained these high-level workstreams (by commit):

- **Position-aware ListMLE (optional)** (`f800436`): Added config toggles in training for position weighting/discounting, and associated code changes.
  - **Config surface** (example): `training.listmle_position_aware`, `training.listmle_position_discount`.
  - **Example config in outputs9 best run**: `outputs9/best_position_aware/config.yaml`.

- **Official-best documentation + analysis writeups** (`cd26550`, plus many added `docs/*.md`): Added/updated docs to capture sweep comparisons and “official best” decisions during that period.
  - Note: those docs lived on the feature branch; this file is the minimal archive intended to survive branch deletion.

- **Standing rank as an input feature (outputs10 plan)** (`9d232a2`): Implemented and documented adding **standing rank** as an explicit model input (and a corresponding sweep plan).
  - The intent was to test whether explicitly providing standings information improves playoff-outcome ranking.
  - In this workspace snapshot, **no outputs10 eval artifacts were present** (no `outputs10/**/eval_report.json` found).

- **Bias/variance reporting + sweep std summaries** (`0a8413b`): Extended evaluation/sweep summaries to report **metric stability** (mean/std across seasons and across sweep trials).
  - This is the “variance” instrumentation that helped diagnose run-to-run instability and generalization noise.

- **Best-run artifacts committed on the branch** (`c7598d0`): A large commit capturing “best run” output folders and sweep summaries for reproducibility in that branch’s history.
  - On `main`, outputs are currently present **locally** (untracked) in this workspace; this doc references them by path.

- **Multi-temp aggregation module for Model A inference** (`54ccf85`): Added `src/models/multi_temp_aggregation.py` to support multi-temperature inference aggregation without import errors.

---

## What we ran (outputs9)

### 1) outputs9 ListMLE sweep (objective: Spearman)

- **Sweep root**: `outputs9/sweeps/outputs9_listmle_spearman/`
- **Mechanics**: 40 combo configs (`combo_0000` … `combo_0039`) with per-combo `outputs/eval_report.json`.
- **Important detail**: All sweep combos were `training.listmle_position_aware: false`.
- **Best combo by ensemble Spearman**: `combo_0016`.
  - Evidence: `outputs9/sweeps/outputs9_listmle_spearman/combo_0016/outputs/run_025/ANALYSIS_01.md` and `.../outputs/eval_report.json`.

Key metrics for `outputs9` sweep **best (combo_16)** (ensemble, test):

- **Spearman**: 0.4772
- **playoff_spearman**: 0.4719
- **rank_mae**: 7.0667
- **rank_rmse**: 8.8506

Source (run summary): `outputs9/sweeps/outputs9_listmle_spearman/combo_0016/outputs/run_025/ANALYSIS_01.md`.

### 2) outputs9 “best_position_aware” run (manual flip)

We took the sweep’s best config (combo_16) and **flipped**:
`training.listmle_position_aware: true` with discounting (`training.listmle_position_discount: log2`), then re-ran the pipeline into:

- **Config**: `outputs9/best_position_aware/config.yaml`
- **Outputs**: `outputs9/best_position_aware/outputs/run_025/`

Key metrics for `outputs9 best_position_aware` (ensemble, test):

- **Spearman**: 0.4194
- **playoff_spearman**: 0.4496
- **rank_mae**: 7.5333
- **rank_rmse**: 9.3274

Source: `outputs9/best_position_aware/outputs/run_025/ANALYSIS_01.md` (and `eval_report.json`).

Interpretation: **position-aware ListMLE did not improve** over the best non-position-aware sweep combo in this setup (it reduced Spearman and worsened rank error).

---

## Comparison to outputs8 (current best sweep family)

Two relevant “best” points in outputs8:

- **Best Spearman combo**: outputs8 `combo_0033`
  Source: `outputs8/sweeps/20260217_042955/combo_0033/outputs/run_025_02-17/ANALYSIS_01.md`
- **Best playoff_spearman combo**: outputs8 `combo_0038`
  Source: `outputs8/sweeps/20260217_042955/combo_0038/outputs/run_025_02-17/ANALYSIS_01.md`

### Headline metric table (ensemble, test)

| Run / config | Spearman | playoff_spearman | rank_mae | rank_rmse |
|---|---:|---:|---:|---:|
| **outputs8 best Spearman** (`combo_0033`) | **0.7771** | 0.8020 | **4.8000** | **5.7793** |
| **outputs8 best playoff_spearman** (`combo_0038`) | 0.7664 | **0.8536** | **4.8000** | 5.9161 |
| outputs9 sweep best (`combo_0016`, ListMLE) | 0.4772 | 0.4719 | 7.0667 | 8.8506 |
| outputs9 best_position_aware (ListMLE + position-aware) | 0.4194 | 0.4496 | 7.5333 | 9.3274 |

**Conclusion from the table:** outputs9 (ListMLE) and outputs9 best_position_aware are **not competitive** with outputs8 on Spearman/playoff_spearman and rank error. For the project’s “playoff-outcome ranking correlation” goal, outputs8 remains the winner.

---

## Final conclusion (branch deletion justification)

- The branch explored a reasonable hypothesis (ListMLE variants + position-aware weighting + standings input).
- The outputs9 experiments produced **worse** results than the outputs8 sweep family on the key objectives we prioritized.
- This doc preserves the intent, implementation scope, and empirical outcome.

After keeping this file on `main`, it is safe to **delete `feature/listmle-position-aware`** without losing the experimental record.

---

## References (artifact paths)

- outputs9 sweep best (combo_16):
  - `outputs9/sweeps/outputs9_listmle_spearman/combo_0016/outputs/run_025/ANALYSIS_01.md`
  - `outputs9/sweeps/outputs9_listmle_spearman/combo_0016/outputs/eval_report.json`
- outputs9 best_position_aware:
  - `outputs9/best_position_aware/config.yaml`
  - `outputs9/best_position_aware/outputs/run_025/ANALYSIS_01.md`
  - `outputs9/best_position_aware/outputs/run_025/eval_report.json`
  - `outputs9/best_position_aware/outputs/run_025/attention_significance.json`
- outputs8 bests:
  - `outputs8/sweeps/20260217_042955/combo_0033/outputs/run_025_02-17/ANALYSIS_01.md`
  - `outputs8/sweeps/20260217_042955/combo_0038/outputs/run_025_02-17/ANALYSIS_01.md`
- Helper script used to summarize outputs9 sweep combos:
  - `scripts/aggregate_outputs9.py`

