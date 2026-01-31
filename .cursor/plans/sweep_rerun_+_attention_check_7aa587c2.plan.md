---
name: Sweep Rerun + Attention Check
overview: Rerun the sweep with a larger validation slice (medium profile) and a higher epoch range, then add a small attention diagnostic so we can answer whether Model A’s attention is producing non-degenerate weights.
todos:
  - id: config-medium-profile
    content: Update `config/defaults.yaml` to medium data profile values
    status: completed
  - id: attention-diagnostic
    content: Add attention stats logging to sweep script (inspect `attn_w` from `DeepSetRank`)
    status: completed
  - id: rerun-sweep
    content: Run sweep with epochs 15,20,25,30,35,40
    status: pending
  - id: report-results
    content: Summarize sweep metrics + attention diagnostics
    status: pending
isProject: false
---

# Sweep Rerun + Attention Check

## Scope and approach

- Use the sweep script with a higher epoch list and the medium data profile (larger val set), then rerun the batch.
- Add a lightweight attention diagnostic that inspects `attn_w` returned by `DeepSetRank` to confirm weights are finite and non-zero.

## Current repo status (already implemented)

### Medium profile config values

Already present in `config/defaults.yaml`:

- `model_a.early_stopping_val_frac = 0.20`
- `training.max_lists_oof = 30`
- `training.model_a_history_days = 120`

### Attention diagnostic

Already present in `scripts/sweep_hparams.py`:

- `_attention_stats(...)` computes `attn_min/mean/max`, `attn_zero_pct`, `attn_nan_pct`
- `main()` appends these metrics into the epoch sweep results

## Execution steps

- Run the sweep with epochs `15,20,25,30,35,40`:
  - `python -u "scripts/sweep_hparams.py" --batch epochs_plus_model_b --epochs 15,20,25,30,35,40`
- Results will be written under `outputs/sweeps/batch_001/` and should be summarized:
  - Best Model A epoch (NDCG/Spearman/MRR)
  - Best Model B combo (RMSE mean and Spearman mean)
  - Attention diagnostic stats (min/mean/max, % zeros, NaNs)

## What this answers

- Whether a larger validation slice changes the “best epoch” for Model A.
- Whether Model B hyperparameters shift with more data.
- Whether Model A attention weights are non-degenerate on real data.

