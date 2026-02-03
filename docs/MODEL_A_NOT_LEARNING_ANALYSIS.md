# Model A Not Learning — Analysis

## Run summary

Train Model A (script 3) was run with:

- **Early stop when not learning:** Process stops when train loss does not improve for 3 consecutive epochs. Loss is printed every epoch.
- **Config unchanged** (no config or hyperparameter changes).

**Result:** Training stopped early in every OOF fold and in the final model phase. Loss was flat at **27.8993** (or 25.59 in fold 1 epoch 1 then 27.90) after the first 1–2 epochs; after 3 epochs with no improvement the process printed the analysis and stopped.

## Why Model A is not learning

Per **docs/CHECKPOINT_PROJECT_REPORT.md** (Section 7, run_020) and the fix_attention plan:

1. **Attention has collapsed:** Set attention returns all-zero or effectively uniform weights (e.g. due to minutes reweighting with degenerate minutes, or softmax saturation). Debug shows `attn_sum_mean=0.0000`, `attn_grad_norm=0.0000`.

2. **Constant pooled representation:** When attention weights are zero or uniform, the pooled vector Z in `DeepSetRank` is the same (or nearly the same) for every team.

3. **Constant scores:** Score = `scorer(Z)`; same Z → same score for every team in a list.

4. **ListMLE loss is fixed:** ListMLE loss depends on the **scores** (one per team). If all scores in a list are equal, the NLL reduces to a function of list length only (sum of `log(L - i + 1)`), so the loss does not depend on model parameters and does not change across epochs.

So: **collapsed attention → constant Z → constant scores → flat ListMLE loss → no learning.**

## What to do next

- **Harden set_attention:** See `.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md`. Already implemented: minutes reweighting only when minutes are meaningful; uniform fallback over valid positions when raw attention is zero so gradients can flow. If loss is still flat, the raw attention from `MultiheadAttention` may still be zero (e.g. key_padding_mask or numerical issues); consider init or architecture changes so attention does not start at zero.
- **Checkpoint report:** Run_021 is the first run where Model A contributed (non-zero attention, primary_contributors). If the current data/config differ from that run, re-align (same stat columns, same roster logic) so Model A can learn again.
- **Verify on dummy data:** `scripts/verify_model_a_training.py` trains on dummy batches; there loss decreases and attention is non-zero. So the issue is data- or config-specific (e.g. real batches yielding zero raw attention).

## References

- **docs/CHECKPOINT_PROJECT_REPORT.md** — Section 7 (Issues: attention all-zero / Model A not contributing), Section 5 (run_020 vs run_021).
- **.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md** — Fix attention collapse and trustworthy run.
- **src/models/set_attention.py** — Minutes reweighting and uniform fallback when raw attention is zero.
- **src/models/listmle_loss.py** — ListMLE loss; constant scores → constant loss.
