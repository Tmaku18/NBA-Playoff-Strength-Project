# Model A negative Spearman with standing rank (outputs10)

## Observation

In outputs10 (Spearman-surrogate + **standing rank as input**), Model A has **negative** Spearman in all 4 evaluated combos (~-0.45 to -0.62). The ensemble is weakly positive only because Model B (XGB) dominates. So the roster model is effectively **inverting** the ranking (high score for bad teams, low score for good teams).

## Checks performed

1. **Training target**  
   - `rel_values = 31 - final_rank`; `final_rank` from `compute_eos_final_rank` with **1 = champion**.  
   - So high rel = good team; loss (spearman_surrogate) encourages **high score for high rel**.  
   - No sign bug in labels.

2. **Standing rank definition**  
   - `standing_rank_norm(rank) = (31 - rank) / 30` → **1 = best** team.  
   - Same “high = good” convention as rel. No sign bug in the feature.

3. **Stat layout**  
   - `build_roster_set` appends `pct_min`, `standing_norm`, then `minutes_norm`, `starter_flag`, `rank_feature`.  
   - Training uses `stat_dim_override = batches[0]["player_stats"].shape[-1]`, so model input dim matches data (25 with standing).  
   - Inference infers `stat_dim` from checkpoint. No stat_dim/column mismatch.

4. **Evaluation**  
   - Spearman(actual_rank, pred_score) with actual_rank = EOS_global_rank (1 = best).  
   - Positive correlation = higher score for better (lower) actual rank.  
   - Evaluation direction is correct.

## Likely cause: standing rank vs outcome at list date

- **Training**: Lists are built at many **as_of_date**s. At each date we feed **current W/L standing** (`standing_rank_norm`).  
- For many dates, the **eventual champion** did not have the best standing (early season, slumps, injuries). So the model sees many **(low standing_rank_norm, high rel)** pairs.  
- It can then learn a **negative** weight on standing: “low standing at list date → high score.”  
- **Test**: We evaluate on test seasons; by end-of-season the champion usually has **good** standing. So we feed **high** standing_rank_norm for the champion → model gives **low** score → champion is under-ranked → **negative** Model A Spearman.

So the inversion is consistent with **over-reliance on standing at list date** when that standing is often misaligned with **eventual** playoff outcome in training.

## Recommendations

1. **Ablate standing in Model A**  
   - Add a config flag, e.g. `model_a.use_standing_rank: false`, and when false pass `team_standing_rank_norm=0` (or omit it and reduce stat_dim) so Model A does not see standing.  
   - Re-run one outputs10 combo (same config except this flag). If Model A Spearman becomes positive, that supports the hypothesis.

2. **Keep standing only in Model B**  
   - Use standing rank as a **team-context** feature for XGB/LR (as now) but **not** in the deep-set input. That preserves the “standing as input” experiment for the tabular model without letting it distort the roster model.

3. **Regularize or down-weight standing in Model A**  
   - If you keep standing in Model A, consider: smaller learning rate for that input, or an auxiliary loss that encourages positive correlation between standing_rank_norm and score on a held-out set where standing and outcome are aligned.

4. **Full 40-trial outputs10**  
   - Run the full sweep with the same setup; a few trials might learn a better balance. The current 4-trial slice may be unlucky.

## Config / code

- **Where standing is added**: `src/features/build_roster_set.py` (team_standing_rank_norm in the player stat vector).  
- **Where it’s filled**: `src/training/data_model_a.py` (standing_map from `standing_rank_as_of_date`, then `standing_rank_norm(standing_map.get(tid, 30))`).  
- **Config**: `config/defaults.yaml` has `stat_dim: 22`; the actual input length with standing is **25** (20 base + 1 pct_min + 1 standing + 3 usage). The model is built from batch shape at train time and from checkpoint at inference, so no bug there; the comment in defaults is just outdated.

## Ablation: disable standing in Model A

A config flag **`model_a.use_standing_rank`** (default `true`) was added. When set to **`false`**:

- Model A’s roster vectors get `team_standing_rank_norm=0` (no standing signal).
- The stat vector length is unchanged (25), so existing checkpoints still load; only that dimension is zeroed at train/inference.

**To test the inversion hypothesis:** Run one outputs10-style combo (Spearman-surrogate, same data) with `model_a.use_standing_rank: false` in the config. If Model A Spearman becomes positive, that supports “standing at list date” as the cause of the inversion. Model B can keep using standing via `TEAM_CONTEXT_FEATURE_COLS`.
