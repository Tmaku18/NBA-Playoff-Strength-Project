# Standing Rank as Input Feature

**Implementation note:** Standing rank is implemented on `main` for **team-context features**. Model A no longer uses standing rank.

Current regular-season standing (rank 1–30, global) is used as an **input feature** to team-context models (Model B / Model C) and optional team-stats models.

## Implementation

- **Model A (DeepSetRank):** Standing rank is **not** used as an input feature (removed from roster-set features).
- **Model B and Model C:** Team-context includes `standing_rank_norm` in the feature set (see `TEAM_CONTEXT_FEATURE_COLS` in `src/features/team_context.py`). Rank is computed from games with `game_date < as_of_date` (no future leakage).

When predicting **playoff rank**, the same feature is used: if `as_of_date` is end of regular season, the value is the **regular season final rank**.

## Hypothesis (10_spearman_surrogate_standing_rank sweep)

In prior comparisons (e.g. output/5_listmle), **standings-trained** ListMLE matched or beat **outcome-trained** when evaluated on playoff outcome. That suggests the model benefits from information that correlates with standings. We therefore **hypothesize** that giving the model **current standings explicitly as an input** (standing rank) should increase accuracy: the model can use this signal directly instead of inferring it only from roster/team stats. The **10_spearman_surrogate_standing_rank** sweep runs the same Optuna setup as 8_spearman_surrogate (Spearman-surrogate, playoff_outcome) but with standing rank as input; results will show whether the hypothesis holds. Path: `output/10_spearman_surrogate_standing_rank`. See [OUTPUTS10_SWEEP_STANDING_RANK.md](OUTPUTS10_SWEEP_STANDING_RANK.md).

## Config

- **model_a.stat_dim:** Model A no longer includes standing rank; current default is `model_a.stat_dim: 24`.
- **model_b:** `standing_rank_norm` is in the default feature list; use `model_b.exclude_features: ["standing_rank_norm"]` to disable for ablation.

## Future work (not yet implemented)

- **Conference-specific rank (1–15):** Extend `standing_rank_as_of_date` with a `scope="conference"` (and team→conference map) so rank is 1–15 per conference instead of 1–30 global.
- **Train East/West separately:** Build lists per conference; train two Model A (and optionally two Model B) instances—one for East, one for West. At inference, score East teams with the East model and West with the West model; combine for conference rankings. Use the **existing** finals logic (e.g. `monte_carlo_championship` or current ensemble) to decide the champion from the two conference winners/top seeds.
