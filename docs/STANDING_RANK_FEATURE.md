# Standing Rank as Input Feature

**Implementation note:** This feature was added on `feature/listmle-position-aware` (and can be merged to `main`). Model A `stat_dim` becomes 22; retrain models after merging. See README § Branches for branch differences.

Current regular-season standing (rank 1–30, global) is used as an **input feature** to all models.

## Implementation

- **Model A (DeepSetRank):** Each team’s roster gets a scalar `standing_rank_norm` = `(31 - rank) / 30` (1 = best), appended to every player’s input vector. Rank is computed from games with `game_date < as_of_date` (no future leakage).
- **Model B and Model C:** Team-context includes `standing_rank_norm` in the feature set (see `TEAM_CONTEXT_FEATURE_COLS` in `src/features/team_context.py`).

When predicting **playoff rank**, the same feature is used: if `as_of_date` is end of regular season, the value is the **regular season final rank**.

## Hypothesis (outputs10 sweep)

In prior comparisons (e.g. outputs5), **standings-trained** ListMLE matched or beat **outcome-trained** when evaluated on playoff outcome. That suggests the model benefits from information that correlates with standings. We therefore **hypothesize** that giving the model **current standings explicitly as an input** (standing rank) should increase accuracy: the model can use this signal directly instead of inferring it only from roster/team stats. The **outputs10** sweep runs the same Optuna setup as outputs8 (Spearman-surrogate, playoff_outcome) but with standing rank as input; results will show whether the hypothesis holds. See [OUTPUTS10_SWEEP_STANDING_RANK.md](OUTPUTS10_SWEEP_STANDING_RANK.md).

## Config

- **model_a.stat_dim:** With standing rank enabled, use 22 (was 21). Training infers dimension from batch shape if batches are built with the new feature.
- **model_b:** `standing_rank_norm` is in the default feature list; use `model_b.exclude_features: ["standing_rank_norm"]` to disable for ablation.

## Future work (not yet implemented)

- **Conference-specific rank (1–15):** Extend `standing_rank_as_of_date` with a `scope="conference"` (and team→conference map) so rank is 1–15 per conference instead of 1–30 global.
- **Train East/West separately:** Build lists per conference; train two Model A (and optionally two Model B) instances—one for East, one for West. At inference, score East teams with the East model and West with the West model; combine for conference rankings. Use the **existing** finals logic (e.g. `monte_carlo_championship` or current ensemble) to decide the champion from the two conference winners/top seeds.
