# Playoff Performance Learning

## Goal

Use playoff stats to train a model for how player performance changes in the playoffs.

## Approaches

### 1. Separate playoff head or fine-tuning

- Train a playoff-only ListMLE or regression on playoff logs.
- **Pros:** Clean separation; can specialize for playoff context.
- **Cons:** Less data; may overfit.

### 2. Regular-season model plus playoff adjustment

- Keep current model; add residual or linear layer on playoff-specific features.
- **Pros:** Reuses learned representations; playoff head learns delta.
- **Cons:** Requires careful feature design for playoff adjustment.

### 3. Single model with playoff flag

- Add a "playoff" binary flag or playoff-specific embeddings.
- **Pros:** One model; can share weights.
- **Cons:** Needs enough playoff samples; risk of dilution.

### 4. Learning over time (one season at a time)

- Train on 2015–16 playoffs, then 2016–17, etc. for temporal robustness.
- **Pros:** Reduces distribution shift; more interpretable.
- **Cons:** More complex pipeline; fewer samples per step.

## Recommended path

Start with **Approach 2** (residual layer on playoff features) or a lightweight **Approach 1** (playoff-only head). Optionally add **Approach 4** for robustness.

## Data

- `playoff_player_game_logs` with game_date, season
- `playoff_team_game_logs`, `playoff_games`
- `compute_playoff_contribution_per_player()` for per-player playoff contribution
