# NBA Analyst Metrics Investigation

## Overview

Investigation of metrics NBA analysts use beyond basic box-score (points, rebounds, assists) and their potential usefulness as model inputs or analysis tools.

## Metrics

| Metric | Description | Computable from our DB? | Useful for inputs/analysis? |
|--------|-------------|-------------------------|-----------------------------|
| **RAPM** (Regularized Adjusted Plus-Minus) | Plus-minus adjusted for lineup, opponent, and regularized | No — requires large lineup play-by-play data and RAPM solver | Could be evaluation proxy if we had it; not for inputs |
| **BPM** (Box Plus-Minus) | Box-score-derived, regression-based estimate of per-100 impact | Partially — we have box stats; BPM formula is public | Could implement as feature or evaluation |
| **VORP** (Value Over Replacement Player) | BPM × minutes, minus replacement level | Partially — needs BPM first | Could implement as evaluation metric |
| **PIPM** | Player Impact Plus-Minus (partially public) | No — proprietary multi-year RAPM | Evaluation proxy only if obtained externally |
| **LEBRON** | BBall-Index metric | No — proprietary | N/A |
| **EPM** (Estimated Plus-Minus) | Public EPM model | No — external API/dataset | Could use as evaluation proxy |
| **Win Shares** | Offensive + Defensive Win Shares from box stats | Partially — we have box stats; formulas exist | Could implement |
| **Usage rate** | FGA + 0.44×FTA + TOV per 100 possessions | Yes — from player logs | Good input feature |
| **TS%** (True Shooting %) | PTS / (2 × (FGA + 0.44×FTA)) | Yes — from player logs | Good input feature |
| **On/off** | Team margin when player on vs off court | No — needs lineup/play-by-play | Not feasible with current data |
| **Net rating** | Team offensive/defensive rating differential | Yes — from team_game_logs (for baselines only; excluded from model inputs) | Baselines only; never as model input |

## Recommendations

- **As inputs:** Usage rate, TS% — computable from our `player_game_logs` / `playoff_player_game_logs`; would enrich player features.
- **As evaluation proxies:** BPM, Win Shares — if we implement them from our data; VORP once BPM exists.
- **As baselines only:** Net rating — already used in baselines; excluded from model inputs per project rules.
- **Not feasible:** RAPM, PIPM, LEBRON, EPM, on/off — require external data or proprietary models.

## Next steps

1. Add TS% and usage rate to rolling features if beneficial.
2. Implement BPM-style or Game Score contribution for playoff players (see `compute_playoff_contribution_per_player`).
3. Document which metrics we use in ANALYSIS.md when we add them.
