---
name: Integrate RAPTOR Metrics
overview: Integrate FiveThirtyEight RAPTOR player metrics into the pipeline. RAPTOR data is available as CSV from GitHub. LEBRON is documented as future manual integration.
todos: []
isProject: false
---

# Integrate RAPTOR Metrics

## Data Source

FiveThirtyEight RAPTOR CSV files on GitHub:

- **URL:** [https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/](https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/)
- **File:** `historical_RAPTOR_by_player.csv` (or `modern_RAPTOR_by_player.csv` for 2014+)
- **Columns:** player_name, player_id (Basketball-Reference ID, e.g. "antetgi01"), season (e.g. "2022"), raptor_offense, raptor_defense, raptor_total, war_total, predator_*

## Player ID Mapping

RAPTOR uses Basketball-Reference IDs; our DB uses numeric NBA API player_ids. Strategy:

- Match on **player_name** against [src/data/db_loader.py](src/data/db_loader.py) players table
- Normalize names (strip Jr./III, handle accents) to reduce mismatches
- Season mapping: RAPTOR "2022" -> config "2022-23" (season start year)

## Implementation

### 1. Download script and loader

- **Script:** `scripts/1c_download_raptor.py` — fetch CSV from GitHub raw URL, save to `data/raw/raptor/historical_RAPTOR_by_player.csv`
- **Loader:** `src/data/raptor_loader.py` — `load_raptor_by_player(path, players_df, seasons_cfg)` returns DataFrame with columns: player_id (our NBA id), season (our format "2022-23"), raptor_offense, raptor_defense, raptor_total

### 2. Integration points


| Location                     | Use                                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model B team context**     | Per [.cursor/plans/comprehensive_feature_and_evaluation_expansion.plan.md](.cursor/plans/comprehensive_feature_and_evaluation_expansion.plan.md): add `raptor_offense_sum_top5`, `raptor_defense_sum_top5` — sum RAPTOR of top-5 by minutes per roster, merge in [src/features/team_context.py](src/features/team_context.py) |
| **Injury adjustment**        | Optional: use `raptor_total` as player impact metric instead of `on_court_pm_approx` when RAPTOR enabled (config flag in [src/features/injury_adjustment.py](src/features/injury_adjustment.py))                                                                                                                              |
| **Model A (optional later)** | Could add raptor_offense/defense as player stats — would increase stat_dim; defer unless beneficial                                                                                                                                                                                                                           |


### 3. Config

Add to [config/defaults.yaml](config/defaults.yaml):

```yaml
raptor:
  data_path: "data/raw/raptor/historical_RAPTOR_by_player.csv"
  enabled: false
```

### 4. team_context wiring

In [src/features/team_context.py](src/features/team_context.py):

- When `config.raptor.enabled` and path exists: load RAPTOR via raptor_loader
- For each (team_id, as_of_date): get roster top-5 by minutes, sum raptor_offense and raptor_defense, merge as `raptor_offense_sum_top5`, `raptor_defense_sum_top5`
- Add to `get_team_context_feature_cols` when enabled

### 5. Pipeline order

- Run `1c_download_raptor` after DB exists (needs players for mapping validation)
- Enable in config, retrain Model B (4, 4b), run inference (6)

## LEBRON (future)

Document in README: LEBRON requires manual CSV export from BBall Index ([https://www.bball-index.com/lebron-database/](https://www.bball-index.com/lebron-database/)) or the free Google Sheets database. Place in `data/raw/lebron/` with schema: player_id or player_name, season, lebron_total, o_lebron, d_lebron. Add loader + team_context wiring when file present.

## File changes summary

- **New:** [scripts/1c_download_raptor.py](scripts/1c_download_raptor.py)
- **New:** [src/data/raptor_loader.py](src/data/raptor_loader.py)
- **Edit:** [config/defaults.yaml](config/defaults.yaml) — add raptor section
- **Edit:** [src/features/team_context.py](src/features/team_context.py) — load RAPTOR, merge raptor_offense_sum_top5, raptor_defense_sum_top5, extend get_team_context_feature_cols
- **Edit:** [src/features/injury_adjustment.py](src/features/injury_adjustment.py) — optional config to use raptor_total instead of on_court_pm_approx when RAPTOR enabled
- **Edit:** [README.md](README.md) — document RAPTOR and LEBRON (future)

