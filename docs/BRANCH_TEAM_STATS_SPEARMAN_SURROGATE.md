# Branch: feature/team-stats-spearman-surrogate

Purpose: run Model A with **team stats** (eFG, TOV_pct, FT_rate, ORB_pct, pace) as input and train with Spearman-surrogate loss. Config matches **8_spearman_surrogate** best (combo_0033) then adds team-stats: same loss, target, rolling, model_a epochs, model_b HPs; only `stat_dim: 27`, `use_team_stats: true` (and `use_standing_rank: false`) differ.

## Canonical config

- `config/team_stats_spearman_surrogate.yaml`
- **Paths:** `paths.outputs: "output/team_stats_spearman_surrogate"`
- **Training:** `loss_type: spearman_surrogate`, `listmle_target: playoff_outcome`, `rolling_windows: [10, 30]`, `loss_tau: 1.0`
- **Model A:** `stat_dim: 27` (21 + team-stats dims), `use_team_stats: true`, `use_standing_rank: false`, `epochs: 15` (match combo_0033), `team_stats_cols: [eFG, TOV_pct, FT_rate, ORB_pct, pace]`
- **Stacking:** `use_confidence: false`

## Run from project root

Config already sets `paths.outputs: "output/team_stats_spearman_surrogate"`. You can omit `--outputs` or pass it explicitly:

Windows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config config/team_stats_spearman_surrogate.yaml
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/team_stats_spearman_surrogate.yaml
```

To force a subfolder: `--outputs output/team_stats_spearman_surrogate/run_001`

## Related config

- `config/outputs10_sweep_standing_rank.yaml` is the existing sweep setup for standing-rank input with Spearman-surrogate training (writes to `output/10_spearman_surrogate_standing_rank`).
