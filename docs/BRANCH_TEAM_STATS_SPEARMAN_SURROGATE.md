# Branch: feature/team-stats-spearman-surrogate

Purpose: run Model A with team standing rank as an input feature and train with Spearman-surrogate loss.

## Canonical config

- `config/team_stats_spearman_surrogate.yaml`
- Uses `model_a.stat_dim: 22` and `model_a.use_standing_rank: true`
- Uses `training.loss_type: spearman_surrogate`, `training.listmle_target: playoff_outcome`, and `training.loss_tau: 1.0`

## Run from project root

Windows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config config/team_stats_spearman_surrogate.yaml --outputs outputs_team_stats_spearman_surrogate/run_001
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/team_stats_spearman_surrogate.yaml --outputs outputs_team_stats_spearman_surrogate/run_001
```

## Related sweep config

- `config/outputs10_sweep_standing_rank.yaml` is the existing sweep setup for standing-rank input with Spearman-surrogate training.
