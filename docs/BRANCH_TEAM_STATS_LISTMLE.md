# Branch: feature/team-stats-listmle

Purpose: run Model A with team standing rank as an input feature and train with ListMLE.

## Canonical config

- `config/team_stats_listmle.yaml`
- Uses `model_a.stat_dim: 22` and `model_a.use_standing_rank: true`
- Uses `training.loss_type: listmle` and `training.listmle_target: final_rank`

## Run from project root

Windows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config config/team_stats_listmle.yaml --outputs outputs_team_stats_listmle/run_001
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/team_stats_listmle.yaml --outputs outputs_team_stats_listmle/run_001
```

## Related existing config

- `config/outputs4_baseline_standing_rank.yaml` is a baseline-style standing-rank run that writes to `outputs12_baseline_standing_rank`.
