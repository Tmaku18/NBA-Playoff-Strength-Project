# Branch: feature/team-stats-listmle

Purpose: run Model A with team standing rank as an input feature and train with ListMLE. Config matches **7_listmle** best (combo 17): playoff_outcome, rolling [10,30], same model_a epochs and model_b HPs, plus standing rank.

## Canonical config

- `config/team_stats_listmle.yaml`
- **Paths:** `paths.outputs: "output/team_stats_listmle"`
- **Training:** `loss_type: listmle`, `listmle_target: playoff_outcome`, `rolling_windows: [10, 30]`
- **Model A:** `stat_dim: 22`, `use_standing_rank: true`, `epochs: 14` (match 7_listmle combo 17)

For the exact 7_listmle best *without* standing rank, use `config/outputs7_listmle_best.yaml` (writes to `output/7_listmle` or as configured).

## Run from project root

Config already sets `paths.outputs: "output/team_stats_listmle"`. You can omit `--outputs` or pass it explicitly:

Windows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config config/team_stats_listmle.yaml
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/team_stats_listmle.yaml
```

To force a subfolder: `--outputs output/team_stats_listmle/run_001`

## Related config

- `config/outputs4_baseline_standing_rank.yaml` is a baseline-style standing-rank run that writes to `output/12_baseline_standing_rank`.
