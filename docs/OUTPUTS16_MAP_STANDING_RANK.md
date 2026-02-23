# outputs16: MAP run with standing rank

**Purpose:** Dedicated output for the **MAP (Maximum A Posteriori) branch** model run **with standing rank as an input feature**. Same as outputs14_map_run but with `standing_rank_norm` in the feature set (stat_dim 22, use_standing_rank true). When the MAP branch is ready, run the pipeline with this config so results go to **outputs16_map_standing_rank/**. Compare to outputs14_map_run (MAP, no standing), baseline (outputs6), and best (outputs8).

---

## Setup

- **Config:** `config/outputs16_map_standing_rank.yaml`
- **Paths:** `paths.outputs: "outputs16_map_standing_rank"`
- **Input:** Standing rank via `model_a.stat_dim: 22`, `model_a.use_standing_rank: true`
- **Baseline reference:** outputs6_baseline. **Best reference:** outputs8_spearman_surrogate.

---

## How to run (when MAP branch is ready)

From project root:

**WSL:**
```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs16_map_standing_rank.yaml --outputs outputs16_map_standing_rank/map_standing_rank
```

**PowerShell:**
```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config config/outputs16_map_standing_rank.yaml --outputs outputs16_map_standing_rank/map_standing_rank
```

---

## After the run

- Run **per-game** evaluation as needed.
- Compare metrics to **outputs6_baseline** (baseline), **outputs8_spearman_surrogate** (best), and **outputs14_map_run** (MAP without standing rank).

See [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md) for the full lineup.
