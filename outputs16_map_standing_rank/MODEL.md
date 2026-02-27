# outputs16_map_standing_rank — Model and role

**Folder name:** `outputs16_map_standing_rank`. See [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).

**Role:** **MAP (Maximum A Posteriori) branch** run with **standing rank as input**. Same as outputs14_map_run but with **standing_rank_norm** in the feature set (stat_dim 22, use_standing_rank true). Use when running the MAP branch model with standing rank; compare to outputs14_map_run (MAP, no standing), baseline (outputs6), and best (outputs8).

**Model:** MAP branch model (when ready) with **standing_rank_norm** in Model A/B/C features. Config: `config/outputs16_map_standing_rank.yaml`. Results go under `outputs16_map_standing_rank/` (e.g. `outputs16_map_standing_rank/map_standing_rank/`).

**Difference from outputs14:** outputs14 = MAP run **without** standing rank. outputs16 = **MAP + standing rank** as input. Isolates the effect of standing rank for the MAP branch.

**Run in WSL (from project root, when MAP branch is ready):**
```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs16_map_standing_rank.yaml --outputs outputs16_map_standing_rank/map_standing_rank
```

**See also:** [docs/OUTPUTS16_MAP_STANDING_RANK.md](../docs/OUTPUTS16_MAP_STANDING_RANK.md), [docs/OUTPUTS14_MAP_RUN.md](../docs/OUTPUTS14_MAP_RUN.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
