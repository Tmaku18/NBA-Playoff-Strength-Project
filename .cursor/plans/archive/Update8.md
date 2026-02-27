# Update8: outputs2, run_019, sweeps & forward plan

## 4.1 Changes since last update (Update7)

- **Outputs:** Pipeline writes to **outputs2** (config `paths.outputs`); run numbering continues from **run_019** when the outputs directory has no `run_*` subdirs and `inference.run_id_base: 19` is set.
- **Inference:** Per-season inference (predictions and figures per test year, e.g. 2023-24, 2024-25); `pred_vs_playoff_rank` legend outside plot; East = dots, West = squares; conference rank as primary rank output; `actual_conference_rank` (Actual Conference Rank) and removal of `global_rank` from prediction output.
- **Sweeps:** Documented (and optionally stubbed in `scripts/sweep_hparams.py`) to repeat hyperparameter testing in the **foreground**, robust, no timeout; when implemented, sweeps write to the configured outputs path (e.g. `outputs2/sweeps/<batch_id>/`).
- **Config / scripts:** `config/defaults.yaml` sets `paths.outputs: "outputs2"` and `inference.run_id_base: 19`. `scripts/6_run_inference.py` uses `run_id_base` when the outputs dir has no `run_*` folders. `scripts/compare_runs.py` and `scripts/run_manifest.py` resolve the outputs directory from config (no hardcoded `outputs`). `outputs2/` created with `.gitkeep`. README documents outputs2, run_019, and sweeps (foreground, robust).

---

## 4.2 Plan moving forward

1. **Hyperparameter testing (next)**  
   Run or implement sweeps (Model A epochs, Model B grid) as the immediate next step; sweeps run in the **foreground** and write to the configured outputs path (e.g. `outputs2/sweeps/`). Requirements: use config for paths; handle missing dirs; avoid overwriting previous batch results (e.g. `--batch-id` or timestamped batch folder); no artificial timeout; no daemon/background mode.

2. **Primary contributors**  
   Fix outputs so **primary contributor information** is displayed and **primary contributors are calculated** correctly (attention/IG path and any fallbacks).

3. **Next plan: playoff performance learning**  
   Use **playoff stats** to train a model for how **player performance changes in the playoffs**. Possible approaches:
   - Separate playoff head or fine-tuning on playoff logs (e.g. playoff-only ListMLE or regression).
   - Regular-season model plus playoff adjustment (residual or linear layer on playoff features).
   - Single model with a “playoff” flag or playoff-specific embeddings.
   - **Learning over time:** Optionally implement **one season at a time** (e.g. train on 2015–16 playoffs, then 2016–17, etc.) for temporal robustness.

4. **Actual playoff contribution**  
   **Calculate actual playoff contribution** using playoff stats. Define an effective formula (e.g. box-score-based contribution per game or per possession, or plus-minus style from playoff logs) and implement it; exact metric TBD (e.g. playoff BPM-style, or points + assists + rebounds weighted, or RAPM-lite from playoff lineups).

5. **NBA analyst metrics investigation**  
   **Investigate** which metrics NBA analysts use beyond basic box-score (points, rebounds, assists). Examples: RAPM, BPM, VORP, PIPM, LEBRON, EPM, Win Shares, usage rate, TS%, on/off, net rating in playoffs. Note which could be **useful as inputs or analysis** for our model (e.g. as features, as evaluation proxies, or in ANALYSIS.md).
