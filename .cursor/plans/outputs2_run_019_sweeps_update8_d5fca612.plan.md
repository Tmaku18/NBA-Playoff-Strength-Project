---
name: outputs2 run_019 sweeps Update8
overview: Switch pipeline outputs to outputs2 with run numbering from run_019; add run_id_base config; update sweeps for hyperparameter testing (robust, not in background); update README and create Update8.md documenting changes since Update7 and the forward plan (hyperparameter testing, primary contributors, playoff performance learning, actual playoff contribution, NBA analyst metrics).
todos: []
isProject: false
---

# outputs2, run_019, sweeps, and Update8

## 1. outputs2 folder and run numbering from run_019

**Config:** [config/defaults.yaml](config/defaults.yaml)

- Set `paths.outputs: "outputs2"` (instead of `"outputs"`).
- Add `inference.run_id_base: 19` (optional). When the outputs directory is empty or has no `run_*` folders, the first run will be `run_019` instead of `run_001`.

**Run-id logic:** [scripts/6_run_inference.py](scripts/6_run_inference.py)

- `_next_run_id(outputs_dir)` currently returns `run_{max+1:03d}` with `max(numbers, default=0)+1`, so an empty dir yields `run_001`.
- Change: accept an optional base from config (e.g. `inference.run_id_base`). If the outputs dir has no `run_*` subdirs and a base N is set, return `run_{N:03d}`; otherwise keep current behavior (max+1). So first run in `outputs2` becomes `run_019`.

**Evaluation / preflight:** [scripts/5_evaluate.py](scripts/5_evaluate.py), [scripts/preflight_check.py](scripts/preflight_check.py)

- Both already use `config["paths"]["outputs"]` to resolve the outputs dir; no change needed once config points to `outputs2`.

**Hardcoded paths:** [scripts/compare_runs.py](scripts/compare_runs.py) uses `ROOT / "outputs"`; [scripts/run_manifest.py](scripts/run_manifest.py) uses `ROOT / "outputs"`. Update to use `config.get("paths", {}).get("outputs", "outputs")` (or the same resolution as other scripts) so they respect the configured outputs folder.

**Create folder:** Add `outputs2/` (e.g. with a `.gitkeep`) so the directory exists.

---

## 2. Sweeps: hyperparameter testing (robust, not in background)

**Context:** Plans (Update4–6, refined_sweep_rerun) describe a sweep script `scripts/sweep_hparams.py` for Model A epochs and Model B grid; that script does **not** exist in the current repo.

**Options:**

- **A)** Add `scripts/sweep_hparams.py` that: reads config (including `paths.outputs` → e.g. `outputs2/sweeps/<batch_id>/`), runs Model A epoch grid and Model B hyperparameter grid, writes results to CSV/JSON under the outputs path, runs **in the foreground** with no background/daemon mode, and has no fixed timeout so long sweeps can complete.
- **B)** Only document in Update8 (and README) that the next step is to implement or restore hyperparameter sweeps with those properties; implement the script in a follow-up.

**Recommendation:** Document in Update8 that sweeps are the next step; optionally add a minimal `sweep_hparams.py` stub that reads config and writes to `outputs2/sweeps/`, with a note to expand it to full grids (or implement A in a later task).

**Explicit requirements for sweeps (when implemented):**

- **Robust:** Use config for paths; handle missing dirs; avoid overwriting previous batch results (e.g. `--batch-id` or timestamped batch folder).
- **Run as long as needed:** No artificial timeout; run in foreground.
- **Do not run in background:** No daemon/subprocess-background mode; user runs the script in the foreground.

---

## 3. README updates

- State that **outputs** are under `outputs2/` (or `config.paths.outputs`) and that the **first run** in a new outputs folder can start at `run_019` when `inference.run_id_base: 19` is set.
- Mention that **hyperparameter sweeps** (when in place) run in the foreground and write to e.g. `outputs2/sweeps/<batch_id>/`.
- Replace any remaining hardcoded `outputs/` in README with wording like “outputs directory (e.g. `outputs2/`)” or “`config.paths.outputs`”.

---

## 4. Create .cursor/plans/Update8.md

**Content:**

**4.1 Changes since last update (Update7)**  
Summarize what was done after Update7, for example:

- **Outputs:** Pipeline writes to **outputs2**; run numbering continues from **run_019** via `inference.run_id_base`.
- **Inference:** Per-season inference (predictions and figures per test year, e.g. 2023-24, 2024-25); `pred_vs_playoff_rank` legend outside plot; East = dots, West = squares; conference rank as primary rank output; `actual_conference_rank` (Actual Conference Rank) and removal of global_rank from prediction output.
- **Sweeps:** Documented (and optionally stubbed) to repeat hyperparameter testing in the foreground, robust, no timeout.

**4.2 Plan moving forward**

1. **Hyperparameter testing (next)**
  Run or implement sweeps (Model A epochs, Model B grid) as the immediate next step; sweeps run in the foreground and write to the configured outputs path (e.g. `outputs2/sweeps/`).
2. **Primary contributors**
  Fix outputs so **primary contributor information** is displayed and **primary contributors are calculated** correctly (attention/IG path and any fallbacks).
3. **Next plan: playoff performance learning**
  Use **playoff stats** to train a model for how **player performance changes in the playoffs**. Describe possible approaches, e.g.:
  - Separate playoff head or fine-tuning on playoff logs (e.g. playoff-only ListMLE or regression).
  - Regular-season model plus playoff adjustment (residual or linear layer on playoff features).
  - Single model with a “playoff” flag or playoff-specific embeddings.
  - **Learning over time:** Optionally implement **one season at a time** (e.g. train on 2015–16 playoffs, then 2016–17, etc.) for temporal robustness.
4. **Actual playoff contribution**
  **Calculate actual playoff contribution** using playoff stats. State that we will define an effective formula (e.g. box-score-based contribution per game or per possession, or plus-minus style from playoff logs) and implement it; exact metric TBD (e.g. playoff BPM-style, or points + assists + rebounds weighted, or RAPM-lite from playoff lineups).
5. **NBA analyst metrics investigation**
  **Investigate** which metrics NBA analysts use beyond basic box-score (points, rebounds, assists). Examples: RAPM, BPM, VORP, PIPM, LEBRON, EPM, Win Shares, usage rate, TS%, on/off, net rating in playoffs. Note which could be **useful as inputs or analysis** for our model (e.g. as features, as evaluation proxies, or in ANALYSIS.md).

---

## 5. Summary of code/config edits


| Item                                     | Action                                                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| config/defaults.yaml                     | `paths.outputs: "outputs2"`; add `inference.run_id_base: 19`.                                                                              |
| scripts/6_run_inference.py               | Use `run_id_base` when outputs dir has no run_* to get run_019.                                                                            |
| scripts/compare_runs.py, run_manifest.py | Resolve outputs dir from config (not hardcoded `outputs`).                                                                                 |
| outputs2/                                | Create directory (e.g. with .gitkeep).                                                                                                     |
| README                                   | Document outputs2, run_019, sweeps (foreground, robust).                                                                                   |
| .cursor/plans/Update8.md                 | New file: changes since Update7 + forward plan (sweeps, primary contributors, playoff learning, actual playoff contribution, NBA metrics). |
| Sweep script                             | Either add minimal stub under scripts/ and document in Update8, or only document in Update8 for a follow-up.                               |


