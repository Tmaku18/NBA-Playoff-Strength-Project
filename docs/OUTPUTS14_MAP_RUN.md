# outputs14: Future MAP run

**Purpose:** Dedicated output for the **MAP (Maximum A Posteriori) branch** model. When the MAP branch is ready, run the pipeline (or MAP-specific entrypoint) with `config/outputs14_map_run.yaml` so results go to **outputs14/**. Run **per-game** evaluation and compare to **baseline (outputs6)** and **best (outputs8)**.

---

## Setup

- **Config:** `config/outputs14_map_run.yaml`
- **Paths:** `paths.outputs: "outputs14_map_run"`
- **Baseline reference:** outputs6_baseline. **Best reference:** outputs8_spearman_surrogate.

---

## How to run (when MAP branch is ready)

From project root:

```powershell
python -m scripts.run_pipeline_from_model_a --config config/outputs14_map_run.yaml --outputs outputs14_map_run/map_run
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs14_map_run.yaml --outputs outputs14_map_run/map_run
```

If a MAP-specific script is added (e.g. `run_pipeline_map`), use that with the same config and `--outputs outputs14/map_run`.

---

## After the run

- Run **per-game** evaluation as needed.
- Compare metrics to **outputs6_baseline** (baseline) and **outputs8_spearman_surrogate** (best).

See [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md) for the full lineup.
