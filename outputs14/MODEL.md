# outputs14 — Model and role (MAP run)

**Role:** **Future MAP (Maximum A Posteriori) branch** runs. Reserved for the MAP branch model; per-game evaluation to be run and compared to **baseline (outputs6)** and **best (outputs8)**.

**Model:** MAP branch model (to be run when branch is ready). Config: `config/outputs14_map_run.yaml`. Results will go under `outputs14/` (e.g. `outputs14/map_run/` or sweep subdirs if applicable).

**Difference from outputs13:** outputs13 = RMSE surrogate sweep. outputs14 = **MAP branch** (different model/estimation); still to be tested with per-game evaluation.

**See also:** [docs/OUTPUTS14_MAP_RUN.md](../docs/OUTPUTS14_MAP_RUN.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
