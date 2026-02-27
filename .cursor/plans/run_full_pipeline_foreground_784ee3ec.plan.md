---
name: Run full pipeline foreground
overview: Run the full pipeline by executing each of the 8 scripts as a separate, foreground command in sequence (no single chained command, no background). Requires project root and PYTHONPATH set once, then run each script and wait for it to finish before starting the next. Triton enabled; run in WSL.
todos: []
isProject: false
---

# Run Full Pipeline (Foreground)

Run the pipeline as **8 separate foreground commands**: set PYTHONPATH at project root, then run scripts 1–8 in order (download raw → build DB → train model A → train model B → stacking → evaluate → explain → inference). Do not chain commands or run in background; wait for each step to finish before starting the next.

**Full steps and commands:** [docs/RUN_FULL_PIPELINE_FOREGROUND.md](docs/RUN_FULL_PIPELINE_FOREGROUND.md)
