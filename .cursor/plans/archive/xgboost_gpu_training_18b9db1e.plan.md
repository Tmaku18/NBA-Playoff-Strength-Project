---
name: XGBoost GPU training
overview: Enable XGBoost to use GPU during Model B training by adding config-driven tree_method and device parameters and passing them through the XGBRegressor build path.
todos: []
isProject: false
---

# XGBoost GPU training

## Goal

Use the existing GPU-enabled XGBoost build by setting `tree_method` and/or `device` when training the XGB regressor (Model B), so that training runs on CUDA when configured.

## Current behavior

- [src/models/xgb_model.py](src/models/xgb_model.py) builds an `xgb.XGBRegressor` with only: `n_estimators`, `max_depth`, `learning_rate`, `random_state`, `subsample`, `colsample_bytree`. No `tree_method` or `device` is set, so training uses CPU (default).
- [config/defaults.yaml](config/defaults.yaml) `model_b.xgb` has `n_estimators`, `max_depth`, `learning_rate`, `early_stopping_rounds`; no GPU-related keys.
- [src/training/train_model_b.py](src/training/train_model_b.py) and [scripts/4_train_model_b.py](scripts/4_train_model_b.py) call `build_xgb(xgb_cfg)` with config from `config["model_b"]["xgb"]`.
- Inference loads the saved model with joblib and calls `.predict()`; the same model works regardless of whether it was trained on GPU or CPU. No inference change.

## Design: config-driven GPU

Add optional `tree_method` and `device` under `model_b.xgb`. When set, pass them into `XGBRegressor` in `build_xgb`. This keeps CPU-only environments working (omit or set `device: cpu` / `tree_method: hist`) and lets GPU environments opt in with `tree_method: gpu_hist` and `device: cuda` (or, for XGBoost 2.x, `tree_method: hist` with `device: cuda`).

## Implementation plan

### 1. Config — [config/defaults.yaml](config/defaults.yaml)

- Under `model_b.xgb`, add optional keys:
  - `tree_method: null` (or omit). When set to `"gpu_hist"` (or `"hist"` in 2.x with GPU), pass to XGBRegressor.
  - `device: null` (or omit). When set to `"cuda"` (or `"cuda:0"`), pass to XGBRegressor.
- Add a short comment that setting `tree_method: gpu_hist` and `device: cuda` enables GPU training when XGBoost is built with CUDA; leave unset for CPU.

Example addition:

```yaml
model_b:
  xgb:
    n_estimators: 250
    max_depth: 4
    learning_rate: 0.08
    early_stopping_rounds: 20
    # Optional: enable GPU when XGBoost built with CUDA (e.g. tree_method: gpu_hist, device: cuda)
    tree_method: null
    device: null
```

Defaulting to GPU when available (e.g. auto-detect) is possible but would change behavior for users who prefer CPU; config-driven is safer and explicit.

### 2. XGBRegressor build — [src/models/xgb_model.py](src/models/xgb_model.py)

- In `build_xgb(config)`:
  - If `config.get("tree_method")` is set (non-null, non-empty string), add `tree_method=config["tree_method"]` to the kwargs passed to `xgb.XGBRegressor(**kwargs)`.
  - If `config.get("device")` is set, add `device=config["device"]` to kwargs.
- Do not add defaults for `tree_method` or `device` in code so that existing configs and sweeps (which do not set these) remain CPU-only unless the user adds them.

Example logic (conceptually):

```python
kwargs = { ... existing keys ... }
if config.get("tree_method"):
    kwargs["tree_method"] = config["tree_method"]
if config.get("device"):
    kwargs["device"] = config["device"]
return xgb.XGBRegressor(**kwargs)
```

### 3. Sweep — [scripts/sweep_hparams.py](scripts/sweep_hparams.py)

- No change required. Sweep overlays config with trial/combo params and does not set `tree_method` or `device`. So sweep continues to run XGB on CPU unless the base config (e.g. defaults.yaml) sets GPU. If you later want sweep to use GPU, set `tree_method` and `device` in the base config or in the sweep’s config override; no code change in the sweep script.

### 4. Inference and evaluation

- No change. The saved `xgb_model.joblib` is loaded and used for `.predict()`; training device does not affect inference.

### 5. Optional: graceful fallback

- If at runtime CUDA is unavailable (e.g. driver issue, wrong env), XGBoost may raise when fitting with `device="cuda"`. Optional improvement: in `build_xgb` or in the training path, catch that failure and retry with `tree_method="hist"` and `device="cpu"` (or omit device), and log a warning. This is optional and can be a follow-up.

## Enabling GPU in your environment

After implementation, set in config (e.g. [config/defaults.yaml](config/defaults.yaml)):

```yaml
model_b:
  xgb:
    tree_method: gpu_hist   # or "hist" for XGBoost 2.x with device=cuda
    device: cuda
```

Use `gpu_hist` for older XGBoost; for 2.x, `tree_method: hist` with `device: cuda` is also valid. Leave `tree_method` and `device` unset (or `device: cpu`) for CPU-only.

## Files to touch


| File                                               | Change                                                                                              |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [config/defaults.yaml](config/defaults.yaml)       | Add optional `model_b.xgb.tree_method` and `model_b.xgb.device` (e.g. `null`) and a brief comment.  |
| [src/models/xgb_model.py](src/models/xgb_model.py) | In `build_xgb`, pass through `tree_method` and `device` from config to `XGBRegressor` when present. |


No new files; no changes to sweep, inference, or evaluation logic.