# GPU Acceleration

## What uses GPU

| Component | When GPU is used | Config / behavior |
|-----------|------------------|-------------------|
| **Model A (PyTorch)** | Training (script 3), inference (script 6), and explain (script 5b) use CUDA when available. | `model_a.device: null` = auto (cuda when available); `"cuda"` or `"cpu"` to force. Logged as `Model A device: cuda:0` or `Model A device: cpu` in script 3; `Model A device (inference): ...` in script 6; `Model A device (explain): ...` in script 5b. |
| **Model B (XGBoost)** | Training (script 4) uses CUDA when `model_b.xgb.use_gpu` is true and CUDA is available. | `config/defaults.yaml`: `model_b.xgb.use_gpu: true`. Uses `tree_method='hist'`, `device='cuda'`. Logged as `XGBoost device: cuda` or `XGBoost device: cpu`. |
| **RF (Random Forest)** | No GPU support (scikit-learn); always CPU. | — |

## If you don’t see the GPU running

1. **Check logs**  
   At the start of script 3 you should see `Model A device: cuda:0` (or `cpu`). For script 4 you should see `XGBoost device: cuda` or `XGBoost device: cpu`. If both say `cpu`, the process is not using the GPU.

2. **PyTorch**  
   Default `pip install torch` is often **CPU-only**. For CUDA you need a CUDA build, e.g.:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   Then in Python: `import torch; print(torch.cuda.is_available())` should be `True` when a GPU and drivers are present.

3. **NVIDIA drivers and WSL2**  
   On Windows with WSL2 you need [NVIDIA drivers for WSL](https://developer.nvidia.com/cuda/wsl) and a CUDA-capable PyTorch build. Running from Windows (PowerShell) with a GPU uses the same drivers; WSL2 shares the GPU with the host.

4. **XGBoost**  
   With `use_gpu: true`, XGBoost uses CUDA only if it’s available (we use `torch.cuda.is_available()` to decide). If PyTorch is CPU-only, XGBoost will also fall back to CPU. Set `model_b.xgb.use_gpu: false` in config to force XGBoost to CPU.

5. **Docker / sweeps**  
   The GPU sweep plan (`.cursor/plans/gpu_sweep_docker_wsl_19275471.plan.md`) describes running in a CUDA container with `--gpus=all`. For local runs, the above applies: install CUDA PyTorch and ensure drivers are available.

## Notion (section 7 of GPU plan)

The plan’s section 7 is a **Notion update**: add a short note on the relevant Notion page (e.g. “Update8: outputs2, run_019, sweeps”) with “GPU sweeps in Docker (WSL2)” and a link to the plan and run instructions. The Notion MCP server was not available in this workspace; you can paste the link and summary into that page manually.
