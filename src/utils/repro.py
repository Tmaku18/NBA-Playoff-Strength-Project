"""Reproducibility: set seeds for random, numpy, torch; optional full deterministic mode."""
import os
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode() -> None:
    """Make torch runs bit-reproducible: deterministic cuDNN, no TF32, deterministic cuBLAS.

    Must be called before CUDA context creation for CUBLAS_WORKSPACE_CONFIG to take effect.
    warn_only=True so ops without deterministic implementations fall back instead of raising.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not _HAS_TORCH:
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def is_deterministic(config: dict | None) -> bool:
    """Read repro.deterministic from config (default True: reproducible runs)."""
    return bool(((config or {}).get("repro") or {}).get("deterministic", True))
