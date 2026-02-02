"""Verify Model A training: loss decreases and attention debug shows non-zero attn_sum_mean and grad norms."""
from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.training.train_model_a import get_dummy_batch, train_model_a


def main() -> None:
    config_path = ROOT / "config" / "defaults.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ma = config.setdefault("model_a", {})
    ma["attention_debug"] = True
    ma["epochs"] = 6
    stat_dim = int(ma.get("stat_dim", 14))
    num_emb = ma.get("num_embeddings", 500)
    device = "cpu"
    batches = [get_dummy_batch(4, 10, 15, stat_dim, num_emb, device) for _ in range(8)]
    out_dir = ROOT / "outputs3" / "verify_model_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = StringIO()
    old_stdout = sys.stdout
    sys.stdout = cap
    try:
        train_model_a(config, out_dir, device=device, batches=batches)
    finally:
        sys.stdout = old_stdout
    log = cap.getvalue()

    losses = [float(m.group(1)) for m in re.finditer(r"epoch \d+ loss=([\d.]+)", log)]
    attn_sum_means = [float(m.group(1)) for m in re.finditer(r"attn_sum_mean=([\d.]+)", log)]
    attn_grad_norms = [float(m.group(1)) for m in re.finditer(r"attn_grad_norm=([\d.]+)", log)]

    assert len(losses) >= 2, f"Expected at least 2 epoch losses, got {len(losses)}. Log:\n{log}"
    loss_decreased = losses[-1] < losses[0]
    assert loss_decreased, (
        f"Expected loss to decrease: first={losses[0]:.4f} last={losses[-1]:.4f}. "
        "Model A attention fix may not be applied or learning rate/epochs too low."
    )
    assert len(attn_sum_means) >= 1, f"Expected at least one attn_sum_mean in log. Log:\n{log}"
    assert len(attn_grad_norms) >= 1, f"Expected at least one attn_grad_norm in log. Log:\n{log}"
    assert attn_sum_means[-1] > 0, (
        f"Expected non-zero attn_sum_mean (got {attn_sum_means[-1]}). "
        "Attention may still be collapsed."
    )
    assert attn_grad_norms[-1] > 0, (
        f"Expected non-zero attn_grad_norm (got {attn_grad_norms[-1]}). "
        "Gradients may not be flowing through attention."
    )
    print("Verify Model A training: OK")
    print(f"  Loss: first={losses[0]:.4f} last={losses[-1]:.4f} (decreased={loss_decreased})")
    print(f"  attn_sum_mean (last)={attn_sum_means[-1]:.4f}")
    print(f"  attn_grad_norm (last)={attn_grad_norms[-1]:.4f}")


if __name__ == "__main__":
    main()
