# Model A training and attention (review notes)

Aligned with PyTorch `nn.MultiheadAttention` and training best practices.

## SetAttention ([src/models/set_attention.py](../src/models/set_attention.py))

- **Query:** Built from masked mean of `x` over valid positions (where `key_padding_mask` is False). Ensures query is in the same space as keys/values and avoids attending to padding.
- **key_padding_mask:** True = ignore (PyTorch MHA uses -inf for masked positions in softmax). Mask is applied to keys so padded positions get zero attention weight.
- **Minutes bias:** After MHA, we blend raw attention `w` with minutes-normalized weights so that high-minute players get a floor; uses `minutes_bias_weight` and `minutes_sum_min`. Only applied when `mins_sum > minutes_sum_min` to avoid division by zero.
- **Fallback (harden):** If raw attention is (nearly) zero on valid positions (sum < 1e-6), we replace with fallback weights so gradients can flow. Configurable via `model_a.attention_fallback_strategy`:
  - **`minutes`** (default): Use normalized minutes on valid positions. *Pros:* Domain-aligned (playing time ≈ importance); smooth gradients. *Cons:* Can reinforce imbalance if model already under-weights low-minute players.
  - **`uniform`**: Use 1/n_valid on valid positions. *Pros:* Neutral; equal gradient to all valid players; robust. *Cons:* Ignores playing time; may slow convergence for impact-based ranking.

## DeepSetRank ([src/models/deep_set_rank.py](../src/models/deep_set_rank.py))

- **Pipeline:** Embedding → concat stats → encoder → SetAttention → linear scorer. Encoder output dim is used as `embed_dim` for MHA.
- **Gradient flow:** Pooled vector Z is used for the score; attention weights are returned for explainability. Fallback in SetAttention ensures Z varies across teams when attention would otherwise be zero.

## Training ([src/training/train_model_a.py](../src/training/train_model_a.py))

- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after backward to avoid exploding gradients.
- **Numerical stability:** `torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)` before ListMLE; ListMLE uses logsumexp.
- **stat_dim_override:** Model is built with `stat_dim` from the first batch’s `player_stats.shape[-1]` when batches are provided, so config and data stay in sync.
- **Attention debug:** When `model_a.attention_debug` is true (or `--attention-debug` in script 3), we log attention stats on the first batch before training and when NOT_LEARNING_PATIENCE triggers.

## Config (run_021 spec)

- `model_a.stat_dim: 17`, `epochs: 28`, `early_stopping_patience: 0`, `attention_heads: 4`, `minutes_bias_weight: 0.3`, `minutes_sum_min: 1e-6`. See [config/defaults.yaml](../config/defaults.yaml) and [README.md](../README.md).
