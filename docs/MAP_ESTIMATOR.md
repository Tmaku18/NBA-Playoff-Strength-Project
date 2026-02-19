# MAP estimator (Model A)

This project can optionally train **Model A (DeepSetRank)** with a **MAP (Maximum A Posteriori)** objective.

## What “MAP” means here

MAP maximizes the posterior:

\[
\arg\max_\theta \; p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta)
\]

If we use a **zero-mean Gaussian prior** on weights, \(-\log p(\theta)\) becomes an **L2 penalty** on parameters. In practice, this is implemented as **weight decay** in the optimizer.

## How to enable

In [`config/defaults.yaml`](../config/defaults.yaml) (or any overlay), set:

- `model_a.weight_decay: 0.0` (default) → **MLE** (no prior)
- `model_a.weight_decay: > 0` → **MAP** (Gaussian prior / L2 regularization)

Suggested starting values to try:

- `1e-5`, `1e-4`, `1e-3`

## Where it’s applied

Model A’s Adam optimizer reads `model_a.weight_decay` in:

- [`src/training/train_model_a.py`](../src/training/train_model_a.py)

