"""Tests for src.training.train_model_a."""
from __future__ import annotations

import torch
import pytest

from src.training.train_model_a import (
    get_dummy_batch,
    _split_train_val,
    train_epoch,
    _build_model,
)


def test_get_dummy_batch_shapes():
    batch = get_dummy_batch(batch_size=4, num_teams_per_list=10, num_players=15, stat_dim=17)
    assert batch["embedding_indices"].shape == (4, 10, 15)
    assert batch["player_stats"].shape == (4, 10, 15, 17)
    assert batch["minutes"].shape == (4, 10, 15)
    assert batch["key_padding_mask"].shape == (4, 10, 15)
    assert batch["rel"].shape == (4, 10)
    assert batch["key_padding_mask"][:, :, 10:].all()
    assert not batch["key_padding_mask"][:, :, :10].any()


def test_split_train_val_no_frac():
    batches = [{"x": i} for i in range(3)]
    train, val = _split_train_val(batches, 0.0)
    assert len(train) == 3 and len(val) == 0


def test_split_train_val_few_batches():
    batches = [{"x": i} for i in range(4)]
    train, val = _split_train_val(batches, 0.2)
    assert len(val) == 0  # n < 5 => no val split


def test_split_train_val_splits():
    batches = [{"x": i} for i in range(10)]
    train, val = _split_train_val(batches, 0.2)
    assert len(train) == 8 and len(val) == 2
    assert train[-1]["x"] == 7 and val[0]["x"] == 8


def test_model_a_learning_with_fallback():
    """With attention fallback, Model A loss should be finite and scores non-constant."""
    config = {
        "model_a": {
            "stat_dim": 21,
            "num_embeddings": 500,
            "embedding_dim": 32,
            "encoder_hidden": [128, 64],
            "attention_heads": 4,
            "dropout": 0.2,
            "minutes_bias_weight": 0.3,
            "minutes_sum_min": 1e-6,
            "attention_fallback_strategy": "minutes",
        },
        "repro": {"seed": 42},
    }
    device = torch.device("cpu")
    model = _build_model(config, device, stat_dim_override=21)
    batches = [
        get_dummy_batch(batch_size=2, num_teams_per_list=8, num_players=15, stat_dim=21),
        get_dummy_batch(batch_size=2, num_teams_per_list=8, num_players=15, stat_dim=21),
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(4):
        loss = train_epoch(model, batches, optimizer, device)
        losses.append(loss)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # Scores should differ across teams (no collapse)
    model.eval()
    with torch.no_grad():
        b = batches[0]
        B, K, P, S = b["embedding_indices"].shape[0], b["embedding_indices"].shape[1], b["embedding_indices"].shape[2], b["player_stats"].shape[-1]
        embs = b["embedding_indices"].to(device).reshape(B * K, P)
        stats = b["player_stats"].to(device).reshape(B * K, P, S)
        minutes = b["minutes"].to(device).reshape(B * K, P)
        mask = b["key_padding_mask"].to(device).reshape(B * K, P)
        score, _, _ = model(embs, stats, minutes, mask)
        score = score.reshape(B, K)
    assert score.shape == (B, K)
    # At least one row should have non-constant scores (variance > 0)
    row_vars = score.var(dim=1)
    assert (row_vars > 1e-8).any(), "Model A scores should vary across teams (attention fallback working)"
