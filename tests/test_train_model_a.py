"""Tests for src.training.train_model_a."""
from __future__ import annotations

import torch
import pytest

from src.training.train_model_a import get_dummy_batch, _split_train_val


def test_get_dummy_batch_shapes():
    batch = get_dummy_batch(batch_size=4, num_teams_per_list=10, num_players=15, stat_dim=7)
    assert batch["embedding_indices"].shape == (4, 10, 15)
    assert batch["player_stats"].shape == (4, 10, 15, 7)
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
