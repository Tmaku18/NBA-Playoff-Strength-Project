"""Train Model A (DeepSet + ListMLE). Checkpointing, walk-forward train/val seasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from src.models.deep_set_rank import DeepSetRank
from src.models.listmle_loss import listmle_loss
from src.utils.repro import set_seeds


def get_dummy_batch(
    batch_size: int = 4,
    num_teams_per_list: int = 10,
    num_players: int = 15,
    stat_dim: int = 7,
    num_embeddings: int = 500,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Dummy batch for ListMLE: B lists, each with K teams; each team has P players."""
    B, K, P, S = batch_size, num_teams_per_list, num_players, stat_dim
    embs = torch.randint(0, num_embeddings, (B, K, P), device=device)
    stats = torch.randn(B, K, P, S, device=device) * 0.1
    minutes = torch.rand(B, K, P, device=device)
    mask = torch.zeros(B, K, P, dtype=torch.bool, device=device)
    mask[:, :, 10:] = True  # last 5 are padding
    rel = torch.rand(B, K, device=device)  # fake relevance

    return {
        "embedding_indices": embs,
        "player_stats": stats,
        "minutes": minutes,
        "key_padding_mask": mask,
        "rel": rel,
    }


def train_epoch(
    model: nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in batches:
        B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
        embs = batch["embedding_indices"].to(device).reshape(B * K, P)
        stats = batch["player_stats"].to(device).reshape(B * K, P, S)
        minutes = batch["minutes"].to(device).reshape(B * K, P)
        mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
        rel = batch["rel"].to(device)

        score, _, _ = model(embs, stats, minutes, mask)
        score = score.reshape(B, K)
        loss = listmle_loss(score, rel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n if n else 0.0


def eval_epoch(
    model: nn.Module,
    batches: Iterable[dict],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = (
                batch["embedding_indices"].shape[0],
                batch["embedding_indices"].shape[1],
                batch["embedding_indices"].shape[2],
                batch["player_stats"].shape[-1],
            )
            embs = batch["embedding_indices"].to(device).reshape(B * K, P)
            stats = batch["player_stats"].to(device).reshape(B * K, P, S)
            minutes = batch["minutes"].to(device).reshape(B * K, P)
            mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
            rel = batch["rel"].to(device)

            score, _, _ = model(embs, stats, minutes, mask)
            score = score.reshape(B, K)
            loss = listmle_loss(score, rel)
            total += loss.item()
            n += 1
    return total / n if n else 0.0


def train_model_a(
    config: dict,
    output_dir: str | Path,
    train_batches: list[dict] | None = None,
    val_batches: list[dict] | None = None,
    device: str | torch.device | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config.get("repro", {}).get("seed", 42))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    emb_dim = ma.get("embedding_dim", 32)
    stat_dim = 7
    enc_h = ma.get("encoder_hidden", [128, 64])
    heads = ma.get("attention_heads", 4)
    drop = ma.get("dropout", 0.2)

    model = DeepSetRank(num_emb, emb_dim, stat_dim, enc_h, heads, drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if train_batches is None:
        train_batches = [get_dummy_batch(4, 10, 15, stat_dim, num_emb, device) for _ in range(5)]
    epochs = int(ma.get("epochs", 3))
    patience = int(ma.get("early_stopping_patience", 0) or 0)
    min_delta = float(ma.get("early_stopping_min_delta", 0.0) or 0.0)

    best_val = None
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_batches, optimizer, device)
        msg = f"epoch {epoch+1} loss={loss:.4f}"
        if val_batches:
            val_loss = eval_epoch(model, val_batches, device)
            msg += f" val_loss={val_loss:.4f}"
            if best_val is None or val_loss < (best_val - min_delta):
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
        print(msg)

        if patience and bad_epochs >= patience:
            print(f"early stopping after {epoch+1} epochs (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    path = output_dir / "best_deep_set.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, path)
    return path
