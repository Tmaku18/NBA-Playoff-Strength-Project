"""Debug Model A training: inspect data, loss, gradients."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from src.data.db_loader import load_training_data
from src.training.data_model_a import build_batches_from_lists
from src.training.build_lists import build_lists
from src.models.deep_set_rank import DeepSetRank
from src.models.listmle_loss import listmle_loss

with open("config/defaults.yaml") as f:
    cfg = yaml.safe_load(f)

db = Path(cfg["paths"]["db"])
games, tgl, teams, pgl = load_training_data(db)
lists = build_lists(tgl, games, teams)
print(f"Total lists: {len(lists)}")

if lists:
    # Find a list from mid-November 2015 (a few weeks into the season with game data)
    mid_season_idx = 0
    for i, lst in enumerate(lists):
        date = lst["as_of_date"]
        # Find a date ~3 weeks into 2015-16 season
        if date >= "2015-11-15":
            mid_season_idx = i
            break
    
    sample = lists[mid_season_idx]
    print(f"\nSample list (mid-Nov 2015-16 season): {len(sample['team_ids'])} teams, date={sample['as_of_date']}")
    print(f"Win rates: {sample['win_rates'][:5]}")
    
    # Test a mid-season list (should have both roster data and prior season baseline)
    test_lists = [lists[mid_season_idx]]
    print(f"\nTesting lists: {[l['as_of_date'] for l in test_lists]}")
    batches, metas = build_batches_from_lists(test_lists, games, tgl, teams, pgl, cfg)
    if batches:
        b = batches[0]
        print(f"\nBatch shapes:")
        print(f"  embedding_indices: {b['embedding_indices'].shape}")
        print(f"  player_stats: {b['player_stats'].shape}")
        print(f"  minutes: {b['minutes'].shape}")
        print(f"  rel: {b['rel'].shape}")
        
        print(f"\nRel (win_rates): {b['rel']}")
        print(f"\nStats sample (first team, first 3 players, first 3 features):")
        print(b["player_stats"][0, 0, :3, :3])
        print(f"\nMinutes sample (first team, first 5 players):")
        print(b["minutes"][0, 0, :5])
        print(f"\nMask sample (first team, first 5 players):")
        print(b["key_padding_mask"][0, 0, :5])
        
        # Test forward pass
        ma = cfg.get("model_a", {})
        stat_dim = int(b["player_stats"].shape[-1])
        attn_cfg = ma.get("attention", {})
        model = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            stat_dim,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
            minutes_bias_weight=float(ma.get("minutes_bias_weight", 0.3)),
            minutes_sum_min=float(ma.get("minutes_sum_min", 1e-6)),
            fallback_strategy=str(ma.get("attention_fallback_strategy", "minutes")),
            attention_temperature=float(attn_cfg.get("temperature", 1.0)),
            attention_input_dropout=float(attn_cfg.get("input_dropout", 0.0)),
            attention_use_pre_norm=bool(attn_cfg.get("use_pre_norm", True)),
            attention_use_residual=bool(attn_cfg.get("use_residual", True)),
        )
        
        B, K, P, S = b["embedding_indices"].shape[0], b["embedding_indices"].shape[1], b["embedding_indices"].shape[2], b["player_stats"].shape[-1]
        embs = b["embedding_indices"].reshape(B * K, P)
        stats = b["player_stats"].reshape(B * K, P, S)
        minutes = b["minutes"].reshape(B * K, P)
        mask = b["key_padding_mask"].reshape(B * K, P)
        rel = b["rel"]
        
        print(f"\nForward pass:")
        score, Z, attn = model(embs, stats, minutes, mask)
        score = score.reshape(B, K)
        print(f"  Scores: {score}")
        print(f"  Attention weights (first team, first 5 players): {attn[:5]}")
        print(f"  Z (pooled features, first team): {Z[0, :5]}")
        
        # Test loss and gradient
        loss = listmle_loss(score, rel)
        print(f"\nLoss: {loss.item():.4f}")
        
        model.zero_grad()
        loss.backward()
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        print(f"\nGradient norms (top 5):")
        for name, norm in sorted(grad_norms.items(), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {norm:.6f}")

print("\nFirst 5 lists:")
for i, lst in enumerate(lists[:5]):
    wr = lst["win_rates"]
    date = lst["as_of_date"]
    conf = lst.get("conference", "?")
    n_teams = len(lst["team_ids"])
    wr_min, wr_max = min(wr), max(wr)
    print(f"  {i}: date={date}, conf={conf}, teams={n_teams}, win_rate=[{wr_min:.3f}, {wr_max:.3f}]")

# Find first list from 2015-16 season (Oct 2015+)
print(f"\nFirst 5 lists from 2015-16 season (Oct 2015+):")
count = 0
for i, lst in enumerate(lists):
    if lst["as_of_date"] >= "2015-10-01" and count < 5:
        wr = lst["win_rates"]
        date = lst["as_of_date"]
        conf = lst.get("conference", "?")
        n_teams = len(lst["team_ids"])
        wr_min, wr_max = min(wr), max(wr)
        print(f"  {i}: date={date}, conf={conf}, teams={n_teams}, win_rate=[{wr_min:.3f}, {wr_max:.3f}]")
        count += 1

print("\nLast 5 lists:")
for i, lst in enumerate(lists[-5:]):
    wr = lst["win_rates"]
    date = lst["as_of_date"]
    conf = lst.get("conference", "?")
    n_teams = len(lst["team_ids"])
    wr_min, wr_max = min(wr), max(wr)
    idx = len(lists) - 5 + i
    print(f"  {idx}: date={date}, conf={conf}, teams={n_teams}, win_rate=[{wr_min:.3f}, {wr_max:.3f}]")

# Test prior season stats computation
print("\n--- Testing Prior Season Stats ---")
from src.features.rolling import get_prior_season_stats, PLAYER_STAT_COLS_L10_L30
prior_2015 = get_prior_season_stats(pgl, "2015-10-01", stat_cols=PLAYER_STAT_COLS_L10_L30, lookback_days=365)
print(f"Prior season stats for 2015-16 (from 2014-15): {len(prior_2015)} players")
if not prior_2015.empty:
    print(f"Sample stats (first 3 players):")
    print(prior_2015.head(3))
