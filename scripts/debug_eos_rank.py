"""Debug EOS final rank: print playoff wins and order for 2024-25."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from src.data.db_loader import load_playoff_data, load_training_data
from src.evaluation.playoffs import (
    get_playoff_wins,
    get_reg_season_win_pct,
    compute_eos_final_rank,
    _filtered_playoff_tgl,
)

with open(ROOT / "config" / "defaults.yaml") as f:
    cfg = yaml.safe_load(f)
db = Path(cfg["paths"]["db"])
games, tgl, teams, pgl = load_training_data(db)
pg, ptgl, _ = load_playoff_data(db)

season = "2024-25"
seasons_cfg = cfg.get("seasons", {})
if season in seasons_cfg:
    season_start = seasons_cfg[season]["start"]
    season_end = seasons_cfg[season]["end"]
else:
    season_start = season_end = None

pw = get_playoff_wins(pg, ptgl, season, season_start=season_start, season_end=season_end)
reg_wp = get_reg_season_win_pct(games, tgl, season, season_start=season_start, season_end=season_end)

# Team id to name
tid_to_name = dict(zip(teams["team_id"].astype(int), teams["abbreviation"])) if not teams.empty else {}

print("=== Playoff wins 2024-25 (all teams with playoff data) ===")
sorted_pw = sorted(pw.items(), key=lambda x: (-x[1], -(reg_wp.get(x[0], 0) or 0)))
for i, (tid, wins) in enumerate(sorted_pw[:20], 1):
    name = tid_to_name.get(tid, str(tid))
    wp = reg_wp.get(tid, 0) or 0
    print(f"  {i:2}. {name} (id={tid}): {wins} playoff wins, reg_win_pct={wp:.3f}")

boston_tid = teams.loc[teams["abbreviation"] == "BOS", "team_id"].iloc[0]
print(f"\nBoston (BOS) team_id={boston_tid}, playoff_wins={pw.get(int(boston_tid), 'N/A')}")

print("\n=== EOS final rank (first 20) ===")
all_team_ids = sorted(teams["team_id"].astype(int).unique().tolist()) if not teams.empty else None
eos = compute_eos_final_rank(pg, ptgl, games, tgl, season, all_team_ids=all_team_ids, season_start=season_start, season_end=season_end)
sorted_eos = sorted(eos.items(), key=lambda x: x[1])
for tid, rank in sorted_eos[:20]:
    name = tid_to_name.get(tid, str(tid))
    wins = pw.get(tid, 0)
    print(f"  rank {rank:2}: {name} (id={tid}), playoff_wins={wins}")

print(f"\nBoston EOS_global_rank in result: {eos.get(int(boston_tid), 'N/A')}")
