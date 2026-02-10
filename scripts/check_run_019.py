"""One-off script: compare run_019 vs run_014 predictions (EOS rank source, playoff data).

What this does:
- Loads predictions from run_019 and run_014, prints eos_rank_source and sample team ranks.
- Used for debugging EOS/playoff rank changes. Not part of main pipeline."""
import json

p19 = json.load(open('outputs/run_019/predictions_2024-25.json'))
p14 = json.load(open('outputs/run_014/predictions.json'))

print("=== EOS Rank Source ===")
print(f"run_019: {p19.get('eos_rank_source')}")
print(f"run_014: {p14.get('eos_rank_source', 'not found')}")

print("\n=== Playoff Data Availability ===")
teams_with_playoff = [t for t in p19['teams'] if t['analysis'].get('post_playoff_rank') is not None]
print(f"Teams with post_playoff_rank in run_019: {len(teams_with_playoff)}")

print("\n=== Sample Teams (run_019 vs run_014) ===")
for team_name in ["Boston Celtics", "Milwaukee Bucks", "Cleveland Cavaliers"]:
    t19 = [t for t in p19['teams'] if t['team_name'] == team_name][0]
    t14 = [t for t in p14['teams'] if t['team_name'] == team_name][0]
    print(f"\n{team_name}:")
    print(f"  run_019: EOS_global={t19['analysis']['EOS_global_rank']}, post_playoff_rank={t19['analysis'].get('post_playoff_rank')}, EOS_playoff_standings={t19['analysis'].get('EOS_playoff_standings')}")
    print(f"  run_014: EOS_global={t14['analysis']['EOS_global_rank']}, post_playoff_rank={t14['analysis'].get('post_playoff_rank')}")
    print(f"  Classification change: '{t14['analysis']['classification']}' -> '{t19['analysis']['classification']}'")
