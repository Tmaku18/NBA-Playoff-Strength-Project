# Memoization for speed

The pipeline and analysis scripts use **memoization** wherever it helps: caching expensive, deterministic results keyed by config + data identity so repeated runs (e.g. sweeps, re-runs, plot scripts) skip recomputation.

**Recommendation:** Leave `paths.batch_cache` and `paths.feature_cache` set in config (defaults enable both). Set to `null` only if you need to force a full rebuild (e.g. after changing DB or feature logic and wanting to invalidate caches).

---

## Where memoization is used

| Component | What is cached | Config | Key |
|-----------|----------------|--------|-----|
| **Script 3 (Model A)** | Built lists and batches (OOF + final) | `paths.batch_cache` (default `data/processed/batch_cache`) | Config (listmle_target, rolling_windows, train_seasons, …) + DB path + DB mtime/size |
| **Script 4 (Models B & C)** | `build_team_context_as_of_dates` output | `paths.feature_cache` (default `data/processed/feature_cache`) | Config (model_b include/exclude, elo, massey, …) + DB + team_dates hash. Shared: `src.features.feature_cache`. |
| **Script 5b (Explain)** | Same team-context features | `paths.feature_cache` | Same key as script 4; reuse across 4 → 5b when config/DB/team_dates match. |
| **Script 4c (Clone classifier)** | Same team-context features | `paths.feature_cache` | Same key as script 4; reuse when config/DB/team_dates match. |
| **Plot scripts** (`plot_feature_rank_vs_playoff_outcome`, `plot_feature_rank_vs_playoff_outcome_pdf`) | Same team-context features | `paths.feature_cache` | Same key; reuse when config/DB/team_dates match. |
| **Inference (script 6)** | Team-context features for all target specs | `paths.feature_cache` | Same config+DB+team_dates logic; file prefix `inf_` so inference cache does not collide with training cache. |
| **DB loader** | Loaded games, tgl, teams, pgl (in-process) | — | `db_path` + mtime; same process does not re-read DB. |

---

## Config

In `config/defaults.yaml`:

```yaml
paths:
  batch_cache: data/processed/batch_cache   # script 3; set null to disable
  feature_cache: data/processed/feature_cache  # script 4, 5b, 4c, plot, inference; set null to disable
```

Sweep combo configs inherit these from the config you pass to the sweep; only `paths.outputs` is overwritten per combo. So sweeps get batch and feature cache for speed by default.

---

## Shared feature cache

Script 4, 5b, 4c, and the plot scripts use the same key logic and cache dir so that:

- After script 4 runs, 5b or 4c with the same config/DB can hit the same cache.
- After script 4 or a plot run, re-running the plot script with the same config/DB hits the cache.

Implementation: `src.features.feature_cache` (`get_feature_cache_dir`, `compute_feature_cache_key`, `load_feature_cache`, `save_feature_cache`).

Inference uses the same config bits and DB/team_dates hash but writes to `inf_<key>.parquet` so training and inference caches do not overwrite each other.

---

## Disabling

- Set `paths.batch_cache: null` to disable script 3 batch cache (every run rebuilds lists and batches).
- Set `paths.feature_cache: null` to disable feature cache everywhere (script 4, 5b, 4c, plot, inference).

Deleting the cache directories also forces full rebuilds on next run.
