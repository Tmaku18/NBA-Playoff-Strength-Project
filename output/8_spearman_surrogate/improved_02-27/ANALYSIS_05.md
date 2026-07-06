# Analysis 05 — Partial re-inference after DB fix (Jul 6)

**Run folder:** `output/8_spearman_surrogate/improved_02-27/outputs/run_026_07-03_0553`  
**Eval timestamp:** 2026-07-06 01:33  
**Config:** `config/8_spearman_improved.yaml` (`db: nba_build.duckdb` fixed; `outputs` still `improved_02-27/outputs`)

---

## Executive summary

This eval is **not a valid full run**. Inference at 01:31 only rewrote **2 of 4** prediction files; eval then pooled **3 mismatched checkpoints** (mix of Jul 3 and Jul 6 artifacts). Pooled Ens Spearman **0.435** is an artifact of that partial overwrite — **do not compare** to production **0.750**.

**Good news:** newly written predictions use `eos_rank_source: eos_final_rank` — the DB-path fix works; 2025-26 playoff outcome ranks are available in `nba_build.duckdb`.

**Still needed:** a **full** `run_pipeline_from_model_a` into a **clean output root** (e.g. `improved_07-06/outputs`), not a partial infer/eval into the old `improved_02-27` folder.

---

## What happened

| File | Last modified | Status |
|------|---------------|--------|
| `predictions_2023-24.json` | Jul 3 05:52 | **Stale** (pre-DB-fix era) |
| `predictions_2024-25.json` | Jul 3 05:52 | **Stale** |
| `predictions_2023-24@2024-03-01.json` | Jul 6 01:31 | Fresh (`eos_final_rank`) |
| `predictions.json` | Jul 6 01:31 | Fresh |
| `predictions_2025-26*.json` | — | **Missing** |

Eval pooled only: `2023-24`, `2023-24@2024-03-01`, `2024-25` → 3 checkpoints, N≈79 teams (not the standard 180 for 6 checkpoints).

---

## Metrics (this invalid pool)

| Metric | This eval | Production (`improved_07-03`, 6 ckpt) |
|--------|-----------|----------------------------------------|
| Ens Spearman | 0.435 | **0.750** |
| Ens NDCG@4 | 0.195 | 0.258 |
| Ens MAE | 6.71 | 4.10 |
| Model B Spearman | 0.503 | 0.760 |
| EOS source | eos_final_rank | eos_final_rank |

Champion picks on this stale/mixed pool: 2023-24 Boston ens **#2** (miss); 2024-25 OKC ens **#6** (miss).

---

## Valid runs to compare

| Run | Valid? | Ens Spearman | Notes |
|-----|--------|--------------|-------|
| **improved_07-03 / run_025_07-03_2349** | ✅ | **0.750** | **Production best** (6 ckpt, outcome rank) |
| improved_07-05 / run_026_07-06_0025 | ❌ | 0.817 | Wrong DB (`nba_build_run`); 2025-26 used standings |
| improved_02-27 / run_026 (Jul 3 full) | ✅ | 0.520 | Pre-07-03 fixes; 2 test seasons only |
| improved_02-27 / run_026 (Jul 6 partial) | ❌ | 0.435 | Mixed stale/fresh predictions |

---

## Recommended next command

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18

python -m scripts.run_pipeline_from_model_a \
  --config config/8_spearman_improved.yaml \
  --outputs output/8_spearman_surrogate/improved_07-06/outputs
```

Also update `paths.outputs` in `8_spearman_improved.yaml` to match (currently still `improved_02-27/outputs`).
