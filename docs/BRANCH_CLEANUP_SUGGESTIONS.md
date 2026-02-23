# Branch cleanup suggestions

**Purpose:** Suggest which branches can be deleted to simplify the repo.

**Execution (Feb 2026):** Main was pushed; **11 remote branches** were deleted; **10 local branches** were deleted. The 4 worktree-blocked locals (`docs/implementation-roadmap`, `feature/sweep-and-hparam-docs`, `implement-plan-phases-fpg`, `update-readme-and-pipeline`) were removed by force-removing their worktrees (axi, net, fpg, hbf), then the branches were deleted. **Remaining local branches:** main, Per_Game_Prediction, feature/model-a-map-estimator, feature/post-checkpoint-updates.

**Current default branch:** **main** (Spearman surrogate line; merged with baseline-listmle so main has full docs, outputs8 best, outputs13/15/16 docs, RMSE findings, and loss logging).

---

## Summary table

| Branch | Dedicated output folder? | Productive? | Suggestion | Reason |
|--------|---------------------------|-------------|------------|--------|
| **main** | — | Yes | **Keep** | Default; has full docs + Spearman surrogate. |
| **main-spearman-surrogate** | — | Yes | Optional | Same commit as main (or sync to main). Keep only if you want a branch name with "spearman surrogate"; otherwise **can delete** after main is pushed. |
| **baseline-listmle** | — | Yes | Optional | Old main merged into main. **Can delete** after merge is pushed; history preserved in main. |
| **feature/train-spearman-surrogate** | outputs8 | Yes | **Can delete** | Renamed to main; remote may still have it. Redundant with main. |
| **feature/train-rank-rmse-surrogate** | outputs13 | **No** | **Suggest delete** | RMSE surrogate deemed unproductive. All findings in [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md) and OUTPUTS13 docs; code (rank_rmse_surrogate) remains in main for reference. |
| **feature/sweep-and-hparam-docs** | — | Partial | **Can delete** | Doc-only; main has newer sweep/hparam docs. |
| **feature/model-a-map-estimator** | outputs14 (future) | Maybe | **Keep for now** | MAP run (outputs14) still to test; branch may have MAP-specific code. |
| **feature/post-checkpoint-updates** | — | Unclear | Review | No dedicated output folder; review if changes are in main then **can delete**. |
| **docs/implementation-roadmap** | — | Partial | **Can delete** | Doc-only; ensure roadmap is in main or .cursor/plans then delete. |
| **implement-plan-phases** | — | Partial | **Can delete** | Plan phases; if plans are in main/.cursor/plans, **can delete**. |
| **implement-plan-phases-fpg** | — | Partial | **Can delete** | Same as above. |
| **update-readme-and-pipeline** | — | Partial | **Can delete** | If README/pipeline updates are in main, **can delete**. |
| **plan-readme-trapdoor-fixes** | — | Partial | **Can delete** | If fixes are in main, **can delete**. |
| **local_main** | — | — | **Can delete** | Usually local mirror of main; **can delete** if same as main. |
| **Per_Game_Prediction** | outputs_player_game? | Maybe | Review | Per-game prediction; if you have a dedicated output folder and still use it, **keep**; else review. |

---

## Recommended order of operations (no deletions performed here)

1. **Push main** (and optionally main-spearman-surrogate) so remote is up to date.
2. **Delete remote branches** you don’t need (e.g. `feature/train-rank-rmse-surrogate` after RMSE findings are in main):
   - `git push origin --delete feature/train-rank-rmse-surrogate`
   - Similarly for `feature/train-spearman-surrogate`, `feature/sweep-and-hparam-docs`, etc., after you confirm their content is in main.
3. **Delete local branches** after remote is updated:
   - `git branch -d feature/train-rank-rmse-surrogate`
   - etc.

---

## Unproductive / no dedicated output folder

- **feature/train-rank-rmse-surrogate** — Unproductive (RMSE surrogate underperformed; see [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md)). **Strong candidate for delete.** Outputs13 folder and docs remain in main; only the branch is redundant.
- **feature/sweep-and-hparam-docs** — No dedicated output folder; sweep/hparam docs are on main and are newer. **Can delete.**
- **docs/implementation-roadmap**, **implement-plan-phases**, **implement-plan-phases-fpg**, **update-readme-and-pipeline**, **plan-readme-trapdoor-fixes**, **local_main** — No dedicated output folders; if their content is merged into main, **can delete.**

---

## Keep (for now)

- **main** — Default; keep.
- **feature/model-a-map-estimator** — Keep until MAP run (outputs14) is tested and either kept or documented and retired.
- **Per_Game_Prediction** — Keep if you still use per-game prediction and its output folder.

---

## Notes

- **main-spearman-surrogate:** Same content as main (or one commit behind). Useful if you want the words “spearman surrogate” in a branch name; otherwise safe to delete after main is pushed.
- **baseline-listmle:** Old main; merged into main. Safe to delete after merge is on remote; history is in main.
- RMSE surrogate findings are fully documented in [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md) and [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md); deleting **feature/train-rank-rmse-surrogate** does not lose findings.
