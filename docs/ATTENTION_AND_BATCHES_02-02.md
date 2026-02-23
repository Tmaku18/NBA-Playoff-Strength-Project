# Attention collapse and all-masked batches

## What is NaN and why does it “collapse” attention?

**NaN** (“Not a Number”) is a floating-point value that represents an invalid or undefined result (e.g. 0/0, ∞−∞). In PyTorch, once a tensor has NaN, it propagates through operations (NaN × x = NaN, etc.), and gradients become NaN so training stops updating weights.

In **attention**, NaN can appear when:

- The softmax inside `MultiheadAttention` receives **all -inf** (e.g. every key is masked), so the output is undefined.
- Logits are **extremely large or small**, leading to numerical overflow/underflow in softmax.

When that happens we say attention **collapsed**: the weights are no longer valid numbers, gradients don’t flow, and the model can’t learn. The fix in `SetAttention` is to treat **non-finite** (NaN or inf) attention as “collapsed” and replace it with fallback weights (minutes-based or uniform) so outputs and gradients are finite again.

---

## Why are all rows masked? (“all_masked=15”)

**key_padding_mask** is True for **padding** (ignore), False for **valid** players. So “all masked” for a team means: for that team, **all 15 player slots** have `key_padding_mask == True`, i.e. every slot is padding and there are **no valid players**.

That happens when **roster lookup returns 0 players** for that team:

1. `get_roster_as_of_date(pgl, team_id, as_of_date, season_start=...)` returns an **empty** DataFrame.
2. `build_roster_set` then pads to 15 slots with padding and sets `key_padding_mask = True` for all 15.

So “all rows masked” does **not** mean “players are missing from the DB.” It means: for that **(team_id, as_of_date)** (and optional **season_start**), the **query** we run over the DB returned **no rows**. Common causes:

- **Season window too narrow:** We filter `game_date >= season_start` and `game_date < as_of_date`. If the DB has no games in that window (e.g. first games are after `as_of_date`, or `season_start` is wrong), we get 0 rows.
- **Date/types:** `as_of_date` or `game_date` types/formats can make the comparison wrong so no rows match.
- **team_id mismatch:** In theory team IDs could differ between tables; in practice they come from the same DB so this is rare.

So “roster lookup returning no players” means “the **query** (with that team, date, and season window) returned no rows,” even if players and games exist elsewhere in the DB. The code now:

1. **Fallback in roster lookup:** If the season-scoped query returns empty, we retry **without** `season_start` (any games for that team before `as_of_date`) so we use data when it exists.
2. **Skip all-masked batches:** When building batches, we skip any list where **every** team would end up with an empty roster (all-masked), so we never add a batch that has zero valid players.
