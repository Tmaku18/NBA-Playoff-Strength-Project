# Attention Temperature Semantics

Model A uses multi-head attention with a softmax temperature. This document describes the semantics of different temperatures and the multi-temperature aggregation design.

## Temperature as "Effective Top-K"

Attention temperature controls how sharp or diffuse the attention distribution is over players:

- **Lower temp** → sharper attention (fewer players dominate; more star-dependent)
- **Higher temp** → softer attention (more spread across roster; more depth-aware)

Interpreted as "effective top-k" for ranking focus:

| Temp | Focus | Semantics |
|------|-------|-----------|
| 1 | Top player | Star-driven; highest star-dependent risk |
| 3 | Top 3 | Stars / core |
| 5 | Top 5 | Starting lineup (most representative) |
| 6 | Top 6 | Main rotation + 6th man |
| 8 | Top 8 | Main rotation depth |
| 10 | Top 10 | Extra depth / full rotation |

## Hypotheses (H1–H4)

**H1 — Injury-prone stars:** Players with low game-day availability (high injury rate) should be discounted. Their contribution should receive less weight; teams built around injury-prone stars rank lower than full-strength would suggest.

**H2 — Durable stars:** Players with consistently high availability get full value. Their contribution is not discounted.

**H3 — Multi-temperature robustness:** Running at multiple temperatures and aggregating reduces sensitivity to any single player being out or roster variation. Temp 1 alone amplifies star-dependent noise; multi-temp dampens it.

**H4 — Injury rating over time:** Availability (availability_L10, availability_L30) tracks players over the season. The system adapts as availability evolves.

## Multi-Temperature Aggregation

When `multi_temp_enabled: true`:

1. Model A is run at temps [1, 3, 5, 6, 8, 10].
2. Per team: rank from each temp's scores; **agreement** = 1/(1+std(ranks)).
3. **Base weights:** Temp 5 (starting lineup) highest (1.0); temp 1 and 10 lowest.
4. **Effective weight:** base_weight × (0.5 + 0.5×agreement) × availability modifier.
5. **Final score:** weighted sum of per-temp scores.

**Confidence c_A:** Agreement across temperatures. High agreement → high confidence; low agreement (ranks disagree) → low confidence.

## Availability Modulation

- **In attention (`use_availability_in_attention`):** `minutes_eff = minutes × availability` (0.5×L10 + 0.5×L30). Injury-prone players get lower effective weight in the attention blend.
- **In aggregation (`use_availability_in_aggregation`):** `starter_availability` = mean availability of top-5 by minutes. High starter_avail → boost temp 5 weight; low → boost temp 8/10 (depth matters).

## References

- `config/defaults.yaml` — `model_a.attention` section
- `src/models/multi_temp_aggregation.py` — aggregation implementation
- `docs/CONFIDENCE_WEIGHTED_ENSEMBLE_OPTIONS_02-12.md` — ensemble use of conf_a
