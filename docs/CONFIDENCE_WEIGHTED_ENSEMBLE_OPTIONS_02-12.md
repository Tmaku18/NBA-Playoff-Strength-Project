# Confidence-Weighted Ensemble: Options and Tradeoffs

**Goal:** Give higher weight in the ensemble to the model that is *more confident* on each prediction (per team / per instance), so the final score is not just a fixed linear blend of Model A + XGB.

---

## What “confidence” can mean in this codebase

- **Model A (Deep Set):** Produces a single score per team; no built-in probability or variance. Possible proxies:
  - **Attention entropy** (per team): \( -\sum_p w_p \log w_p \) over player attention weights. *Low entropy* = attention concentrated on few players → “sure”; *high entropy* = diffuse → “uncertain”.
  - **Attention max weight:** Higher max = more decisive focus on one player (proxy for confidence).
  - **Score extremity:** \( |\text{score} - \text{median}(\text{scores})| \) on the current batch. More extreme = further from “average” → could be treated as more committed (or more extreme, depending on interpretation).
  - **Z-norm / latent spread:** If we had multiple samples (e.g. dropout at inference), variance of score could indicate uncertainty; not available without changing inference.

- **XGB (regression):** Single `predict()` output per team. No native uncertainty. Possible proxies:
  - **Tree-level variance:** If we use `pred_contribs` or iterate trees and take std across trees, higher variance → less confident (common in XGB for uncertainty).
  - **Distance from (train) distribution:** Score far from training mean/median might be “extreme” (could be confident or out-of-distribution).
  - **Learned confidence:** A small meta-model or calibration layer that predicts “how much to trust XGB” from features or from OOF residuals (e.g. where XGB was wrong in the past).

So in practice we can define **per-team confidence** \( c_A(i), c_X(i) \) for Model A and XGB using the above (e.g. 1 − normalized entropy for A, inverse of tree std for XGB), then use them in one of the schemes below.

---

## Implemented approach: Option 2 (meta-learner with 4 inputs)

The codebase implements **Option 2**: the meta-learner (RidgeCV) is trained on **4 inputs** (s_A, s_X, c_A, c_X) when confidence columns are present in OOF.

- **Model A confidence c_A:** From attention weights per team. Formula: c_A = entropy_weight * (H / H_max) + max_weight_weight * (1 - max_p w_p). High entropy = diffuse = high confidence; low entropy or high max weight = star-dependent = high risk = low confidence. Config: `model_a.confidence.entropy_weight`, `model_a.confidence.max_weight_weight` (defaults 0.5, 0.5).
- **XGB confidence c_X:** From tree-level variance: per-sample std across trees, then c_X = 1/(1+std). Higher variance = lower confidence.
- **Training:** Script 3 writes conf_a to oof_model_a.parquet; script 4 writes conf_xgb to oof_model_b.parquet. Script 4b merges and, when both columns exist, trains RidgeCV on 4 columns. Inference computes c_A, c_X and passes them; if the loaded meta has 4 coefficients, the ensemble uses the 4-col input.
- **Backward compatibility:** OOF without confidence columns yields 2-col meta; meta with 2 coefs uses 2-col input only.

Other options below remain as future variants.

---

## Option 1: Soft confidence-weighted average (no meta-learner change)

**Idea:**  
\( \text{ens}_i = \frac{c_A(i)\, s_A(i) + c_X(i)\, s_X(i)}{c_A(i) + c_X(i)} \).  
Weights sum to 1 per instance; the more confident model gets higher weight.

**Pros:**
- Simple, interpretable, no retraining.
- Directly implements “higher confidence → higher weight”.
- Works with any scalar confidence proxies.

**Cons:**
- No learning: optimal blend might not be “proportional to confidence” (e.g. one model might be better overall and you’d want a floor/ceiling).
- Sensitive to how confidence is normalized (scale of \( c_A, c_X \)).

**Variants:** Use log-odds style weighting \( w_i \propto \log(c_i/(1-c_i)) \) so small differences in confidence don’t dominate; or \( w_A = c_A^\alpha \) with \( \alpha > 1 \) to sharpen.

---

## Option 2: Meta-learner with confidence as extra inputs (learned blend)

**Idea:** Keep RidgeCV (or another meta-learner) but feed it **4 inputs**: \( (s_A, s_X, c_A, c_X) \). The meta-learner learns how to use confidence (e.g. “trust XGB more when \( c_X \) is high and \( c_A \) is low”).

**Pros:**
- Data-driven: optimal use of confidence is learned from OOF.
- Can capture non-linear effects (e.g. “only downweight A when both A is uncertain and XGB is confident”).
- Still one model at inference; same pipeline, just wider OOF table.

**Cons:**
- Need to define and compute \( c_A, c_X \) at **training time** for every OOF row (and store them or recompute). For A, need attention (or entropy) in the OOF loop; for XGB, need e.g. tree variance in script 4.
- More hyperparameters (e.g. alpha for Ridge with 4 features); risk of overfitting if OOF is small.
- Interpretability of the blend is lower than a fixed formula.

---

## Option 3: Entropy-based weighting (Model A only, or both if XGB has a proxy)

**Idea:** Use only Model A’s attention entropy. Define \( \text{conf}_A = 1 - H_A / \log P \) (normalized so 1 = full confidence). Weight A by \( \text{conf}_A \); XGB by \( 1 - \text{conf}_A \) or by a constant (e.g. 0.5) if XGB has no confidence.  
So: \( \text{ens} = \text{conf}_A \cdot s_A + (1-\text{conf}_A) \cdot s_X \) (with normalizing denominator if you want weights to sum to 1).

**Pros:**
- Theoretically motivated: entropy measures “spread” of the model’s focus.
- Only requires attention weights from Model A (already available in inference).
- No retraining of the stacker if you use a fixed formula.

**Cons:**
- Entropy might not align with “accuracy” (e.g. sometimes diffuse attention could still be right).
- XGB has no natural entropy; you’d either treat it as constant confidence or add a separate proxy (e.g. tree variance).

---

## Option 4: Metric-based switching (e.g. “NDCG model when confident, Spearman model when not”)

**Idea:** You have (or could train) two stackers or two base-model configs: one tuned for NDCG, one for Spearman. At inference, for each team, compute a “confidence” (e.g. agreement between A and XGB, or entropy). If confidence is high → use the NDCG-optimized blend; if low → use the Spearman-optimized blend (or vice versa). Blend could be a soft switch: \( \text{ens} = \lambda \cdot \text{ens}_{\text{NDCG}} + (1-\lambda) \cdot \text{ens}_{\text{Spearman}} \) with \( \lambda = \sigma(\text{confidence}) \).

**Pros:**
- Directly targets your two metrics: use the model that is better for the metric you care about in each regime.
- Can be simple (two fixed blends + one threshold or soft weight).

**Cons:**
- Requires training and maintaining two meta-models (or two full pipelines).
- “Confidence” here is really “which regime we’re in”; you need a clear definition (e.g. when do we prefer NDCG vs Spearman? e.g. when rankings are clear vs when they’re noisy).
- Risk of discontinuities at the switch boundary unless you use a soft blend.

---

## Option 5: Gating network (small NN that outputs per-instance weights)

**Idea:** Train a tiny NN that takes \( (s_A, s_X, c_A, c_X) \) (and optionally team/context features) and outputs \( (w_A, w_X) \) with \( w_A + w_X = 1 \). Train it to minimize ranking loss (e.g. ListMLE or NDCG) on OOF or a validation set.

**Pros:**
- Maximally flexible: can learn complex rules (e.g. “trust A in early season, XGB late” if you feed date).
- Can directly optimize NDCG or Spearman via differentiable approximation or black-box optimization.

**Cons:**
- More code and tuning; risk of overfitting.
- Needs OOF (and ideally confidence) for many samples; might be overkill for 2 models.

---

## Option 6: Bayesian / posterior weighting (agnostic Bayesian learning of ensembles)

**Idea:** Treat meta-weights as uncertain; use holdout performance to form a posterior over which predictor is better, then combine with Bayesian model averaging (e.g. weight by posterior probability that each model is best, or by expected utility under a ranking metric).

**Pros:**
- Principled uncertainty over “which model to trust”; can incorporate prior (e.g. prefer simpler blend).
- Well-suited when you have a clear metric (e.g. NDCG) and holdout sets.

**Cons:**
- Per-instance weighting is not standard in this framework; usually it’s per-model. To get “confidence” you’d need to define instance-level uncertainty (e.g. from residuals or conformal intervals).
- More complex implementation and possibly more data for stable posteriors.

---

## Option 7: Confidence as “agreement” (weight by A–XGB agreement)

**Idea:** Define confidence per team as agreement between Model A and XGB (e.g. correlation of their ranks in a window, or 1 − normalized rank difference). Where they agree, use a strong blend; where they disagree, use a more cautious blend (e.g. closer to average or to the model that is usually more accurate).

**Pros:**
- No need for internal confidence from A or XGB; only uses their outputs.
- Easy to implement; interpretable (“we’re more confident when both models agree”).

**Cons:**
- Agreement is not the same as “correct”; two wrong models can agree.
- Doesn’t distinguish “both confident and agree” from “both uncertain and similar by chance”.

---

## Summary table

| Option | Complexity | Retrain? | Confidence from | Best when |
|--------|------------|----------|-----------------|------------|
| 1. Soft confidence-weighted avg | Low | No | A: entropy/max; XGB: tree var or constant | You want a simple, interpretable rule |
| 2. Meta with (s_A, s_X, c_A, c_X) | Medium | Yes (OOF) | A: entropy; XGB: tree var | You want data-driven use of confidence |
| 3. Entropy-based (A only) | Low | No | A: attention entropy | You only trust A’s confidence and want minimal change |
| 4. Metric-based switch (NDCG vs Spearman) | Medium–High | Yes (two stackers) | Agreement or entropy | You care about different metrics in different regimes |
| 5. Gating NN | High | Yes | Same as 2 | You want maximum flexibility and have enough OOF data |
| 6. Bayesian weighting | High | Conceptually yes | Holdout + optional instance uncertainty | You want principled uncertainty and have holdout sets |
| 7. Agreement-weighted | Low | No | A–XGB rank agreement | You want something simple without internal confidence |

---

## Recommendation (short)

- **Fastest and robust:** **Option 1** (soft confidence-weighted average) with Model A confidence = \( 1 - \text{normalized attention entropy} \) and XGB confidence = 1 (constant) or inverse of tree std if you add it. No retraining; easy to A/B test.
- **Best if you can retrain:** **Option 2** (meta-learner with 4 inputs): add \( c_A, c_X \) to OOF, refit RidgeCV on \( (s_A, s_X, c_A, c_X) \). Then you learn how much to trust each model from data.
- **If you care about NDCG vs Spearman in different situations:** **Option 4** is the one that explicitly switches (or blends) between a NDCG-optimized and a Spearman-optimized combiner using a confidence/agreement signal.

I can implement any one of these (e.g. Option 1 + Option 3 in inference only, or Option 2 end-to-end with OOF confidence and stacking changes) once you pick a direction.
