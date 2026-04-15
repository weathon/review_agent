Now let me search for calibration papers to determine the appropriate score.Now I have enough information to write the final consolidated review.

---

## Summary

TABDIFF introduces a continuous-time mixed-type diffusion framework for tabular data generation that operates directly in the original data space (not a latent space), jointly modeling numerical features via Gaussian (VE-SDE) diffusion and categorical features via absorbing-state masked diffusion. The two central technical innovations are (1) feature-wise learnable noise schedules (power-mean for numerics, log-linear for categoricals) to handle per-column distribution heterogeneity, and (2) a mixed-type stochastic sampler that adds a small forward perturbation at each reverse step to correct accumulated decoding errors. The paper also extends the framework with classifier-free guidance for missing-value imputation. Experiments across seven datasets and eight metrics show consistent improvements, with up to 22.6% gain over the strongest baseline (TabSyn) on pairwise column correlation (Trend).

---

## Strengths

- **Principled unified framework**: The combination of continuous-time Gaussian and absorbing masked diffusion in the *original* data space is novel and well-motivated. It avoids the encoding overhead of latent-space methods (TabSyn) and the loose ELBO of discrete-time methods (TabDDPM, CoDi). The continuous-time ELBO derivation for masked categorical diffusion (Eq. 9) is technically sound.

- **Feature-wise learnable schedules**: The insight that different tabular columns require different corruption rates is well-grounded. The parametric schedule families (power-mean, log-linear) are simple, constrained enough to stabilize training, and empirically verified to reduce training loss (Figure 2) and boost Trend substantially in the ablation (Table 5: Fix+Det 2.29 → Learn+Sto 1.80).

- **Strong and consistent empirical results on correlation capture**: The 22.6% average improvement over TabSyn on Trend (pairwise column correlation error), with consistent gains across all seven datasets (Table 2), is the paper's clearest and most compelling evidence. Preserving inter-column dependencies is one of the hardest aspects of tabular generation, making this the most important result.

- **Clear ablation support for both contributions**: Table 5 cleanly isolates the effects of learnable schedules and the stochastic sampler, and shows they are complementary rather than redundant. The design is transparent and the ablation is easy to interpret.

- **Code released and end-to-end training**: The framework is trainable end-to-end (noise schedule parameters learned via backpropagation), and code is publicly available, facilitating reproducibility and adoption.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **No computational cost or efficiency comparison** — The stochastic sampler (Algorithm 2) adds a full forward perturbation step before each backward step, effectively doubling the per-step computation relative to a deterministic ODE solver. No wall-clock time, sampling time, or NFE (neural function evaluation) count is reported for TABDIFF or any baseline. For a method claiming practical superiority over strong baselines, this omission is significant. If TABDIFF's gains on Trend come at a 2× or greater sampling cost, that is an important trade-off readers need to evaluate.

- **Imputation evaluation is too narrow to support its claimed contribution** — Section 2.5 and 4.3 present conditional generation via CFG as a meaningful extension of the framework. However, Table 4 compares only against TabSyn and XGBoost. No other diffusion-based baselines (CoDi, STaSy, TabDDPM) are included, nor dedicated imputation methods. Moreover, TABDIFF+CFG underperforms TabSyn on two datasets (Shoppers: 96.4 vs 96.5, Beijing: 0.414 vs 0.386 RMSE). The evaluation is more proof-of-concept than a validated contribution to conditional generation; the claims in Section 2.5 should be scoped accordingly.

### Minor

- **Dataset count inconsistency** — Section 4.1 explicitly states "seven real-world tabular datasets" but immediately lists *eight* names: "Adult, Default, Shoppers, Magic, **Faults**, Beijing, News, and Diabetes." The dataset "Faults" appears in the experimental setup description but nowhere in Tables 1–4 or the ablations. This is a genuine error in the paper text (not a parsing artifact), and should be corrected.

- **Table 1 bolding error** — In Table 1 (Shape metric), TABDIFF's entries for **Default** (1.24 vs TabSyn's 1.01) and **News** (2.35 vs TabSyn's 2.06) are bolded despite TABDIFF being worse than TabSyn there (lower error = better). The "Improv." row correctly shows 0% for both columns, confirming these are not ties. Bolding non-best entries is a presentation error that undermines the table's clarity.

- **No analysis of learned schedule parameters** — The core theoretical claim is that feature-wise schedules enable adaptive capacity allocation and flexible denoising order. Yet the paper never shows what ρ_i and k_j values are actually learned, whether schedules differ meaningfully across features with different marginals, or whether features with simpler distributions get faster schedules. Figure 2 shows lower training loss but does not confirm the intended mechanism. A visualization of the learned schedule curves per feature on at least one dataset would directly justify the core design claim.

- **Ablations only report dataset averages** — Table 5 presents ablation results averaged over all datasets for Shape and Trend. For a method claiming consistent superiority across heterogeneous datasets, dataset-level ablation results are needed to confirm robustness. Given the high variance across datasets (e.g., TABDIFF gains 46% on Diabetes vs 0% on Default/News for Shape), the averages may obscure inconsistencies.

### Trivial

- The typo "archives the closest match" (Section 4.5, should be "achieves") is minor.

---

## Nice-to-Haves

- **Visualization of learned noise schedules per feature**: Plotting σ_ρᵢ(t) and α_kⱼ(t) for a few features with contrasting distributions (e.g., uniform vs. skewed categorical) would directly validate the heterogeneity-handling claim and add significant interpretive value.

- **Sampling trajectory visualization**: Illustrating how individual features evolve across reverse denoising timesteps would make the stochastic sampler's error-correction mechanism intuitive and would support the "flexible denoising order" claim.

- **Ablation on masking vs. uniform categorical diffusion**: The choice of absorbing (masking) diffusion over uniform transition matrices (Austin et al. 2021) is well-motivated but not empirically compared. A brief ablation would strengthen confidence in this design choice.

- **Discussion of scalability to high-dimensional datasets**: All seven benchmarks have at most ~30 features. A brief discussion or experiment on a higher-dimensional dataset would clarify whether feature-wise schedule parameters (M_num + M_cat scalar parameters) remain stable and whether the approach scales.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Privacy claim not substantiated**: The paper explicitly names DCR as the privacy metric in Section 4.2 and states "results are deferred to Appendices A.2 and E." This is standard practice for space-limited submissions. The appendix appears to contain the full DCR results. The criticism that privacy is "advertised but not established" misreads the paper structure. REMOVED.

- **[Harsh Critic] Baseline evaluation too weakly specified to trust headline claims**: The reuse of baseline numbers from Zhang et al. (2024) is explicitly disclosed in a footnote and is standard practice in the tabular generation literature. TabSyn results are re-reproduced by the authors (also disclosed). Critically, TABDIFF is evaluated on *all* seven datasets including the harder ones (News, Diabetes) where some baselines (GReaT, STaSy) report OOM — meaning the current setup is already *unfavorable* to TABDIFF's average metrics relative to those baselines. The "evidential weakness" framing is overstated. REMOVED per hard rule (asymmetry favors baselines, not the authors).

- **[Neutral Reviewer] OOM baselines skew averages against TABDIFF**: GReaT and STaSy with OOM on News/Diabetes have their averages computed only over 5/7 datasets, excluding the hardest ones. This actually *benefits* those baselines in average comparisons. The concern should have been the opposite — if anything, TABDIFF's average is computed more fairly over all 7 datasets. REMOVED per hard rule.

- **[Harsh Critic / Neutral Reviewer] Interaction between numerical and categorical sampling theoretically unjustified**: The ODE approximation for numerics and categorical sampling run jointly within the same step. Demanding formal theoretical justification for this interaction goes beyond the standard for an empirical systems paper in this community. The joint objective and algorithmic procedure are clearly stated; formal coupling analysis is a nice-to-have at most. MOVED to Nice-to-Haves framing.

- **[Harsh Critic] Loss weighting λ_num and λ_cat not analyzed**: While a sensitivity analysis would be nice, the paper trains end-to-end and the λ values are implementation details disclosed in Appendix D. This is a reproducibility nitpick rather than a core concern. REMOVED per reproducibility nitpick rule.

---

## Novel Insights

The most genuinely novel observation surfaced by the reviewers — not fully articulated in the paper itself — is that the stochastic sampler addresses a qualitatively distinct problem in categorical diffusion relative to continuous diffusion: once a categorical token is "unmasked" in the reverse process, it is deterministically frozen for all subsequent steps (Eq. 8). This means accumulated denoising errors on early-decoded features cannot be corrected by the ODE solver (which only applies to numerics). The stochastic sampler's re-masking step is therefore not just an adaptation of predictor-corrector methods from continuous diffusion but addresses a structural deficiency specific to absorbing-state categorical diffusion when coupled with continuous features. The paper would benefit from making this mechanistic justification more explicit.

---

## Suggestions

1. **Report NFE and wall-clock sampling time** in Table 5 or a dedicated efficiency table, comparing TABDIFF (stochastic sampler) against TABDIFF (deterministic) and TabSyn. This directly addresses the cost-benefit question raised by the stochastic sampler.

2. **Fix Table 1 bolding**: Only the globally best entry per column should be bolded. TABDIFF's Default and News entries should not be bolded.

3. **Fix the dataset count**: Either remove "Faults" from the enumeration or add it to the experimental tables with an explanation.

4. **Add a learned schedule visualization**: For one dataset, plot σ_ρᵢ(t) for several numerical features and α_kⱼ(t) for several categorical features on the same axes. This is inexpensive and would directly justify the core design claim.

5. **Broaden the imputation comparison**: Include at minimum TabDDPM and CoDi in Table 4, and frame the CFG contribution as a "proof-of-concept" rather than a validated competitive contribution if full comparison is infeasible.

6. **Dataset-level ablation**: Report Shape/Trend ablation results per dataset (perhaps in an appendix) to demonstrate that the stochastic sampler and learnable schedules help robustly rather than only on average.

---

## Score and Decision

**Calibration:**

- **TabSyn** (4Ay23yeuz0.md, Accept Oral): Scores 8, 6, 8, 5 (avg ~6.75). This is the prior SOTA; incremental but solid contribution, well-executed. TABDIFF clearly *outperforms* TabSyn empirically. However, TabSyn's contribution was arguably more impactful as the first strong latent-diffusion model for tabular data; TABDIFF builds on that foundation.

- **CDTD** (QPtoBPn4lZ.md, Accept Poster): Scores 6, 5, 5, 6 (avg ~5.5). Very similar framing to TABDIFF (also mixed-type, also feature-wise adaptive noise schedules, also concurrent work). CDTD operates in a homogenized continuous embedding space; TABDIFF operates directly in data space with a principled categorical diffusion process. TABDIFF is technically stronger (continuous-time categorical ELBO, stochastic sampler, stronger empirical results), so TABDIFF should score above CDTD.

- **TabDAR** (kkGIbmpCHU.md, Reject): Scores 5, 3, 5, 6. Rejected for narrow novelty, missing timing comparison, weak baselines. TABDIFF is clearly stronger than TabDAR on all these dimensions.

**Positioning**: TABDIFF is above CDTD (poster, ~5.5 avg) but below TabSyn's oral-level reception (~6.75 avg), primarily because TabSyn was a more foundational contribution and TABDIFF, while empirically superior, has the noted gaps (narrow imputation evaluation, no timing, minor presentation errors). A score of **6.5** is appropriate — well above the reject threshold, deserving of poster acceptance.

**Axis summary:**
- *Originality*: Good. Novel joint continuous-time framework for mixed-type data in original space; not purely incremental.
- *Importance*: Good. Pairwise correlation preservation is a longstanding challenge; 22.6% improvement over prior SOTA is meaningful.
- *Claims vs. evidence*: Mostly well-supported for unconditional generation; imputation/CFG claims are overstated relative to evidence.
- *Soundness*: Good. Methods are technically correct; loss functions and algorithms are clearly stated.
- *Clarity*: Good, with minor issues (bolding error, dataset count discrepancy).
- *Value to community*: High. Code is available; method establishes a new strong baseline for tabular generation.

**Score: 6.5 — Accept (Poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>