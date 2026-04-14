=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
Now I have a thorough understanding of the paper. Let me produce the final consolidated review.

---

## Summary

This paper proposes a *parallel budget-controlled A/B test* design for online advertising markets, where market segmentation identifies relatively isolated submarkets and budget-split experiments are run in parallel across them. The authors formally model cross-submarket interference as supply contamination in the First-Price Pacing Equilibrium (FPPE) framework, propose a debiased surrogate estimator based on directional derivatives that eliminates first-order bias, and establish asymptotic normality for the plug-in estimator. The approach is grounded in real Meta production experiments (99 paired comparisons) and further evaluated on semi-synthetic data calibrated to real auction statistics.

---

## Strengths

- **Novel and elegant contamination abstraction for interference.** Mapping cross-submarket interference to supply contamination in the FPPE framework (Section 3) is a principled and clean conceptual contribution that enables rigorous statistical inference where none existed before. The structural identification of Θ_bad as items with positive values for buyers from multiple submarkets (Section 3.2) maps precisely onto the bipartite-graph clustering construction used in practice.

- **Directional-derivative debiasing with formal guarantees.** Theorem 1 establishes that the debiased surrogate ‖β̃* − β*‖₂ = o(α) vs. the Θ(α) bias of the naive estimator — a qualitatively meaningful improvement — and Theorem 3 establishes asymptotic normality with a concrete, estimable covariance structure. The derivation exploits the convex (EG) program structure in a way that is not routine and represents genuine technical depth.

- **Real-world grounding at scale.** Validation with 99 production A/B experiments at Meta (Figure 1) showing 75–79% sign consistency with the gold-standard budget-split design is a concrete, large-scale demonstration that the parallel submarket design is practically competitive. This is unusually strong empirical grounding for a theory paper.

- **Calibrated semi-synthetic setup.** Using heavy-tailed budgets and valuations sampled from actual Meta auction data (rather than synthetic distributions) to calibrate the simulation is a careful methodological choice that substantially increases the realism and credibility of the numerical results.

- **Dual Hessian estimation pathways.** Providing both a general finite-difference Hessian estimator and a simplified closed-form estimator under the bid-gap condition (Theorem 3, Part 2) gives practitioners a feasible path to implementation even when the general estimator is costly.

---

## Weaknesses

### Fatal
None.

### Major

- **Coverage substantially below nominal in semi-synthetic experiments.** Table 1 shows coverage for β̂ ranging from 0.826 to 0.877 against a nominal 95% level — an 8–17 percentage-point shortfall. The paper attributes this to variance under-estimation in finite markets, but this is exactly the gap the variance estimator is supposed to close. For a paper whose central applied claim is reliable inferential procedure for A/B testing decisions, failing to achieve nominal coverage even in a carefully calibrated semi-synthetic setting is a material weakness. The discussion is honest but does not offer a resolution, calling this an open problem for future work.

- **Severe and unexplained discrepancy between analytical and bootstrap confidence intervals for revenue.** Table 1 reports that the analytical CI for revenue achieves coverage of 1.0 (certain coverage) with normalized widths 0.191–0.244, while the bootstrap CI achieves 0.81–0.95 coverage with widths 0.039–0.044 — roughly a 5× difference in width with divergent coverage properties. Figure 4 confirms both converge asymptotically, identifying this as a finite-sample phenomenon, but the paper provides no quantitative diagnosis of why the two methods diverge so dramatically at the sample sizes used in practice. A practitioner using this methodology cannot determine which interval to trust. This inconsistency deserves dedicated analysis beyond a brief note.

- **The α-is-known assumption is strong and its violation is unanalyzed.** Section 3.1 states "we assume α is known throughout the paper" and notes that in practice it can be estimated from the sample (since good/bad item labels are observable in the parallel A/B test setting). While the observation that α is identifiable from data is correct, the paper never quantifies how estimation error in α̂ propagates to bias and coverage of the debiased estimator. If α̂ deviates from the true α by ε, how does this affect ‖β̃* − β*‖₂? Without even a first-order sensitivity result, a practitioner cannot assess robustness.

### Minor

- **Residual bias of the debiased estimator at moderate-to-large α.** Theorem 1 guarantees o(α) bias as α → 0, but Figure 3 shows that at α = 0.50, the debiased surrogate still has ~5% normalized bias. The o(α) guarantee says nothing useful when α is moderate. The paper provides no threshold α₀ below which the debiasing is practically reliable, leaving practitioners without actionable guidance on the regime of validity.

- **The good-item intra-cluster isolation assumption is unvalidated.** The model assumes buyer i ∈ Cₖ has vᵢ(θ) = 0 for all θ ∈ Θₖ′, k′ ≠ k within the good item set — i.e., within-cluster markets are perfectly isolated. The contamination model concentrates all imperfect separation into Θ_bad, which is an approximation. The paper does not validate how sensitive the estimator is to partial cross-valuation that remains within the "good" items after clustering.

- **Bid gap condition (Theorem 3, Part 2) is opaque.** The condition E_α[1/bidgap(β*_α, θ)] < ∞ is stated without intuition or practical characterization. In competitive first-price markets with many bidders, bid gaps can be very small (tight competition), making the inverse gap potentially large or ill-defined. The paper should characterize when this condition holds or fails in realistic ad auction settings.

- **Rate condition η_t = o(1/√t) is demanding with no practical guidance.** The paper mentions using "a separate large historical dataset" for Hessian estimation to achieve this rate, but gives no indication of how large this dataset must be relative to the experimental data. If the required historical sample is much larger than what is available, the normality guarantee in Theorem 3, Part 1 may not be practically achievable.

- **Computational scalability of the Hessian.** H_α is an n × n matrix where n is the number of buyers. For very large ad platforms the paper targets, even the diagonal approximation's statistical properties relative to the full Hessian are not characterized. The paper mentions diagonal approximation briefly but does not analyze its impact on the asymptotic normality guarantees.

### Tiny

- **The "81.5% optimal agreement" figure is not derived in the text.** The value appears to be 0.9² ≈ 0.81, reflecting that at 90% confidence level two independent experiments must both be correct simultaneously, but this derivation is implicit and the required independence/distributional assumptions are not stated.

- **Only 100 simulations per α level.** Standard error on coverage at the 95% nominal level is ~0.022, adequate to detect the observed shortfall but on the low side for stable conclusions about the shape of coverage across α. More simulations would improve reliability.

---

## Nice-to-Haves

- **Sensitivity experiments for α estimation error.** Even a simulation study varying α̂ around its true value would quantify robustness and give practitioners a margin-of-error budget for α estimation.

- **Real-data validation of the debiasing step.** The 99 production experiments (Figure 1) validate the parallel submarket design concept but predate the debiasing estimator. Applying the debiased estimator to these or similar experiments — even approximately, by comparing debiased point estimates to the gold standard — would substantially strengthen the empirical case.

- **Ablation on Hessian estimation quality.** Varying historical sample size used for Hessian estimation would empirically connect the Hessian estimation error rate η_t to observed coverage, illuminating the dominant source of under-coverage in Table 1.

- **Visualization of estimator distributions.** Histograms of the naive vs. debiased estimator relative to the true value across simulations (rather than only aggregated bias lines in Figure 3) would visually verify the asymptotic normality claim and reveal variance properties hidden by mean-bias summaries.

- **Discuss budget-mediated interference.** Even if items are perfectly partitioned, buyers with global budget management may redistribute spending in ways the contamination model does not capture. A brief discussion acknowledging this additional source of interference — and whether it is likely to be large or small relative to the supply-contamination effect — would strengthen the paper.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Concern about scope being limited to first-price auctions** (Harsh Critic Concern 19): The paper explicitly and properly scopes to first-price auctions, which are "the predominant way display advertising is sold online." Criticizing the absence of second-price or GSP analysis is scope creep.

- **Lack of comparison to switchback designs / cluster-robust variance estimators** (Reviews 2 & 3): The paper's contribution is a model-based inference framework specific to FPPE equilibrium, not a general causal inference comparison study. Demanding side-by-side comparison with switchback designs imposes a scope the paper does not claim. If the design works well for this setting, that is sufficient.

- **Concern about main body structure / appendix placement** (Harsh Critic Concern 3): The full parallel A/B test estimator and revenue analysis in the appendix is a structural choice, not a correctness issue. The main contributions (contamination model, debiased surrogate, asymptotic normality) are fully presented in the main body.

- **Concern about stationarity / non-stationarity of ad markets** (Harsh Critic Concern 17): FPPE is a static equilibrium model; asking for dynamic market analysis is outside the paper's stated scope and would constitute a different paper entirely.

- **Concern about incremental novelty over Liao & Kroer (2023)** (Harsh Critic Concern 4): The contamination framing, the directional-derivative debiased surrogate, and the asymptotic normality result for the contaminated-market estimator are all new contributions not present in prior FPPE inference work. The paper is clear about what is inherited vs. new.

- **Concern about unfair comparison of parallel design vs. full budget-split** (implicit): The production comparison (Figure 1) shows the parallel design against a gold standard that requires more budget. This asymmetry is intentional and favorable to the baseline, making the 75–79% agreement result a conservative understatement of the parallel design's value. This is not a flaw.

---

## Novel Insights

The most genuinely novel insight in this paper — beyond the mechanical contributions — is the realization that supply contamination in FPPE provides exactly the right abstraction for interference in clustered experimental designs. The "bad items" that attract bids across submarket boundaries act as a contaminating distribution on the supply, and this framing allows the entire machinery of FPPE sensitivity analysis (directional derivatives of the EG program, Hessian-based variance characterization) to be redeployed for debiasing without bespoke interference theory. This reuse is elegant and potentially generalizable: any system where equilibrium inference is well-understood under clean data, but where a known fraction of the data is "corrupted," could in principle benefit from the same directional-derivative debiasing approach. The result that the debiased surrogate achieves o(α) bias rather than Θ(α) is a concrete payoff of this abstraction.

The empirical finding that coverage degrades with α (Table 1) while bias is well-controlled (Figure 3) is itself a useful empirical regularity: the debiasing procedure successfully shifts the error from bias to variance, but the variance estimator does not keep pace, pointing to Hessian estimation quality as the likely bottleneck. This is an actionable diagnostic for future work.

---

## Suggestions

1. **Diagnose and fix the coverage gap.** Report the decomposition of estimation error into Hessian error, δ_α estimation error, and irreducible sampling variance. An ablation varying the historical dataset size for Hessian estimation would isolate which component dominates the under-coverage and guide a targeted fix.

2. **Provide a first-order sensitivity bound for α misspecification.** Derive or numerically estimate how much coverage degrades when α̂ = α ± ε. Even a simple simulation study sweeping ε would give practitioners a practical robustness criterion.

3. **Explain and reconcile the analytical vs. bootstrap CI discrepancy for revenue.** Either derive analytically why the analytical CI (Eq. 22) is so conservative in finite samples relative to bootstrap, or provide a bias-corrected version of the analytical variance. Without this, practitioners are left with two unreliable and contradictory intervals.

4. **Clarify the inferential target.** Be explicit throughout whether confidence intervals are for β̃* or β*. Since Theorem 1 guarantees β̃* ≈ β* only to o(α), and coverage results in Table 1 are measured against β̃*, the paper should explicitly discuss the implied under-coverage relative to β* when α is moderate.

5. **Characterize the bid gap condition practically.** Provide either a market structure result (e.g., "if buyers have generic valuations, the bid gap condition holds a.s.") or an empirical check on the semi-synthetic data confirming the condition is satisfied, to justify applying Theorem 3 Part 2.

---

**Overall assessment by axis:**

- **Novelty**: High. The contamination framing for FPPE interference is original, and the directional-derivative debiasing is a non-trivial technical contribution.
- **Technical soundness**: Good, with caveats. The theory is carefully developed under stated conditions. The conditions themselves (SMO, SCS, bid gap) are strong and their practical validity is partially unverified.
- **Empirical support**: Moderate. Bias reduction is convincingly demonstrated; coverage performance is below par and unexplained in the regime of practical interest; real-data validation of the debiasing step is absent.
- **Significance**: High for the ad-tech / market design community; moderate for ICLR's general ML audience.
- **Clarity**: Good overall; the exposition of FPPE theory and the debiasing construction is precise and well-organized. The coverage results in Table 1 deserve more analysis than they receive.

# Actual Human Scores
Individual reviewer scores: [8.0, 5.0, 8.0]
Average score: 7.0
Binary outcome: Accept
