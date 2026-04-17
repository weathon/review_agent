## Summary

This paper studies sparse signal recovery from mixed-quality observations: a small set of high-quality measurements with low noise variance σ₁² and a larger set of low-quality measurements with higher noise variance σ₂². The authors establish sufficient conditions on sample sizes (n₁, n₂) for support recovery in both agnostic (decoder unaware of per-sample variances) and informed (decoder knows variances) settings, introducing the "Price of Quality" γ measuring how many low-quality samples substitute for one high-quality sample. They also prove that the LASSO recovery threshold in the agnostic setting depends only on total sample size and average noise variance, meaning high- and low-quality data contribute equally at the algorithmic level.

## Strengths

- **Novel and well-motivated problem formulation.** The mixed-quality data setting captures a practically important scenario (e.g., expert vs. crowd-sourced labels, multi-site sensors) that had not been analyzed in the sparse recovery literature. The distinction between agnostic and informed settings is clean and captures realistic scenarios.

- **Strong LASSO result (Theorem 3) with nontrivial technical contribution.** The extension of Wainwright's LASSO phase transition to heterogeneous noise via Gram–Schmidt decomposition and Haar measure arguments is technically demanding and genuinely novel. The result is both necessary and sufficient, and the finding that the threshold depends only on n₁+n₂ and σ²_avg—a striking robustness to heterogeneity—is a valuable insight.

- **Interpretable "Price of Quality" framework (Theorems 1–2).** The sufficient conditions α₁n₁ + α₂n₂ ≥ (1+ε)n★ yield the intuitive trade-off γ = α₁/α₂, with clean asymptotic expressions across SNR regimes (equations 13–14, 19–21). This provides actionable guidance on how to trade data quantity for quality.

- **Transparent about limitations.** Remark 3.2 explicitly acknowledges the looseness of the agnostic sufficient condition. Remark 4.2 honestly discusses why the informed LASSO extension is nontrivial. The discussion of generalizations (Remark 3.4) is useful.

## Weaknesses

### Major

- **The "Price of Quality" in the agnostic setting is a property of a specific sufficient condition, not of the underlying information-theoretic problem, but the paper frames it as more fundamental than it is.** Theorem 1 provides only a sufficient condition via a relaxed Chernoff bound. As the authors acknowledge (Remark 3.2), optimizing the Chernoff exponent—which yields tight thresholds in the homogeneous case—leads to a cubic equation whose relaxation introduces unknown looseness. The claim that "one high-quality sample is never worth more than two low-quality samples" (abstract; §1.2.1; eqs. 13–14) is technically qualified with "for this sufficient condition to hold" in the abstract, but this hedge is often dropped or de-emphasized elsewhere (e.g., §3.1: "we conclude, in the low SNR regime, that under our sufficient condition, one high-quality sample can be replaced by up to two low-quality samples" — the "under our sufficient condition" caveat appears sporadically). More critically, the contrast between "agnostic γ ≤ 2" and "informed γ → ∞" is presented as exposing "a fundamental difference" (abstract, conclusion) between settings, but the agnostic side is derived from a loose bound while the informed side is from a tighter (though still only sufficient) bound. This asymmetry in tightness makes the comparison interpretively precarious.

- **Comparison between agnostic and informed settings conflates estimator-specific properties with information-theoretic limits.** Theorem 1 analyzes a particular estimator (constrained least-squares over B_{p,s} in eq. 8), not the optimal agnostic decoder. No lower bound is provided showing no agnostic decoder can do better. Theorem 2 analyzes the MLE with known Σ, which is canonical for the informed model. The "Price of Quality" comparison between these two analyses thus reflects the difference between two estimators with different optimality status and two bounds of different tightness, not necessarily a structural property of agnostic versus informed knowledge. The paper's framing—particularly "This highlights a key practical implication" and "fundamental difference"—goes beyond what the results establish.

- **"Information-theoretic threshold" language is used for what are only sufficient conditions, without matching converses.** The introduction correctly notes that the homogeneous case features sharp information-theoretic and algorithmic thresholds (n_INF and n_ALG). In the heterogeneous case, however, Theorems 1 and 2 provide only sufficient conditions, with no impossibility results. The paper nonetheless uses "threshold" language (§1.2: "Sampling complexity of sparse recovery"; conclusion: "the informed information-theoretic threshold... is sharp"). The informed condition (16) may indeed be near-sharp—the Chernoff exponent is optimized exactly—but necessity is not proven, and calling it a "threshold" without proof of impossibility below it is misleading.

### Minor

- **LASSO result does not cover the informed setting.** The paper identifies the informed setting as practically important (multi-site trials, sensor networks) and shows the Price of Quality can be arbitrarily large there information-theoretically, but provides no algorithmic result for this case. Remark 4.2 explains the proof barrier (loss of Wishart structure with Σ⁻¹), but no alternative algorithm is proposed or conjectured.

- **Different signal assumptions across results impede direct comparison.** Theorems 1–2 assume binary β★ ∈ {0,1}^p; Theorem 3 assumes real-valued β★ with min |β★ᵢ| ≥ ρ and recovers signed support. While both are standard in their respective literatures and Remark 3.1 discusses the binary assumption, the "fundamental difference" between IT and algorithmic thresholds is being drawn across results under different assumptions.

- **No numerical validation.** Simulations confirming the phase transitions and testing the tightness of the sufficient conditions would be valuable, especially given the acknowledged looseness of the agnostic bound.

### Trivial

- **Remark 3.4 generalizations (eqs. 22–23) are stated without proof.** These are natural extensions but should either be proven or clearly marked as conjectures.

## Nice-to-Haves

- **Pursue a necessary condition for at least the informed IT setting.** Since the Chernoff exponent is optimized exactly there (Remark 3.3), a Fano-type converse may be achievable, which would make the Price of Quality interpretation rigorous in the informed case.
- **Investigate variance-weighted agnostic estimators.** The 1/Y²ᵢ reweighting idea in Remark 3.2 is interesting; even a partial analysis would clarify whether γ ≤ 2 is an artifact of the unweighted estimator.
- **Add simulations** showing the actual recovery boundary as a function of (n₁, n₂) versus the predicted sufficient conditions, to empirically assess tightness.

## Removed Points

- **"The agnostic estimator involves combinatorial optimization"** — This is a standard theoretical device (equivalent to the MLE in the homogeneous setting, analogous to brute-force in Gamarnik & Zadik 2022). The paper separately addresses tractability through the LASSO. Not a meaningful weakness.

- **"No algorithmic result for informed setting"** was listed as "minor" above rather than "major" because the paper explicitly scopes this out (Remark 4.2) and the LASSO result in the agnostic setting is a standalone contribution. Demanding both settings would be scope creep.

- **"Gaussian design assumption is restrictive"** — Standard in the sparse recovery literature (Wainwright 2009; Reeves et al. 2019; Gamarnik & Zadik 2022). Extending to sub-Gaussian or correlated designs is meaningful future work but not a core flaw.

- **"Two-source noise model is restrictive"** — The paper addresses this in Remark 3.4, generalizing to arbitrary invertible Σ. The two-source case is the natural starting point for studying the trade-off.

- **"No comparison of LASSO vs. informed weighted LASSO (no experiments)"** — This would strengthen the paper but is beyond its stated theoretical scope.

- **"The binary signal assumption is restrictive"** — Standard in the literature, and the paper discusses the reduction in Remark 3.1.

- **"How to select λ_p in practice when σ_avg is unknown"** — The agnostic setting assumes the decoder doesn't know per-sample variances, but the average noise level may still be estimable (e.g., from the residual). This is a practical consideration but not a theoretical flaw.

## Novel Insights

The most striking finding—that the LASSO threshold is completely invariant to noise heterogeneity, depending only on total sample size and average noise—creates a curious structural insight: at the algorithmic level for LASSO, it literally does not matter how your noise budget is distributed across samples. This is interesting precisely because it contrasts with the information-theoretic picture (even under the loose agnostic bound), where noise variance allocation matters. Whether this contrast is sharp or an artifact of the one-sided bounds remains the key open question raised by this work.

## Suggestions

- Clearly and consistently qualify the agnostic Price of Quality claim throughout the paper (not just the abstract) by emphasizing it is a property of the specific sufficient condition, not a proven information-theoretic limit.
- Pursue a converse for the informed condition (Theorem 2) to establish the Price of Quality as a true information-theoretic quantity in at least one setting.
- Add a figure or table summarizing which results are sufficient vs. necessary-and-sufficient, and which "thresholds" have matching impossibility results.

## Score and Decision

**Calibration.** I compared against:
- *How Much is a Noisy Image Worth?* (qZwtPEw2qN, avg 6.8, Accept Poster): Theory + experiments on mixed-quality data for diffusion models. Simpler theory but empirical validation. **Current paper has deeper theory but no experiments and overclaims.**
- *Provable weak-to-strong generalization* (4vzGQcVUG8, avg 6.5, Accept Poster): Theory-only paper with stylized model. **Current paper has comparable technical depth but broader scope.**
- *Statistical theory of data selection under weak supervision* (HhfcNgQn6p, avg 7.75, Accept Oral): Strong theory + experiments on data quality tradeoffs. **Current paper is weaker on empirical validation and tightness of bounds.**

The LASSO result (Theorem 3) is a solid, novel, and non-trivial contribution. The information-theoretic results are interesting but weakened by the sufficiency-only nature and the gap between what is proven and what is claimed. The overclaiming of the "Price of Quality" as a fundamental property rather than a bound-dependent coefficient is the main issue preventing a higher score. The paper is above the acceptance bar on the strength of the LASSO contribution and the overall framework, but not comfortably above.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>