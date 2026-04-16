Now I have a good understanding of the calibration papers and their scores. Let me now synthesize the final review.

Key calibration points:
- RJSD paper (WnqD3EiylC): Scores 5,6,5,6,3 → Reject. Very similar pattern: new divergence measure, marginal improvements over MMD, kernel normalization restrictions.
- Regularized MMD (yqaN7MfkFU): Scores 3,5,6,3,5 → Withdrawn/Reject. Marginal improvements, conditions on theoretical results.
- Kernel CI test (GPcSYm89wK): Scores 5,5,3,5 → Reject. Marginal power improvements, conditions for theoretical results, kernel selection challenges.
- Model Equality Testing (QCDdI7X3f9): Scores 8,6,6,6 → Accept Poster. Applied MMD testing, clear practical contribution.
- MMD-MP for MGT (3fEKavFsnv): Scores 6,6,8,6 → Accept Poster. Applied MMD variant, clear practical contribution.
- Deep Kernel Relative Test (z9j7wctoGV): Scores 6,6,6 → Accept Poster. Relative testing for MGT detection.

Now let me carefully analyze the paper under review and verify key claims.

Let me verify specific points from the harsh reviewer against the paper:

1. **"MMD is less informative" motivation**: The harsh critic argues this is structurally weak. Let me check - the paper's argument is that at fixed MMD value, different distribution pairs with different norms have different p-values in two-sample testing, and thus different "closeness levels." The critic argues this confuses effect size with noise level. The paper's Figure 1 argument is: at same MMD value, distributions with higher norms have lower standard deviations of the MMD estimator, leading to smaller p-values. So NAMMD = MMD/(4K - ||μ_P||² - ||μ_Q||²) captures this. This is actually a reasonable observation - the test power depends on both effect size and variance, and scaling by norms captures this. However, the critic's point is partially valid: this doesn't mean MMD is "less informative" as a metric - it means MMD alone doesn't determine test power, which is true by construction. The paper's contribution is identifying that including norm information in the statistic can improve power. The "less informative" framing is somewhat overclaiming.

2. **"Higher test power" claims**: Theorems 10 and 12. Let me check what they actually prove vs what's claimed.
- Thm 10: First part says if MMD rejects, NAMMD also rejects with high probability. Second part says there exist events where MMD fails but NAMMD succeeds, with probability ≥ 1/65. This is a weak result - it shows non-zero improvement, not uniform dominance. The claim "achieves higher test power" is overstated.
- Thm 12: Requires ||μ_{P1}|| + ||μ_{Q1}|| < ||μ_{P2}|| + ||μ_{Q2}||. This is a non-trivial condition. The claim "NAMMD achieves higher test power" is conditional.

3. **Canonne comparison**: The paper compares NAMMD-based closeness testing against Canonne's TV-based test on discrete distributions. These test different null hypotheses (NAMMD ≤ ε vs TV ≤ ε'). The comparison is indeed testing different hypotheses with different metrics, though both are framed as "distribution closeness testing."

4. **Empirical improvements in Table 1**: Looking at the actual numbers - e.g., blob/Gaussian: 0.600 vs 0.616 (within one std dev of 0.090). These improvements are very small relative to variance.

5. **Comparison fairness in Figure 2**: The paper states "we set the test sample size for NAMMDFuse, MMDFuse, MMDAgg, and ACTT to be twice that of other methods, as these methods do not require training for kernel selection." This is mentioned, but it means the comparison isn't purely about sample efficiency - some methods get double the test samples.

6. **The NAMMD denominator**: 4K - ||μ_P||² - ||μ_Q||². The paper doesn't prove this is optimal in any sense. Could other normalizations work as well or better? No ablation.

7. **Kernel form restriction**: κ(x,x') = Ψ(x-x') ≤ K, Ψ(0) = K. This is a real restriction. Deep kernels as used in experiments may not satisfy translation invariance κ(x,x')=Ψ(x-x'), though the paper claims they do. Actually, many deep kernels are not translation-invariant.

Let me now write the final review.</think>## Summary

The paper proposes *norm-adaptive maximum mean discrepancy* (NAMMD), a modification of MMD that scales by the RKHS norms of the distribution embeddings: NAMMD(P,Q,κ) = MMD²(P,Q,κ) / (4K − ‖μ_P‖² − ‖μ_Q‖²). The motivation is that MMD can be "less informative" when comparing closeness across multiple distribution pairs sharing the same MMD value but with different norms (and thus different test power characteristics). The authors use NAMMD to formulate kernel-based distribution closeness testing (DCT), derive asymptotic distributions, prove Type-I error control and sample complexity bounds, and show conditions under which NAMMD has higher test power than MMD. Experiments cover two-sample testing, comparisons with Canonne's TV-based DCT, and three case studies on ImageNet variants and adversarial perturbation detection.

## Strengths

- **Novel and natural problem formulation.** Extending distribution closeness testing from TV on discrete domains to kernel methods on complex data fills a recognized gap in the literature. The observation that MMD does not account for norm-dependent variance when comparing closeness levels across distribution pairs is legitimate and motivates a genuine research direction.

- **Comprehensive theoretical framework.** The paper derives asymptotic distributions for the NAMMD estimator (Theorem 2), establishes concentration bounds (Lemma 6–7) and sample complexity (Theorem 8), proves Type-I error control (Theorem 5), and provides power comparisons with MMD (Theorems 10 and 12). This is a thorough theoretical treatment that matches the standard of the kernel testing literature.

- **Creative practical applications.** The case studies evaluating model performance across ImageNet variants without labels (Section 5.2) demonstrate a genuinely useful application of DCT. The connection between NAMMD closeness and accuracy/confidence margins is an interesting direction for label-free domain shift assessment.

- **Consistent empirical improvements over MMD.** Across all 20 cells in Table 1 (5 datasets × 4 kernels), NAMMD never underperforms MMD, and the same holds in the DCT experiments (Table 2). While gains are often modest, this consistency is noteworthy.

## Weaknesses

### Major:

- **The "higher test power" claim overstates what the theory proves.** Theorems 10 and 12 are the paper's central theoretical results, but they establish conditional and weak forms of superiority rather than the uniform dominance claimed. Theorem 10 shows: (i) if MMD rejects, NAMMD also rejects with high probability; (ii) there exists a probability ≥ 1/65 that MMD fails while NAMMD succeeds. Part (ii) only guarantees *some non-zero additional power in some region*, not that NAMMD uniformly dominates MMD. Theorem 12 adds the condition ‖μ_{P1}‖ + ‖μ_{Q1}‖ < ‖μ_{P2}‖ + ‖μ_{Q2}‖, which is not universally satisfied. The claim that NAMMD "achieves higher test power" (abstract, Section 1, Theorem headings) should be substantially qualified — the paper shows NAMMD is *never worse* and *sometimes better*, under specific conditions, which is meaningfully weaker than "higher test power."

- **Empirical improvements in two-sample testing are marginal and often within noise.** In Table 1, improvements over MMD with the same kernel are consistently tiny relative to standard deviations (e.g., blob/Gaussian: 0.600→0.616 with std ~0.090; mnist/Laplace: 0.873→0.881 with std ~0.016). No statistical significance tests (e.g., paired t-tests) are reported. The average improvement across all 20 cells is approximately 0.007, well within typical variance. This raises questions about whether the theoretical power advantages materialize meaningfully in standard two-sample testing, which is the setting of Theorem 10. The NAMMDFuse results (Figure 2) are more favorable but use twice the test samples for some methods, making direct comparison difficult (see below).

- **The comparison with Canonne's TV-based DCT is methodologically mismatched.** In Table 2, NAMMD (testing H₀: NAMMD ≤ ε via a kernel method on Mahalanobis kernel) and Canonne's test (testing H₀: TV ≤ ε' via occurrence counts) are testing *different null hypotheses measured on different metrics*. NAMMD can leverage the kernel's smoothing over the 50-element support, while the count-based TV estimator suffers from high variance at limited sample sizes. A fair comparison would require both methods targeting the same metric, or careful analysis showing NAMMD can serve as a proxy for TV closeness testing. As presented, the comparison mainly shows that kernel methods regularize better than naive count estimates on structured discrete data — not that NAMMD is superior for closeness testing per se.

- **The denominator 4K − ‖μ_P‖² − ‖μ_Q‖² is introduced heuristically without optimality justification.** The Remark after Definition 1 provides intuition ("we separate more effectively at same MMD distance with larger norms"), but the paper does not establish whether this specific normalization is optimal in any sense (e.g., maximizes Pitman efficiency, minimizes asymptotic variance). Natural alternatives — such as normalizing MMD by its estimated standard deviation, or using MMD/(2K) — are not discussed or compared. Given that the entire contribution hinges on this denominator choice, its lack of justification is a significant gap.

### Minor:

- **The restriction to kernels κ(x,x') = Ψ(x−x') ≤ K with Ψ(0) = K** excludes polynomial kernels and general deep kernels that lack translation invariance, yet the paper uses "deep kernels" in experiments. The relationship between deep kernels and the required kernel form κ(x,x')=Ψ(x−x') is not clearly established — many deep kernels do not satisfy translation invariance. This limitation is only addressed in Appendix C.4, which is easy to miss.

- **Experimental fairness in Figure 2:** The paper gives NAMMDFuse/MMDFuse/MMDAgg/ACTT twice the test sample size compared to methods that use training samples for kernel selection. While this practice is explained, it means the "test power vs. sample size" curves are not comparing methods at the same total data budget. A fairness adjustment (e.g., counting training samples toward total budget) would make the comparison more informative.

- **The condition ‖μ_{P1}‖ + ‖μ_{Q1}‖ < ‖μ_{P2}‖ + ‖μ_{Q2}‖ in Theorem 12** is stated to be "often met" because norms correlate with MMD values. This claim is plausible but not rigorously justified or empirically verified. No experiments specifically test the DCT scenario when this condition is violated.

### Trivial:

- The expression for σ²_{P,Q} in Theorem 2 contains a square root nested inside the variance formula, which is unusual for variance notation and may be a typographical issue.

## Nice-to-Haves

- Compare NAMMD against the standard *studentized MMD statistic* (MMD divided by its estimated null standard deviation), which naturally accounts for variance and is a widely used normalized baseline. This would clarify whether the norm-based denominator specifically helps, versus any reasonable normalization.

- Report paired significance tests (e.g., paired permutation tests on test power estimates) for Table 1 to confirm whether improvements are statistically meaningful.

- Test NAMMD in scenarios violating the norm condition of Theorem 12 to characterize when it may underperform or degrade relative to standard MMD DCT.

- Add a brief discussion or experiment on numerical stability when denominator 4K − ‖μ_P‖² − ‖μ_Q‖² approaches zero (highly concentrated distributions near the kernel bandwidth).

## Removed Points

- **"MMD is less informative as a closeness measure" is a fundamental conceptual flaw.** The harsh reviewer argues that MMD being the same for pairs with different norms is "not a defect" but simply means MMD measures distance, not p-values. However, the paper's argument is more nuanced: it observes that the *testability* — i.e., whether a closeness test can detect that two pairs differ in their distance — depends on both MMD and norm information. The paper's framing (Figure 1c) that p-values differ for the same MMD is a valid empirical observation about test power, even if saying MMD is "less informative" slightly overclaims. The underlying insight — that norm information can improve test power — is sound, so this is primarily a framing issue, not a fatal conceptual flaw.

- **Comparison with Canonne's test should be dismissed because the hypotheses are different.** While the hypotheses differ (NAMMD vs TV), this is inherent to comparing kernel-based vs TV-based closeness testing, which is exactly the paper's contribution space. The comparison is fair in the sense that both methods aim to solve the same practical problem (is distribution pair 2 farther than pair 1?), just with different metrics. The weakness is already captured above: the comparison should be more transparent about this distinction.

- **The ImageNet experiments are "only loosely validated surrogates" for model performance.** While the connection between NAMMD distance and accuracy margins is more illustrative than rigorous, these are presented as case studies, not as rigorous evaluations of a label-free model performance predictor. This is a reasonable scope for the paper.

- **Demand for other normalized MMD baselines.** While this would strengthen the paper, it is a request for additional experiments rather than a flaw in what is presented.

## Novel Insights

The paper identifies a real and previously unnoticed gap: standard MMD treats all pairs with the same MMD value as equally distinguishable, but in testing practice, pairs with higher RKHS norms (more concentrated distributions) yield more powerful tests. While the specific normalization chosen (4K − ‖μ_P‖² − ‖μ_Q‖²) lacks optimality justification, the insight that incorporating norm information into the test statistic can improve power in closeness testing is genuinely novel. The application to domain shift assessment without labels using closeness testing is also a creative contribution that bridges theory and practice, though it requires more rigorous validation.

## Suggestions

- Qualify the "higher test power" claim throughout the paper to accurately reflect what the theorems prove: NAMMD is never worse than MMD under the same kernel, and can provide strictly higher power under certain norm conditions, rather than making a blanket superiority claim.

- Add one additional baseline: MMD normalized by its estimated standard deviation (the standard studentized MMD test). This is the most natural alternative and would clarify whether the improvement comes from norm-adaptation specifically or from normalization generally.

- Provide statistical significance tests for Table 1, and explicitly discuss the small magnitude of improvements in two-sample testing relative to variance.

- For the DCT comparison (Table 2), add a column or discussion showing what happens when both NAMMD and Canonne's test target the same metric, to make the comparison interpretable.

## Score and Decision

Calibration against similar papers:
- RJSD (WnqD3EiylC): New kernel divergence with marginal improvements over MMD, kernel normalization restrictions, scores 5,6,5,6,3 → Reject. NAMMD is conceptually cleaner than RJSD and has a better-motivated specific application (DCT), but shares the pattern of marginal empirical gains and restricted kernel class.
- Regularized MMD (yqaN7MfkFU): MMD variant with conditions and marginal improvements, scores 3,5,6,3,5 → Reject. NAMMD has stronger theory but similarly marginal empirical gains.
- Kernel CI test (GPcSYm89wK): Kernel testing with marginal power improvements, scores 5,5,3,5 → Reject. Similar profile to NAMMD.
- Model Equality Testing (QCDdI7X3f9): Applied MMD testing with clear practical contribution, scores 8,6,6,6 → Accept Poster. NAMMD aims for a more theoretical contribution but the practical gains are less clear-cut.

NAMMD sits between these: it has a legitimate theoretical contribution (extending DCT to kernels, deriving proper asymptotic theory), a genuine insight (norm information improves closeness testing), and interesting practical applications. However, the overclaiming of uniform power superiority, the marginal empirical improvements in the core two-sample setting, the heuristic denominator choice, and the mismatched Canonne comparison are significant weaknesses. The paper is above the clearly rejected papers (GPcSYm89wK, yqaN7MfkFU) because it has a more coherent theoretical story and a genuinely novel application, but below the accepted applied MMD papers (QCDdI7X3f9, 3fEKavFsnv) which have clearer practical impact and more substantial empirical gains. Its profile is closest to RJSD (WnqD3EiylC), but with somewhat stronger theory and a better-motivated application domain.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>