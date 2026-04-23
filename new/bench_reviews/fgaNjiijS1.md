## Summary

The paper proposes Norm-Adaptive MMD (NAMMD), which normalizes MMD² by (4K − ‖μ_P‖² − ‖μ_Q‖²) to create a closeness measure in [0,1] that scales with RKHS norms of distributions. This is motivated by the observation that distribution pairs with the same MMD value but different norms yield different p-values in testing, and NAMMD better correlates with these p-values. The paper extends distribution closeness testing (DCT) from discrete TV-based methods to complex data via kernel methods, provides theoretical analysis (asymptotic distributions, sample complexity, Type-I error control, power comparisons with MMD), and demonstrates practical utility on ImageNet variant assessment, domain adaptation confidence margins, and adversarial perturbation detection.

## Strengths

- **Extends DCT to complex data via kernels**: The paper identifies a genuine gap—existing DCT methods use TV on discrete 1D distributions—and proposes a kernel-based solution applicable to images and other complex data. This is a meaningful and underexplored direction (Sections 1–2, Table 2, Figures 3–5).

- **Clean and interpretable NAMMD definition**: Definition 1 normalizes MMD² by the "unexplained variance" term 4K − ‖μ_P‖² − ‖μ_Q‖², keeping the statistic in [0,1] and making it increase with norms. The formula has a natural interpretation as capturing how effectively two distributions are separated relative to their spread (Definition 1, Remark, Section 3).

- **NAMMDFuse consistently outperforms MMDFuse and other SOTA methods**: Figure 2 shows NAMMDFuse (which replaces MMD with NAMMD in the MMDFuse fusion framework) achieves higher test power across blob, higgs, hdgm, and mnist datasets. This is the strongest evidence for NAMMD's specific contribution, since NAMMDFuse and MMDFuse share the same fusion framework and differ only in the base statistic.

- **Compelling practical case studies**: Section 5.2 demonstrates real utility through ImageNet variant similarity assessment (Figure 3), confidence margin evaluation for domain adaptation (Figure 4), and adversarial perturbation detection (Figure 5). The NAMMD distance correctly reflects closeness relationships consistent with ground-truth accuracy margins.

- **Comprehensive theoretical framework**: The paper provides asymptotic distribution analysis (Theorem 2), Type-I error control (Theorem 5), concentration inequalities (Lemmas 6–7), and sample complexity bounds (Theorem 8), all within a coherent framework.

## Weaknesses

### Fatal

None.

### Major

- **Theoretical guarantee for NAMMD's power advantage over MMD is essentially vacuous**: Theorems 10 and 12 establish that NAMMD has "higher test power" than MMD, but the key result—that NAMMD can reject when MMD does not—holds only with probability ≥ 1/65. This is an existence result showing the possibility of improvement, not a practical guarantee. The required sample size constant C' is distribution-dependent and unquantified (lines 245–249, 277–281). A 1/65 probability bound provides almost no assurance that NAMMD will meaningfully outperform MMD in practice.

- **Empirical improvements over MMD in two-sample testing are marginal and lack significance tests**: Table 1 shows NAMMD outperforming MMD in all 20 comparisons, but most differences are far within one standard deviation (e.g., higgs/Gaussian: 0.566 ± 0.075 vs 0.563 ± 0.073; cifar10/Gaussian: 0.222 ± 0.020 vs 0.219 ± 0.017). The average improvement across datasets is ~1% of test power for most kernels. While the consistency is notable, no statistical significance tests are reported, and the differences are too small to establish meaningful superiority in this setting. This directly undermines the paper's central claim of "higher test power" for two-sample testing.

- **Comparison with Canonne's test conflates kernel advantage with NAMMD-specific advantage**: Table 2 compares NAMMD (with Mahalanobis kernel) against Canonne's occurrence-based test on 50-element discrete distributions. The advantage partly comes from using a kernel at all—which captures geometric structure—rather than from the NAMMD normalization specifically. The paper states "NAMMD for distribution closeness testing achieves better performances than Canonne's test" (Section 5.2), but this overstates the conclusion. A fairer comparison would include an MMD-based DCT with the same kernel, testing whether MMD(P₂,Q₂) > MMD(P₁,Q₁), to isolate the effect of NAMMD's normalization from the effect of using a kernel.

### Minor

- **Theorem 12's norm condition is asserted without rigorous justification**: The condition ‖μ_{P₁}‖ + ‖μ_{Q₁}‖ < ‖μ_{P₂}‖ + ‖μ_{Q₂}‖ is stated to be "often met in practice as norms of mean embeddings are typically positively correlated with MMD value" (line 283). However, this correlation is not guaranteed: pairs of broad, overlapping distributions can have small norms while being far apart in MMD, violating the condition. The scope of Theorem 12 is narrower than claimed, and the failure mode (where NAMMD could have *lower* power than MMD) is not analyzed.

- **Type-I error for ε > 0 with plug-in variance estimator needs empirical verification in the main paper**: For DCT with ε > 0, the threshold τ_α (Eq. 2) uses the empirical variance σ_{X,Y} as a plug-in for the true σ_{P,Q}. Lemma 4 only provides a bias bound |E[σ²_{X,Y}] − σ²_{P,Q}| = O(1/√m), not a high-probability concentration bound. The Type-I error experiments are relegated to the appendix (Section D.6, not in the main paper), making it impossible to verify whether the Gaussian threshold with plug-in variance properly controls Type-I error.

- **Motivation framing is imprecise**: The paper claims MMD is "less informative" because "MMD value can be the same for many pairs of distributions that have different norms" (Abstract, Section 1). This conflates the distance metric (which correctly gives the same value for equidistant pairs) with test power (which varies with norms). Figure 1c already shows that the MMD *test* accounts for norms through the estimator's variance. The actual contribution is a new test statistic that re-encodes the signal-to-noise ratio into the distance itself, not a fix for a deficiency in MMD as a distance metric. More precise framing would strengthen the paper.

## Trivial

- The σ²_{P,Q} notation in Theorem 2 may have a dimensional inconsistency (the formula appears to give a standard deviation divided by a constant, not a variance), but this could be a PDF extraction artifact and is difficult to verify without the original.

## Nice-to-Haves

- Include significance tests (e.g., paired t-tests) for the Table 1 power comparisons to properly assess whether NAMMD's improvement over MMD is statistically significant.
- Add a comparison against an MMD-based DCT with the same kernel (testing MMD(P₂,Q₂) > MMD(P₁,Q₁)) to isolate the NAMMD normalization effect from the kernel effect in the DCT setting.
- Analyze the failure mode: when Theorem 12's norm condition is violated, how much does NAMMD lose relative to MMD?
- Add error bars or confidence intervals to Figures 3–5 for the DCT case studies.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Denominator approaches zero causing instability"**: The critic claimed the denominator 4K − ‖μ_P‖² − ‖μ_Q‖² approaches zero when distributions are near-point-masses. This is incorrect: since 0 ≤ ‖μ_P‖² ≤ K and 0 ≤ ‖μ_Q‖² ≤ K, the denominator is always ≥ 2K > 0. The theoretical NAMMD is always well-defined.

- **"Variance formula dimensional inconsistency"**: While the σ²_{P,Q} formula in Theorem 2 looks potentially inconsistent (square root in numerator rather than full variance expression), this is likely a PDF extraction artifact. Cannot verify without the original submission.

- **"Canonne comparison tests different null hypotheses"**: While technically true (TV ≤ ε' vs NAMMD ≤ ε), the practical comparison is fair in the sense that both tests address the same underlying question (is the test pair further from the reference pair?) on the same data distributions. The real issue (kept above) is that the advantage comes partly from using a kernel, not from NAMMD normalization specifically.

- **"Demanding missing related works"**: Not verifiable without external sources.

- **"NAMMD and MMD have the same test power estimator"**: The paper itself acknowledges this in Section 4.2 (line 251), noting that for two fixed distributions, NAMMD is just MMD scaled by a constant. This is not a weakness but a property the paper already discusses.

## Novel Insights

The paper reveals an interesting structural insight: normalizing MMD² by the "unexplained variance" 4K − ‖μ_P‖² − ‖μ_Q‖² = 4Var(P,κ) + 4Var(Q,κ) − 2MMD²(P,Q) effectively converts MMD from an absolute distance into a signal-to-noise-like ratio. This is why NAMMD better correlates with p-values and can achieve higher test power—especially in the DCT setting where you compare relative distances across pairs with different norms. However, the empirical evidence shows this conversion yields only marginal improvements in standard two-sample testing (Table 1), suggesting the benefit is primarily relevant when comparing distances across distribution pairs with differing norms (the DCT setting), rather than for a single pair.

## Suggestions

- Report paired statistical significance tests for Table 1; if the improvements are not significant, this should be acknowledged rather than overstated.
- Add an MMD-based DCT baseline (same kernel, testing MMD(P₂,Q₂) > MMD(P₁,Q₁)) to Table 2 or the DCT experiments to isolate NAMMD's specific contribution.
- Provide a concrete example or quantitative analysis of when Theorem 12's norm condition holds vs. fails, including empirical demonstration of NAMMD's behavior in the failure regime.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| GZ6AcZwA8r (MMD Graph Kernel) | 7.5 | Stronger: solid novel kernel framework with impressive empirical results and clear theoretical contributions. This paper's contributions are narrower and less well-supported. |
| z9j7wctoGV (Deep Kernel Relative Test) | 6.0 | Comparable: both extend kernel testing to practical settings. That paper had stronger practical motivation and clearer contribution, despite concerns about assumptions. This paper's improvements are more marginal. |
| yqaN7MfkFU (Regularized MMD) | 4.4 | Weaker: that paper had serious theoretical gaps. This paper has a cleaner formulation and more comprehensive framework, placing it above. |
| Hh0Cg4epYY (Bayes Error Bounds) | 2.33 | Much weaker: fundamentally flawed theory. This paper is coherent and well-organized, well above this level. |
| PPxyXlCAOJ (Learning Representations for Independence Testing) | 5.5 | Similar: both propose modifications to kernel test statistics. That paper was rejected despite reasonable ideas due to concerns about the evidence for improvement. |

The paper proposes a reasonable and interpretable extension of MMD for distribution closeness testing, with a comprehensive theoretical framework and interesting practical applications. However, the central claim of "higher test power" is supported by weak theoretical guarantees (1/65 probability bound) and marginal empirical improvements in the most direct comparison (Table 1). The more convincing evidence comes from the NAMMDFuse integration and DCT case studies, but even these have limitations (no error bars, no fair MMD-based DCT baseline). The paper extends DCT to complex data, which is a genuine contribution direction, but overclaims the evidence for NAMMD's advantage over MMD specifically. Relative to the calibration anchors, this paper falls between the regularized MMD paper (4.4) and the deep kernel relative test paper (6.0), closer to the middle.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>