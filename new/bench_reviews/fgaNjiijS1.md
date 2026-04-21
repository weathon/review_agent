Now I have a thorough understanding of the paper and the calibration anchors. Let me write the consolidated review.

## Summary

The paper proposes Norm-Adaptive MMD (NAMMD), a variance-normalized variant of Maximum Mean Discrepancy, for kernel-based distribution closeness testing (DCT). NAMMD rescales MMD² by $1/(4K - \|\mu_\mathbb{P}\|^2 - \|\mu_\mathbb{Q}\|^2)$, which increases with the RKHS norms of the two distributions, yielding a statistic in [0,1] that improves test power when distributions are more concentrated. The paper provides a complete theoretical analysis including asymptotic distributions, Type-I error control, sample complexity bounds, and formal power comparison theorems showing NAMMD dominates MMD under certain conditions. Experiments compare NAMMD against MMD on two-sample testing, against Canonne's TV-based test on DCT, and demonstrate practical case studies on ImageNet variants and adversarial perturbation detection.

## Strengths

- **Extends distribution closeness testing from discrete TV-based settings to kernel-based testing on complex data.** Prior DCT methods rely on total variation over discrete one-dimensional distributions (Section 1, background in Section 2). By introducing NAMMD with kernel mean embeddings, the paper enables DCT on continuous and high-dimensional domains, with experiments demonstrating this on CIFAR-10 and ImageNet variants (Table 2, Figure 3).

- **Complete theoretical package.** Theorem 2 provides asymptotic distributions (Gaussian for ε∈(0,1], weighted chi-squared for ε=0); Theorem 5 controls Type-I error at α; Theorem 8 gives sample complexity bounds scaling as 1/(NAMMD−ε)²; Lemma 6 provides concentration inequalities; Theorems 10 and 12 establish power dominance over MMD under specific conditions. This is a thorough and rigorous theoretical treatment.

- **Consistent empirical improvement over MMD across all 20 kernel×dataset combinations in Table 1.** While the improvements are small (discussed in weaknesses), NAMMD outperforms MMD systematically, never underperforming, which supports the theoretical predictions of Theorem 10.

- **Practical case studies with real-world relevance (Figures 3–5).** The three applications — evaluating ImageNet variant similarity without labels (Figure 3), confidence margin assessment for domain adaptation (Figure 4), and adversarial perturbation level detection on CIFAR-10 (Figure 5) — demonstrate that NAMMD reflects the same closeness ordering as accuracy/confidence margins and outperforms MMD.

## Weaknesses

### Fatal

None.

### Major

- **Table 2 comparison with Canonne's TV-based test does not control for effect size differences across metrics.** The experiment constructs alternatives where TV$(\mathbb{P}_{50}, \mathbb{Q}_{50}^A) = \epsilon' + 0.2$ (a constant 0.2 gap from the null threshold in TV), but no control is placed on the NAMMD gap NAMMD$(\mathbb{P}_{50}, \mathbb{Q}_{50}^A) - \epsilon$. If the NAMMD effect size relative to its threshold is much larger than the TV effect size relative to its threshold, higher test power trivially follows — it reflects different relative effect sizes in different metrics, not methodological superiority. The paper should report the actual NAMMD distances for both null and alternative distributions to allow the reader to assess whether the comparison is fair. Without this, the large apparent advantage in Table 2 (e.g., .968 vs .856 on blob at ε'=0.1) is uninterpretable.

- **The core motivation that "MMD is less informative" mischaracterizes a well-understood statistical property as a deficiency of the distance metric.** MMD correctly quantifies the distance between distributions. The observation (Figure 1) that p-values differ across distribution pairs with the same MMD but different norms reflects the standard fact that test power depends on estimator variance (lower for more concentrated distributions). NAMMD effectively rescales MMD² by a quantity related to within-distribution spread ($4K - \|\mu_\mathbb{P}\|^2 - \|\mu_\mathbb{Q}\|^2$), making it a variance-normalized (signal-to-noise) statistic. This is a useful construction for testing, but it conflates inter-distribution distance with intra-distribution concentration — a fundamentally different semantic object from a metric. The paper's claim that MMD has a "limitation" (lines 41–44) is misleading; the correct claim should be that for *closeness testing purposes*, normalizing by variance improves test power. The current framing overclaims the contribution by casting a standard statistical observation as a novel discovery about MMD's inadequacy.

### Minor

- **Improvements over MMD in Table 1 are marginal relative to noise.** NAMMD's improvements are consistently 1–3 percentage points (e.g., .600→.616 on blob+Gaussian, .563→.566 on higgs+Gaussian) while standard deviations are 7–10 points. Though the consistency across all 20 settings provides some evidence, the paper would be strengthened by reporting statistical significance tests (e.g., paired t-tests) or confidence intervals on the test power differences.

- **The norm condition in Theorem 12 (∥μ_{P1}∥ + ∥μ_{Q1}∥ < ∥μ_{P2}∥ + ∥μ_{Q2}∥) is only heuristically justified.** The paper states this condition is "often met in practice as norms of mean embeddings are typically positively correlated with MMD value" (line 283), but provides neither proof nor empirical validation of this correlation. If norms decrease while MMD increases, NAMMD's denominator shrinks (reducing NAMMD), potentially working against the claimed power improvement. An ablation or analysis examining when this condition fails would strengthen the paper.

- **Theorem 10's quantitative bound on NAMMD outperforming MMD is very weak (probability ≥ 1/65).** While the first part of the theorem (NAMMD rejects whenever MMD does) is valuable, the second part's bound is too conservative to provide meaningful practical assurance. This is typical of such theoretical guarantees but should be acknowledged more explicitly.

- **Figure 2 confounds sample size with method.** NAMMDFuse, MMDFuse, MMDAgg, and ACTT use twice the test sample size because they don't require kernel training data (line 311). While the rationale is explained, the reader cannot disentangle how much of NAMMDFuse's advantage comes from the NAMMD normalization versus simply having more test samples. A comparison at equal effective test sample sizes would help isolate NAMMD's contribution.

### Trivial

- None.

## Nice-to-Haves

- Report NAMMD distances alongside TV distances in Table 2 to address the effect size concern.
- Provide scatter plots of NAMMD vs. MMD test power across many distribution pairs to reveal when NAMMD wins, ties, or loses vs. MMD.
- Discuss connections to known variance-stabilized MMD estimators or energy distance normalizations in the related work.
- Explicitly discuss settings where NAMMD would be expected to underperform MMD (violating the norm condition of Theorem 12), to set realistic expectations.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic's claim that the theoretical analysis uses asymptotic thresholds but experiments use permutation tests as a weakness.** This is standard practice in kernel hypothesis testing — permutation tests are the norm for two-sample testing, and the asymptotic theory establishes the validity of the statistic. The paper explicitly uses permutation tests for ε=0 and asymptotic thresholds for ε>0 (lines 147, 141). Not a real weakness.

- **Harsh critic's concern about the variance estimator σ_{X,Y} complexity and lack of finite-sample validation.** The paper provides Lemma 4 showing $|E[\sigma_{X,Y}^2] - \sigma_{\mathbb{P},\mathbb{Q}}^2| = O(1/\sqrt{m})$, a standard quality guarantee for variance estimators. Requesting additional finite-sample validation beyond this is a nitpick.

- **Strength finder's claim that Figure 1d "provides direct evidence that MMD is less informative."** This conflicts with the verified major weakness that the "less informative" framing mischaracterizes a variance effect as a distance metric deficiency. Since the weakness is verified, this strength is dropped.

- **Harsh critic's point about the NAMMD ∈ [0,1] claim requiring ⟨μ_P, μ_Q⟩ ≥ 0.** This is implicitly satisfied for non-negative kernels, which is the class the paper considers (κ(x,x') = Ψ(x-x') ≤ K with positive-definite Ψ). This is a minor technical observation, not a weakness.

- **Harsh critic's point about Figure 4 setup naturally favoring NAMMD.** This is speculative — the confidence margin setup involves testing whether distribution pairs with larger margins exceed the reference threshold, but there is no inherent reason this would systematically favor NAMMD's normalization. Removed as insufficiently grounded.

- **Missing related works concerns.** Not verifiable without external sources.

## Novel Insights

The paper reveals an interesting conceptual tension in its construction: NAMMD is simultaneously presented as a new "distance" for distribution closeness (with a [0,1] range like a proper metric) and as a variance-normalized test statistic (signal-to-noise ratio). These are semantically different objects — a distance that scales with intra-distribution concentration is not a metric in the traditional sense, as it can rank distribution pairs differently than their actual distributional discrepancy. The paper's strongest contribution is extending DCT to kernel settings; NAMMD itself is best understood as a test-power-maximizing statistic for a specific class of testing problems, not as a more "informative" distance measure.

## Suggestions

- In Table 2, report the NAMMD distances for both null (NAMMD(P₅₀, Q₅₀)) and alternative (NAMMD(P₅₀, Q₅₀^A)) distributions alongside the TV distances, so readers can assess whether the effect sizes are proportionally comparable.
- Reframe the introduction: instead of claiming MMD is "less informative," state that for DCT purposes, test power depends on both the distance and the distributional concentration, and NAMMD explicitly incorporates both.
- Report p-values or confidence intervals on the test power differences in Table 1, given the large standard deviations relative to the improvement magnitudes.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| MMD Graph Kernel (GZ6AcZwA8r) | 7.50 | More substantial novelty (deep learnable graph kernels) with strong empirical gains. This paper is clearly below this. |
| Deep Kernel Relative Test (z9j7wctoGV) | 6.00 | Similar profile: extends kernel testing to new setting, moderate empirical gains, solid but incremental. This paper is somewhat below it due to the fairness issue in Table 2 and overclaiming. |
| Practical Kernel Learning for CI Testing (GPcSYm89wK) | 4.50 | Marginal power gains with limited evaluation. This paper has more complete theory and more diverse experiments, placing it above. |
| Shapley Value k-Additive (lLzeKG6t52) | 4.00 | Theory without sufficient empirical validation. This paper has better empirics. |
| LLM Stock Prediction (ICwdNpmu2d) | 1.50 | Fundamentally flawed. Not comparable. |

This paper sits between the GPcSYm89wK kernel testing paper (4.50, rejected for marginal gains) and the Deep Kernel Relative Test (6.00, accepted poster). It is stronger than the former due to more complete theory and practical case studies, but weaker than the latter due to the Table 2 fairness concern, overclaiming in motivation, and marginal MMD improvements. The extension of DCT to kernels is a genuine and useful contribution with complete theoretical backing, but NAMMD itself is a simple rescaling whose practical advantages are not yet convincingly demonstrated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>