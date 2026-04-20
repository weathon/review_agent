## Summary
The paper introduces NAMMD (Norm-Adaptive MMD), a scaled version of the Maximum Mean Discrepancy that divides by $4K - \|\mu_P\|^2 - \|\mu_Q\|^2$, to address the claim that MMD is "less informative" when comparing closeness of distribution pairs with different RKHS norms. The paper provides theoretical guarantees (asymptotic normality, Type-I error control, sample complexity bounds, and comparative power theorems) and experiments on two-sample testing, synthetic DCT, and case studies involving ImageNet variants, confidence margins, and adversarial perturbation detection.

## Strengths

- **Complete theoretical package for the proposed statistic.** The paper derives asymptotic behavior of the estimator (Theorem 2), Type-I error guarantees (Theorem 5), concentration bounds (Lemmas 6-7), and sample complexity bounds (Theorem 8). These provide a formal foundation for using NAMMD in kernel-based closeness testing.

- **NAMMD provably achieves higher test power than MMD under the same kernel.** Theorem 10 formally shows that when MMD correctly rejects the null, NAMMD also rejects, and there exist cases (with probability ≥ 1/65) where NAMMD rejects while MMD does not. Table 1 confirms this with NAMMD uniformly outperforming MMD across all 20 kernel×dataset combinations, including Gaussian, Laplace, Mahalanobis, and Deep kernels.

- **NAMMD extends kernel-based DCT to complex data beyond discrete settings.** Unlike prior DCT methods limited to total variation on discrete one-dimensional distributions, NAMMD applies to continuous and image data (CIFAR-10, MNIST, ImageNet variants), as demonstrated in Section 5.2 and Figures 3–5.

- **Figure 1 provides an effective visual motivation.** Panels (c) and (d) illustrate that when MMD is held constant at 0.15, NAMMD increases monotonically with norms, matching the p-value trend — making the norm-scaling intuition accessible.

## Weaknesses

### Fatal
None.

### Major

- **The core motivation conflates finite-sample estimator variance with population-level distributional distance.** (Section 1, Figure 1). The paper claims MMD is "less informative" because two distribution pairs with the same MMD value but different RKHS norms yield different empirical p-values. However, this is a standard property: tighter distributions naturally produce lower-variance estimators, leading to smaller p-values for the same sample size. This is a feature of hypothesis testing, not a deficiency of MMD as a metric. NAMMD's scaling does not resolve a genuine population-level identifiability issue; it reweights the statistic based on empirical concentration. The theoretical advantage in Theorems 10 and 12 does not contradict this — it shows improved finite-sample power, but this is an empirical phenomenon rather than a fundamental metric improvement. The motivation, as presented, mischaracterizes the statistical phenomenon.

- **The plug-in $\epsilon$ estimation procedure breaks the theoretical Type-I error guarantees.** (Section 3, "Performing Distribution Closeness Testing in Practice"). The paper advocates setting $\epsilon = \text{NAMMD}(\mathbb{P}_1, \mathbb{Q}_1, \kappa)$ estimated from a reference distribution pair. However, Theorem 5's Type-I error bound assumes $\epsilon$ is a fixed, known constant. When $\epsilon$ is estimated from finite samples, its variance propagates into the decision threshold, potentially inflating the actual false-positive rate beyond the nominal $\alpha$. The paper provides no theoretical correction (e.g., via union bounds, simultaneous inference, or conformal calibration) for this estimation error. This means the theoretical guarantees do not apply to the exact testing procedure the authors recommend for practice.

### Minor

- **Empirical gains on two-sample testing are marginal and statistically unsubstantiated.** In Table 1, the improvements of NAMMD over MMD are consistently small — often differences of 0.003–0.006 (e.g., 0.689 vs. 0.692 for Deep kernel on average). These differences are within one standard deviation of the reported values (e.g., $\pm 0.072$), suggesting they may be within stochastic variation. No paired statistical significance tests or confidence intervals are provided across the 10 runs, which weakens the claim of consistent superiority. While NAMMD does strictly dominate across all 20 combinations, the effect sizes are small enough that the practical significance is unclear.

- **No DCT experiment with known ground-truth distances on continuous or high-dimensional data.** Table 2 tests on synthetic discrete distributions with 50 support points, which does not validate the method's ability to perform DCT on complex data — the very motivation for extending DCT beyond TV-based methods. The case studies (Figures 3–5) use accuracy/confidence margins as proxies for ground-truth $\epsilon$, which measure classifier robustness, not statistical distributional distance. A controlled DCT benchmark with a known population distance (e.g., synthetic continuous distributions where NAMMD/MMD distances are analytically tractable) would be needed to validate the Type-I error control and power claims for $\epsilon > 0$ on complex data.

- **Theorem 12's finite-sample advantage for DCT depends on an unverified condition.** The condition $\|\mu_{P1}\| + \|\mu_{Q1}\| \leq \|\mu_{P2}\| + \|\mu_{Q2}\|$ restricts the theoretical advantage to cases where the test distributions have larger kernel mean embedding norms than the reference. The paper asserts this is "often met in practice" but provides no empirical evidence for when this holds across the case studies. This assumption effectively biases the claimed advantage.

### Trivial
None.

## Nice-to-Haves
- Analyze the sensitivity of NAMMD to kernel bandwidth and the denominator scaling term, especially when $4K - \|\mu_P\|^2 - \|\mu_Q\|^2$ approaches zero for highly concentrated distributions.
- Report empirical Type-I error rates for the plug-in $\epsilon$ procedure to assess whether calibration is maintained in practice.
- Compare NAMMD with a kernel-based DCT method rather than Canonne's TV test in Table 2 for a more distribution-class-consistent evaluation.

## Removed Points
These points are flagged to be removed; treat them with caution since they were removed per the hard rules.

- **Reviewer claiming MMD scaling is "arbitrary" or "does not correct a genuine metric deficiency."** While the motivation discussion (Major Weakness 1) notes the conflation issue, the specific scaling in Definition 1 is a concrete proposal that the paper analyzes extensively. Dismissing it as purely arbitrary without demonstrating a better alternative is a strawman.

- **Reviewer concern about denominator convergence ("not explicitly bounded relative to the numerator").** For characteristic kernels with $0 \leq \kappa(x,y) \leq K$, the denominator is bounded away from zero in practice, and the paper operates within this standard assumption.

- **Request for larger datasets or more models.** The paper already tests on 5 datasets (blob, higgs, hdgm, mnist, cifar10) with 4 kernel types, plus ImageNet case studies. This is adequate for the scope.

- **Concerns about reproducibility details (hyperparameters, training logs).** These are standard nitpick reproducibility concerns that should be removed per the hard rules.

- **Criticism about missing related work on "kernelized adaptations of tolerance regions."** Per the hard rules, do not mention missing related works.

## Novel Insights
The paper usefully reframes the role of RKHS norms in kernel-based hypothesis testing — not as a fundamental metric defect of MMD, but as a factor that influences finite-sample test power through estimator concentration. The NAMMD scaling captures this empirically: by reweighting MMD based on the norms of mean embeddings, the statistic becomes more sensitive to distributional differences in high-concentration regimes. This insight, while not fundamentally altering the population-level properties of MMD, provides a practically useful heuristic for improving test power when comparing distribution pairs under a fixed kernel.

## Suggestions
- Re-examine the framing in Section 1 and the introduction. Rather than claiming MMD is "less informative" as a distance metric, motivate NAMMD as a *variance-aware test statistic* that improves finite-sample power by accounting for estimator concentration. This more honestly positions the contribution.
- Add a section or explicit caveat discussing the gap between the oracle-$\epsilon$ theory (where Theorem 5 holds) and the plug-in $\epsilon$ procedure used in practice. Even a brief empirical study showing Type-I error under plug-in estimation would strengthen the paper significantly.
- Include a paired statistical test (e.g., Wilcoxon signed-rank or paired bootstrap) on Table 1 results to confirm that the marginal improvements, while small, are statistically significant across the 10 runs.

## Score and Decision
I calibrated against the following papers:

- **High anchors (7+):** GZ6AcZwA8r (MMD graph kernels, accepted spotlight, 8,8,6,8) — stronger empirical breakthrough and more impactful claims. QCDdI7X3f9 (Model Equality Testing, accepted poster, 8,6,6,6) — clearer problem framing and practical impact. Ip6UwB35uT (conditional hypothesis testing via conformal p-values, accepted poster, 6,6,8,8) — stronger finite-sample validity guarantees.

- **Medium anchors (5–6):** Pf85K2wtz8 (Deep MMD Gradient Flow, accepted poster, 6,6,5,6) — similar incrementality with competitive experiments on image generation. z9j7wctoGV (kernel relative test for text detection, accepted poster, 6,6,6) — straightforward method with clear experiments.

- **Low anchors (<5):** yqaN7MfkFU (Regularized MMD, withdrawn, 3,5,6,3,5) — theoretical questions and limited baselines led to rejection despite soundness.

This paper sits between the medium and high anchors. It has a complete theoretical package (like yqaN7MfkFU, but with stronger results), consistent but marginal empirical gains, and a motivation that partially mischaracterizes the statistical phenomenon. The method is incremental — scaling MMD by a known quantity — but it works consistently across benchmarks. The epsilon estimation gap and the lack of continuous DCT benchmarks with ground-truth are real weaknesses, but not fatal. The paper is comparable to Pf85K2wtz8 (6,6,5,6) in scope and quality but slightly weaker in empirical substantiation of claims. I score it at **5.5**, positioning it as a borderline poster.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>