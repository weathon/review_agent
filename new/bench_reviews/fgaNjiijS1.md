Now I have sufficient information. Let me write the consolidated final review.

---

## Summary

This paper proposes the Norm-Adaptive Maximum Mean Discrepancy (NAMMD), a new kernel-based statistic for Distribution Closeness Testing (DCT) on complex data. The core insight is that standard MMD is "less informative" when comparing closeness across multiple distribution pairs under a fixed kernel: at equal MMD values, pairs with larger RKHS norms have smaller estimation variance and therefore lower p-values (i.e., are statistically more distinguishable). NAMMD addresses this by scaling MMD by a factor involving the norms of the mean embeddings, yielding a statistic in [0,1] that correlates better with test power. The paper provides theoretical guarantees (asymptotic distributions, Type-I error control, sample complexity, power dominance theorems) and experiments on synthetic and image datasets.

---

## Strengths

- **Novel extension of DCT to complex data.** The existing DCT literature is confined to discrete 1D distributions with TV distance. This paper is the first to extend DCT to arbitrary complex distributions via kernel methods, a genuinely valuable contribution that opens a new application domain for hypothesis-testing tools.

- **Substantial theoretical framework.** The paper proves asymptotic normality of the NAMMD estimator (Theorem 2), Type-I error control (Theorem 5), large-deviation bounds and sample complexity (Lemmas 6–7, Theorem 8), and power dominance over MMD under both two-sample testing (Theorem 10) and DCT (Theorem 12). This is a non-trivial body of theory.

- **NAMMDFuse vs. SOTA two-sample tests (Figure 2).** The fused variant consistently achieves higher or comparable test power to competitive methods (MMDFuse, MMDAgg, MEMabid, MMD-D, ACTT, AutoML) across diverse benchmarks (blob, higgs, hdgm, mnist), with the advantage of requiring no training samples.

- **Intuitive and computationally lightweight modification.** NAMMD requires only computing norms of mean embeddings in addition to standard MMD, making it easy to drop into existing kernel-testing pipelines.

- **Practical case studies.** The three case studies (ImageNet variant ranking, confidence-margin based closeness, adversarial perturbation detection) demonstrate that NAMMD better aligns with known ground-truth orderings (accuracy margins) compared to MMD, and can do so without ground-truth labels.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Marginal empirical gains in the same-kernel setting (Table 1), without statistical significance evidence.** Table 1—the cleanest test of the core normalization idea—shows improvements of roughly 0.003–0.025 in test power (e.g., 0.563→0.566 for Gaussian/higgs, 0.332→0.334 for Deep/hdgm). These deltas are often within or near the reported standard deviations, and with only 10 repetitions per setting, no statistical significance test is reported (e.g., paired t-test or Wilcoxon). This matters because the theoretical claim is that NAMMD achieves *higher* test power than MMD under the same kernel; the empirical evidence is too weak to confirm a practically significant improvement in this regime. The gains in DCT and NAMMDFuse settings are larger and more compelling, but the same-kernel baseline comparison is the most direct test of the normalization mechanism.

- **Theorem 12 (DCT power advantage) depends on a restrictive and opaque condition.** The theorem requires $\|\mu_{\mathbb{P}_1}\| + \|\mu_{\mathbb{Q}_1}\| < \|\mu_{\mathbb{P}_2}\| + \|\mu_{\mathbb{Q}_2}\|$—i.e., the test pair must have larger norm-sum than the reference pair. The paper remarks this "is often met in practice as norms of mean embeddings are typically positively correlated with MMD value," but this is stated without justification or experimental verification. More importantly, the $\Delta \in (0, 1/2)$ condition depends on unknown population quantities and sample size, and the paper provides no interpretable sufficient conditions or discussion of how restrictive this is. The scope of the theoretical advantage for DCT is therefore unclear.

- **DCT comparison is against a single baseline using a different discrepancy notion (Table 2).** Canonne's test measures TV on discrete distributions, while NAMMD tests its own kernel-induced threshold. Although both are evaluated on the same constructed datasets (with TV parameterizing the distributions), the tests are not asking exactly the same statistical question. The paper's explanation for NAMMD's superior performance—"the kernel trick can effectively capture intrinsic structures and complex patterns"—is plausible, but since no kernel-based DCT competitor exists (this being the paper's own contribution), the comparison is inherently limited. The conclusion that NAMMD is a superior DCT method is only weakly supported by this one heterogeneous comparison.

### Minor

- **NAMMD is not verified to satisfy metric properties.** The paper refers to "NAMMD distance" throughout, but NAMMD is a ratio (a signal-to-noise-ratio-style quantity) and there is no proof or discussion of whether it satisfies the triangle inequality. Calling it a "distance" without establishing metric properties may mislead readers. Using "NAMMD statistic," "discrepancy," or "divergence" would be more precise.

- **No analysis of failure regimes.** Theorems 10 and 12 show NAMMD dominates under certain conditions, but no experiment or analysis examines when normalization hurts—e.g., when the denominator $4K - \|\mu_\mathbb{P}\|^2 - \|\mu_\mathbb{Q}\|^2$ is small, estimation noise in norms could cause large fluctuations in NAMMD. At least one synthetic example showing failure or near-failure would establish the method's boundaries.

- **Kernel selection for DCT with multiple pairs remains open.** The paper explicitly defers this ("an important future work") and uses a reference-pair-selected kernel for all comparisons. In settings where the reference pair is very different from the test pair, the selected kernel may be suboptimal, but this limitation is only acknowledged qualitatively.

### Trivial

- The practical case studies (Section 5.2) are qualitative alignment checks (does NAMMD ordering match accuracy-margin ordering?), not a quantitative validation of the claim that passing/failing NAMMD predicts acceptable performance drop. These are described as "case studies" and are appropriately framed as such, but the introduction's stronger language ("help us decide if we really need to adapt a model") slightly overpromises.

---

## Nice-to-Haves

- **Statistical significance on Table 1 gains.** Even a simple paired t-test over the 10 runs per cell would clarify whether the same-kernel improvements are meaningful.
- **Empirical validation of Type-I error for the Gaussian threshold (Eq. 2) in finite samples.** The paper refers to Appendix D.6 for Type-I error experiments; making explicit that these cover the $\epsilon \in (0,1)$ regime (not just $\epsilon=0$) in the main text would strengthen credibility.
- **Empirical test of the norm-condition for Theorem 12.** Constructing a synthetic scenario where $\|\mu_{\mathbb{P}_1}\| + \|\mu_{\mathbb{Q}_1}\| > \|\mu_{\mathbb{P}_2}\| + \|\mu_{\mathbb{Q}_2}\|$ and reporting NAMMD vs. MMD performance there would clarify the scope of the theorem.
- **Comparison with at least one alternative normalization** (e.g., dividing by $\sqrt{\|\mu_\mathbb{P}\|^2 + \|\mu_\mathbb{Q}\|^2}$) to justify that the specific denominator $4K - \|\mu_\mathbb{P}\|^2 - \|\mu_\mathbb{Q}\|^2$ is the right design choice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic – "Central premise is a conceptual flaw; NAMMD measures detectability not closeness" (claimed as structural/fatal):**  
The paper explicitly frames DCT as a hypothesis testing problem and operationally *defines* closeness via NAMMD (Definition 3). The observation that at fixed MMD, higher-norm distributions are more statistically distinguishable (lower p-value, Figure 1c) is both correct and well-illustrated in the paper. The paper's claim that NAMMD is a *better test statistic for DCT* (not a purer geometric distance) is internally coherent. The critic's charge that this "conflates distance and test statistic" has merit as a terminological imprecision (see Minor weakness above), but does not rise to a structural/conceptual flaw that undermines the paper's core contribution. Removed as an overclaimed severity.

**Harsh Critic – "Table 2 comparison is unfair because different discrepancy notions make power comparisons uninterpretable" (classified structural):**  
This is a valid concern (retained as a Major weakness above) but the harsh critic overstates it as making the comparison *entirely uninterpretable*. Both tests are evaluated on the same data constructions, and NAMMD's advantage is explicitly attributed to the kernel's ability to capture data structure. The experimental setup is acknowledged to be limited (no kernel DCT competitor exists), not a deliberate unfair comparison.

**Harsh Critic – "Practical application claim is entirely unvalidated":**  
Partially valid (retained in Minor/Trivial above), but the case studies DO demonstrate qualitative alignment with known ground-truth orderings—which is a non-trivial empirical check. The harsh framing that "falls well short of establishing the practical claim" ignores the paper's own moderate framing of these as "case studies."

**Neutral Reviewer – "Limited experiments outside computer vision; needs text/audio/tabular domains":**  
This is generic scope-creep criticism. The paper uses benchmark datasets standard in kernel two-sample testing (blob, higgs, hdgm, MNIST, CIFAR, ImageNet). Demanding validation on other data types is outside the paper's stated scope and not standard in this literature. Removed per soft rules.

**Neutral/Spark – "No evaluation in high-dimensional/small-sample regimes":**  
Generic criticism not specific to NAMMD's failure modes. Theorem 8 provides sample complexity analysis; demanding comprehensive empirical coverage of all dimensionality/sample-size regimes is scope creep. Removed as generic.

---

## Novel Insights

The paper makes a genuinely useful observation: because the variance of the MMD estimator under a fixed kernel depends on the concentration (norms) of the distributions, two pairs with equal MMD can be very differently distinguishable in finite samples. NAMMD is a principled signal-to-noise normalization that turns this into an actionable test statistic. While normalizing by variance or scale is conceptually familiar in statistics, the specific RKHS-norm-based formulation for DCT—with a clean closed-form estimator, asymptotic Gaussian null distribution for $\epsilon > 0$, and provable power dominance—is a genuine novel contribution to the kernel hypothesis testing toolkit. The case study framing of DCT as a label-free proxy for model performance degradation under distribution shift is a practically useful synthesis.

---

## Suggestions

1. Report paired statistical tests (e.g., Wilcoxon signed-rank) on Table 1 differences to confirm same-kernel improvements are not within Monte Carlo noise.
2. Add a synthetic failure-mode experiment showing NAMMD vs. MMD performance when the norm condition of Theorem 12 is violated.
3. Clarify throughout that NAMMD is a test statistic / divergence, not a metric (no triangle inequality is established), or else prove the triangle inequality.
4. Provide interpretable sufficient conditions for the $\Delta \in (0, 1/2)$ range in Theorem 12 so readers can assess when the DCT power advantage applies in practice.
5. Add a brief comparison with at least one alternative normalization to justify the denominator design choice.

---

## Score and Decision

**Calibration:**

- **z9j7wctoGV** (Deep Kernel Relative Test, Accept, 6/6/6): Extends kernel testing methods to a new task (MGT detection), with solid theory and empirical validation. Comparable in scope to NAMMD, with similarly solid theory. Accepted at 6.
- **WnqD3EiylC** (RJSD, Reject, 5/6/5/6/3): Novel kernel divergence for two-sample testing and generative modeling, moderate empirical gains, theoretical contributions. Mixed scores; rejected partly due to unclear motivation for the normalization and limited empirical advantage—very similar pattern to NAMMD.
- **QCDdI7X3f9** (Model Equality Testing, Accept 8/6/6/6): Strong applied paper on MMD-based testing with clear real-world impact and strong experiments. Stronger than NAMMD empirically.
- **yqaN7MfkFU** (Regularized MMD, Withdrawn/Reject, 3/5/6/3/5): Similar kernel testing paper with methodological concerns about the core optimization; scored lower due to soundness issues.

**Positioning:** The paper is above yqaN7MfkFU (has sound theory, clear contribution, credible experiments) and roughly comparable to WnqD3EiylC (novel kernel divergence, solid theory, modest empirical gains, limited baselines) and the lower end of z9j7wctoGV. The paper's core theoretical contributions are solid but the empirical validation of the core same-kernel improvement is weak, the DCT baseline comparison is limited, and Theorem 12's restrictive condition narrows the claimed advantage. These are real but non-fatal weaknesses.

**Score: 5.0** — Borderline. Genuine contribution to an underexplored problem (kernel DCT), solid theory, but modest and incompletely validated empirical claims on the paper's primary contribution.

**Decision: Reject** — The paper is close to the acceptance threshold, but the combination of marginal same-kernel empirical gains (Table 1) lacking statistical support, the restrictive and inadequately analyzed condition for the main DCT theorem (Theorem 12), and the limited DCT experimental regime leave the central empirical claims insufficiently validated. With stronger statistical validation of Table 1, analysis of Theorem 12's scope, and at least one experiment probing failure modes, the paper would be in solid accept territory.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>