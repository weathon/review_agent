Now I have sufficient calibration. Let me write the final review.

## Summary
This paper proposes Norm-Adaptive Maximum Mean Discrepancy (NAMMD), a normalized variant of MMD designed for distribution closeness testing on complex data. The method scales MMD by a denominator term involving distribution norms to address the claim that MMD is "less informative" when comparing multiple distribution pairs with different RKHS norms. The paper provides asymptotic theory, sample complexity bounds, and experiments showing marginal test power improvements over standard MMD.

## Strengths
- **Extends distribution closeness testing to complex data**: The paper successfully adapts the DCT framework beyond discrete one-dimensional distributions (where TV-based methods dominate) to complex data like images using kernel methods. Table 2 demonstrates NAMMD outperforming Canonne's TV-based test on discrete distributions (0.934 vs 0.800 at ε'=0.1), showing kernel methods can be effective even in settings traditionally favoring count-based approaches.

- **Solid theoretical foundation with asymptotic analysis**: The paper derives asymptotic distributions for the NAMMD estimator (Theorem 2), provides concentration bounds (Lemmas 6-7), and establishes sample complexity guarantees (Theorem 8). The Type-I error control is proven (Theorem 5), and Lemma 9 correctly establishes NAMMD=0 iff P=Q for characteristic kernels.

- **Thoughtful case studies connecting theory to practice**: The ImageNet variant experiments (Figure 3) and adversarial perturbation analysis (Figure 5) demonstrate practical utility for label-free model evaluation. The observation that NAMMD ordering aligns with accuracy margins across ImageNet variants ({0.529, 0.564, 0.751, 0.827}) provides concrete evidence the statistic captures meaningful distributional differences.

- **Drop-in compatibility with existing kernel frameworks**: NAMMD can replace MMD in established methods like kernel fusion, deep kernels, and Mahalanobis kernels without architectural changes. Figure 2 shows NAMMDFuse competing favorably against SOTA two-sample tests (MMDFuse, MMDAgg, MMD-D) across blob, higgs, hdgm, and mnist datasets.

## Weaknesses

### Fatal
None

### Major
- **The normalization denominator lacks principled justification**: The NAMMD denominator `4K - ‖μ_P‖² - ‖μ_Q‖²` is chosen to map the range to [0,1] and ensure monotonicity in norms, but the paper provides no derivation showing this arises from optimizing a natural statistical quantity (e.g., signal-to-noise ratio, Pitman efficiency, or normalized effect size). Alternative normalizations like `‖μ_P - μ_Q‖² / (Var(P,κ) + Var(Q,κ))` could have clearer statistical interpretations. This is a core conceptual gap because the entire novelty of NAMMD rests on this specific scaling choice. Without principled justification, it remains unclear whether this normalization is genuinely superior or merely one of many possible ad hoc choices happen to show small gains in specific settings.

- **Empirical improvements over MMD are marginal and lack statistical significance analysis**: Table 1 shows NAMMD outperforming MMD with the same kernel by typically 0.003-0.01 absolute test power—differences often smaller than the reported standard deviations. For example, on blob with Gaussian kernel: MMD 0.600±0.090 vs NAMMD 0.616±0.090, where the 0.016 gain is well within one standard deviation. Yet the paper declares "better performance" without confidence intervals, hypothesis tests, or any quantification of whether these differences exceed Monte Carlo variability. For a paper whose central claim is "NAMMD achieves higher test power than MMD," this absence of statistical rigor in validating the claimed improvement is a significant evidential weakness.

- **Theoretical power dominance claims are oversold relative to theorem content**: Theorems 10 and 12 are conditional implications with nontrivial probabilistic qualifiers (e.g., "with probability ς ≥ 1/65 there exist samples where MMD fails but NAMMD rejects"), not unconditional power orderings. These results show that under specific conditions and on particular events, NAMMD can reject when MMD does not—but this is compatible with the two tests having nearly identical average power. The prose upgrades these modest, assumption-heavy results into broader claims ("NAMMD test achieves better performance than MMD") without careful scoping. This discrepancy between the narrow theorem statements and the sweeping narrative claims is substantive and misleading.

### Minor
- **The motivation that MMD is "less informative" is somewhat circular**: The paper defines "closeness" via p-values of hypothesis tests, then proposes a statistic that correlates better with those p-values. Figure 1 shows NAMMD increasing with norms while p-values decrease, but p-values depend on sample size, kernel choice, and test variant—they are not an intrinsic property of the distribution pair. Using test performance to justify the metric's superiority creates a self-referential argument: NAMMD is better because it aligns with p-values, and we know the p-values reflect true closeness because... they come from hypothesis tests. A more principled approach would define a task-independent notion of closeness (e.g., via a specific loss function or risk) and show NAMMD captures it better than MMD.

- **Variance expression in Theorem 2 appears notationally inconsistent**: Line 117 defines `σ²_{P,Q} = sqrt(4E[H_{1,2}H_{1,3}] - 4(E[H_{1,2}])²) / (...)`, placing a square root in the numerator of something labeled as a variance. This is dimensionally suspicious (variance should not have square root structure if already denoted σ²) and likely a typo that should be clarified. While this may not affect correctness if the appendix contains a proper derivation, it creates confusion for readers trying to verify asymptotic claims.

- **Comparison with Canonne's TV-based test is asymmetric in setup**: Table 2 applies a sophisticated kernel method (NAMMD with learned Mahalanobis kernel on real-valued data) against a count-based estimator over a discretized support of size 50. The paper acknowledges Canonne's method is handicapped "especially when data is limited," which is precisely the regime tested. This makes the comparison more a demonstration that kernel methods work well on structured continuous data than a fair evaluation of closeness testing methodology per se. A more balanced comparison would either discretize the data before applying NAMMD or give Canonne's method access to the full continuous representation.

### Trivial
- **Two different null calibration methods for ε=0 vs ε>0 lack unified treatment**: The paper uses permutation tests when ε=0 (two-sample testing) and Gaussian approximation with estimated variance when ε>0 (closeness testing), but does not cleanly integrate these into a unified framework or discuss the implications of switching procedures. This is a presentation issue rather than a fundamental flaw.

- **Global kernel selection remains unresolved**: The paper acknowledges that selecting an appropriate global kernel for comparing multiple distribution pairs is "an open question and poses a significant challenge," yet all theoretical comparisons assume such a kernel exists. This is honestly stated but means practical deployment requires solving a problem the paper explicitly leaves open.

## Nice-to-Haves
- **Ablation isolating NAMMD vs MMD under identical configurations**: Run experiments where the *only* difference is replacing MMD with NAMMD—same kernel, same network architecture, same sample size, same permutation strategy—with confidence intervals and statistical significance tests. This would definitively separate the effect of normalization from other design choices.

- **Analysis of denominator near-singularity regime**: Test NAMMD in settings where `‖μ_P‖² + ‖μ_Q‖²` approaches `4K`, making the denominator small. Assess estimator stability, Type-I error control, and sensitivity to small estimation errors in norms.

- **Counterexamples where feature closeness does not imply performance closeness**: Construct scenarios with covariate shift that preserves marginal feature similarity but changes label conditionals, demonstrating limits of unlabeled distribution testing as a proxy for model performance. This would calibrate the practical claims made in Section 5.2.

- **Scatter plots of MMD vs NAMMD vs p-value across many pairs**: Beyond the single constructed example in Figure 1, show many real distribution pairs plotting both statistics against observed p-values to concretely illustrate NAMMD's improved correlation with testing difficulty.

## Removed Points
The following points were flagged by the harsh critic but are removed with caution:

1. **"The central problem with MMD is largely manufactured"** — While the motivation is somewhat circular, the paper does identify a genuine phenomenon (equal MMD for different norm pairs yielding different test power) and proposes a working solution. This is a conceptual weakness but not a manufactured problem—the issue is the *framing* not the existence of the phenomenon itself. Moved to Minor under circular motivation.

2. **"Claims about p-values being intrinsic are wrong"** — The paper does acknowledge p-values depend on sample size and kernel; the criticism is slightly overstated. The real issue is using p-values as the ground truth for "closeness" without external validation.

3. **"Ad hoc denominator is a core conceptual weakness undermining interpretability"** — Retained in Major but softened: the lack of principled derivation is a gap, but the denominator does produce consistent (if small) gains. This doesn't "undermine" the method but does limit its generalizability.

4. **"Theorems are assumption-heavy and narrow"** — Retained in Major but clarified: the theorems are technically correct but the prose oversells them. The issue is rhetorical overreach, not mathematical error.

5. **"Gains are within noise"** — Retained in Major as "lack statistical significance analysis." The harsh critic's phrasing implied the gains might be fake; the more precise criticism is that the paper doesn't *demonstrate* they're real through proper statistical testing.

6. **"ImageNet case studies loosely connected to theory"** — Weakened to Minor: the case studies are suggestive but not rigorous validation. This is appropriate for a "case study" section but shouldn't be over-interpreted as the paper sometimes implies.

7. **Harsh critic's claim that "no amount of additional experiments fixes the conceptual weakness"** — This is too strong. Better experiments (significance testing, ablations, counterexamples) would substantially strengthen the empirical case even if the motivation remains partially circular.

## Novel Insights
The paper's central observation—that for fixed MMD value, distribution pairs with larger RKHS norms yield smaller p-values in two-sample testing—is genuinely interesting and underexplored in the MDD literature. This connects the geometry of the embedding (via norms) to test power in a way that standard MMD treatments ignore. However, the paper stops short of providing a principled explanation for *why* this relationship should hold beyond empirical observation and an informal argument about concentration. A deeper insight would derive this from the variance structure of the MMD U-statistic, showing how norms affect the signal-to-noise ratio and thus test power. The paper hints at this (mentioning variance decreases with norms) but doesn't formalize it into a general principle that would justify the specific normalization chosen.

## Suggestions
1. **Provide principled derivation of the normalization**: Either show the NAMMD denominator optimizes some statistical criterion (e.g., asymptotic test power, signal-to-noise ratio) or explicitly frame it as a heuristic choice validated empirically. If the former, add the derivation; if the latter, soften claims about NAMMD being "the" solution.

2. **Add statistical significance analysis to Table 1**: For each MMD vs NAMMD comparison, report whether the difference exceeds what would be expected from Monte Carlo noise (e.g., via paired t-test or bootstrap confidence intervals). If differences are not significant, acknowledge this and temper claims accordingly.

3. **Reframe Theorems 10 and 12 with precise scope**: State clearly these are conditional results holding with specific probabilities under specific assumptions, not universal power dominance. Remove or qualify prose that suggests NAMMD is "strictly better" in general.

4. **Clarify the variance expression in Theorem 2**: Fix the notational inconsistency in line 117 (σ² defined with sqrt) and verify the expression matches standard asymptotic variance formulas for U-statistics.

5. **Add ablation studies**: Show NAMMD vs MMD with everything held constant (same kernel learned on same data, same test procedure, same sample size) to isolate the normalization effect from other factors.

## Score and Decision

**Calibration reasoning:**

I compared this paper against several anchors:

- **Pf85K2wtz8.md** (accepted poster, scores 6,6,5,6): MMD-based method with competitive empirical performance and marginal improvements over baselines. Similar to NAMMD's situation—solid theory, modest gains. This paper was accepted.

- **rP7rghI7yt.md** (rejected, scores 5,8,3,5): PHI-S normalization for distillation with tiny performance gaps (<0.3%) over standard normalization. Reviewers questioned whether the proposed normalization was meaningfully better. This is highly analogous to NAMMD's situation and was rejected.

- **GPcSYm89wK.md** (rejected, scores 5,5,3,5): Kernel learning for CI testing with "only marginal improvements in power" compared to median heuristic. Despite solid methodology, the small effect size contributed to rejection.

- **Ip6UwB35uT.md** (accepted, scores 6,6,8,8): Conditional testing with solid theory and numerical simulations. Accepted despite modest experiments because theory was strong and claims were appropriately scoped.

- **6QBHdrt8nX.md** (rejected, scores 1,3,6): Explicitly rejected for lacking statistical significance analysis when claiming one method outperforms another—directly parallel to NAMMD's weakness.

NAMMD sits between the accepted papers with marginal but real contributions (Pf85K2wtz8, Ip6UwB35uT) and rejected papers where tiny gains without rigorous validation led to rejection (rP7rghI7yt, GPcSYm89wK). The distinguishing factors:

**Why NAMMD is stronger than the rejected anchors:**
- Has more complete asymptotic theory (variance derivation, sample complexity)
- Case studies are more thoughtful than typical marginal-gain papers
- The method is clearly described and reproducible

**Why NAMMD is weaker than the accepted anchors:**
- Empirical gains (0.003-0.01) are smaller than Pf85K2wtz8's "competitive" performance
- Claims are oversold relative to theorem content, unlike Ip6UwB35uT's appropriately scoped contributions
- No statistical significance analysis despite claiming superiority—this is the critical gap that aligned with 6QBHdrt8nX's rejection

The paper has genuine technical merit (theory is mostly sound, method works, case studies are interesting) but the Major weaknesses—particularly the ad hoc normalization and lack of statistical validation for claimed improvements—prevent it from being a clear accept. This is a **borderline paper** that would need stronger evidence the gains are real and principled to reach acceptance threshold.

Compared to PHI-S (rejected with similar normalization/marginal-gain profile), NAMMD has slightly stronger theory but similarly tiny empirical gains. Compared to the accepted MMD-flow paper (Pf85K2wtz8), NAMMD's empirical case is weaker because it lacks statistical rigor.

**Center of anchor cluster: 5-6 range.** Given the oversold claims and missing statistical analysis, I lean toward the lower end.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>