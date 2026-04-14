=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
Now I have sufficient understanding of the paper to synthesize the reviews. Let me produce the final consolidated review.

---

## Summary

This paper provides an empirical comparison of Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on six categories of functions distinguished by regularity: smooth regular, non-differentiable-but-continuous, jump discontinuities, singularities, coherent oscillations, and noisy variants of the above. Parameter counts are matched between architectures, and the effects of training sample size and optimizer choice (Adam vs. L-BFGS) are examined. The main finding is that KAN does not universally outperform MLP — it excels on singular and oscillatory functions but underperforms on functions with non-differentiable points or jump discontinuities.

---

## Strengths

- **Parameter-controlled comparison:** The authors explicitly match parameter counts between KAN and MLP configurations (Table 2), which is a concrete methodological choice that enables a meaningful architectural comparison rather than a capacity comparison. This is non-trivial given KAN's unusual parameterization via grid size and spline order.
- **Differentiated performance profile by function regularity:** The paper's central finding — that KAN outperforms MLP on singularities and coherent oscillations, but MLP outperforms on non-differentiable and jump-discontinuous functions — is a specific, nuanced result rather than a generic "no winner" conclusion, and it provides practitioner-relevant guidance for architecture selection.
- **Optimizer analysis with practical implications:** The Adam vs. L-BFGS comparison for KAN specifically (Tables 3 and 4) surfaces an important practical trade-off: L-BFGS converges in fewer epochs but incurs 10–70× greater wall-clock cost. This optimizer-specific behavior for KAN had not been systematically documented for irregular function fitting.

---

## Weaknesses

### Fatal
*(None that single-handedly invalidate the core claim, but the contradiction below must be resolved before publication.)*

### Major

- **Direct contradiction between Section 3.3 and Figure 3 caption.** Section 3.3 states unambiguously: *"Results show that the MLP outperforms the KAN"* for jump discontinuity functions $f_5$ and $f_6$. The caption of Figure 3 states the opposite: *"In all cases, KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line)."* The conclusion (Section 5) confirms the Section 3.3 claim ("MLP outperforms KAN on test functions $f_3$–$f_6$ on stabilized test loss"), meaning the Figure 3 caption is almost certainly erroneous — but this ambiguity is left unresolved in the paper. A reader relying on the figure caption and the reader relying on the text would draw opposite conclusions about a key experimental result. This must be corrected.

- **All experiments are 1D, limiting generalizability and significance.** Every test function is univariate ($f: \mathbb{R} \to \mathbb{R}$). This is a genuine limitation that the paper does not acknowledge. KAN's architectural motivation is rooted in the Kolmogorov-Arnold theorem for *multivariate* functions — the 1D case is trivially expressible without the multi-level sum structure. It is entirely possible that relative performance would shift significantly in higher dimensions where KAN's decomposition structure becomes non-trivial. The absence of any multivariate experiment, or even a discussion of whether the findings transfer, leaves the core question about KAN's practical utility unanswered.

- **No mechanistic analysis.** The paper describes *that* KAN fails on jump discontinuities and *that* it succeeds on singularities, but provides no analysis of *why*. For example: why would spline-based activations struggle more with jump discontinuities than fixed activations (likely Gibbs-like oscillations from smooth basis functions, but this is never articulated)? Why do singularities favor KAN (possibly due to spline grid adaptation near high-gradient regions)? Without mechanistic reasoning or even hypotheses, the findings are purely observational and difficult to generalize or build upon.

- **Inconsistent optimizer protocol across function categories without controlled cross-comparison.** Sections 3.1–3.3 use L-BFGS; Sections 3.4–3.5 switch to Adam (for both architectures). The justification is that Adam yields lower final loss for singularity and oscillation functions. While this is partially reasonable as "using the best available optimizer per function type," it conflates optimizer effect with architecture effect. Any cross-category performance comparison is confounded because the optimizer regime changes. The paper should either (a) report results under a fixed optimizer for cross-category comparisons, or (b) explicitly acknowledge that category-level comparisons are not optimizer-controlled.

- **No statistical repeats.** All results appear to be single-run. Neural network training — especially KAN with B-spline grids and L-BFGS — is sensitive to initialization. A single run could be misleading, particularly when the claimed performance difference between architectures on some functions appears small. The absence of error bars or seed-averaged results makes it impossible to assess whether reported differences are reliable.

### Minor

- **KAN grid hyperparameters fixed ($G=3$, $k=3$) without ablation.** The capacity of KAN scales with grid resolution. Failures on jump discontinuities (where splines are forced to approximate a non-smooth function with a low-resolution smooth basis) may partly stem from under-parameterization of the spline grid rather than an intrinsic architectural limitation. Without sweeping $G$ and $k$, the paper cannot cleanly attribute the failure to architecture versus capacity.

- **Noise distribution and SNR computation undefined.** Section 4 compares architectures under SNR values of 0, 4, and 10, but the paper never formally defines the noise distribution (Gaussian? Uniform?), how SNR is computed, or what amplitude corresponds to these levels. The noisy experiments are therefore not reproducible.

- **Computational cost asymmetry not discussed in context.** Tables 3–4 show KAN requiring 40–70× more wall-clock time than MLP (e.g., 588s vs. 8s for $f_7$ with L-BFGS). The paper notes this but does not discuss when KAN's accuracy advantage would justify this cost. The paper claims "KAN outperforms MLP" for singularities, but a Pareto perspective (accuracy vs. compute) would substantially change the practical conclusion.

- **"Section D" dangling reference.** Section 3.5 states "taking a similar approach as described in section D" — an appendix that does not appear in the submission. This leaves the experimental procedure for $f_9$/$f_{10}$ incompletely described.

### Tiny

- The claim in Section 3.4 that "L-BFGS achieves faster convergence" is ambiguous. It converges in fewer *epochs* but takes significantly more *wall-clock time*. This should be qualified explicitly.
- The conclusion section adds little beyond restating body results without synthesis.

---

## Nice-to-Haves

- **Multivariate function experiments** (e.g., $f: \mathbb{R}^2 \to \mathbb{R}$ with singularities or discontinuities along curves) would test whether the findings transfer to the regime where the KA theorem's multivariate decomposition is meaningful, and would substantially strengthen the paper's significance.
- **Domain-resolved error profiles** (e.g., plotting $|f(x) - \hat{f}(x)|$ across the input domain) would sharpen the analysis of *where* each model fails — e.g., whether MLP error on singularities is concentrated near the singular point or distributed globally.
- **Visualization of learned spline activations** in KAN after training on jump-discontinuous functions would provide direct evidence for or against the hypothesis that smooth spline bases are the architectural bottleneck.
- **KAN 1.0 vs. KAN 2.0 ablation** specifically for jump discontinuities: the paper notes multiplication nodes are used but minimal improvement expected; confirming this empirically would strengthen the claim.
- **Extension to a small set of real-world tabular regression tasks** with known irregular behavior would contextualize the synthetic findings.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Conceptual gap" from 1D scope relative to the KA theorem** (harsh critic, framed as a fatal flaw): The theorem is cited as motivation, not as a design constraint for the experiments. The paper's stated scope is function approximation under regularity conditions; 1D is a principled choice to isolate regularity effects. This is a scope limitation (kept as Major above) but not a conceptual invalidation of the methodology.
- **"Unprincipled function selection within categories"** (harsh critic): The paper does not claim to be a comprehensive survey of each category — it studies representative examples. Demanding principled coverage of each function class exceeds the paper's stated scope.
- **Criticism that the optimizer switch to Adam for f7–f10 makes comparisons unfair to MLP** (spark finder): In Sections 3.4–3.5, Adam is used for both MLP and KAN. The concern is about cross-category confounding (kept as a Major weakness), not unfairness to the baseline. The asymmetric optimizer criticism (giving MLP its best optimizer) is not applicable here — both architectures receive the same optimizer regime per category.
- **Demand for theoretical proofs of architectural behavior** (implicit in multiple reviews): This is an empirical benchmarking paper, and theoretical derivations of why splines fail on jumps are outside its scope. Kept as Nice-to-Have only.
- **"Timely and relevant topic" as a specific strength** (positive reviewer): This is generic and applies to any early KAN benchmark paper. Removed from strengths.
- **"Well-written" or "extensive experiments"** type generic strengths: Not included as they do not distinguish this paper from others in the area.

---

## Novel Insights

The most actionable novel finding beyond what prior KAN benchmarks have documented is the *regularity-specific inversion of rankings*: KAN is better at singularities and coherent oscillations (functions with unbounded derivatives at isolated points, remaining smooth everywhere else) while MLP is better at non-differentiable-but-continuous and jump-discontinuous functions. This suggests a structural hypothesis worth investigating: KAN's B-spline activations, being inherently smooth, may adapt concentration of basis function nodes near singular points (analogous to adaptive quadrature), but the smoothness constraint prevents them from representing true jumps or cusps regardless of capacity. If this mechanism were confirmed via visualization of learned spline shapes (as suggested above), it would constitute a genuine contribution to understanding the inductive biases of learnable-activation networks.

---

## Suggestions

1. **Resolve the Section 3.3 vs. Figure 3 contradiction.** Audit every figure caption against the text claims, especially for f5 and f6. This is the highest priority fix.
2. **Separate the optimizer confound.** Include supplemental tables or figures reporting results under a single fixed optimizer (e.g., Adam for all functions) so that cross-category comparisons are controlled. The optimizer-per-function results can be retained as supplementary "best-effort" results.
3. **Add at least 3–5 random seeds with error bands** to all main loss curve plots. For small-scale experiments, this is not computationally expensive and substantially strengthens the reliability of claimed performance differences.
4. **Define the noise model formally.** State the distribution (e.g., $\mathcal{N}(0, \sigma^2)$), how SNR is defined (e.g., signal power / noise power), and provide the $\sigma$ values for SNR = 0, 4, 10.
5. **Add a mechanistic section or paragraph** hypothesizing why each architecture fails where it does — even a qualitative argument connecting B-spline smoothness to Gibbs-like artifacts on jumps would raise the paper's scientific contribution substantially.
6. **Remove or fix the "section D" reference** in Section 3.5 so the experimental procedure is self-contained.
7. **Qualify all "faster convergence" claims** with explicit clarification of whether this means epoch-wise or wall-clock-wise convergence, and add a brief cost-benefit analysis for singular/oscillatory functions where KAN's advantage is claimed.

---

**Overall assessment:** The paper addresses a legitimate question and its function-type-specific performance breakdown is useful to practitioners. However, the unresolved internal contradiction (Section 3.3 vs. Figure 3), the restriction to 1D functions, the absence of statistical repeats, inconsistent optimizer protocols across categories, and purely descriptive findings without mechanistic depth collectively place this paper below the ICLR acceptance bar in its current form. The work is closer to a technical report or workshop contribution. Addressing the contradictions and statistical deficiencies would make the paper publishable at a workshop level; adding multivariate experiments and mechanistic analysis would be needed for a main-track ICLR contribution.

- **Novelty:** Low-to-moderate. The framing by function regularity category is new, but single-run 1D experiments with no theoretical analysis limit the advance over prior KAN-vs-MLP benchmarks.
- **Technical soundness:** Moderate, significantly damaged by the unresolved Figure 3 vs. Section 3.3 contradiction and lack of statistical repeats.
- **Empirical support:** Weak. Single runs, 1D only, two architecture sizes, unspecified noise model.
- **Significance:** Low for ICLR. Findings are narrowly scoped and not clearly connected to the practical use cases of KAN.
- **Clarity:** Mixed. Function categorization is clear; optimizer methodology is poorly organized and the dangling "section D" reference degrades reproducibility.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 1.0]
Average score: 2.5
Binary outcome: Reject
