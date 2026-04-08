=== CALIBRATION EXAMPLE 27 ===

# Final Consolidated Review
## Summary

The paper introduces a new Iteratively Reweighted Least Squares (IRLS) algorithm for ℓ_p regression that achieves state-of-the-art iteration complexity O(p²n^{3p/(p−2)} log(n/ε))—matching the best known theoretical bound of Adil et al. (2019a; 2024)—while retaining the practical simplicity of an IRLS framework. The core innovation is a primal-dual approach where update rules are derived from an invariant on the dual energy objective, allowing large coordinate-wise updates unlike standard mirror descent or multiplicative weights methods. Experiments on synthetic and real-world datasets demonstrate consistent 1–2.6× speedups over the prior practical IRLS method (p-IRLS) and orders-of-magnitude improvements over CVX solvers.

## Strengths

- **Bridges the theory–practice gap in ℓ_p regression IRLS:** The paper's central contribution is concrete and well-motivated. Prior work had either strong theory with impractical algorithms (Adil et al., 2019a; 2024) or practical algorithms with weaker guarantees (Adil et al., 2019b). This work achieves the best of both via a simpler primal-dual IRLS scheme, which is a meaningful advance for this well-studied problem.

- **Elegant algorithmic design with longer steps:** Unlike standard mirror descent or multiplicative weights approaches that regularize the norm or use mirror maps, the proposed update rule allows coordinates of the dual solution to change by large polynomial factors per step (Section 2.2, "our update scheme allows our algorithm to take much longer steps"). This is the key mechanism enabling the improved convergence rate and is a clean conceptual contribution.

- **Consistent empirical improvements:** Across random matrices, random graphs, and six real-world UCI datasets, the algorithm outperforms p-IRLS in both iteration count and wall-clock time. The speedup gap widens with increasing p and problem size (Figures 1–2, Table 1), which is consistent with the theoretical advantage.

## Weaknesses

### Major:

- **Limited evaluation across p values on real-world data:** Table 1 reports results for only p = 8 on all six real-world datasets. The paper claims contributions for all p ≥ 2 (and via reduction, 1 < p < 2), yet the real-world validation covers a single value. The synthetic experiments test a wider range (p = 3–50), but real-world structures may expose different behavior. Without testing at least a few additional p values (e.g., p = 3, 4, 10) on real data, it is unclear whether the speedups generalize across the full claimed range or are specific to p = 8.

- **Behavior near p = 2 is not discussed:** The complexity bound involves the exponent 3p/(p−2), which blows up as p → 2⁺. This is a practically important regime (ℓ₂ regression is classical least squares; p slightly above 2 appears in robust variants). The paper's theorems and algorithms focus on p ≥ 2 but never explicitly address how the algorithm degrades near this boundary, what the practical iteration counts look like, or whether alternative methods should be preferred. This is a notable gap for practitioners.

### Minor:

- **Hard-constraint vs. soft-constraint formulation gap:** Theorems 1.1 and 1.2 solve min_{Ax=b} ‖x‖_p (affine-constrained), while Section 4 experiments solve min ‖Ax−b‖_p (standard regression). The reduction is in Appendix B (Lemma B.1) but is never mentioned in the main text or experimental section, which can confuse readers expecting the standard regression formulation. A brief clarification in Section 4 would help.

- **Discontinuous regime switch at the κ boundary:** Algorithm 3 sets κ = 1 if p ≤ log n/(log log n − 1), else κ = p/(p−2). For n = 10⁶, this threshold is approximately p ≈ 8.75—right near the p = 8 used in most experiments. The paper does not discuss whether performance changes abruptly at this boundary, or whether a smooth transition exists. This matters for practical deployment.

- **Linear solver treated as a black box:** All iteration complexity and runtime claims depend on solving linear systems of the form ADA^⊤φ = z. The paper does not discuss the choice of solver (direct vs. iterative, preconditioning), which becomes the practical bottleneck for large-scale instances. The reported MATLAB timings implicitly use a particular solver, but the scalability implications of this choice are unanalyzed.

- **Large constant factors in convergence rates:** Lemma E.2 shows the function value gap decreases by a factor of (1 − 1/(2^{13}pκ)) per progress iteration. For p = 8, κ = 1, this is approximately (1 − 1/32768)—extremely slow per-iteration progress that is amortized by the O(p² log n log(n/ε)) outer iterations. The paper would benefit from briefly discussing whether these constants reflect practical behavior or are artifacts of the analysis.

- **Case 2 averaging correctness not formalized in main text:** Algorithm 2 returns the average s^{t'}/t' of primal solutions in Case 2. While Lemmas 2.2 and 2.3 address Cases 1 and 3, the justification for Case 2 is not stated as a separate lemma in the main text and the reader must reconstruct it from the convergence analysis in Appendix D (around Lemma D.1). Making this explicit would improve self-containment.

### Trivial:

- Notation shifts between Sections 2 and 3 (‖x‖_{2p}² in the low-precision regime vs. ‖x‖_p^p in the high-precision regime) are deliberate but could confuse a casual reader. A one-sentence reminder when the shift occurs would help.

## Nice-to-Haves

- **Convergence trajectory plots:** Plotting objective gap vs. iteration count and wall-clock time would reveal whether the empirical speedup comes from fewer iterations, cheaper per-iteration cost, or both. This is important for validating the "simpler IRLS" efficiency claim beyond final iteration counts.

- **Overconstrained regime benchmarks:** The paper assumes d = Θ(n) for simplicity but notes that the regime n ≫ d is important (citing Jambulapati et al., 2022). Explicit benchmarks with fixed d and varying n would verify whether the theoretical dependence on dimensions translates to practice.

- **Ill-conditioned input tests:** IRLS methods can be numerically unstable for large p or ill-conditioned A. Testing on matrices with varying condition numbers would define the method's operational envelope and is relevant for the robust regression applications cited in the introduction.

- **Additional solver baselines:** Comparing against modern first-order convex solvers (e.g., SCS, OSQP) beyond CVX's interior-point methods would better contextualize the speedups within the broader optimization landscape.

- **Non-MATLAB implementation:** A Python or C++ implementation would improve accessibility for the broader ML community, though the MATLAB code in the supplementary material is a reasonable starting point.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Formatting inconsistencies in complexity expressions:** These are parser artifacts, not paper errors. Removed.
- **Reproducibility concerns (random seeds, exact versions, dependencies):** Per hard rules, these are nitpicks about implementation details impractical to include in a submission. Removed.
- **Demanding a derivation of the Sherman-Morrison formula:** This is a standard linear algebra identity; its inclusion would be a pedagogical nicety, not a gap. Removed.
- **Demanding implementation of the Adil 2019a theoretical algorithm as a baseline:** The paper explicitly argues this algorithm is impractical due to complex subroutines and tuning requirements. Demanding the authors implement a method they reasonably argue is infeasible is unreasonable. Removed.
- **Broader societal impact discussion:** This is a pure optimization algorithm paper. Requiring societal impact analysis for such work is scope creep beyond ICLR norms for theory-heavy contributions. Removed.
- **Missing concurrent work claims:** Per hard rules, I cannot verify existence of uncited concurrent work and will not flag this. Removed.
- **Binary search initialization failure scenarios:** The ℓ₂ initialization is always well-defined, and the range is derived rigorously. This concern is hypothetical without evidence of actual failure. Removed.
- **Statistical significance testing for optimization experiments:** The experiments report means and standard deviations over multiple runs. Demanding formal hypothesis tests for deterministic optimization algorithms with small variance is not standard in this community. Removed.
- **Lewis weight sampling integration:** This is a dimensionality-reduction technique for the overconstrained regime by different authors. Demanding its inclusion is scope creep. Removed.
- **Proof details deferred to appendix:** Standard practice for theory papers at ICLR. Key intuitions are provided in the main text. Removed as formatting/style nitpick.

## Novel Insights

The paper's primal-dual invariant (Eq. 1)—requiring that the ratio of energy increase to dual norm increase is at least M²—is a unifying structural insight that connects ℓ_p regression to width-independent multiplicative weights methods for packing-covering LPs, while allowing exponentially longer steps. This invariant acts as a "bang-for-buck" ratio constraint per coordinate, and the algorithm's convergence is driven by the geometric fact that you cannot simultaneously maintain low dual norm growth and high energy gains across many coordinates for many steps (Lemmas 2.4–2.5). This coordinate-wise decomposition of a global invariant, combined with aggressive multiplicative updates, is the conceptual novelty that distinguishes this from both classical mirror descent (which regularizes to enable bounded steps) and the Adil et al. iterative refinement (which relies on Taylor expansion residuals). The observation that this framework generalizes to the mixed ℓ_p + ℓ_2 objective by carefully controlling the dual norm initialization (Algorithm 4) is a non-trivial extension that the paper handles but does not emphasize enough.

## Suggestions

- Test the algorithm on at least 3 additional p values (e.g., p = 3, 4, 10) on the real-world datasets from Table 1, and report both iteration counts and runtimes. This would take minimal additional effort and substantially strengthen the empirical contribution.

- Add a brief paragraph in Section 2 or 3 explicitly discussing the behavior as p → 2⁺: what does the iteration bound look like, at what point does the method become impractical, and what alternative (e.g., standard least squares with small regularization) should practitioners use instead?

- Include one convergence trajectory plot (objective gap vs. iteration) for a representative instance to visually confirm that the theoretical convergence rate translates to practice, and to reveal whether the algorithm converges smoothly or exhibits plateaus.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 6.0]
Average score: 6.5
Binary outcome: Accept
