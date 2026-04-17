## Summary

This paper extends decision trees and random forests to product manifolds—Cartesian products of hyperbolic, hyperspherical, and Euclidean spaces—via an angular reformulation of splits. By parameterizing splits as angles in 2D projections, the method ensures geodesically convex, composable splits across all constant-curvature manifolds while recovering Euclidean and hyperbolic DTs as special cases and introducing hyperspherical DTs. The method is evaluated on 57 benchmarks spanning synthetic data, graph embeddings, VAE latents, and empirical datasets.

## Strengths

- **Elegant angular formulation**: The unification of splits across Euclidean, hyperbolic, and spherical manifolds through angular projections (Eqs. 15–16) is conceptually clean and yields O(1) decision complexity per split while maintaining geometric properties. This generalizes the sparse hyperplane idea from Chlenski et al. (2024) to all constant-curvature spaces.

- **Strong single-manifold results**: On the single-curvature synthetic benchmarks (Figs. 3–4), the product DT/RF achieves top-1 performance on 21 of 22 benchmarks, convincingly outperforming ambient-space and tangent-plane baselines. This empirically validates the core geometric insight.

- **Broad empirical scope**: The 57 benchmarks cover classification, regression, and link prediction across diverse geometries and data types. The top-2 rate of 93% (53/57) demonstrates consistent competitiveness.

- **Qualitative interpretability**: Figure 5 (landmass classification on S²) effectively illustrates that geometry-aware RFs produce smoother, more natural decision boundaries than Euclidean/tangent or k-NN alternatives.

- **Honest limitations discussion**: Section 5 candidly acknowledges dependence on good embeddings, signature selection challenges, the privileged basis problem, and tradeoffs with GNNs.

## Weaknesses

### Major:

1. **Unsupported "maximum-margin" claim**: The abstract and introduction prominently state that splits are "maximum-margin," but this is never formally defined or proven. The method selects splits maximizing information gain (Eq. 13) and places thresholds at geodesically equidistant midpoints—this is a locally balanced split, not maximum-margin in the standard sense (as in SVMs). The term "maximum-margin" carries specific connotations that the algorithm does not substantiate. This should either be formalized with a theorem and proof specifying what margin is being maximized, or the claim should be retracted.

2. **Limited baseline comparison, especially for product manifolds**: The baselines are primarily ambient/tangent DTs/RFs, k-NN, and a product-space perceptron. While MLP and GNN comparisons exist in Appendix I, the main narrative (Tables 2–3) lacks manifold-aware nonlinear baselines. The product-space SVM from Tabaghi et al. (2021)—explicitly cited in related work—is evaluated only in its perceptron variant, which the paper itself notes "never achieved competitive results." A stronger geometric baseline like the product SVM would better contextualize performance. On the product-manifold benchmarks specifically (the paper's claimed novelty), k-NN outperforms the proposed method on 7 of 8 synthetic multi-curvature classification tasks (Table 2), making stronger baselines essential for the core claim.

3. **Modest empirical advantage on product manifolds, the central novelty**: The paper's primary contribution is DTs/RFs for product manifolds, yet the product-manifold results are notably weaker than single-manifold results. The abstract reports 18/35 top-1 on product benchmarks versus 21/22 on single manifolds. On synthetic multi-K classification, k-NN dominates every single setting (e.g., H⁴: 47.5 vs 40.0 F₁). For regression (Table 3), product and ambient models are often within 0.001–0.003 RMSE, and ambient wins on Temperature and Traffic. The paper's framing ("strong preliminary evidence favoring mixed-curvature DTs and RFs") is more assertive than the product-manifold evidence supports.

### Minor:

4. **Small-scale and dated benchmark datasets**: The graph datasets (CiteSeer, Cora, PolBlogs, etc.) are well-known but small and old. No evaluation appears on modern large-scale benchmarks, leaving scalability concerns unaddressed in the main text (runtime analysis is in Appendix J).

5. **Missing ablation of projection sets**: Section 3.4 claims the ability to search over all $\binom{D}{2}$ projections as an advantage, but no ablation compares this against simpler basis-only projections. It is unclear whether performance gains come from the angular formulation's geometric benefits or simply from having more candidate features.

6. **Confusing Euclidean equivalence presentation**: Eq. 19 appears garbled (applying arctan to angles already computed via arctan), and the claim of "complete equivalence" to standard axis-aligned CART is deferred to Appendix C without a clear sketch in the main text. While the equivalence might hold, the presentational gap may confuse readers.

7. **Geodesic convexity claimed without proof**: The statement that "homogenous hyperplanes are geodesically convex in any constant-curvature manifold" is a standard result in Riemannian geometry, but the paper states it without reference or proof. A citation to a differential geometry textbook would suffice, but as written the reader must take this on faith.

### Trivial:

- The tangent plane definition in Eq. 1 uses points of the manifold rather than tangent vectors, which is non-standard notation but does not affect correctness.

## Nice-to-Haves

- Ablation studies varying projection set size (basis-only vs. all pairs) to isolate the geometric contribution from feature expansion.
- Signature sensitivity analysis: evaluate degradation when the product manifold signature is misspecified (wrong curvatures, wrong number of components).
- Visualization of decision boundaries on a simple product manifold (e.g., the S¹×S¹ torus from Figure 1) to complement the S²-only visualization in Figure 5.
- Formal definition and proof (or retraction) of the "maximum-margin" property, or reformulation as "geodesically equidistant" which is more accurate.

## Removed Points

- **"The Euclidean equivalence is fundamentally wrong"** (harsh critic): The claim is supported by a proof in Appendix C. While the main-text presentation is confusing (especially Eq. 19), dismissing the equivalence without reading the appendix proof oversteps. Downgraded to a presentation issue (Minor #6).

- **"Geodesic convexity is not established and may not hold"** (harsh critic): The claim that homogeneous hyperplanes intersect constant-curvature manifolds in geodesically convex regions is a standard result in Riemannian geometry for spaces of constant curvature. For spheres, hemispheres cut by great circles are geodesically convex; for hyperboloids, an analogous result holds. The product case preserves convexity per-component. The issue is lack of citation, not incorrectness. Downgraded to Trivial #7.

- **"Lack of neural network baselines"** as a fatal flaw (spark): MLP and GNN comparisons actually exist in Appendix I. The concern about their placement in the appendix is valid, but claiming their complete absence is factually wrong.

- **"Composability is vague marketing"** (harsh critic): The paper defines composability concretely in Section 3.4—splits across different component manifolds can be independently composed within a single tree. This is a specific structural property, not just marketing.

- **"Single 80/20 split without cross-validation"** (spark): The paper reports 95% confidence intervals and applies Bonferroni-corrected Wilcoxon signed-rank tests, suggesting multiple trials. The methodological concern is minor for a 57-benchmark evaluation.

- **"Format/style nitpicks"** (e.g., Eq. 1 notation): Removed per rules.

## Novel Insights

The angular reformulation's key insight—that thresholding on arctan(x₀/x_d) in 2D projections unifies axis-aligned CART (Euclidean), hyperbolic DTs (Chlenski et al., 2024), and introduces hyperspherical DTs under a single mechanism—is genuinely elegant. However, the empirical results reveal an important tension: the method's strongest performance is on single-manifold settings where the geometry is correctly specified, while in genuinely mixed-curvature product settings, the simpler k-NN baseline remains highly competitive. This suggests that the value of geometry-aware decision trees may be strongest when the manifold structure is simple and well-understood, and weaker when signature selection and embedding quality introduce compounding uncertainties—an important caveat the community should internalize.

## Suggestions

- Replace "maximum-margin" with the more precise "geodesically equidistant" or "centered" throughout, or provide a formal theorem with proof specifying the margin definition.
- Add product-space SVM (not just perceptron) from Tabaghi et al. (2021) as a baseline to properly contextualize nonlinear manifold-aware classification performance.
- Include an ablation comparing basis-only projections vs. all $\binom{D}{2}$ projections to isolate the geometric contribution.
- Soften claims about product-manifold superiority to reflect the nuanced empirical picture—use "competitive" rather than "superior" for the multi-curvature setting, and explicitly discuss k-NN dominance on synthetic product classification.

## Score and Decision

**Calibration comparison:**

- **HyperDT** (TTonmgTT9X.md): scores 5,6,8,6,8 (avg ~6.6), accepted poster. This extends DTs to hyperbolic space only. Our paper has broader scope (all constant-curvature + product) but weaker empirical results on the novel portion.
- **Spectro-Riemannian GNN** (2MLvV7fvAz.md): scores 6,5,6,6 (avg ~5.75), accepted poster. Product manifold GNN with similar baseline concerns and modest empirical gains.
- **Mixed-curvature Transformers** (AN5uo4ByWH.md): scores 1,5,5 (avg ~3.7), rejected. Substantially weaker contribution.

Our paper is below HyperDT (narrower but cleaner results, no overclaiming) and around or slightly below Spectro-Riemannian GNN (similar baseline gaps, similar scope). The overclaiming of "maximum-margin" and the modest product-manifold results offset the broader scope and elegant formulation.

MY FINAL SCORE: 5.5
MY FINAL DECISION: <orange>Reject</orange>