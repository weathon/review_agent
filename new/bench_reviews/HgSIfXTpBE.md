Now I have all the information I need to write the final review.

## Summary

The paper introduces decision trees (DTs) and random forests (RFs) for product manifolds—Cartesian products of hyperbolic, hyperspherical, and Euclidean components—via a unified angular-split reformulation. Splits are parameterized as angle thresholds under 2-D projections of the manifold coordinates (Eqs. 15–16), with manifold-specific midpoint formulas ensuring geodesic equidistance. This framework generalizes existing hyperbolic DTs, introduces novel hyperspherical DTs, and provides product-manifold DTs/RFs that span all components simultaneously. The method is evaluated on 57 benchmarks spanning synthetic, graph, VAE, and empirical data.

## Strengths

- **Unified angular reformulation across all constant-curvature manifolds:** The key insight—reframing DT splits as angle thresholding under 2-D projections (Eqs. 15–16)—provides a clean, consistent framework that works across Euclidean, hyperbolic, and spherical spaces. This generalizes Chlenski et al. (2024)'s hyperbolic approach and introduces a novel hyperspherical DT algorithm (Section 3.3, Eq. 22). The observation that homogeneous hyperplanes are geodesically convex in *any* constant-curvature manifold (Section 3, paragraph 2) is a solid geometric justification.

- **Comprehensive benchmark evaluation:** 57 benchmarks across synthetic data, graph embeddings, VAE latent spaces, and empirical datasets (Table 1) is more thorough than typical work in this niche. Top-2 on 53/57 (93%) benchmarks overall is a strong result.

- **Strong single-curvature results:** On single-manifold synthetic benchmarks, the method dominates: 10/11 classification wins (Figure 3) and 11/11 regression wins (Figure 4), most with Bonferroni-corrected statistical significance.

- **Compelling qualitative visualization:** Figure 5 clearly demonstrates that product-space RF produces smooth, geodesically coherent decision boundaries on S², while Euclidean/tangent RFs produce blocky artifacts—providing intuitive evidence for the value of geometry-aware splitting.

- **Practical computational advantages:** The angular formulation preserves O(1) decision complexity per split and enables feature subsampling in RFs without enforcing manifold constraints at inference time (Section 3.4, advantages 1–2), making the approach practical for higher-dimensional product manifolds.

## Weaknesses

### Fatal
None.

### Major

- **"Maximum-margin" claim is asserted without proof or precise definition.** The abstract and contribution #1 (line 39) prominently claim splits are "maximum-margin," but the paper never proves this property or precisely defines what "maximum-margin" means in this context. What the method actually does is place splits at geodesic midpoints between boundary points (Section 3, p. 264: "geodesically equidistant the two points to either side of it"), which is a *local* property of the greedy split—not the global SVM notion of maximum margin. The paper would benefit from either proving the claim under a stated definition or replacing "maximum-margin" with the more precise "geodesically equidistant" or "midpoint" throughout. The current phrasing misleadingly suggests a stronger theoretical guarantee than the method provides.

- **k-NN dominates on synthetic multi-curvature classification, with no discussion or analysis.** On all 8 synthetic multi-curvature classification benchmarks (Table 2, rows 1–8), k-NN achieves the best F₁ score, with gaps as large as 7.5 points (e.g., H⁴: 47.5 vs. 40.0). This is the controlled setting where ground-truth manifold structure is known—the very setting the method is designed for. The paper does not acknowledge or analyze this pattern anywhere. The abstract emphasizes the 21/22 single-manifold result while burying the 18/35 product-manifold result, which is partly propped up by VAE/empirical datasets. Understanding *why* k-NN wins here (axis-aligned restriction? depth limits? fundamental DT limitation in product manifolds?) is important for practitioners to know when to deploy this method versus simpler alternatives.

### Minor

- **max(DT, RF) reporting convention inflates all methods and complicates win-counting.** Tables 2–3 report max(mean(Ambient DT), mean(Ambient RF)) per cell, selecting the better of two models post hoc. While this inflates every method equally, it makes the "top-1"/"top-2" counting in Table 1 less reliable and obscures whether the performance comes from geometry-aware splits or from ensembling. Reporting DT and RF separately would clarify the individual contributions (Section 4.3, Table 2 caption).

- **Limited nonlinear baseline comparison on product manifolds.** The baseline set includes ambient/tangent DTs, RFs, k-NN, and the product perceptron—but no kernel methods, oblique DTs, or neural baselines beyond the brief appendix comparison. Tabaghi et al. (2021) also describes a product-manifold SVM, which is a stronger linear competitor than the perceptron that the paper already dismisses. The paper notes it "omit[s] product space perceptrons, which never achieved competitive results" but does not evaluate the SVM.

### Trivial
None.

## Nice-to-Haves

- A decision boundary visualization on product-manifold data (not just S²) showing how splits interact across components would strengthen the geometric intuition for the method's core contribution.
- An ablation investigating why k-NN dominates on synthetic multi-curvature classification (e.g., depth limit, axis-aligned restriction, component-wise allocation).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Composable" is trivially true of any DT** (Harsh Critic): The paper uses "composable" in a specific geometric sense—that angular splits on different components can be composed within a single tree to form valid decision regions on the product manifold (Section 3.4). This is not trivially true of, e.g., tangent-plane DTs, which project all components into a flat space and lose the per-component geometry. Removed as a strawman criticism.

- **Missing Appendix C proof for Euclidean DT equivalence** (Harsh Critic/Strength Finder): The parser strips appendices; Appendix C exists in the original submission. Removed as a parser artifact complaint.

- **Demand for product-manifold SVM comparison as a fatal issue** (Harsh Critic, elevated): The paper's stated scope is extending DTs/RFs to product manifolds; the SVM from Tabaghi et al. is a linear classifier and is a different algorithm class. Not including it is a reasonable baseline selection, not a fatal flaw. Demoted to Minor.

- **Insufficient related works** (implicit): Not verifiable without external knowledge. Removed per rules.

- **Reproducibility concerns about undisclosed hyperparameters** (implicit): The paper references standard Scikit-Learn hyperparameters and provides algorithm pseudocode in Appendix B. Removed as a generic nitpick per rules.

## Novel Insights

The paper's most underappreciated tension is between the elegant theoretical framework and the empirical reality that k-NN with product-manifold distances is a surprisingly strong baseline for classification. This suggests that the value of product DTs/RFs lies less in classification accuracy per se and more in their dual advantage of interpretability (Figure 5's smooth, understandable boundaries) plus their ability to win on regression tasks and VAE/empirical data where the manifold structure may be less clean. The paper's framing as a "straightforward yet powerful new tool" would be better served by explicitly positioning the method as a geometry-aware, interpretable alternative rather than competing head-to-head with k-NN on pure accuracy.

## Suggestions

- Replace "maximum-margin" with "geodesically equidistant" or "midpoint" throughout the abstract and contributions, or provide a formal definition and proof of the maximum-margin property under that definition.
- Add a paragraph in Section 4.4 analyzing the k-NN dominance on synthetic multi-curvature classification and discussing when product DTs/RFs are the right tool (e.g., interpretability, regression, empirical/VAE data with less clean structure).
- Report DT and RF results separately in at least one key table so readers can assess whether the performance is driven by manifold-aware splitting or by ensembling.

## Score and Decision

**Evaluation summary:**
- **Originality:** The unified angular reformulation is a genuine conceptual contribution that simplifies and generalizes prior hyperbolic-only approaches. The hyperspherical DT is novel.
- **Research question importance:** Extending DTs/RFs to product manifolds fills a real gap in the non-Euclidean ML toolkit.
- **Claim support:** Claims are partially undermined by the overclaimed "maximum-margin" property and the unacknowledged k-NN dominance on the most controlled product-manifold benchmarks.
- **Experimental soundness:** 57 benchmarks is comprehensive; the max(DT,RF) convention is a minor methodological concern; baseline set is adequate but thin for nonlinear methods.
- **Clarity:** Generally well-written, with clean mathematical exposition and good visualizations.
- **Community value:** The method fills a practical gap for interpretable inference on product manifold data.

**Calibration:**
- *High anchors (avg >7):* Oblique Decision Forests (7.0) — stronger theoretical grounding, narrower scope but cleaner claims. This paper's overclaiming puts it below.
- *Medium anchors (avg 4–6):* RankSHAP (6.5) — extends classical method with moderate contribution; this paper has a similar profile. CUSP mixed-curvature GNN (5.75) — comparable domain but better baseline coverage. Solv geometry (4.5) — comparable overclaiming but weaker methodology.
- *Low anchors (avg <3):* Optimal spherical codes (2.5) — this paper is clearly stronger with genuine contributions and thorough evaluation.

This paper sits between the medium anchors: it has a cleaner conceptual contribution than CUSP but weaker empirical analysis of failure modes than RankSHAP. The overclaimed "maximum-margin" and unaddressed k-NN gap on synthetic data are real issues but don't invalidate the core contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>