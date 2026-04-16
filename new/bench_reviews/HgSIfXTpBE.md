## Summary

The paper introduces decision trees (DTs) and random forests (RFs) for mixed-curvature product manifolds—Cartesian products of spherical, Euclidean, and hyperbolic components—by reformulating splits as angle thresholding on 2D projections. This angular formulation ensures splits are geodesically convex and composable across components, providing a unified framework that subsumes prior Euclidean and hyperbolic DTs while introducing novel spherical DTs. The method is evaluated on 57 benchmarks spanning synthetic data, graph embeddings, VAE latent spaces, and empirical datasets.

## Strengths

- **Unified and principled angular framework.** The reformulation of splits as angle thresholding on 2D projections (Eqs. 15–16) provides a clean, unified treatment across all three constant-curvature geometries. The observation that homogeneous hyperplanes yield geodesically convex decision regions in any constant-curvature manifold, and that this reduces to angular thresholding, is elegant and well-motivated.

- **Strong single-curvature results.** On synthetic single-curvature benchmarks, the method achieves top-1 F1 in 10/11 classification signatures (Figure 3) and top-1 RMSE in all 11 regression signatures (Figure 4), with statistical significance (Bonferroni-corrected) in most cases. This convincingly demonstrates the value of geometry-aware splits on single-manifold data.

- **Novel hyperspherical DT.** The paper introduces DTs for hyperspherical manifolds (Section 3.3), filling a genuine gap in the literature with a notably simple midpoint formula (Eq. 22).

- **Extensive evaluation.** The 57 benchmarks across diverse tasks (classification, regression, link prediction), data types (synthetic, graph, VAE, empirical), and signatures provide a thorough assessment. The world-map visualization (Figure 5) is a compelling qualitative demonstration.

- **Practical algorithmic design.** The angular formulation enables O(1) decision complexity, allows subsampling angular features in RFs (Section 3.4, advantage 2), and permits incorporating metadata as additional Euclidean components—practical benefits beyond the theoretical contribution.

## Weaknesses

### Major

- **"Maximum-margin" claim is unsupported.** The abstract and contributions (contribution 1) claim splits are "maximum-margin," but the paper never formally defines margin on a manifold, nor proves that the angular threshold maximizing information gain (Eq. 13) achieves maximum margin under any such definition. The impurity-based greedy split selection in CART is different from margin maximization. This is a substantive overclaim on a property central to the paper's narrative about geometry-awareness. Without a definition, proof, or even empirical margin analysis, this claim should be retracted or significantly qualified.

- **Product-manifold results are mixed despite being the paper's core contribution.** On the 8 synthetic multi-K classification benchmarks (Table 2, the primary test of product-manifold advantage), k-NN achieves top-1 on all 8, with product DTs/RFs consistently second and often substantially behind (e.g., H⁴: 47.5 vs 40.0; (H²)²: 41.5 vs 37.0). On link prediction, k-NN is best on 5/6 datasets. On empirical product-manifold regression, the product method loses to both ambient RF and k-NN on Temperature and Traffic (Table 3). The abstract claims "ranked first on 21 of 22 single-manifold benchmarks and 18 of 35 product manifold benchmarks," but this obscures that on the canonical product-manifold benchmarks, the method is usually second-best to a simple distance-based method. The paper should position product DTs/RFs as interpretable, competitive alternatives rather than uniformly superior.

- **Missing comparison to prior hyperbolic DT/RF methods.** The paper explicitly claims to generalize Chlenski et al. (2024) and Doorenbos et al. (2023), yet never benchmarks against them on hyperbolic data. If the angular reformulation truly subsumes these methods, a direct comparison showing equivalent or improved performance would strengthen the generalization claim. Without it, it is impossible to verify whether the reformulation preserves performance on the single-curvature setting it subsumes.

### Minor

- **Notation confusion in Euclidean reformulation (Eq. 19).** The expression $\tan^{-1}(\theta_u)$ where $\theta_u$ is already an angle is dimensionally confusing; it appears to mean $\tan^{-1}$ applied to an angle, which would compose arctan with arctan. Eq. 18 appears correct (the midpoint angle for axis-aligned threshold $(u_d+v_d)/2$ lifts to $\text{atan}(2/(u_d+v_d))$ under the embedding $\phi$), but Eq. 19 is opaque as written. The appendix proof is referenced but not in the main text. This does not undermine the method (Eq. 18 is sufficient), but it mars the paper's claim to a clean unified presentation.

- **No product vs. per-component ablation.** Section 3.4 argues that allowing a single DT to span all components enables the model to "independently allocate splits across components according to their relevance," but no experiment validates this design choice against independently training trees on each component. Without this ablation, it is unclear whether the product structure provides benefit beyond feature access.

- **Signature must be known a priori, with no sensitivity analysis.** The method requires specifying the manifold signature (component types, dimensions, curvatures) in advance. While acknowledged in Limitations, no analysis is provided on how performance degrades under signature misspecification—e.g., treating a hyperbolic component as Euclidean, or using incorrect curvature.

- **Key neural baselines relegated to appendix.** The main text compares only against k-NN, ambient/tangent DTs/RFs, and perceptron. MLP and GNN comparisons are mentioned in the appendix but not surfaced in the main results. Given that product-manifold representations are often produced by neural models, these comparisons matter for practical utility.

## Nice-to-Haves

- Visualize decision boundaries on hyperbolic and product manifolds (only S² is visualized in Figure 5); this would directly verify whether the claimed geodesically convex splits produce meaningful boundaries in non-spherical geometries.
- Report DT and RF results separately rather than max(DT, RF), so readers can assess the contribution of the tree algorithm independently of the ensemble.
- Runtime comparison on identical hardware across all methods, to help practitioners weigh the angular formulation's preprocessing overhead.
- Ablation on higher-dimensional manifolds (all benchmarks use D=2 or D=4 component manifolds) to validate the generality claim.

## Removed Points

- **Claim that Euclidean reformulation is "mathematically inconsistent."** The harsh critic alleged Eq. 18 is inconsistent with conventional thresholding. However, Eq. 18 itself is derivable: the standard threshold $x_d > (u_d+v_d)/2$ under the lift $\phi(u)=(1,u)$ becomes $\text{atan}(1/((u_d+v_d)/2)) = \text{atan}(2/(u_d+v_d))$, which matches Eq. 18. The issue is confined to Eq. 19's notation—confusing, but not evidence of mathematical inconsistency in the method itself. Appendix C presumably contains the formal proof.
- **Criticism about "lack of formal description" for product-tree training (Section 3.4).** The reviewer demanded main-text pseudocode, but the paper references Appendix B for this. Pseudocode in an appendix is standard practice and not a methodological gap.
- **CART's greedy suboptimality as a weakness.** This is inherent to all CART-based methods and not specific to the paper's contribution; it applies equally to standard RFs.
- **Demand for confidence intervals/reproducibility concerns about hyperparameters.** Single-run evaluation with Wilcoxon tests is standard in this community; undisclosed hyperparameters are trivial relative to the contribution.
- **Criticism that ambient/tangent baselines are "weak."** These are natural baselines for manifold-structured data; the fact that they lack geometry awareness is precisely the point of comparison.

## Novel Insights

The paper reveals an important empirical asymmetry: geometry-aware decision trees achieve substantial gains over naïve approaches on single-manifold data, but on product manifolds—where the geometry is most complex and the method should have its greatest advantage—a simple k-NN on intrinsic manifold distances frequently outperforms. This suggests that the value of product DTs/RFs lies more in interpretability and composability than in raw predictive accuracy, and that the angular split formulation may not be expressive enough to fully exploit mixed-curvature structure. The paper's own Limitations section hints at this (lack of privileged basis in embeddings), but the empirical evidence makes the point more sharply than the text acknowledges.

## Suggestions

- Retract or substantially qualify the "maximum-margin" claim; either formally define and prove it, or replace it with the accurate description ("splits are geometry-aware and respect manifold structure").
- Re-position product DTs/RFs as interpretable, geometry-respecting methods that are competitive (often top-2) but not uniformly dominant on product-manifold data. Acknowledge k-NN's strength as a baseline and discuss regimes where trees are preferable (interpretability, latency, high dimensions).
- Add a direct comparison with Chlenski et al. (2024) on hyperbolic benchmarks to validate the generalization claim.
- Include a product-vs-per-component ablation to justify the core design choice.

## Score and Decision

**Calibration papers:**
- HyperDT (extending DTs to hyperbolic space): human scores 5,6,8,6,8 → poster
- CUSP (mixed-curvature GNN): scores 6,5,6,6 → poster
- Curve Your Attention (mixed-curvature transformers): scores 1,5,5 → reject

This paper has broader scope than HyperDT (product manifolds vs. just hyperbolic) and a stronger empirical sweep (57 benchmarks), but it overclaims ("maximum-margin") and its core product-manifold results are weaker than the narrative suggests (k-NN wins on most multi-curvature benchmarks). Relative to HyperDT (avg ~6.5), this paper's stronger overclaiming and weaker product-manifold results offset its broader scope. Relative to CUSP (avg ~5.75), this paper has a clearer methodological contribution but similar issues with mixed empirical results. I place this below HyperDT but comparable to CUSP.

MY FINAL SCORE: 5
MY FINAL DECISION: Reject