## Summary

This paper introduces mixed-curvature decision trees (DTs) and random forests (RFs) for product manifolds—Cartesian products of hyperbolic, hyperspherical, and Euclidean components. The core technical contribution is an angular reformulation of decision splits, parameterizing boundaries as angles in 2D subspaces, which the authors claim ensures geodesic convexity and maximum-margin separation uniformly across all constant-curvature manifolds. The approach is evaluated on 57 benchmarks spanning synthetic data, graph embeddings, VAE latent spaces, and empirical geospatial/temporal datasets.

---

## Strengths

- **Elegant unified angular framework.** Representing data and splits as angles in 2D projected subspaces (Eqs. 15–16) provides a single algorithm that naturally handles Euclidean, hyperbolic, and hyperspherical components without distinct branches per curvature type. The equivalence of the angular Euclidean split to classical thresholding (Appendix C) is a non-trivial but clean result.

- **Strong single-manifold performance with statistical rigour.** On single-curvature benchmarks, the method ranks first on 21 of 22 tasks (10/11 classification, 11/11 regression in Figures 3–4), with Bonferroni-corrected Wilcoxon significance marking. These results are consistent and convincing, not marginal.

- **Geometrically coherent decision boundaries on spherical data.** Figure 5 provides a compelling qualitative demonstration: the product-space RF learns smooth, great-circle-aligned boundaries on the landmasses dataset, whereas Euclidean and tangent baselines produce block-like artifacts and k-NN produces fragmented regions. This directly illustrates the inductive advantage of the approach.

- **Strong VAE latent-space results.** On Blood, CIFAR-100, and Lymphoma datasets embedded via mixed-curvature VAEs, the product method outperforms both ambient and k-NN baselines (Table 2), suggesting the method genuinely exploits product manifold structure when the embedding quality is appropriate.

- **Genuinely comprehensive benchmark suite.** 57 diverse benchmarks across classification, regression, and link prediction in multiple geometric settings, with confidence intervals and corrected significance tests, exceeds typical evaluation depth in geometric ML papers.

---

## Weaknesses

1. **The "maximum-margin" claim is unsubstantiated and likely incorrect as stated.** The abstract and Contribution 1 assert that splits are "maximum-margin," yet the algorithm in Section 2.2 is CART-based greedy information gain maximization (Eq. 13)—a standard impurity criterion with no margin-maximization objective. The term "maximum-margin" has a specific meaning (SVM-style geometric margin). No definition of margin in this curved context is given, no derivation is provided, and no connection to a margin-maximization objective is made anywhere in the main text. This is a materially misleading claim about the method's theoretical properties and must either be precisely defined and proven or removed.

2. **k-NN dominates on all 8 synthetic product-manifold classification tasks—the setting most favorable to the proposed method.** Table 2 shows k-NN with the correct product distance outperforming the proposed method on every single "Synthetic (multi-K) Gaussian" row, often by substantial margins (e.g., H⁴: k-NN 47.5 vs. Product 40.0). Since this synthetic data is *generated from Gaussian mixtures in the exact product manifold assumed at inference time*, this is the hardest possible test for k-NN (it must use the right distance) and the easiest for DTs (the generative model is compatible with the split structure). The paper provides no explanation for this gap in the main text. This substantially weakens the claim that geometry-aware DTs outperform distance-based methods on product manifold data.

3. **Unexplained failures on empirical regression datasets.** In Table 3, the proposed Product method is the worst performer on both Temperature (S²×S¹) and Traffic (E¹×(S¹)⁴): Product RMSE 7.130 vs. Ambient 4.531 on Temperature, and Product 0.534 vs. Ambient 0.505 on Traffic. These are precisely the datasets where non-Euclidean geometry should provide the strongest benefit (geospatial data on a sphere, cyclic time series on circles). The main text offers no analysis of why the geometry-aware method performs worse than ignoring manifold structure. Without explanation, these failures call into question the generality of the approach.

4. **Arbitrary north-pole choice for hyperspherical splits introduces unexamined rotation sensitivity.** Section 3.3 acknowledges there is no canonical x₀ in S^n and fixes the first embedding dimension as the "north pole." This means that two equivalent embeddings related by a rotation yield entirely different split structures. The Limitations section mentions "lack of a privileged basis" but does not analyze the practical impact. For graph embeddings where the rotation of the learned embedding is arbitrary, this could significantly affect performance reproducibility and is a principled gap in the method's foundation.

5. **Neural baselines confined to appendix.** Comparisons to MLPs and GNNs appear only in Appendix I, with no main-text result. Given that the paper positions the method as a practical tool between "underpowered linear classifiers and powerful but uninterpretable neural networks," knowing the performance gap relative to neural approaches is essential context for assessing the method's practical utility.

---

## Nice-to-Haves

- **Analyze the k-NN vs. Product gap on synthetic data.** Even a brief discussion of whether this reflects the curse of dimensionality in product spaces, or a fundamental incompatibility between Gaussian mixture decision boundaries and axis-aligned splits, would substantially improve the paper's scientific contribution.
- **Ablate geometric midpoint formulas vs. Euclidean averaging** to isolate whether the manifold-specific midpoint calculations or simply the angular parameterization drive performance gains.
- **Test robustness to misspecified signatures**, e.g., running a model assuming E⁴ on data that lives in H²S², to characterize practical robustness when the signature is imperfect.
- **Summarize runtime comparisons in the main text.** The O(D²) split search is mentioned in Section 3.4 but runtime results are relegated to Appendix J. Even a single-sentence summary of overhead vs. ambient-space DTs would be helpful.
- **Move at least one MLP/GNN comparison table into the main paper** or a condensed comparison figure, even if only for a subset of benchmarks.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Contribution synthesis" novelty concern (Harsh Critic).** The critic flags that the paper merely combines Chlenski et al. (2024) and Tabaghi et al. (2021). While the inspiration is explicitly acknowledged, the extension to hyperspherical space, the unified product-manifold framework, and the composability argument represent genuine additions, not just concatenation of prior work.
- **Eq. 19 typographic inconsistency (Harsh Critic).** The formula in Eq. (19) is unusual-looking but the paper explicitly defers to Appendix C for the proof of equivalence. Without access to the appendix proof, claiming this is an error is unfounded.
- **Contribution 3 (Gaussian sampling) is overstatement (Harsh Critic).** While this is a secondary contribution, it enables the entire synthetic benchmark suite and is properly described as a novel extension of non-Euclidean sampling to product manifolds.
- **Signature selection data leakage concern (Harsh Critic).** Selection is based on metric distortion using graph structure (not labels), so this is not label leakage in the traditional sense. This is a known practice in product manifold embedding and the concern is overstated.
- **"Dependency on pre-computed embeddings" as a weakness (Balanced Reviewer).** The paper explicitly scopes itself as a downstream inference tool and the Limitations section addresses this clearly. Criticizing it for not doing joint embedding+inference is scope creep.
- **"Requesting confidence intervals in Tables 2/3" (Harsh Critic).** Tables 2 and 3 do report 95% CIs (shown in the ± notation). The significance asterisks appear in Figures 3/4 where per-curvature comparisons are appropriate. The reporting convention is adequate.
- **"Comparison unfairness" concerns.** The baselines (Ambient, Tangent) intentionally use Euclidean-space classifiers on non-Euclidean data—this favors the baselines (they avoid O(D²) search costs) and the asymmetry benefits the baseline, not the authors' method.

---

## Novel Insights

The most genuinely insightful observation across the three reviews—one the authors themselves do not fully develop—is the **tension between k-NN optimality and tree-based splitting on the same product manifold data**. On synthetic data generated *from the exact generative model the product DT is designed for*, k-NN dominates. This suggests that axis-aligned angular splits, while geodesically coherent, may be poorly matched to the decision boundaries induced by Gaussian mixtures in curved product spaces. Conversely, on VAE latent spaces (where the generative process is more complex and not aligned with any simple geometric prior), the product DT shines. This pattern suggests the method's true niche is *empirical and latent-space data* where the manifold structure is imposed by a learned encoder rather than a synthetic Gaussian process—a distinction the paper does not draw explicitly but that would sharpen its contribution statement considerably.

---

## Suggestions

1. **Either prove or remove the maximum-margin claim.** If the angular CART splitting procedure has a maximum-margin interpretation in any precise geometric sense, derive it (even a sketch in the main text with a full proof in the appendix). If not, replace with "geodesically convex" throughout—a claim that is supported.

2. **Add a dedicated paragraph explaining the k-NN gap on synthetic data**, even if only to hypothesize the mechanism. Options include: (a) k-NN benefits from exact distance computation whereas DTs use coordinate projections; (b) Gaussian mixture boundaries in product spaces are not well-approximated by axis-aligned angular splits; (c) dimensionality effects. Any explanation is more satisfying than silence.

3. **Add a paragraph explaining the Temperature and Traffic regression failures.** Are these due to poor embedding quality (i.e., the signature S²×S¹ or E¹×(S¹)⁴ does not fit the data well)? If so, that is a limitation of signature selection, not the DT algorithm, and should be stated clearly.

4. **Directly address rotation sensitivity for hyperspherical components.** A brief experiment (e.g., measuring performance variance under random SO(n) rotations of the embedding before fitting the DT) would quantify the practical impact of the arbitrary north-pole choice and either reassure readers or identify a genuine robustness problem.

5. **Move the most informative MLP/GNN comparison row(s) into the main text.** This does not require a full table—a single well-chosen comparison (e.g., on Cora or CIFAR-100) gives readers the information they need to calibrate the method's position in the landscape.

---

**Evaluation on key axes:**

- **Novelty:** Moderate. The angular unification is elegant and the product-manifold extension is new; however, the reliance on Chlenski et al. (2024) for the angular perspective and Tabaghi et al. (2021) for the product-space framework means the novelty is incremental synthesis rather than a conceptual breakthrough.
- **Technical soundness:** Moderate. The geodesic convexity arguments are principled, the angular reformulation is well-defined, and the equivalence proof for Euclidean DTs is deferred but cited. The "maximum-margin" claim is an unsupported assertion that overstates the theory.
- **Empirical support:** Mixed. Single-manifold benchmarks are compelling. Product-manifold benchmarks are disappointing on the synthetic and two empirical regression tasks. VAE results are encouraging. The unexplained failures are the most significant empirical issue.
- **Significance:** Moderate. Fills a genuine gap in inference tools for product manifold embeddings. Practical utility is real but narrower than claimed—the method works best on learned latent representations, less well on distance-based synthetic data.
- **Clarity:** Good for the geometric core; too brief in Section 3.4 and silent on empirical failures.

MY FINAL SCORE: <pineapple>5.2</pineapple>