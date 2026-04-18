Now I have a thorough understanding of the calibration papers. Let me now synthesize the final review based on the actual paper content and the three input reviews.

Key calibration anchors:
- HyperDT paper (TTonmgTT9X.md): scores 5,6,8,6,8, accepted as poster — similar topic (hyperbolic DTs), the predecessor work
- Mixed-Curvature Transformers (AN5uo4ByWH.md): scores 1,5,5, withdrawn/reject — similar product manifold topic but with novelty issues
- Spectro-Riemannian GNNs (2MLvV7fvAz.md): scores 6,5,6,6, accepted poster — mixed-curvature ML
- Deepfake in Curved Spaces (G4D6jClNFl.md): scores 5,6,5,3, rejected — novelty concerns with mixed-curvature methods
- Graph representations in Normed Spaces (rcbKbaLxGd.md): scores 3,8,3,3, withdrawn/reject — novelty and application concerns

Let me now carefully evaluate the paper's claims against the reviewer criticisms.

**Verification of harsh critic's key claims:**

1. **Mathematical dubiousness of angular split construction**: Let me check the specific claims.

- Eq. 15 uses θ(x,d) = atan(x₀/x_d). For Euclidean case with x₀=1 (from the lift in Eq. 17), this becomes θ = atan(1/x_d). The split S(x,d,θ) = I{θ(x,d) ∈ [θ, θ+π)}. For the Euclidean lifted case, since x₀=1, θ = atan(1/x_d) which is a monotonically decreasing function of x_d, so thresholding on θ IS equivalent to thresholding on x_d. The appendix proof is referenced. So the equivalence claim seems correct.

- Eq. 18-19: The formulas look like they have a notation issue with nested atan, but the paper says "In our implementation, we use the PyTorch atan2 function" and refers to the appendix. The formulas in the main text may be garbled in rendering. Let me check more carefully... Eq. 18 says m_E(u,v) = atan(2/(u_d + v_d)) and Eq. 19 says = atan(tan(θ_u)·tan(θ_v)/(tan(θ_u) + tan(θ_v))). Since θ_u = atan(1/u_d) and θ_v = atan(1/v_d), tan(θ_u) = 1/u_d, tan(θ_v) = 1/v_d. So Eq. 19 becomes atan((1/u_d · 1/v_d)/(1/u_d + 1/v_d)) = atan(1/(u_d+v_d)/2) = atan(2/(u_d+v_d)). This matches Eq. 18! So the formulas are actually consistent. The critic's claim that Eq. 18-19 are "very likely incorrect" appears to be wrong — they are equivalent expressions for the same midpoint.

- **Spherical x₀ choice**: The critic raises a valid concern that fixing the first coordinate as "north pole" breaks rotational symmetry. This is true and acknowledged in the paper ("we adopt the convention of fixing the first dimension"). However, this is somewhat analogous to choosing a coordinate system — the method iterates over all choices of d, so it does explore splits relative to different axes. The choice of x₀ is an inductive bias, not a bug, but the critic is right that it's not intrinsic to the manifold.

- **"arbitrary" projections claim**: The paper states in Section 3.4 advantage 1 that they "can consider angles under arbitrary linear projections... while maintaining O(1) decision complexity. For instance, we can easily search over all (D choose 2) 2-dimensional projections if we wish." Looking at the actual implementation, if they use all (D choose 2) projections, each projection considers splitting on a pair of coordinates, which means the number of features is O(D²). The paper says they "consider all (D choose 2) projections — for a total of 3 features per 2-dimensional component manifold." Wait, let me re-read: "we consider all (D choose 2) projections—for a total of 3 features per 2-dimensional component manifold." This is saying there are D=2 per component, so (2 choose 2) = 1 projection → 3 features (x₀ coordinate, the angle θ, and... actually for a 2D subspace with coordinates x₀ and x_d, the features include x₀, x_d, and maybe the angle). Actually the phrasing "3 features per 2-dimensional component manifold, just like ambient space methods use" means that for each component manifold of dimension D_i, they have D_i * (D_i+1)/2 or actually... Hmm. Let me re-read: "we consider all (D choose 2) projections—for a total of 3 features per 2-dimensional component manifold, just like ambient space methods use." This is a bit confusing. For a component manifold with dimension D (embedded in D+1), the number of possible sparse projections is D (choosing which dimension d to pair with the fixed x₀). If they also allow all (D choose 2) projections... Actually, re-reading, the "3 features" seems to refer to the fact that for each 2D subspace (x₀, x_d), you get one angle feature. Plus the raw coordinates. But the overall claim about "arbitrary" projections appears to be aspirational rather than fully implemented — the actual implementation uses only x₀-x_d pairs (the "sparse" approach from Chlenski et al.).

Actually wait, re-reading Section 3.4 more carefully: "allowing for a single DT to span all components" and "we can consider angles under arbitrary linear projections (not just projections onto basis dimensions) while maintaining O(1) decision complexity." This is a claimed advantage of the angular formulation — that you COULD also search over arbitrary 2D projections. But in the experiments, they say "we set hyperparameters identically to Scikit-Learn DTs and RFs, except we consider all (D choose 2) projections—for a total of 3 features per 2-dimensional component manifold." This is confusing but it appears they mean they use the basis projections (x₀ paired with each x_d), not truly arbitrary rotations. The (D choose 2) is probably a typo or misstatement — they likely mean D projections (one for each dimension d paired with x₀).

The "arbitrary projections" advantage is a claimed advantage that is NOT implemented in the experiments. This is a valid criticism but not necessarily fatal — it's an aspirational claim about the framework's extensibility.

- **"maximum-margin" claim**: The paper says splits are "maximum-margin" in the abstract and contribution list. Looking at the paper, in Section 3, after Eq. 16, it says "geodesically convex, maximum-margin, and composable." The "maximum-margin" comes from the midpoint choice — by choosing the hyperplane that is equidistant (in geodesic terms) between the two nearest points of opposite classes, they're making a maximum-margin-like choice. However, this is a heuristic rather than a formal maximum-margin guarantee, because in CART-style trees, the split is chosen greedily to maximize information gain, and the midpoint is used as the threshold after sorting by angle. The midpoint construction ensures the split is balanced, but calling it "maximum-margin" is indeed an overclaim without formal proof.

2. **Baseline configuration**: The baselines include ambient DT/RF, tangent DT/RF, k-NN, and product perceptron. The product perceptron is noted as "never competitive." Missing baselines like product-manifold SVMs, manifold neural networks, etc. are valid concerns. However, the paper is about DTs and RFs — it's reasonable to compare to other DT/RF approaches as primary baselines and k-NN as a simple manifold-aware baseline.

The feature space asymmetry is a valid point — product DTs use angular features while baselines use raw coordinates. But this is the POINT of the contribution: using manifold-appropriate features. The question is whether it's a fair comparison.

Looking at Tables 2-3, k-NN beats the product DT/RF on ALL 8 synthetic multi-K classification benchmarks and several other benchmarks. This is a significant weakness that the paper glosses over.

3. **Conflation of embedding quality vs. tree quality**: This is a valid concern but somewhat unfair — the paper's stated scope is "given product manifold embeddings, provide good downstream inference." The paper explicitly states "we view our work as downstream of signature selection and embedding generation."

Let me now write my consolidated review.

**Summary of key findings:**

The paper genuinely extends DTs/RFs to product manifolds via an angular reformulation. The core idea is sound — angles in 2D projections of constant-curvature manifolds give geodesically convex splits. The method achieves top-2 on 53/57 benchmarks. However:

Real weaknesses:
- "Maximum-margin" claim is not formally justified — it's a heuristic midpoint choice, not a proven margin maximizer
- The spherical case's arbitrary pole choice breaks rotational invariance, which is a real limitation
- Product-manifold results are notably weaker than single-manifold results (46% top-1 vs 95% top-1), with k-NN consistently beating the method on synthetic multi-K classification
- Missing stronger manifold-aware baselines (SVM, logistic regression)
- Some overclaiming (arbitrary projections claim, "maximum-margin")
- The product DT essentially treats component angles as independent features, which is less geometrically deep than advertised

Calibration: This paper is directly comparable to HyperDT (scores 5,6,8,6,8, mean ~6.6, accepted as poster). It extends HyperDT to product manifolds and adds spherical DTs. The contribution is incremental relative to HyperDT but broader in scope. The empirical results are more comprehensive but also more mixed. The mathematical gaps in the spherical/Euclidean derivations are concerning but likely fixable. Compared to HyperDT's reception, this paper has similar strengths (clean extension, good empirical coverage) but also has more weaknesses (weaker relative performance on the main claimed contribution, product manifolds; mathematical gaps in non-hyperbolic cases).

I would score this around 5.5-6 because it's a solid but incremental extension that makes some overclaims, with genuinely useful but modest contributions.</think>## Summary

This paper extends decision trees (DTs) and random forests (RFs) to product manifolds—Cartesian products of hyperbolic, hyperspherical, and Euclidean spaces—by reformulating splits as angular thresholds in 2-dimensional subspaces. The angular parameterization is claimed to yield geodesically convex, maximum-margin, and composable decision boundaries, and simplifies to existing Euclidean and hyperbolic DT formulations in special cases. The method is evaluated on 57 benchmarks spanning synthetic data, graph embeddings, VAE latent spaces, and empirical datasets.

## Strengths

- **Principled unification across curvature regimes.** The observation that homogeneous hyperplane splits in constant-curvature manifolds reduce to angular thresholding in 2D projections is elegant and enables a single algorithmic framework to handle Euclidean, hyperbolic, and spherical components. This genuinely unifies prior work on hyperbolic DTs (Chlenski et al., 2024) and extends it naturally.

- **Novel hyperspherical DT algorithm.** The spherical DT contribution (Section 3.3) appears to be new, with a simple midpoint formula (Eq. 22) that averages projection angles. This fills a gap in the non-Euclidean ML toolkit.

- **Comprehensive empirical evaluation.** The 57 benchmarks across classification, regression, and link prediction—covering synthetic, graph embedding, VAE, and empirical datasets—is substantially broader than typical papers in this area. The statistical significance testing with Bonferroni correction on the main synthetic experiments (Figures 3–4) adds credibility.

- **Strong single-manifold results.** Top-1 performance on 21/22 single-manifold benchmarks, with convincing margins over ambient, tangent, and k-NN baselines (Figures 3–4), demonstrates that the angular formulation works well when the manifold geometry is correctly specified.

- **Practical accessibility.** Building on the CART/RF framework with scikit-learn-style hyperparameters makes the method readily usable by practitioners.

- **Clear visualizations.** The landmasses example (Figure 5) effectively shows that manifold-aware decision boundaries produce more geometrically natural partitions than naive Euclidean/tangent approaches.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed "maximum-margin" property.** The abstract and introduction state that splits are "maximum-margin," and this is listed as a core contribution. However, the paper provides no formal proof of this property. The midpoint construction (especially Eq. 22 for the spherical case) ensures approximate balancing of splits, but maximum-margin in the geodesic sense would require proving that the chosen hyperplane maximizes the minimum geodesic distance to points of both classes—a claim not established, particularly for the spherical case where the angular coordinate is not a linear function of geodesic distance. This is a meaningful overclaim: the property is central to the method's conceptual narrative, yet only heuristic support is given.

- **Weaker product-manifold results undermine the central contribution.** The paper's main novelty is product-manifold DTs/RFs, yet on product-manifold classification benchmarks, the method achieves top-1 in only 11/24 cases (46%), with k-NN consistently outperforming it on all 8 synthetic multi-K classification benchmarks (Table 2). The abstract highlights "21 of 22 single-manifold benchmarks" but the product-manifold results—which are the paper's titular contribution—are noticeably weaker and this discrepancy is not discussed. This raises questions about the method's added value in the product-manifold setting specifically.

- **The spherical construction's arbitrary coordinate choice.** In Section 3.3, the first ambient coordinate is fixed as the "north pole" ($x_0$) for computing angles. This choice is not intrinsic to the manifold and breaks rotational invariance: different coordinate orientations would yield different splits and different trees. The paper does not discuss or analyze this inductive bias, which is a significant conceptual gap for a method that claims to respect manifold geometry. The hyperbolic case naturally has a preferred origin (due to the timelike coordinate), but the spherical case has no such preferred direction.

- **Missing manifold-aware baselines beyond k-NN.** The baselines are ambient-space DT/RF, tangent-space DT/RF, k-NN, and the product perceptron (which is excluded for being non-competitive). The paper's related work cites Tabaghi et al. (2021) who describe product-space perceptron AND SVM, yet no product SVM is included. No Riemannian logistic regression or manifold-aware shallow neural methods are compared. For a paper claiming to provide "powerful new tools for data analysis in product manifolds," the absence of other manifold-aware classifiers makes it difficult to assess whether DTs/RFs are the right tool rather than simply whether manifold-appropriate features help.

### Minor

- **The "arbitrary 2D projections" claim (Section 3.4, Advantage 1) is aspirational rather than demonstrated.** The paper states "we can consider angles under arbitrary linear projections (not just projections onto basis dimensions) while maintaining O(1) decision complexity," but the actual implementation uses only sparse basis projections ($x_0$ paired with $x_d$). The advantage is claimed but not empirically validated or complexity-analyzed for the general case.

- **Product DTs are essentially standard DTs over engineered angular features.** The product extension (Section 3.4) iterates over all component-wise angles using per-component midpoint formulas, without modeling cross-component geodesics or joint curvature effects. This is a reasonable engineering choice, but the conceptual framing suggests deeper geometric integration than is actually implemented. The paper would benefit from acknowledging this more explicitly.

- **Euclidean equivalence proof relegated to appendix.** Since the paper claims complete equivalence to standard CART for the Euclidean case, this proof is foundational to the generality claim and deserves at least a sketch in the main text.

- **Some real-data benchmarks show notable underperformance.** On Temperature regression (Table 3), the product model's RMSE (7.130) is far worse than ambient (4.531), yet this is not discussed. Understanding when and why the method fails would strengthen the paper.

### Trivial

- Equation notation in Eqs. 18–19 could be clearer; the nested arctangent expressions are difficult to parse without careful derivation.

## Nice-to-Haves

- Comparison to the Chlenski et al. (2024) hyperbolic RF on single-manifold hyperbolic benchmarks, to verify that the angular reformulation faithfully reproduces prior work's performance.
- Ablation studying robustness to signature misspecification (e.g., using a wrong signature) to assess practical applicability when the manifold structure is not known a priori.
- Sensitivity analysis regarding the choice of "north pole" in the spherical case.
- MLP/GNN baselines on the graph tasks to contextualize the DT/RF performance niche.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh critic: "Eq. 18–19 are very likely incorrect."** Verified against the paper: Eq. 18 gives the midpoint directly as atan(2/(u_d+v_d)), and Eq. 19 re-expresses it in terms of the projection angles. Since tan(θ_u)=1/u_d and tan(θ_v)=1/v_d, Eq. 19 reduces to atan(1/((1/u_d+1/v_d)/(1/(u_d·v_d)))) = atan(2/(u_d+v_d)), matching Eq. 18. The formulas are equivalent and correct. Removed because it is factually wrong.

- **Harsh critic: "Perceptron baseline is omitted from tables, reducing transparency."** The paper explicitly states in Figure 3 caption and Section 4.3 that the perceptron "never achieved competitive results" and is excluded for brevity. This is a reasonable editorial choice, not a transparency issue, especially given that full results are available. Removed as nitpick.

- **Neutral reviewer: "Computational cost discussion relegated to appendix."** The paper mentions runtime analysis is in Appendix J, which is standard for a methods paper. The O(1) per-decision claim is for individual split evaluations, which is correctly stated. Removed as minor scope creep.

- **Harsh critic: "The objective function for signature selection is not discussed."** The paper explicitly scopes its contribution as "downstream of signature selection and embedding generation" (Section 5). Criticizing absence of signature selection methodology is scope creep.

- **Harsh critic: "No comparison to trees on original graph features."** The paper's scope is classification/regression on product manifold embeddings, not graph learning per se. The graph embeddings are one source of manifold-valued data. Removed as scope creep.

- **Human finder: "Most graphs are small—scalability concerns."** The paper provides complexity analysis and runtime comparisons in Appendix J, addressing this concern. Removed as already addressed.

- **Harsh critic: "In many datasets (especially synthetic), the labeling function may be more naturally expressed in terms of manifold distances or angles."** This is precisely the design motivation—manifold-appropriate representations should outperform Euclidean ones when the data is naturally manifold-structured. This is a feature, not a bug. Removed as misunderstanding the method's purpose.

## Novel Insights

The reviews surface an important tension that the paper does not adequately address: on product manifolds—the paper's central contribution—simple k-NN with manifold distances consistently outperforms the proposed DT/RF on synthetic benchmarks. This suggests that the angular-split inductive bias is most beneficial when the manifold structure is simple (single-curvature) and the data geometry aligns with the coordinate pole choice, but provides less advantage in the more complex product-manifold setting where k-NN's non-parametric flexibility is harder to beat. This finding actually supports the method's niche as an "interpretable yet competitive" alternative rather than a uniformly superior one, a framing the paper could adopt to strengthen rather than weaken its contribution.

## Suggestions

- Downgrade the "maximum-margin" claim to "geodesically midpoint-balanced" or provide a formal proof. This is the single most important revision for intellectual honesty.
- Add an explicit discussion section analyzing the product-manifold performance gap relative to k-NN, identifying when the method is preferred (interpretability, speed at inference) versus when simpler methods suffice.
- Report product perceptron results in an appendix table for completeness, even if non-competitive.

## Score and Decision

**Calibration anchors:**
- HyperDT (TTonmgTT9X.md): scores 5,6,8,6,8 (mean 6.6, accepted poster). This is the direct predecessor, and this paper extends it to product manifolds. The contribution is comparable in scope—cleaner formulation (angular splits), broader manifold coverage—but less novel since it builds on existing work.
- Spectro-Riemannian GNN (2MLvV7fvAz.md): scores 6,5,6,6 (mean 5.75, accepted poster). Similar topic (mixed-curvature ML), similar scope.
- Mixed-Curvature Transformers (AN5uo4ByWH.md): scores 1,5,5 (withdrawn/reject). Novelty and motivation concerns similar in spirit.
- Deepfake in Curved Spaces (G4D6jClNFl.md): scores 5,6,5,3 (rejected). Incremental combination of existing ideas.

This paper has a solid core idea (angular reformulation of splits), genuinely new contributions (spherical DTs, product manifold extension), and comprehensive experiments. However, it overclaims the "maximum-margin" property, has weaker results on the titular product-manifold setting, and the spherical construction has an unacknowledged inductive bias. Compared to HyperDT, this paper has broader scope but less depth per manifold type and more overclaiming. It is above the reject threshold due to genuinely useful contributions, but below a strong accept due to the overclaims and mixed product-manifold results.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>