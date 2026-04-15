Now let me search for calibration papers to inform my scoring.Now I have enough context to write the final review. Let me synthesize everything.

---

## Summary

This paper introduces mixed-curvature decision trees (DTs) and random forests (RFs) for product manifolds—Cartesian products of hyperbolic, spherical, and Euclidean component manifolds. The core technical contribution is an angular reformulation of DT splits via 2D projections, which yields a unified framework across all constant-curvature manifolds and composes naturally to product settings. The method is evaluated on 57 benchmarks spanning synthetic data, graph embeddings, VAE latent spaces, and empirical datasets. Single-manifold results are strong; product-manifold results are more mixed, with the method competitive but not clearly dominant across the board.

---

## Strengths

- **Principled unification across geometries.** The angular-split perspective elegantly subsumes Euclidean axis-aligned splits, hyperbolic DTs (Chlenski et al., 2024), and introduces novel hyperspherical DTs—all under one framework. The claim that homogeneous hyperplanes are geodesically convex in *any* constant-curvature manifold is a genuine conceptual insight that motivates the approach cleanly (Sections 3.1–3.3, Figure 2).

- **Strong single-manifold results.** The method achieves top-1 on 21/22 single-curvature classification benchmarks (Figure 3) and top-1 on all 11 regression benchmarks (Figure 4), with statistically significant improvements over ambient, tangent, and k-NN baselines in most configurations—a convincing empirical case for the proposed algorithm in the single-manifold setting.

- **Computational efficiency preserved.** The angular reformulation maintains O(1) per-split decision complexity and enables feature subsampling in RFs, matching the efficiency of Euclidean DTs while respecting manifold geometry (Section 3.4). This is practically important for adoption.

- **Composability across components.** A single DT spanning all components can allocate splits adaptively across components based on task relevance—a clear architectural advantage over ensembles of per-component DTs (Section 3.4).

- **Broad and diverse evaluation.** The 57 benchmarks cover classification, regression, and link prediction across synthetic, graph-embedded, VAE-latent, and empirical data. The visualization in Figure 5 provides qualitative evidence that geometry-aware splits produce more natural decision regions than Euclidean or tangent-plane approaches.

- **Candid Limitations section.** The authors explicitly acknowledge dependence on embedding quality, signature selection challenges, computational cost of embedding generation, and the lack of a privileged basis—an unusually forthright assessment that improves scientific credibility.

---

## Weaknesses

### Fatal
*None that invalidate the paper's core contribution.*

### Major

- **Synthetic product-manifold results are dominated by k-NN with no explanation provided.** Table 2 shows that k-NN outperforms Product DTs/RFs on all 8 synthetic multi-K classification benchmarks—precisely the setting designed to showcase product structure. The method is consistently second-best here, not worst, but these are the benchmarks that most directly test the paper's central claim. The paper omits any discussion of *why* k-NN wins in this regime. Is it a dimensionality effect, a failure of the angular split to capture cross-component structure, or a signature-selection artifact? Without this analysis, the paper leaves its most important failure mode unexplained. This materially weakens the empirical case for product manifold DTs as a "powerful" tool.

- **The "maximum-margin" claim in the contributions is not formally established in the main paper.** Contribution #1 explicitly states splits are "geodesically convex, maximum-margin, and composable." Geodesic convexity follows from the homogeneous hyperplane construction and is well-motivated. But the maximum-margin property is never proven or even formally stated as a theorem in the submission body—the midpoint/equidistance construction (Section 3, Eq. 16) guarantees a geodesically-equidistant split, which is not the same as a maximum-margin split for an impurity-optimized criterion. This is an unsupported claim in a prominent location.

- **Abstract and Introduction overstate the empirical evidence.** The abstract says the results "highlight the value of product DTs and RFs as straightforward yet powerful new tools," and the Introduction claims the framework "achiev[es] more accurate results than competing models." But Table 1 shows only 46% top-1 rate on product-manifold classification—less than half of those benchmarks. The method is competitive and often top-2, but the broad-superiority framing is not supported by the product-manifold evidence, which is the novel contribution of this paper over prior hyperbolic/single-manifold work.

### Minor

- **Temperature and Traffic regression failures are not discussed.** Table 3 shows Ambient RF substantially outperforming Product RF on Temperature (4.531 vs. 7.130 RMSE) and Traffic (0.505 vs. 0.534). These are real-world datasets—arguably the hardest test—and the model fails on both. These results deserve at least a brief analysis of why the product formulation hurts here.

- **Collapsed DT/RF columns in Tables 2–3 obscure the source of gains.** The footnote explains that "Ambient" means max(mean(Ambient DT), mean(Ambient RF)). This makes tables readable but hides whether product gains come from the geometry-aware split mechanism or simply from ensembling (RF vs. DT). Reporting DT and RF separately, even in supplementary tables, would strengthen the methodological claim.

- **Basis-dependence in hyperspherical splits is a structural limitation, acknowledged but not mitigated.** Section 3.3 fixes the first embedding dimension as x₀ ("north pole"), an arbitrary gauge choice. While this is acknowledged in Limitations, the paper offers no empirical assessment of sensitivity to this choice (e.g., results under random rotations of the embedding). For learned non-Euclidean embeddings without a canonical axis, this could materially affect performance.

- **Link prediction uses non-standard evaluation.** Section 4.2 frames link prediction as binary classification evaluated with F₁. Standard protocols in the graph-learning community use MRR and Hits@k with negative sampling. Without these metrics, it is unclear whether performance reflects genuine link prediction capability or classifier behavior on the paper's custom binary setup.

- **Equation 19 notation appears suspicious.** The expression $m_{\mathbb{E}}(\mathbf{u},\mathbf{v}) = \tan^{-1}\left(\frac{\tan^{-1}(\theta_u)\tan^{-1}(\theta_v)}{\tan^{-1}(\theta_u)+\tan^{-1}(\theta_v)}\right)$ applies arctan to results of arctan operations, which is dimensionally unusual. The equivalence is deferred to Appendix C, but this formula in the main text—even if a parser artifact—should be stated more carefully.

### Trivial

- **Perceptron baseline dismissal lacks a supporting table.** The perceptron is omitted from Figure 3 with the comment "never achieved competitive results." Given that Tabaghi et al. (2021) is cited as prior work, a brief supplementary comparison would close this loose end.

---

## Nice-to-Haves

- **Analyze why k-NN dominates synthetic multi-K classification.** Even a qualitative explanation (e.g., comparing decision region shapes, examining feature importance by component) would substantially strengthen the paper's product-manifold claims and guide practitioners on when to prefer product DTs over k-NN.

- **Empirical sensitivity analysis for the hyperspherical "north pole" choice.** Evaluating performance under random rotations of the embedding for S^D datasets would quantify the practical impact of this acknowledged limitation and help motivate the rotation forests direction mentioned in Future Work.

- **Separate DT vs. RF results in product manifold tables.** Reporting them separately (or in supplementary tables) would clarify whether wins come from the geometry-aware mechanism or from ensemble variance reduction.

- **Richer baselines as supplementary comparison.** Gradient-boosted trees (e.g., XGBoost) applied to ambient coordinates are arguably the strongest practical Euclidean baseline; including them in the appendix comparison (alongside the existing MLP/GNN comparisons in Appendix I) would better contextualize the practical value.

- **Visualization of product-manifold decision boundaries.** Figure 5 shows only the S² landmasses case. A visualization on a product manifold (e.g., S¹ × S¹ torus, which the introduction features) would provide more targeted evidence of what geometry-aware splits look like vs. ambient/tangent counterparts.

- **Scalability discussion for higher-dimensional product manifolds.** The paper considers all $\binom{D}{2}$ 2D projections as split candidates; for larger products, this could become prohibitive. A brief runtime analysis (already partially in Appendix J) would help set expectations for practitioners.

---

## Removed Points

*These points are flagged for removal; treat them with caution—they may reflect reviewer knowledge gaps or misreadings rather than genuine paper flaws.*

- **Harsh Critic: "Evaluation conflates representation quality and classifier quality."** The paper explicitly scopes itself as downstream of embedding and signature selection (Section 5, Limitations: "we view our work as downstream of signature selection and embedding generation"). While this framing limits generalizability claims, it is explicitly stated—and the claim is about trees-as-classifiers, not about embedding methods. The criticism that "the paper's claims are often phrased as if they isolate the advantage of the DT/RF algorithm" is overstated; the paper is reasonably careful about this framing.

- **Harsh Critic and Human Finder: Missing comparison to global-optimal DT methods (e.g., Quant-BnB).** The predecessor paper (Chlenski et al., 2024) was already accepted without such a comparison. These methods are niche within ML and not a standard baseline for representation-learning-adjacent work. Moved to nice-to-have if anything.

- **Human Finder: Lack of theoretical analysis for *why* geodesic splits outperform.** Demanding theoretical learning-guarantees for an empirical algorithmic paper is not standard practice in this community. The empirical results constitute the justification.

- **Harsh Critic: "Source of repeated trials underlying CIs is not clear."** Section 4.1 states an 80:20 split is used. CIs on F₁ or RMSE over a finite test set can be computed via bootstrapping or exact binomial confidence intervals without requiring multiple random data splits. This is a routine methodological detail, not a reproducibility concern.

- **Harsh Critic: Baseline set "too narrow" for general claims.** The paper's primary contribution is a tree-based algorithm; the relevant comparators are other tree-based methods and k-NN (non-parametric). MLP and GNN comparisons are included in Appendix I. Criticizing the absence of broad ML baselines in the main tables overstates the paper's scope.

---

## Novel Insights

The most genuinely novel observation across the reviews is that the angular reformulation of DT splits is not merely a computational trick but a principled geometric unification: by recasting splits as 2D angular projections, the framework inherits geodesic convexity "for free" across all constant-curvature manifolds without requiring separate derivations for each geometry. The introduction of hyperspherical DTs (absent from prior work) is a concrete byproduct of this unification. The most important unresolved insight—left for future work—is whether the per-component independence of angular splits is a fundamental limitation on product manifolds (i.e., whether the model structurally cannot capture decision boundaries that depend on cross-component interactions), which would explain the synthetic multi-K failures and help practitioners know when to prefer k-NN or neural alternatives.

---

## Suggestions

1. **Address the k-NN dominance on synthetic multi-K tasks directly.** Provide at least a qualitative or geometric analysis of why product DTs/RFs lose on Gaussian-mixture data in product spaces. This is the single most damaging gap in the current submission.

2. **Qualify the "maximum-margin" language** in Contribution #1 and the abstract. Either prove it (or restrict to the specific sense in which equidistance implies maximum margin under the angular parameterization) or replace with "geodesically convex and composable," which the paper does support.

3. **Discuss the Temperature and Traffic failures** with even one or two sentences. Geospatial and cyclical time-series data are precisely the application area motivating non-Euclidean representations; a regression failure there deserves acknowledgment.

4. **Report DT and RF results separately** for product-manifold tables (even in supplementary form), so readers can assess whether the geometric mechanism or the ensemble effect drives improvements.

---

## Score and Decision

**Calibration:**

- **TTonmgTT9X** (Fast Hyperboloid Decision Tree Algorithms, the direct predecessor): Accepted poster, average score ~6.6. That paper addressed a single manifold (hyperbolic), had clean results, and comparable algorithmic rigor.
- **NkGDNM8LB0** (Hyperbolic Genome Embeddings): Accepted poster, average ~6.5. Strong domain results with imperfect understanding of failure cases.
- **tdbK3TGFl1** (Asymmetric Embedding for Hierarchical Retrieval): Rejected, average ~3.5. Weak theoretical and empirical contributions.
- **EyWKb7Ltcx** (Riemannian Classifiers on SPD Manifolds): Rejected, average ~5. More theoretical framework but weak empirical support.

**Positioning:** The paper under review is more ambitious than TTonmgTT9X (product manifolds, not just hyperbolic), achieves stronger single-manifold results, and introduces genuine novelty (hyperspherical DTs, unified angular framework). However, the headline product-manifold claim is supported only partially—k-NN dominates the synthetic multi-K regime with no explanation, and the abstract/intro overstate the evidence. The "maximum-margin" claim is prominent and unproven. The evaluation has structural weaknesses (collapsed columns, non-standard link prediction metrics).

This places the paper *marginally below* TTonmgTT9X in terms of empirical support for its central claims, despite being broader in scope. The single-manifold contributions alone would clear the bar; the product-manifold case needs more work. I score this **5.5**: clearly above weak papers (2–4) that lack a contribution, but below the acceptance threshold for papers with clean empirical support for their headline claims (~6.5+).

**Originality:** Good — angular unification and hyperspherical DTs are genuine contributions.  
**Importance of research question:** Moderate-High — tools for product manifold inference are underdeveloped.  
**Support for claims:** Mixed — single-manifold claims well-supported; product-manifold claims overstated.  
**Soundness of experiments:** Fair — broad benchmark suite, but key failure mode unexplained and evaluation has structural issues.  
**Clarity of writing:** Good — well-written, limitations honestly stated.  
**Value to community:** Moderate — useful toolkit contribution but requires a more honest positioning of where it wins and where it doesn't.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>