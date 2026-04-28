## Summary
This paper proposes the Spherical Tree-Sliced Wasserstein (STSW) distance, extending tree-sliced optimal transport to hyperspherical manifolds via a novel spherical Radon transform on "spherical trees." The method provides closed-form expressions for efficient computation and demonstrates orthogonal invariance. Experiments on gradient flows, self-supervised learning, density estimation, and autoencoders show improvements over recent spherical sliced Wasserstein baselines.

## Strengths
- **Formal theoretical construction with metric proofs**: The paper establishes that spherical trees form metric spaces with tree metrics (Theorem 3.3), proves injectivity of the spherical Radon transform under O(d+1)-invariant splitting maps (Theorem 4.3), and demonstrates STSW is a valid orthogonally invariant metric (Theorem 5.2). These proofs distinguish the work from heuristic approximations.
- **Closed-form computational expression**: Equation (19) provides a closed-form approximation for STSW that enables highly parallelizable GPU implementation, avoiding the supercubic complexity of general OT solvers. This is evidenced by runtime results in Tables 1-4 showing competitive or superior speed compared to baselines.
- **Consistent empirical improvements**: Across four distinct tasks (gradient flow, SSL, density estimation, SWAE), STSW achieves better or competitive results than ARI-S3W, RI-S3W, SSW, and SW baselines. For example, Table 1 shows STSW achieves log W₂ of -4.69 vs -4.39 for ARI-S3W, and Table 2 shows 80.53% vs 80.08% SSL accuracy.

## Weaknesses

### Fatal
None

### Major
- **Overstated novelty regarding "tree" structure**: The paper motivates the method by claiming adaptation of "tree systems" to capture "topological information" better than lines (Abstract, Introduction), describing them as "intricate structures." However, the constructed spherical trees (Definition 3.2) are **star graphs (depth-1 trees)**: k rays glued at a single root x with no hierarchical branching. This is explicitly visible in Figure 2a and the construction in Section 3. The "tree" aspect effectively amounts to correlating k slices at a random point x, rather than introducing multi-scale hierarchical structure as in standard Tree-Wasserstein methods (e.g., quadtrees or clustering-based trees). This undermines the core novelty claim and misleads readers about the method's topological expressivity. Similar concerns were raised in the Mixed-Curvature TSW paper (e439wJl5sT, score 6.0) where a reviewer noted "the paper's construction is limited to star-shaped trees at a single point x... This axis-aligned, star-shaped probe may not be the most effective geometry for capturing complex, non-axis-aligned correlations."

- **Missing critical hyperparameters preventing verification of efficiency claims**: The number of trees L (Monte Carlo samples) and number of rays per tree k are not reported in the main text for STSW experiments. Table 1 states "N_R = 30 rotations for ARI-S3W" but does not disclose L for STSW. Table 2 mentions "ARI-S3W and RI-S3W use 5 rotations" but again omits STSW's L. Without knowing L, the runtime comparisons (e.g., STSW 1.89s vs ARI-S3W 20.25s in Table 1) are unverifiable—if STSW uses L=5 and ARI-S3W uses L=30, the speedup is trivial. This transparency issue is comparable to problems in papers scoring 4.0-4.5 (e.g., V4zln7XiJj, lbEUvx1ILN) where missing hyperparameter disclosure undermined experimental claims.

### Minor
- **Inconsistent NLL metric reporting across tables**: Table 1 shows NLL values around -5000, Table 3 shows values around 0-2, and Table 4 shows values around -0.005 to 0.001. It is unclear whether these represent total log-likelihoods, per-sample NLLs, or different normalizations. This inconsistency prevents reliable assessment of generative modeling performance and raises questions about whether likelihood calculations include normalization constants consistently across baselines. The paper should clarify the metric definition and ensure consistent reporting.

- **Unanalyzed geometric singularity at the antipode**: The spherical tree construction identifies rays only at the root x, treating endpoints at the antipode -x as distinct leaf nodes (Figure 2a caption explicitly acknowledges: "even when endpoints... are all identical to -x on the sphere, the spherical tree treats these as five distinct points"). This creates geometric distortion: points arbitrarily close to -x on different rays are distance ≈2π apart in the tree metric but ≈0 on the sphere. While Monte Carlo integration over random roots is intended to average this out, the paper provides no analysis of how this singularity affects convergence topology or gradient flow behavior. For tasks like Section 6.1 where local geometry matters, this distortion could cause artifacts that are not discussed.

### Trivial
- **Ablation on k (rays per tree) would strengthen claims**: The paper should show how performance and runtime scale with k. If k=1 performs similarly to k=5, the "tree" structure adds unnecessary complexity. This is a standard ablation that would validate whether the multi-ray construction provides tangible benefits.

## Nice-to-Haves
- Clarify the tuning parameter ζ in the splitting map α (Equation 15): The paper notes ζ is a "tuning parameter" affecting sparsity but provides no guidance on selection or sensitivity analysis.
- Transport plan visualization near the antipode would help demonstrate whether the splitting mechanism causes unnatural transport paths.
- Future work could explore hierarchical spherical trees (recursive subdivision) rather than depth-1 star graphs to better align with the "Tree-Sliced" literature's multi-scale geometry capture.

## Removed Points
The following points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim about "unfair comparison" if STSW uses fewer trees than baselines**: This is speculation without evidence. The asymmetry in experimental design (if it exists) would need to be verified against the appendix, which the parser stripped. However, the core issue—missing L disclosure—is retained as a Major weakness.

- **Strength Finder's generic claim "this paper addressed an important problem"**: Removed as it lacks specific citation or concrete content.

- **Strength Finder's claim about "superior empirical efficiency" without noting the hyperparameter transparency issue**: The empirical results are genuine strengths, but the efficiency claims cannot be fully verified without L disclosure. Retained the empirical improvement evidence but flagged the verification issue as a Major weakness.

- **Any criticism about missing appendix content**: The parser strips appendix sections from all papers; proofs and experimental details mentioned as "in Appendix" are assumed to exist in the original submission.

## Novel Insights
The paper's core limitation—that "spherical trees" are star graphs rather than hierarchical trees—reveals a tension in the tree-sliced Wasserstein literature: the computational tractability of closed-form solutions (Equation 19) comes from collapsing angular information into a single scalar distance per ray, which limits topological expressivity. This trade-off is not unique to this work (similar concerns appear in MC-TSW and Tree-Sobolev IPM reviews) but is particularly acute here because the paper's motivation emphasizes "intricate structures" and "topological information" while delivering a depth-1 construction. The antipode singularity issue is a genuine geometric artifact of the stereographic projection approach that deserves analysis, as it could affect gradient-based optimization on spherical manifolds.

## Suggestions
1. **Revise novelty claims**: Temper language about "intricate tree systems" and "topological information" to accurately reflect the star-graph (depth-1) structure. Consider renaming "spherical trees" to "spherical ray systems" or similar to avoid confusion with hierarchical tree structures.
2. **Disclose hyperparameters in main text**: Report L (number of trees) and k (rays per tree) for all STSW experiments in the main text, not just the appendix. This is essential for verifying runtime comparisons.
3. **Clarify NLL metric**: Specify whether NLL values are total, per-sample, or normalized, and ensure consistent scaling across all tables.
4. **Add ablation on k**: Show performance/runtime vs. k to demonstrate whether multi-ray construction provides benefits over single-ray slicing.
5. **Discuss antipode singularity implications**: Add a brief analysis (theoretical or empirical) of how the antipode distortion affects convergence, particularly for gradient flow tasks.

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Decision | Comparison to STSW |
|-------|-----------|----------|-------------------|
| HHNQSXaLkF (Tree-Sliced Sobolev IPM) | 6.00 | Accept | Similar theoretical depth (Radon transform on tree spaces, metric proofs), but better hyperparameter transparency and no overstated novelty claims |
| e439wJl5sT (Mixed-Curvature TSW) | 6.00 | Accept | Same star-graph limitation, but clearer about it; better experimental disclosure |
| l3KtyVZde3 (Double-Sliced Wasserstein) | 7.00 | Accept | Stronger theoretical contribution (Banach space generalization), more thorough sensitivity analysis |
| JQ0SIA2IA1 (Rigid-Invariant SW) | 5.33 | Reject | Missing baseline comparisons and transparency issues similar to STSW |
| V4zln7XiJj (Reasoning Trees) | 4.50 | Accept Poster | Hyperparameter sensitivity without principled guidance—similar transparency gap |
| uq6nIOoPGG (Hyperspherical InfoMax) | 2.50 | Reject | Weak experimental validation, but STSW has stronger empirical results |

**Reasoning:** The paper has solid theoretical foundations (Theorems 3.3, 4.3, 5.2) and consistent empirical improvements across four tasks, which aligns it with the 6.0-score anchor papers (HHNQSXaLkF, e439wJl5sT). However, two significant issues prevent a clear accept: (1) the overstated novelty claims about "intricate tree systems" when the construction is a depth-1 star graph, and (2) missing hyperparameters (L, k) that make runtime comparisons unverifiable. These transparency and claim-accuracy issues are comparable to problems in papers scoring 4.5-5.3 (JQ0SIA2IA1, V4zln7XiJj). The empirical results are genuinely strong and the theory is sound, so the paper is not as weak as the 2.5-4.0 anchors. Positioning relative to anchors suggests a borderline score.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>