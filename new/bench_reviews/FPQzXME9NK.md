Now I have all the information I need. Let me write the final review.

## Summary

The paper proposes the Spherical Tree-Sliced Wasserstein (STSW) distance, a new metric for probability measures on the sphere that replaces the 1D great semicircle projections of prior spherical sliced OT methods with spherical trees—multi-branch structures built via stereographic projection. By constructing an O(d+1)-invariant splitting map that distributes mass across tree edges, the authors prove injectivity of the resulting spherical Radon transform and show STSW is a proper metric. A closed-form computation (Eq. 19) yields efficient, highly parallelizable evaluation, and the method is demonstrated on gradient flows, self-supervised learning, earth density estimation, and SWAE.

## Strengths

- **Novel and elegant spherical tree construction (Section 3, Definition 3.2):** The mapping from rays in the hyperplane to great semicircles on the sphere via stereographic projection φ_x (Eq. 4) is natural and well-justified, yielding clean isometry with [0,π] that enables the tree metric structure (Eq. 6). The topological construction via quotient of disjoint unions is rigorous.

- **Closed-form tree-Wasserstein computation (Eq. 19):** The derivation that pushforward measures on spherical trees admit the tree-Wasserstein closed form is the key practical contribution. The O(n log n + nk) cost per tree, combined with avoidance of the binary search needed for circular sorting in SSW, gives a genuine computational advantage. Table 1 demonstrates ~10× speedup over ARI-S3W while achieving better log W₂.

- **Theoretical rigor (Theorems 4.3, 5.2):** The injectivity of the spherical Radon transform for O(d+1)-invariant splitting maps, and the resulting orthogonally invariant metric property, are non-trivial theoretical results. The O(d+1)-invariance is particularly valuable for spherical applications where rotational symmetry matters.

- **O(d+1)-invariant splitting map β (Eq. 14):** The specific construction using arccos(⟨y,y_i⟩/√(1−⟨x,y⟩²))·√(1−⟨x,y⟩²) is geometrically motivated and its invariance properties are a genuine technical contribution.

- **Consistent empirical improvements across four diverse tasks:** STSW outperforms or is competitive with SSW, S3W, RI-S3W, and ARI-S3W on gradient flows (Table 1), SSL (Table 2), density estimation (Table 3), and SWAE (Table 4).

## Weaknesses

### Fatal
None.

### Major

- **Missing ablations on key design parameters (k, ζ):** The paper's central methodological claim is that spherical trees with multiple edges capture information better than single great semicircles. Yet there is no ablation studying k (number of edges per tree) or ζ (splitting parameter in Eq. 15). Without showing that k > 1 (a genuine tree) outperforms k = 1 (a single spherical ray, reducing to SSW-like projection), the paper cannot establish that the tree structure itself—rather than simply using more projections—drives the improvements. Similarly, without comparing ζ > 0 against ζ = 0 (uniform splitting), the splitting map's role is untested. This is the same critical gap flagged in the predecessor paper (Tran et al., 2025d, tree-sliced Wasserstein on systems of lines), and its persistence here weakens the empirical validation of the core claim.

- **No projection-count-matched comparison:** Each spherical tree with k edges yields k effective 1D transport sub-problems per tree. If STSW uses L trees with k edges, it solves L·k total sub-problems, while baselines like SSW/S3W use L projections. The paper does not compare STSW(L trees, k edges) against SSW/S3W with L·k projections, which would be the natural fair comparison. Runtime columns partially address this but are an imperfect proxy since per-projection costs differ across methods. Without this comparison, the reader cannot determine whether STSW's advantages come from the tree structure or from a larger effective projection budget.

### Minor

- **Overclaiming about "topological information":** The paper repeatedly asserts that spherical trees "enhance the ability to capture topological information" (Abstract, Section 1, Conclusion). However, the constructed spherical trees are star-shaped (k rays meeting at a root)—topologically trivial structures. The real advantage of the tree framework appears to be directional resolution via the splitting map, which preserves azimuthal information within each latitude. This is a statement about richer projection structure, not about topology in any meaningful mathematical sense. The "topological" framing risks misleading readers about the actual mechanism.

- **Missing hyperparameter values (L, k, ζ) in main text tables:** The number of sampled trees L, edges per tree k, and splitting parameter ζ for STSW are not reported in any of the four experiment tables, making it impossible to assess computational fairness from the main text alone. The runtime columns are a partial substitute but do not reveal the projection budget.

- **No error bars or significance tests in SSL experiment (Table 2):** The improvement of STSW over ARI-S3W(5) in SSL accuracy is marginal (80.53% vs 80.08%), and no error bars or statistical significance tests are reported, making it difficult to assess whether this difference is meaningful.

- **Different training epochs in density estimation (Table 3):** STSW trains for 10K epochs while ARI-S3W uses 20K epochs. While the paper argues this shows faster convergence, the comparison at unequal epochs complicates interpretation—should NLL be compared at equal epochs or equal wall-clock time? This deserves clarification.

### Trivial
None.

## Nice-to-Haves

- A toy example showing two distributions that are indistinguishable by SSW but distinguishable by STSW would make the advantage of the tree structure tangible and intuitive.
- Discussion of failure cases or limitations, e.g., for distributions concentrated near the antipodal point −x where all edges converge.
- Ablations on k and ζ would strengthen the paper substantially (promoted from nice-to-have to major weakness above, since these directly test core claims).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Notation shift between R^σ_α and R^α:** The critic claimed a confusing notation shift, but reading the paper shows R^σ_α (Eq. 8–9) is the operator for a single tree T, while R^α (Definition 4.1) is the full transform over all trees. This is consistent notation for two related but distinct objects—removed as a non-issue.

- **σ notation confusion (distribution on T_k^d vs uniform on sphere):** The critic noted σ is used for both, but the paper explicitly defines σ as the joint distribution on the space of spherical trees (line 153), which includes a uniform component on S^d. This is standard practice and not confusing—removed.

- **Derivation of β deferred to appendix:** The critic argued this makes the main text insufficient. However, the paper gives the explicit formula for β (Eq. 14) and states its key properties (continuity, O(d+1)-invariance). Deferring the derivation to the appendix is standard practice for technical proofs in venue-constrained papers—removed as nitpick.

- **Reproducibility concerns about undisclosed hyperparameters:** While the missing L, k, ζ values in the main text are a legitimate concern (moved to Minor weakness above), the broader reproducibility claim is weakened by the fact that the code is publicly available. Readers can verify the exact settings from the code—removed the broader reproducibility framing.

- **SWAE BCE slightly worse (Table 4):** The critic noted STSW's BCE (0.6341) slightly underperforms SSW (0.6309). This is a marginal difference and the paper already acknowledges it. Not a substantive weakness—removed.

- **Request for larger dataset experiments / more models:** Generic scope-creep demand not tied to the paper's stated goals—removed.

## Novel Insights

The critical observation across reviews is that the paper's mechanism for improved performance is likely the splitting map's directional resolution (preserving azimuthal information within each latitude band), not the "topological" structure of the trees themselves—which are star-shaped and topologically trivial. This distinction matters: if k=1 with the splitting map performs as well as k>1, then the tree structure is unnecessary overhead and the paper's contribution reduces to a new splitting-based spherical Radon transform rather than a tree-based one. Conversely, if k>1 matters but ζ=0 (uniform splitting) performs equally well, then the splitting map design is unnecessary. Without these ablations, the paper cannot disentangle which component drives the improvements, and the "topological" framing may be obscuring the real mechanism.

## Suggestions

- Add ablations on k ∈ {1, 2, 5, 10, 20} and ζ ∈ {0, 1, 5, 10, ∞} to directly test whether the tree structure and splitting map each contribute to the observed improvements.
- Report L, k, ζ values in the main text tables (or as footnotes) so readers can assess computational fairness.
- Add a projection-count-matched comparison: e.g., STSW with L=10 trees and k=5 edges vs. SSW/S3W with L=50 projections.
- Replace or clarify "topological information" with more precise language like "directional resolution" or "richer projection structure" unless the topological claim can be substantiated.
- Add error bars to Table 2 (SSL) to enable assessment of whether small accuracy differences are significant.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LSSOT (Spherical Sliced OT + Linear OT) | fgUFZAxywx | 7.5 | More complete empirical validation, cleaner framing, no ablation gaps. STSW is below this due to missing ablations and overclaiming. |
| QMC for 3D SW | Wd47f7HEXg | 7.5 | Thorough experimental comparison, well-scoped contribution. STSW's empirical gaps are more significant. |
| Diffeomorphic Mesh via SW | gxhRR8vUQb | 7.0 | Well-executed application paper. STSW has stronger theoretical depth but weaker empirical validation. |
| TSW-SL (predecessor, tree-sliced on lines) | EKaVO0ceh8 | 6.0 | Same group, same missing-ablation weakness. STSW is moderately stronger due to non-trivial spherical extension, closed-form computation, and more diverse experiments. |
| Imitation Learning via Gen. SW | 8rN439jpkT | 4.4 | Weaker contribution and more limited scope than STSW. |
| IFGW (flawed OT metric) | Aku2I3z4aV | 2.6 | Fundamentally flawed. STSW is far above this. |

STSW is clearly above the low-scoring anchors and the weakest medium anchors. It is stronger than its predecessor (TSW-SL, avg 6.0) due to the non-trivial spherical extension, the O(d+1)-invariant splitting map construction, and more diverse empirical evaluation. However, it shares the same critical weakness (missing ablations on k and ζ) that contributed to the predecessor's rejection, and adds overclaiming about "topological information." It falls below the high-scoring anchors (LSSOT, QMC-SW) which have more complete empirical validation and cleaner framing. The paper's theoretical contributions (spherical tree construction, injectivity proof, closed-form computation) are genuine and non-trivial, but the empirical validation has important gaps that prevent confident attribution of the improvements to the tree structure itself.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>