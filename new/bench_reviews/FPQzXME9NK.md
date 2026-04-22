Now I have a solid understanding of the paper and the calibration anchors. Let me compose the final review.

## Summary

The paper introduces the Spherical Tree-Sliced Wasserstein (STSW) distance, which adapts the recently proposed tree-sliced optimal transport framework from Euclidean space to probability measures on the hypersphere. The key construction involves "spherical trees"—k spherical rays (great semicircles) glued at a root point—yielding tree-metric spaces where OT admits closed-form solutions. A novel spherical Radon transform on these trees, combined with O(d+1)-invariant splitting maps, ensures injectivity and produces an orthogonally invariant metric with an efficient closed-form approximation (Eq. 19). Experiments across four tasks (gradient flow, self-supervised learning, density estimation, auto-encoders) show consistent improvements over existing spherical OT baselines with notable efficiency gains.

## Strengths

- **Complete and rigorous theoretical chain**: The paper provides a fully coherent framework—Definition 3.2 (spherical trees), Theorem 3.3 (tree metric), Theorem 4.3 (injectivity of the Radon transform under O(d+1)-invariant splitting), and Theorem 5.2 (STSW is a metric with orthogonal invariance)—all carefully established. The injectivity result is particularly non-trivial, given that classical spherical transforms (e.g., Funk–Radon) have non-trivial kernels.

- **Genuine computational advantage from closed-form expression**: Equation (19) provides an explicit closed-form for tree-Wasserstein on spherical trees, avoiding both the binary search needed for circular 1D OT and the multiple rotations required by rotation-invariant baselines. This translates to empirical gains: STSW runs in 1.89s vs. 20.25s for ARI-S3W(30) in gradient flow (Table 1)—approximately 10× faster—while achieving lower log W₂.

- **Orthogonal invariance as an intrinsic property**: Theorem 5.2 guarantees STSW is invariant under O(d+1) transformations, a property prior spherical sliced methods achieve only approximately (S3W) or through costly explicit ensembling of rotations (ARI-S3W). This is a meaningful theoretical distinction.

- **Consistent empirical improvements across diverse tasks**: STSW outperforms all spherical OT baselines in gradient flow (Table 1: −4.69 vs −4.39 log W₂), self-supervised learning (Table 2: 80.53% vs 80.08%), density estimation (Table 3: best NLL on all three datasets), and SWAE (Table 4: best log W₂ and NLL). The code is publicly available.

- **Effective illustration of geometric constructions**: Figures 1 and 2 make the abstract spherical tree and Radon transform constructions accessible.

## Weaknesses

### Fatal
None.

### Major

- **No ablation study on critical hyperparameters (k, L, ζ)**: The method has three key design choices—the number of tree edges k, the number of sampled trees L, and the splitting sharpness ζ—yet no experiment in the main text varies these systematically. This makes it impossible to determine whether STSW's empirical gains come from the richer integration domain (the tree structure itself) or from the implicit computational budget (effectively kL "projections" per tree sample). A controlled ablation varying k while holding kL constant, or comparing baselines given matched wall-clock time, is essential for validating the core motivation that "trees capture topological information that lines miss." All results in Tables 1–4 are affected by this gap. While the code is available and defaults may be in the appendix, the main text should present this analysis for the paper to be self-contained on its central claim.

### Minor

- **Missing variance information in Tables 2 and 4**: Table 1 and Table 3 report ± standard deviations, but Tables 2 (SSL) and 4 (SWAE) do not. The 0.45% accuracy gap between STSW and ARI-S3W in Table 2—80.53% vs. 80.08%—could fall within run-to-run variance. Without standard deviations, statistical significance of improvements on these two tasks cannot be assessed.

- **"Topological information" motivation is asserted but not directly verified**: The introduction (line 59) and conclusion (line 376) claim that spherical trees capture topological information lost by lines or circles. While the empirical results are consistent with this intuition, no experiment specifically isolates this mechanism—for example, by checking whether increasing k (tree edges) improves performance while total computation (kL) is held constant. The claim remains plausible but unverified as a causal mechanism.

- **Construction of β is not intuitively explained in the main text**: Equation (14) defines the splitting map β with a formula involving arccos and square root terms, but the construction is deferred to Appendix A.2 with only a one-line remark (line 231). Given that β determines how mass distributes across tree edges—a central design choice—some geometric intuition in the main text would aid understanding. The behavior at the boundary y = ±x (where β = 0 and α becomes uniform) is also not discussed.

- **Minor overclaiming in conclusion**: The conclusion states STSW "significantly outperforms recent spherical Wasserstein variants" (line 376). While improvements are consistent, some are modest (e.g., 0.45% accuracy gap in SSL, similar NLL for Flood in Table 3) and the BCE in Table 4 actually slightly favors SSW (0.6309 vs 0.6341). "Consistently outperforms" would be more accurate than "significantly outperforms."

### Trivial
None.

## Nice-to-Haves

- Convergence/variance analysis of the Monte Carlo approximation STSŴ as a function of L and k, analogous to the known O(L^{−1/2}) rate for standard sliced Wasserstein.
- Visualization of how the splitting map distributes mass near ray boundaries under different ζ values, which would help practitioners choose ζ.
- Controlled comparison where baselines (SSW, S3W) are given wall-clock time equal to STSW's, to directly test whether gains are algorithmic or computational.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released / cannot be independently verified" concerns**: Any doubt about the existence of Tran et al. (2025b;d) or other cited works is not a valid criticism—the paper cites them and they are assumed to exist.

- **"Overclaimed scope unsupported by experiments" (from harsh critic on topological information)**: Partially retained as a minor weakness above. The core claim is about the metric's validity and efficiency, which the experiments do support. The "topological information" claim is motivational, not the central technical claim.

- **"Table 3 unequal training epochs is a weakness"**: The paper explicitly states STSW uses 10K epochs vs 20K for ARI-S3W and still achieves better results. This is a strength (faster convergence), not a weakness. The harsh critic's framing that "it is difficult to disentangle whether the distance metric or the training schedule drives the improvement" is inaccurate—the point is precisely that STSW requires less training to achieve better results.

- **"Only star trees, not deeper trees"**: Star trees are the natural starting point and enable the clean closed-form in Eq. (19). Deeper tree structures would be a meaningful future direction but are not a weakness of the current paper.

- **"Complexity analysis O(n·k·L) not stated explicitly"**: The paper provides the closed-form expression and Algorithm 1 in the appendix. Computational complexity is straightforward to derive from Eq. (19) and is not a gap.

- **Formatting/notation nitpicks and reproducibility concerns about undisclosed hyperparameters**: The code is publicly available, and hyperparameter details are in the appendix. The main concern about lacking an ablation is retained above as major; the fact that specific values aren't in the main text is a presentation issue, not a methodological one.

- **Request for statistical/convergence analysis of the Monte Carlo approximation**: This is a nice-to-have, not a requirement—the same standard is not applied to prior sliced Wasserstein papers.

- **Missing references / related works**: Cannot verify these without external sources.

- **BCE slightly worse in Table 4**: Already noted as a minor overclaiming issue. The BCE difference (0.6341 vs 0.6309) is negligible and STSW leads on the other two metrics.

## Novel Insights

The paper reveals an interesting design tension: the splitting map α (Eq. 15) must balance specificity (large |ζ| → hard assignment to nearest ray) against smoothness (ζ → 0 → uniform mass distribution), and this trade-off does not arise in standard sliced Wasserstein where projection is deterministic. This makes the "soft projection" mechanism of STSW fundamentally different from hard 1D slicing—it is closer in spirit to kernel-based OT metrics than to standard slicing, which could explain its lower variance in stochastic approximation (as seen in the ±0.01 std in Table 1).

## Suggestions

- Add an ablation table (even for one task, e.g., gradient flow) varying k ∈ {3,5,10,20} and L such that kL is held constant, to show whether more edges per tree genuinely improves the distance approximation beyond what additional random trees provide. This would directly validate or refute the "topological information" motivation.
- Report standard deviations in Tables 2 and 4; if the accuracy difference in SSL is not statistically significant, acknowledge this honestly.
- Replace "significantly outperforms" with "consistently outperforms" in the conclusion.

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LSSOT (Spotlight) | fgUFZAxywx | 7.50 | Spherical sliced OT paper with novel linear embedding. STSW is comparable in theoretical completeness and efficiency gains, but LSSOT had more novel applications (cortical registration, shape embedding). STSW's incremental nature (extension of tree-sliced to sphere) is less novel. |
| TSW-SL (Reject) | EKaVO0ceh8 | 6.0 | Direct predecessor on the Euclidean side. STSW clearly improves on TSW-SL: new domain (sphere), non-trivial theoretical adaptation (orthogonal invariance, new injectivity proof), larger efficiency gains. STSW is noticeably stronger. |
| QMC Spherical SW (Spotlight) | Wd47f7HEXg | 7.50 | Another high-quality spherical SW paper with better MC estimation. STSW has a more substantial theoretical contribution but less novelty in the empirical methodology. |
| OSC-LSH (Reject) | 0SgPbbyrWh | 2.50 | Poor writing, limited experiments, questionable utility. STSW is far superior in every dimension. |

STSW is stronger than the TSW-SL predecessor (6.0) but falls short of the Spotlight-level spherical OT papers (7.5) due to: (1) incremental nature relative to the tree-sliced framework, (2) missing ablation study that directly tests its core motivation, and (3) modest empirical gains on some tasks. The paper sits between these anchors.

**Score: 6.5** — The theoretical framework is sound and the efficiency gains are real, but the experimental validation is incomplete for establishing the central motivation, and the contribution is more incremental than the high-scoring spherical OT papers.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>