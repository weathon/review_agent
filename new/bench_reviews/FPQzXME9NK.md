Now let me search for calibration papers to help score this work.Now I have enough information to write the final consolidated review.

---

## Summary

This paper introduces the Spherical Tree-Sliced Wasserstein (STSW) distance, adapting the tree-sliced Wasserstein framework from Euclidean to spherical geometry. The core idea is to construct "spherical trees" via stereographic projection, define a novel spherical Radon transform on these trees, and exploit the resulting tree metric structure for closed-form OT computation. The authors prove that STSW is an orthogonally-invariant metric and demonstrate consistent improvements over recent spherical sliced Wasserstein variants across four tasks: gradient flow, self-supervised learning, earth density estimation, and SWAE.

---

## Strengths

- **Rigorous theoretical foundation**: The paper provides complete proofs that spherical trees are metric spaces with tree metrics (Theorem 3.3), establishes injectivity of the spherical Radon transform under O(d+1)-invariant splitting maps (Theorem 4.3), and proves STSW is an orthogonally-invariant metric (Theorem 5.2). The theoretical development is systematic and self-contained, and O(d+1)-invariance is a meaningful and practically desirable property.

- **Closed-form computation and demonstrated efficiency**: Equation (19) enables a parallelizable implementation. The efficiency gains shown in Table 1 are substantial: STSW achieves 1.89s vs. 20.25s for ARI-S3W(30) while also producing better log W₂ (−4.69 vs. −4.39). This is not a marginal gain—the speedup is over an order of magnitude.

- **Consistent empirical improvements across four tasks**: STSW outperforms baselines on gradient flow (Table 1), SSL (Table 2), earth density estimation (Table 3), and SWAE (Table 4 — best in log W₂ and NLL, competitive in BCE and runtime). The breadth of evaluation is appropriate for an OT metric paper.

- **Well-motivated design**: The O(d+1)-invariant splitting map β (Equation 14) and softmax-based α (Equation 15) are concrete and come with clear geometric intuition. The connection to stereographic projection and the spherical ray construction is clearly explained and illustrated.

- **Public code**: Code is publicly available at the URL in the abstract, enabling reproducibility.

---

## Weaknesses

### Fatal
*None.*

### Major

- **No ablation studies on key hyperparameters** — The method introduces three consequential design choices: the number of tree edges *k*, the number of sampled trees *L*, and the softmax temperature *ζ*. The main paper does not report any sensitivity analysis for these. Because *k* directly controls expressivity and computational cost, and *ζ* controls sparsity of the splitting map, their effect is central to interpreting why STSW outperforms baselines. It is possible that increasing *k* or *L* is the primary driver of gains rather than the spherical tree geometry per se. A table varying *k* and *L* with performance-vs-runtime trade-offs is essential and absent.

- **Missing variance in SSL and SWAE tables** — Table 2 (SSL) reports single numbers (e.g., 80.53 vs. 80.08 for ARI-S3W). These differences are small enough that without multi-run confidence intervals, the reported superiority is not statistically convincing. Table 4 (SWAE) similarly reports single runs. The paper uses multi-run variance in Tables 1 and 3 (gradient flow and density estimation), which makes its absence in Tables 2 and 4 an inconsistency.

### Minor

- **Efficiency comparison involves non-matched approximation budgets**: In Table 1, ARI-S3W uses 30 rotations; in Table 2, 5 rotations; in Table 3, 50 rotations and 20K epochs vs. STSW's 10K epochs. The lack of a controlled comparison at matched computational budgets means one cannot cleanly decompose how much of the runtime advantage is due to algorithmic design versus fewer projection objects. The paper does explicitly acknowledge the epoch difference in §6.3 and frames it as a feature. The gains are large enough to likely survive matched-budget comparisons, but this should be verified.

- **No formal complexity characterization**: The paper claims "efficient metric" and "highly parallelizable implementation" (§5.2, §7), but the main text never states the computational complexity as a function of support size *n*, number of edges *k*, and number of sampled trees *L*. Readers cannot compare STSW's asymptotic cost against alternatives without consulting the appendix.

- **Core mechanistic claim unverified**: The introduction asserts that line/circle projections suffer from "loss of topological information" and that spherical trees remedy this. While the empirical results support that STSW outperforms baselines, no experiment isolates whether the improvement comes from the tree topology specifically (vs. increased expressivity from the multi-ray structure). This does not invalidate the results but overstates the mechanistic interpretation.

- **Incremental novelty relative to TSW-SL / Db-TSW**: The framework is a natural and non-trivial adaptation of Tran et al. (2025d) to the sphere, but many proofs follow "similar" arguments. The primary novelty is the spherical domain adaptation, the orthogonal (rather than Euclidean) invariance, and the experimental demonstration. This is meaningful but reviewers in closely related work (Db-TSW) have noted the same concern.

### Trivial

- The SWAE BCE result (Table 4, 0.6341 vs. 0.6309 for SSW) shows STSW is not best on all metrics; the paper acknowledges this honestly, but the acknowledgment is brief.

---

## Nice-to-Haves

- **Ablation on splitting map design**: Testing alternative choices of α (e.g., uniform splitting, hard assignment) would clarify how much the softmax construction and temperature ζ contribute to performance, as suggested by the analogy with Db-TSW's design choices.

- **Formal comparison with entropic OT on the sphere**: Since STSW is positioned as an efficient approximation of spherical OT, a comparison to the quantity it approximates (even on small examples) would validate that STSW is a good proxy.

- **Topological characterization**: A formal statement about what weak topology STSW metrizes and whether it agrees with the spherical Wasserstein distance would strengthen the theoretical section.

- **Evaluation at higher sphere dimensions**: The gradient flow and density estimation experiments use S², while SSL uses S⁹. A systematic study over dimension would reveal if performance scales well.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — "efficiency advantage not established under fair comparison" (Claim 1, classified as "Evidential")**: While the concern about non-matched approximation budgets has partial merit, the harsh critic overstated this as a near-fatal flaw. In Table 1, STSW achieves a 10× speedup over ARI-S3W(30) while producing better accuracy. Even if settings were equalized, the margin is too large to explain away. This is **kept as a minor weakness** (non-matched budgets) rather than a major flaw invalidating the efficiency claim.

- **Harsh Critic — "the conclusion overstates"**: The conclusion says "significantly outperforms" which reviewers found too strong given missing uncertainty in some tables. This is **kept as a minor concern** under the SSL/SWAE variance weakness above, but is not a standalone flaw warranting removal.

- **Neutral Reviewer — "Limited comparison scope" / "no entropic OT baseline"**: Comparing only within sliced variants is entirely reasonable for a paper proposing a new sliced variant. The paper does not claim to be the best spherical OT method overall, only the best among sliced variants. Moved to **nice-to-have**.

- **Neutral Reviewer — "Key proofs entirely deferred"**: Deferring proofs to appendices is standard in ML papers. The proofs are present and referenced. Removed entirely.

- **Neutral Reviewer / Harsh Critic — "scalability absent"**: Requesting formal empirical scaling studies is a reasonable suggestion but not a standard requirement for a metric paper in this field. Moved to **nice-to-have**.

- **Human Finder — "root-concurrent structure limits flexibility"**: This is a structural property of the construction (shared root), not a bug. The paper builds on a principled design (spherical rays from a common root). There is no evidence this causes empirical failures. Removed as a speculative concern.

---

## Novel Insights

The paper's most genuinely novel observation is that the tree-sliced Wasserstein framework, when adapted to the sphere via stereographic projection, yields an orthogonally-invariant metric with closed-form computation that dramatically outpaces great-circle slicing baselines (by over an order of magnitude in Table 1). This combination—spherical geometry-aware structure, orthogonal invariance, and closed-form tree-OT—had not previously been achieved in a single method. The result is that spherical trees provide a principled middle ground between computationally cheap but topology-losing line projections and computationally expensive but topology-preserving full OT, and the empirical gaps are large enough to be practically meaningful even absent ablation studies.

---

## Suggestions

1. **Add ablation on k and L**: A single table in the appendix varying k ∈ {2, 5, 10} and L ∈ {10, 50, 100} for at least one task (gradient flow is the cheapest) would address the most common criticism of tree-sliced methods and dramatically strengthen the empirical case.

2. **Report variance for SSL and SWAE**: Re-run Tables 2 and 4 with at least 3 seeds and report mean ± std, consistent with Tables 1 and 3.

3. **State complexity in main text**: Add one paragraph to §5.2 stating the computational complexity as O(Ln·k) (or whatever the correct form is) to allow direct comparison with baselines.

4. **Provide a proof sketch for Theorem 4.3 in the main text**: Even two sentences on why O(d+1)-invariance of α implies injectivity would help readers assess the argument without consulting the appendix.

5. **Temperature ζ discussion**: Report what value of ζ is used in each experiment and include a brief sensitivity analysis (even just ζ ∈ {1, 5, 10}) to guide practitioners.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Decision | Scores | Notes |
|---|---|---|---|
| EKaVO0ceh8 (TSW-SL) | Reject | 8, 5, 5, 6 (avg 6.0) | Direct ancestor, similar structure, weaker results |
| OiQttMHwce (Db-TSW) | Accept Poster | 6, 6, 6, 8 (avg 6.5) | Closest analog — Euclidean invariant extension of TSW-SL |
| fgUFZAxywx (LSSOT) | Accept Spotlight | 6, 10, 6, 8 (avg 7.5) | Spherical OT domain, more novel linear embedding idea |

STSW is most comparable to Db-TSW (OiQttMHwce): both extend TSW-SL with a geometric-invariance property, both have similar theoretical depth, and both show consistent improvements over prior work on multiple tasks. STSW's efficiency gains over ARI-S3W are notably larger (10× vs. comparable in Db-TSW), and it tackles the spherical domain which has its own practical relevance. On the other hand, STSW shares the same gap as Db-TSW: missing ablations on k/L/ζ and lacking a complexity analysis. The SSL table's missing variance is a mild additional concern. TSW-SL was rejected partly because its improvements over vanilla SW were less compelling and had similar experimental gaps; STSW's improvements are more substantial. LSSOT, scored higher, had a more novel embedding idea and stronger experimental diversity.

**Assessment**: Solid poster-level contribution. The spherical tree construction is principled and fills a real gap. The empirical results are consistently strong and the efficiency gains are large. The weaknesses (missing ablations, missing variance in two tables) are significant but not fatal, and are fixable in revision. Positioned slightly above Db-TSW due to stronger runtime results, and below LSSOT due to more incremental framing.

**Score: 6.5 — Accept (Poster)**

**Axis summary:**
- *Originality*: Moderate — non-trivial adaptation of tree-sliced OT to the sphere; orthogonal invariance is well-motivated but the high-level framework mirrors Euclidean predecessors.
- *Importance*: Good — spherical distributions are genuinely important in many ML applications, and an efficient invariant metric fills a real gap.
- *Claims well-supported*: Mostly — strong empirical evidence but efficiency claims need more controlled baselines and ablations.
- *Experimental soundness*: Adequate — four tasks, multi-run variance in most tables, but two tables lack variance and hyperparameter sensitivity is untested.
- *Clarity*: Good — well-written, figures are helpful, proofs properly deferred to appendix.
- *Value to community*: Good — code is released, method is practical, and it extends a growing family of efficient OT variants.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>