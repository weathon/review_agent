Now let me search for calibration papers.Now I have enough calibration data. Let me compile the final review.

---

## Summary

NEUROLIFTING proposes to reparameterize discrete MRF decision variables via a randomly initialized GNN, then minimize the MRF energy directly as an unsupervised loss using gradient descent. The method handles arbitrary-order cliques via tensor products, uses GraphSAGE with Jumping Knowledge aggregation, and is motivated by a conceptual parallel to classical "lifting" in optimization. Experiments cover synthetic Erdős–Rényi graphs, UAI 2022 competition benchmarks, and real-world Physical Cell Identity instances.

---

## Strengths

- **Superior performance on large-scale MRFs (Table 1, 2, 5):** At 50k-node pairwise instances (Table 1), NEUROLIFTING achieves the best energy on 8/12 large instances (e.g., P.potts_4: 11,679 vs. LBP's 12,411 and Toulbar2's 12,955). On all five large-scale high-order synthetic instances (Table 2), NEUROLIFTING outperforms Toulbar2, including two cases where Toulbar2 returns NA or an inferior solution by large margins.

- **Natural and principled extension to high-order MRFs (Section 3.4, Tables 2, 4):** The tensor-product formulation in Eq. 5–6 cleanly generalizes to arbitrary-order cliques, and LBP/TRBP cannot handle these cases. This is a genuine advantage over most prior GNN-for-combinatorial-optimization work, which is limited to pairwise interactions.

- **Concrete padding strategy for variable-cardinality MRFs (Section 3.2, Fig. 2):** Padding with per-term maximum energies to discourage virtual state selection is a practically motivated and reproducible design choice; the Remark explicitly discusses and dismisses alternative strategies that cause convergence to infeasible solutions.

- **Diverse evaluation across three domains:** Synthetic ER graphs, UAI 2022 competition data (pairwise and high-order), and real-world PCI instances provide meaningful coverage and make the empirical story more credible than a single-domain evaluation.

- **Motivated GNN backbone and aggregator selection (Section 3.3, Fig. 3):** The argument for choosing GraphSAGE (equal-weight neighbor aggregation, consistent with the MRF assumption) over GCN or GAT is principled, and Fig. 3 provides empirical validation across three dataset families.

---

## Weaknesses

### Fatal
None.

### Major

- **Abstract overclaims contradict the experimental results at medium scale.** The abstract asserts NEUROLIFTING "significantly surpasses existing approximate methods." Table 1 directly contradicts this: at 5k nodes (P.potts_2), LBP achieves −111,319 while NEUROLIFTING achieves −105,953 — LBP wins by ~5%. At 10k nodes (P.potts_3): LBP −221,567 vs. NEUROLIFTING −209,925, again ~5% gap favoring LBP. On random-energy counterparts (P.random_2/3), Toulbar2 dominates and NEUROLIFTING finishes last among the three methods by wide margins. The abstract's claim is only true in the large-scale (50k) regime; at moderate scale (1k–10k), NEUROLIFTING is often the weakest method. The paper needs to scope this claim to the regime it actually applies to.

- **No wall-clock runtime data anywhere in the paper.** The abstract claims efficiency and the paper claims linear complexity, yet not a single table or figure reports actual GPU/CPU runtimes for NEUROLIFTING. Toulbar2's and LBP's time limits are mentioned, but NEUROLIFTING's corresponding wall-clock time is entirely absent. The per-iteration complexity analysis in Section 3.5 is formally valid but meaningless for evaluating end-to-end competitiveness — it says nothing about the number of iterations required or the constant factors of GPU vs. CPU computation. For a paper whose core claim includes computational efficiency, this omission is severe.

- **Catastrophic failure on ProteinFolding_12 (250 nodes) with no diagnosis.** Table 3 shows NEUROLIFTING returns 16,051.798 vs. Toulbar2's optimal 3,562.387 — a factor of ~4.5× worse, on an instance with only 250 nodes and 1,848 edges. The paper neither flags nor explains this result. The claim in Section 3.4 that "the discrepancy between L(θ) and E({v_i}) is minor" after convergence is stated without evidence and is likely violated in this case. Understanding when and why the method fails so dramatically on small instances is essential to establishing trustworthiness.

- **Most directly relevant prior baselines (GNN-based CO) are omitted.** Schuetz et al. (2022) and Cappart et al. (2023) are cited in the introduction as "recent heuristics utilizing GNNs for solving combinatorial problems" — the paper's own closest relatives — yet neither appears in any experiment. Without this comparison, the paper cannot demonstrate that its specific design choices (lifting interpretation, GraphSAGE with JK, energy-as-loss) add value over naive GNN-for-CO approaches. This is not a scope argument; these baselines are directly applicable to MRF MAP problems.

### Minor

- **Ablation studies report loss curves, not post-rounding discrete energy values.** Figures 3 and 4 compare GNN backbones and optimizers via continuous loss L(θ), but the final evaluation metric is the discrete energy E({v_i}) after rounding. It is unclear whether convergence differences on the continuous loss translate to meaningful differences in post-rounding solution quality.

- **Loss landscape visualization performed on a single instance.** The landscape analysis in Fig. 4 (Segmentation_19) is the primary visual evidence for the lifting/smoothing claim. One example is insufficient to establish a general principle; systematic comparison across multiple instance types and scales would be needed to test the hypothesis properly.

- **"Lifting" framing is more analogy than theorem.** Section 3.5 argues that GNN reparameterization mirrors classical lifting by expanding the problem space. The LBP-to-GraphSAGE analogy in Section 3.3 rests on "message aggregation," which is too broad to justify the specific design choices. The conceptual contribution is still valuable, but the "non-parametric lifting" claim is informal and would benefit from at least a statement of what formal property is being claimed.

### Trivial
- Section 4.1 characterizes medium-scale performance as "comparable solution quality even when problem sizes are small" — this characterization undersells a clear systematic weakness identified above.

---

## Nice-to-Haves

- **Runtime table (GPU time for NEUROLIFTING, CPU time for Toulbar2/LBP/TRBP) across all instance sizes.** This would convert the efficiency claim from theoretical to empirical.
- **Multiple runs with variance reporting** on representative instances, given NEUROLIFTING starts from random GNN initialization.
- **Rounding gap analysis** per instance: reporting both L(θ) and E({v_i}) side-by-side would help characterize when the continuous approximation is reliable and when it breaks down (as in ProteinFolding_12).
- **Conditions for tightness**: Even an empirical characterization of what instance properties predict a small rounding gap would strengthen the paper considerably.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. **"Lifting connection is unsupported because it lacks formal certificates"** (Harsh Critic, Weakness 3) — *Removed/weakened to Minor.* The paper does not claim a rigorous hierarchy or approximation guarantee; it argues by structural analogy. For an empirical systems paper, this level of formalism is standard. Demanding SDP-level proofs is scope creep. Kept as a minor note.

2. **"Table 2, H.Instances_3 energy scale discrepancy — possibly not apples-to-apples"** (Harsh Critic, Section-by-Section Notes) — *Removed.* Both methods use the same MRF format and transformation (Section 4, "MRF format and transformation"); the paper states it interprets files identically to Toulbar2. Questioning the validity of a result without evidence of a bug is not actionable.

3. **"Abstract claims 'comparable solution fidelity' with Toulbar2 — contradicted everywhere"** — *Partially removed.* The claim is true on some subsets (large-scale synthetic, PCI) and false on others (UAI pairwise, medium synthetic). Kept only as part of the broader "overclaiming" major weakness above, not as a separate fatal issue.

4. **Strength Finder: "principled connection between LBP and GNN reparameterization provides formal justification"** — *Weakened.* The connection is by analogy, not formal derivation. Retained as a supporting observation but not elevated to a primary strength.

5. **Strength Finder: "empirically validates linear scalability by successfully handling 50k-node instances"** — *Weakened.* Successfully running on 50k nodes does not validate linear scaling; it merely shows the method can run at that scale. Without runtime measurements, this is unverifiable. Not listed as a strength.

---

## Novel Insights

The most genuinely novel contribution is the combination of (a) a fully unsupervised, instance-specific optimization objective (the MRF energy itself as loss), (b) a GNN reparameterization that provides a smooth, higher-dimensional parameter space for gradient descent, and (c) natural extension to high-order cliques via tensor products. The observation that GNN depth correlates with smoother, more navigable loss landscapes (Fig. 4–5) is an interesting empirical finding that could generalize to other combinatorial relaxations. The PCI application — transforming a real 5G network assignment problem into a pairwise MRF and demonstrating NEUROLIFTING's dominance at scale — is a concrete industrial case study that adds credibility beyond benchmark-only work.

---

## Evaluation on Key Axes

- **Originality:** Moderate. The core idea (GNN reparameterization for unsupervised CO) is not new (Schuetz et al., 2022), but the adaptation to general MRFs (arbitrary order, heterogeneous state spaces) with a clear lifting interpretation is a meaningful extension.
- **Importance of research question:** High. Scalable MAP inference in large MRFs is a fundamental and widely used problem.
- **Claims well supported:** Weak. The abstract's headline claims do not accurately reflect the results tables; the efficiency claims are unverifiable without runtime data.
- **Soundness of experiments:** Mixed. Diverse benchmarks and baselines are provided, but critical omissions (runtimes, GNN-CO baselines) and an unexplained catastrophic failure case undermine confidence.
- **Clarity of writing:** Acceptable. The method description is generally clear; the mismatch between abstract and results sections is the main presentational problem.
- **Value to community:** Moderate. The large-scale high-order MRF results and the PCI case study are genuinely useful; the paper would be more valuable if claims were accurately scoped.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to NEUROLIFTING |
|---|---|---|---|
| ROS (GNN relaxation for Max-k-Cut) | CpiJWKFdHN | 5.67 (Reject) | Most topically similar; same approach (GNN softmax relaxation + CO loss). Rejected for missing baselines and limited novelty. NEUROLIFTING is broader in scope but has similar missing-baseline and overclaiming issues. |
| Quasi-Quantum Annealing (gradient-based CO with timing) | 9EfBeXaXf0 | 6.75 (Accept-Poster) | Same domain (continuous relaxation for CO), but explicitly includes timing comparisons, cleaner claims, stronger performance. NEUROLIFTING is weaker on experimental rigor. |
| Gaussian Ensemble Belief Propagation | PLskiLUBDW | 7.00 (Accept-Poster) | Topically related (belief propagation, scalable inference), stronger theoretical grounding and honest claims. |
| Primal-Dual GNN for NP-hard CO | 4Hd7u3LHlZ | 5.25 (Reject) | Similar positioning (GNN for NP-hard combinatorial), rejected for similar reasons (limited contribution, missing baselines). |
| Low anchor — GCN feature transformation | HYsU5X4kE5 | 3.00 (Reject) | Clearly weaker paper with no real novelty; NEUROLIFTING is substantially better. |

NEUROLIFTING sits between the rejected ROS paper (5.67) and the accepted QQA paper (6.75). The ROS paper was rejected in part because of missing baselines and limited novelty — NEUROLIFTING has the same issues plus missing runtime data and an overclaimed abstract, but a somewhat broader scope (general MRFs, high-order, real-world PCI). The QQA paper includes timing comparisons and more honest claims. NEUROLIFTING's missing runtimes (critical for an efficiency-claiming paper) and the ProteinFolding_12 failure without explanation push it below the QQA bar.

**Final score: 4.5 (Reject)**

The paper's strongest results (large-scale high-order MRFs, PCI) are genuinely promising but are overshadowed by a systematically overclaimed abstract, the complete absence of runtime comparisons that would be needed to validate any efficiency claim, an unexplained catastrophic failure on a small instance, and missing comparisons to the most natural prior baselines. These are revision-addressable issues that make the paper not ready for acceptance in its current form.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>