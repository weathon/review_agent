Now let me run calibration searches to anchor the score.Now I have enough information to write the final review.

---

## Summary

GECO is a GNN explainability method for graph classification that applies modularity-based community detection to partition a graph into communities, then scores each community by passing it as a stand-alone subgraph through the trained GNN and comparing the output probability to a mean threshold. Communities exceeding the threshold form the explanation mask. The method is evaluated on six synthetic and four real-world datasets against five baselines, showing strong synthetic results and a dramatic computational efficiency advantage (≈18 s vs. ≈700 s for competitors on real-world datasets).

---

## Strengths

- **Novel structural prior for explanation**: GECO is the first method (in this line of work) to use community structure—motivated by the AGGREGATE step of GNN message-passing (Section 3.1–3.2, Eq. 1)—as the explanatory unit. This is a principled and original design choice distinct from perturbation masks, MCTS, or graphical-model surrogates.

- **Ground-truth-validated improvement (GEA)**: Beyond fidelity metrics, GEA in Table 1 shows GECO consistently improves over baselines using ground-truth motif membership. For example, ba\_cycle\_wheel: 0.553 vs. 0.380 (SubgraphX), and ba\_cycle\_wheel\_grid: 0.561 vs. 0.527 (SubgraphX). Because GEA directly measures precision against the known motif, it partially validates the results independently of possible fidelity-metric artifacts.

- **Exceptional computational efficiency**: Section 4.1–4.2 shows GECO runs in under 3 s on synthetic datasets and ≈18 s on real-world datasets, versus 100–700+ s for SubgraphX, GNNExplainer, and PGExplainer. This is a genuine practical differentiator traceable to the method's algorithmic simplicity (Algorithm 2).

- **Robust experimental protocol**: 100 random train/test splits with standard deviations reported throughout (Tables 1–2) provide significantly stronger variance estimates than the single-split evaluations common in GNN explainability literature.

---

## Weaknesses

### Fatal
None.

### Major

- **Explanation mask size never reported, compromising interpretation of Fid⁻ ≈ 0**: The paper's headline result is near-perfect Fid⁻ (0.000–0.002) across all six synthetic datasets and comparable values in Table 2. Both Fid⁺ and Fid⁻ are known to be sensitive to mask size: a very small mask causes the GNN to receive an out-of-distribution (OOD) micro-subgraph, which for a 3-layer GCN trained on 30–100+ node graphs may simply output a near-constant probability—consistent with the correct label if the majority class dominates. The random baseline's Fid⁻ ≈ 0.40–0.52 (Table 1), reflecting ~50% node selection, strongly suggests a size-dependent effect. Without reporting the average number of nodes in each method's explanation mask, it is impossible to attribute GECO's Fid⁻ advantage to explanation quality rather than explanation compactness. This is not a presentation issue; it undermines the interpretability of the most striking numbers in the paper. Every established GNN explainability benchmark controls for explanation size (e.g., fixing the top-k nodes or top-k%). Re-running with size-controlled comparisons is required for these results to be interpretable.

- **Synthetic datasets are structurally designed to favour modularity-based community detection**: All six datasets attach structurally dense motifs (wheel, house, cycle) to sparse Barabási-Albert or Erdős-Rényi base graphs (described in Appendix A.1). A wheel or house subgraph attached to a BA graph has high internal edge density and low external connectivity—the textbook profile of a high-modularity community. GECO's Clauset et al. greedy modularity algorithm will naturally identify these motifs as communities almost deterministically, since they correspond exactly to the peaks of the modularity objective. The ground-truth explanation is, in effect, the highest-modularity community in the graph. This creates a deep alignment between the evaluation setup and GECO's algorithmic mechanism that the paper neither acknowledges nor controls for. The large performance gap on synthetic datasets (e.g., Fid⁺ 0.929 vs. 0.478 for ba\_house\_cycle) versus real-world data (where GECO loses Fid⁺ and charact on Mutagenicity) is consistent with this explanation. The synthetic results therefore do not constitute evidence that GECO generalises to settings where the classification-relevant subgraph is not a modularity-dominant community.

### Minor

- **Out-of-distribution validity of community subgraph forward passes not discussed**: Algorithm 2 Step 3 feeds isolated community subgraphs (potentially 3–10 nodes) to a GNN trained on full graphs (30–100+ nodes). Node embeddings computed via neighbourhood aggregation in a small induced subgraph are drawn from a different distribution than those in the full graph, which the GNN has never seen during training. The paper provides no theoretical or empirical justification that the ranking of communities by p_i reflects their actual causal contribution to the GNN's full-graph prediction. This concern is shared to varying degrees by all perturbation-based explainers (which also create OOD inputs), but the severity here is larger due to the extreme graph-size difference and the fundamental dependence of Algorithm 2 on these p_i values being meaningful. At minimum, an empirical check showing that communities containing the ground-truth motif consistently receive higher p_i than non-motif communities would help.

- **Two methods are both labelled "PGExplainer" in the results tables**: Table 2 has two consecutive rows labelled "PGExplainer" with different numbers, referencing Luo et al. (2020) and Vu & Thai (2020) (the latter is PGMExplainer). Table 1 correctly uses "PGMExplainer" for Vu & Thai. The inconsistency makes it impossible to determine which method produces which result in Table 2 without referring back to the text, and the name collision in related work (Section 2) is also confusing.

- **Threshold τ = mean and weights w+, w− not ablated or reported**: Algorithm 2 fixes τ as the simple mean of p_i with no justification for this choice over the median or other quantiles mentioned in Section 3.2. The weights w+ and w− for the characterization score (Eq. 5) are also never stated in the paper, making the charact metric effectively unreproducible.

- **GNN classification accuracy never reported**: Without knowing whether the underlying GNN achieves good classification, Fid⁺, Fid⁻, and GEA are all ambiguous—a random-accuracy GNN would yield confounded fidelity metrics regardless of explanation quality.

### Trivial

- Figure 2 shows only the final explanation mask, not the underlying community partition. Showing both would allow readers to verify that community boundaries align with meaningful substructures, as claimed.

---

## Nice-to-Haves

- **Ablation over community detection algorithms**: The paper chooses Clauset et al. greedy modularity without comparison to Louvain, Girvan-Newman, or random partitioning. An ablation would clarify whether the gains derive from the community-detection framework broadly or specifically from modularity optimization.
- **Test on GNN architectures beyond GCN**: The forward-pass scoring mechanism is architecture-dependent; results on GAT or GIN would strengthen the generality claim.
- **Analysis of the synthetic/real-world gap**: GECO's advantage over GNNExplainer is 0.451 Fid⁺ points on ba\_house\_cycle but reverses by −0.159 on Mutagenicity. Understanding why would significantly strengthen the paper.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **Harsh reviewer concern: "GECO's out-of-distribution community forward passes completely invalidate the mechanism"** — Weakened rather than removed. While the OOD concern is real, all perturbation-based explainability methods create partially OOD inputs by design. The concern is moved to Minor rather than treated as a fatal flaw.

- **Harsh reviewer concern: "the abstract is misleading by saying 'most real-world datasets'"** — Removed as pure nitpick. GECO achieves best charact on 3/4 real-world datasets; "most" is factually accurate.

- **Harsh reviewer concern about GEA structural bias on negative class** — Removed as a weakness because the paper already transparently acknowledges and explains this limitation in Section 4.2 with appropriate technical detail. The concern is self-contained in the paper.

---

## Novel Insights

The most insightful observation from the reviews—which the paper itself does not fully confront—is the structural co-design problem: all six synthetic evaluation benchmarks are instances of "dense motif attached to sparse random graph," which is exactly the problem community detection via modularity maximization is provably well-suited to solve. This means the synthetic results, while striking, represent a near-trivial case for GECO's mechanism, and the real-world results (which are more mixed) likely reflect the true generalisation capability of the method. Papers proposing structure-exploiting explainability methods should be required to include at least one synthetic benchmark where the ground-truth explanation does *not* correspond to a high-modularity community, to separate the contribution of the algorithmic framework from the evaluation design.

---

## Suggestions

1. Re-run all experiments with fixed-size explanations (select the top-k nodes per method, with k matched across methods) or add a column reporting the average mask size. This is the single most important change needed.
2. Add a synthetic benchmark where the class-discriminative motif is embedded in a region of low local modularity (e.g., a star subgraph inside a regular lattice base) to test whether GECO generalises beyond modularity-aligned scenarios.
3. Clearly label the two PGExplainer variants consistently across all tables.
4. Report the w+ and w− weights used in the characterisation score and perform a sensitivity analysis on τ.

---

## Score and Decision

**Calibration anchors:**

- *KZII3faAs2* ("AIMing for Explainability in GNNs", avg ≈ 3.4, Reject): An XAI-for-GNNs survey/framework paper rejected primarily because key claims were contradicted by existing literature. GECO is clearly stronger—its contribution is original and not contradicted—so it should score above this.

- *zSUXo1nkqR* ("TreeX", avg ≈ 3.4, Reject): GNN explanation via subtree extraction, rejected for narrow scope and thin experimental validation. GECO has broader experiments and real-world evaluation.

- *hXJrQWIoR3* ("Explainable GRL via Graph Pattern Analysis", avg ≈ 5.75, Reject): Has theoretical analysis, multi-setting evaluation, and outperforms baselines. Rejected mainly due to presentation and scalability. Comparable in scope to GECO.

- *AJBkfwXh3u* ("Causality-Inspired Spatial-Temporal Explanations", avg ≈ 6, Accept-Poster): Accepted with rigorous causal modeling and solid baselines. Stronger methodological justification than GECO.

GECO sits between the 3.4-range rejects (which had more fundamental problems) and the 5.75 borderline reject. Its key advantage over the rejects is a genuinely novel, clean contribution with practical utility (speed). Its key disadvantage relative to a ~6-level paper is that both major experimental pillars—the synthetic results and the near-zero Fid⁻ values—have substantive validity concerns that the paper does not address. The real-world results are positive but less dramatic. I place GECO at **4.5**: it clears the threshold of having a real contribution but the evaluation methodology issues (unreported mask sizes, structurally biased benchmarks) are major enough to prevent acceptance without revision.

**Originality**: Moderate — community detection as an explanatory unit is a fresh angle.  
**Importance of research question**: High — GNN explainability is central and practically relevant.  
**Claim support**: Weak — the most dramatic claims rest on potentially confounded metrics.  
**Soundness of experiments**: Insufficient — missing mask-size controls and structurally biased synthetic benchmarks.  
**Clarity**: Adequate — algorithm is clearly described; evaluation reporting has notable gaps.  
**Value to community**: Conditional — the efficiency contribution is real; the accuracy claims need verification.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>