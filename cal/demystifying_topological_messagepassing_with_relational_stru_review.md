=== CALIBRATION EXAMPLE 33 ===

# Final Consolidated Review
## Summary

This paper proposes a unifying framework that models simplicial complexes and their message-passing schemes as relational structures, introducing the concept of *influence graphs* as the key technical tool. By embedding simplicial message passing into the relational structure formalism, the authors extend established GNN oversquashing results — sensitivity bounds, curvature connections, depth analysis, and hidden dimension effects — to topological deep learning (TDL). A practical rewiring heuristic adapts existing graph rewiring algorithms to relational structures via a "collapsed adjacency matrix." The framework is positioned as addressing open questions (directions 2 and 9 of Papamakarios et al., 2024) in the TDL community.

---

## Strengths

- **Influence graph as a unifying analytical tool:** The construction of the aggregated influence matrix (Eq. 6) and its associated influence graph (Definition 3.1) is a concrete and non-trivial contribution. It collapses heterogeneous, multi-arity relational structure into a single weighted directed graph on which GNN-theoretic machinery can operate — this is not automatic and provides a principled object for future TDL analysis.

- **Breadth of architectural coverage:** Remark 2.7 explicitly demonstrates that the relational message-passing model encompasses RGCNs, simplicial networks (Bodnar et al., 2021b), higher-order GNNs (Morris et al., 2019), and CW networks (Bodnar et al., 2021a). This is a genuine unification, not just a re-labeling.

- **Systematic RingTransfer validation:** The synthetic experiments in Section 5.2 are well-designed, varying hidden dimensions, ring size, and rewiring iterations in isolation. Each sub-experiment maps to a distinct theoretical result (Theorem 3.5 / Section 3.4 / Section 4), giving the theory three independent empirical anchors in a controlled setting.

- **Extensibility to cellular complexes:** Section 2 explicitly notes (and Appendix G demonstrates) that the framework extends to cellular complexes and higher-order graphs, making the contribution more broadly applicable than simplicial networks alone.

- **Statistically significant curvature-lifting relationship:** Section 5.3 and Appendix D.2 report a statistically significant linear relationship between weighted curvature of graphs and their lifted clique complexes. This is a genuinely interesting empirical pattern that, if robust, offers a practical handle on how lifting modifies geometric properties.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Likely notation error in Proposition 3.4 (Equation 10).** The left-hand side of Equation 10 reads ‖∂**h**_τ^(2)/∂**h**_τ^(0)‖₁, i.e., the *self-sensitivity* of entity τ at layer 2. Yet the right-hand side involves EFC_G(τ, σ), w_τ^out, and w_σ^in — quantities defined for the directed edge (τ→σ) and involving the distinct entity σ. This is structurally incoherent: a self-sensitivity bound should not depend on properties of an arbitrary neighbor σ, and the asymmetric dependence on σ's in-degree has no natural interpretation in that context. The analogous result in Fesser & Weber (2023, Prop. 3.4), which the paper explicitly extends, bounds the cross-entity Jacobian ‖∂h_σ^(2)/∂h_τ^(0)‖ for the edge (τ→σ) — suggesting the numerator in Eq. 10 should be **h**_σ^(2), not **h**_τ^(2). If this is a typo, it must be corrected and the proof checked. If it is intentional, a clear conceptual justification is required. As stated, the result is either incorrect or its connection to oversquashing (which concerns cross-entity influence) is unclear.

- **GIN/None dominates simplicial models on RingTransfer — the benchmark specifically designed to test long-range propagation.** Figure 2 shows plain GIN with no topological lifting achieving the highest accuracy across all three experimental dimensions (hidden dimension, ring size, rewiring iterations), while RGCN/Ring is consistently the weakest. The paper mentions this only obliquely (Section 5.2 describes "similar trends"), but does not explain *why* lifting into higher-order structures degrades or fails to improve performance on a benchmark where long-range structure should matter. This is a meaningful negative result that directly challenges the practical motivation for the topological approach; it deserves an honest discussion rather than framing the figure as a validation of theoretical trends.

- **Post-hoc selection bias in "Best Rew." column.** Table 1 reports the best performance across three rewiring algorithms (SDRF, FoSR, AFRC) without a held-out selection procedure. Selecting the best of three post-hoc inflates the apparent benefit of rewiring across the board. The paper should either report each rewiring variant separately or use a validation-set procedure to select among algorithms — the current presentation overstates how reliably rewiring helps.

### Minor

- **Proposition 3.4 is restricted to t = 2 layers.** Even if the notation error is resolved, the curvature result applies only at depth 2, while architectures that suffer from oversquashing typically have 4–10 layers. The paper frames this as illustrative (and notes that a balanced-Forman extension is left for future work), but the restriction significantly limits the result's practical interpretability. This should be foregrounded as a limitation rather than understated.

- **Mixed rewiring results in Table 1 without failure-mode analysis.** Rewiring frequently degrades performance (e.g., GIN/None on IMDB-B: 74.7 → 67.1; SIN/Clique on ENZYMES: 51.0 → 46.5; CIN++/Ring on MUTAG: 90.5 → 84.5). The paper briefly attributes variance to dataset- and model-agnostic hyperparameters but provides no analysis of *when or why* rewiring helps versus hurts. Understanding this is important for practitioners and could reveal whether oversquashing is actually the binding bottleneck in these cases (versus, e.g., oversmoothing induced by added edges).

- **Section 3.4 (hidden dimensions) is informal.** The conclusion that β^{(ℓ)} = O(p'_ℓ) is asserted to hold for shallow networks and readers are referred to Appendices B.2 and C.4, but this scaling is initialization- and architecture-dependent and does not hold universally for trained MLPs. The section is presented alongside Lemma 3.2 and Theorem 3.5, suggesting comparable rigor, yet it is an observation relying on an additional assumption not shared by the main results. Clearly labeling it as a corollary under specific initialization conditions would be more accurate.

- **Conceptual gap between theory and rewiring heuristic.** The theoretical bottleneck is identified via the influence graph G(S, B̃), a weighted directed object. The rewiring heuristic operates on the collapsed adjacency matrix A^col, an unweighted undirected graph that discards both edge directions and relation-type distinctions. The paper acknowledges in Section 6 that existing rewiring algorithms "were not originally designed with weighted directed influence graphs in mind," but does not discuss whether rewiring A^col actually reduces distances or increases walk counts in G(S, Ã). Without this link, the heuristic is pragmatic but theoretically decoupled from the analysis.

### Tiny

- The ternary relations R₄ and R₅ (lower/upper adjacency) are stated to be equivalent to the standard binary message-passing formulation, but no formal proof or remark is given. Uniqueness of δ given (σ, τ) — which is what makes the reduction work — holds trivially here (δ = σ ∩ τ for lower, δ = σ ∪ τ for upper), but stating this explicitly would tighten the paper.

---

## Nice-to-Haves

- **Rewiring directly on the influence graph.** Implementing a rewiring algorithm tailored to the weighted directed influence graph G(S, B̃) (rather than A^col) would close the theory–practice gap identified above. Even a pilot experiment on a small example would strengthen the paper considerably.

- **Computational overhead reporting.** No training time or memory usage is reported for topological versus graph models, nor for the influence matrix computation. Given that ternary shift operators are higher-dimensional tensors, practitioners need to know the computational cost.

- **A separation task for higher-order structure.** A synthetic benchmark where higher-order interactions are *strictly required* (not merely helpful) would provide a sharper argument for topological message passing, separately from the oversquashing question. The current RingTransfer results show GIN dominating, leaving the practical value of higher-order structure undemonstrated.

- **Oversmoothing discussion.** Rewiring adds edges and can exacerbate oversmoothing. A brief discussion (even qualitative) of the oversquashing–oversmoothing trade-off in the topological setting, especially in light of the rewiring failures in Table 1, would be valuable.

- **Tuned baseline for at least one dataset.** Even for a single dataset, tuned results would establish a ceiling and clarify whether the fixed-hyperparameter regime is systematically disadvantaging topological models relative to their true potential.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"Axiomatic label is a stretch."** The harsh critic argues that calling the relational framing "axiomatic" is overselling. This is a terminology preference, not a substantive flaw.

- **"Lemma 3.2 is mathematically trivial."** The critic notes that once the relational framework is in place, the bound follows by standard chain-rule telescoping. While technically accurate, this is not a weakness: the value of unification results lies precisely in enabling routine extensions. This criticism would apply equally to much of the GNN theory literature.

- **"Double-sum over-counting in Eq. 5 makes bounds looser."** The critic identifies that pairwise interactions may be counted multiple times. Since B is used only for upper bounds, this is conservative by design, not a flaw. The paper uses these bounds directionally, not as tight estimates.

- **"Attention mechanisms are excluded."** Attention-based topological networks violate the fixed-operator assumption of Eq. 4. While true, this is a well-understood scope restriction of sensitivity-based oversquashing analysis that applies equally to Topping et al. (2022) and Di Giovanni et al. (2023). Criticizing its absence is scope creep.

- **"No comparison to published state-of-the-art."** The paper explicitly uses fixed, model-agnostic hyperparameters and frames experiments as a controlled study of rewiring effects, not a competitiveness claim. Demanding SoTA comparison misreads the experimental design.

- **"Theorem 3.5 condition M < 1/(2α_max β_max) has no analysis of when it holds."** The critic argues this is incomplete guidance for practitioners. However, this is standard form for analogous results in Di Giovanni et al. (2023) and the paper correctly positions it as a structural insight, not an empirical design rule.

- **"The framing as addressing research directions 2 and 9 of Papamakarios et al., 2024 appears twice — reads like padding."** Pure stylistic criticism.

---

## Novel Insights

The most genuinely interesting observation to emerge from the synthesis is the *negative result* in Figure 2: graph lifting into simplicial or ring complexes *degrades* performance on RingTransfer relative to plain GIN, even on a benchmark designed to reward long-range propagation. This is not adequately discussed in the paper but constitutes a hypothesis-generating finding — it suggests that topological lifting may introduce structural bottlenecks (e.g., via the richer relational graph G(S, B̃)) that offset any benefit from higher-order adjacencies. Systematically studying *when* lifting helps versus hurts through the lens of the influence graph's spectral and geometric properties is a natural and valuable direction enabled by this framework. Additionally, the statistically significant linear relationship between weighted curvature of base graphs and their clique complexes (Section 5.3, Appendix D.2) is an empirical regularity that, if theoretically grounded, could explain both lifting benefits and failure modes.

---

## Suggestions

1. **Resolve the Proposition 3.4 notation error.** Check whether the numerator should read ∂**h**_σ^(2) (cross-entity sensitivity, consistent with Fesser & Weber 2023) or ∂**h**_τ^(2) (self-sensitivity). If the latter is intended, provide an explicit conceptual justification for why self-sensitivity connects to the edge curvature EFC(τ, σ) — this is non-trivial and not currently argued.

2. **Explain the GIN/None dominance on RingTransfer.** Add a paragraph in Section 5.2 or 6 analyzing why plain GIN outperforms all topological variants on the long-range benchmark. Possible explanations include: (a) the influence graph for lifted complexes has larger diameter than the base graph; (b) ring/clique lifting adds structural noise that interferes with task-relevant signals; (c) the fixed hyperparameters disadvantage topological models disproportionately. This analysis would substantially strengthen the paper's scientific contribution.

3. **Replace post-hoc "Best Rew." with per-algorithm results or a validation-based selection.** Report SDRF, FoSR, and AFRC separately (even in an appendix), or select among them using a validation set and report the selection procedure. This makes the rewiring benefit verifiable and removes the selection bias.

4. **Quantify bound tightness empirically.** For at least one synthetic example, compare the theoretical sensitivity upper bound from Lemma 3.2 (or Theorem 3.5) with empirically measured Jacobian norms. This would address the fundamental question of whether the bounds are informative in practice or vacuously loose.

5. **Discuss rewiring-induced oversmoothing explicitly,** especially given the frequent red-text entries in Table 1. Even a brief hypothesis connecting oversmoothing risk to specific topological configurations (e.g., dense clique lifting) would give practitioners useful guidance.

---

**Evaluation summary:**
- *Novelty:* Moderate — the relational structure viewpoint has precedent, but the influence graph construction and systematic oversquashing analysis in TDL are genuinely new contributions to the community.
- *Technical soundness:* Moderate — Lemma 3.2 and Theorem 3.5 are sound extensions of known results; Proposition 3.4 contains a likely notation error that must be resolved; Section 3.4 is informal.
- *Empirical support:* Weak-to-moderate — RingTransfer confirms theoretical trends, but the most salient finding (GIN dominance) is not explained; Table 1 has selection bias and no hyperparameter tuning, limiting interpretability.
- *Significance:* Moderate — the framework is a useful foundation for TDL analysis and addresses recognized open questions, but the practical payoff of rewiring is inconsistent and the negative RingTransfer result is unexplained.
- *Clarity:* Generally good, with the Proposition 3.4 notation issue being the main exception.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 8.0, 8.0]
Average score: 6.8
Binary outcome: Accept
