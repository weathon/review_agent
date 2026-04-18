Now I have enough context for my calibration. Let me synthesize my review.

**Calibration comparisons:**
1. **IA-GPL (Instance-Aware Graph Prompt Learning)** - similar domain, similar concerns about marginal improvements over simpler baselines despite increased parameters. Scores: 6, 5, 5, 5 → avg ~5.25, withdrawn/rejected.
2. **Does Graph Prompt Work?** - similar domain (graph prompt theory), similar concerns about weak/incomplete theory. Scores: 3, 5, 6 → avg ~4.7, withdrawn/rejected.
3. **GPromptShield** - similar domain (graph prompt tuning), accepted as poster with scores 6, 6, 6 → avg 6.
4. **Edge embeddings in GNNs** - edge-related GNN paper, similar concerns about marginal improvements, scalability. Scores: 5, 5, 5, 6 → avg ~5.25, rejected.
5. **DP-GPL** - graph prompt privacy paper with serious theoretical flaws. Scores: 3, 5, 3, 3 → avg 3.5, rejected.

The EdgePrompt paper is:
- Stronger than DP-GPL and "Does Graph Prompt Work?" because its empirical results are solid and comprehensive
- Weaker than IA-GPL or GPromptShield in terms of clean attribution of gains (IA-GPL also had similar concerns but had some novelty in its PHM approach)
- Has a significant issue with the gap between theoretical claims and actual support from the theory

The paper's empirical results are genuinely strong and comprehensive (10 datasets, 4 pre-training strategies, 6 baselines). However, the attribution of gains to "edge-level prompting" vs. "more expressive parameterization" is not cleanly established, and the theoretical claims are overstated relative to what the theory actually proves. This is a solid empirical contribution with a reasonable conceptual idea, but the paper oversells its theoretical contribution and the specific "edge" narrative.

I'd place this around a 5.5 - it's a borderline paper with solid experiments but significant overclaiming in the theory and incomplete empirical analysis for attributing gains specifically to edge-level prompting.</think>## Summary

The paper proposes EdgePrompt and EdgePrompt+, graph prompt tuning methods that place learnable prompt vectors on edges rather than nodes. EdgePrompt assigns a single global prompt vector per GNN layer shared across all edges, while EdgePrompt+ computes edge-specific prompts as weighted averages of learnable anchor prompts using an attention-based scoring mechanism. Theoretical analyses are provided under the contextual stochastic block model (CSBM) and for universality in graph classification. Experiments across 10 datasets and 4 pre-training strategies demonstrate consistent improvements over six baselines.

## Strengths

- **Well-motivated core idea:** The observation that node-level prompts propagate uniformly to all neighbors via message passing (illustrated in Figure 1) is intuitive and clearly articulated. Moving prompts to edges to allow differentiated per-neighbor signal injection is a natural and reasonable design choice.

- **Comprehensive empirical evaluation:** Testing across 10 datasets, 4 pre-training strategies (GraphCL, SimGRACE, EP-GPPT, EP-GraphPrompt), and both node classification (5-shot) and graph classification (50-shot) provides broad coverage. EdgePrompt+ achieves the best or near-best performance in the vast majority of settings, with particularly large gains under EP-GPPT pre-training (e.g., ~28% improvement on Cora from 28.65% to 56.41%).

- **Compatibility and universality as design goals:** The method works with GNNs that do not natively support edge attributes (e.g., GCN, GIN), and applies to both node and graph classification, making it practically useful. The anchor-prompt mechanisml to handle label sparsity for edge prompts is a thoughtful engineering choice.

- **The paper correctly notes the close relationship between EdgePrompt and GPF:** The observation that "GPF and EdgePrompt have relatively small performance gaps (always < 1.8%)" is honest and supports Theorem 2's claim of comparable universal capability between the two methods.

## Weaknesses

### Fatal
None.

### Major

- **The theoretical claims substantially overreach what the theorems actually establish.** Theorem 1 is restricted to a two-class CSBM and merely shows existence of prompts that improve centroid separation; it does not compare edge prompts against node prompts under the same assumptions, nor does it address learnability from limited labeled data. Theorem 2 is an existence statement that could apply to many sufficiently expressive prompt parameterizations, yet is used to claim "comparable universal capability with GPF" and to support universality claims in the abstract and introduction. The conclusion in Section 4.3 that "we can conclude that our proposed EdgePrompt+ benefits pre-trained GNN models for node classification" is a very strong generalization from a narrow existence result. These overclaims are structurally woven throughout the paper and misrepresent the theoretical contribution.

- **The experiments do not cleanly isolate the contribution of edge-level prompting from increased per-message expressivity.** EdgePrompt+ introduces anchor prompts plus per-edge attention weights, which is a more expressive parameterization than prior node-prompt baselines. Crucially, there is no ablation applying the same anchor-and-attention machinery to nodes rather than edges. Since EdgePrompt (the single-global-prompt variant) often performs comparably to GPF, the gains of EdgePrompt+ likely come from the richer per-edge parameterization rather than from the conceptual shift from nodes to edges. Without this control, the central claim that "edge prompt design" is fundamentally superior to "node prompt design" is not adequately substantiated.

### Minor

- **Limited architecture diversity:** Only GCN (node classification) and GIN (graph classification) are tested as backbones, despite the claim of compatibility with "prevalent GNN architectures." Testing on GAT, GraphSAGE, or edge-attribute-aware architectures would strengthen this claim, especially since the integration mechanism (Eq. 2) could interact differently with attention-based architectures.

- **No computational cost or scalability analysis:** EdgePrompt+ requires computing attention scores for every edge at each layer (O(|E| × M_l) per layer). No training/inference time, parameter count, or memory comparison is provided. This is a notable omission for a method that claims practical applicability, especially given that edge counts can be quadratic in dense graphs.

- **Narrow few-shot settings tested:** Only 5-shot node classification and 50-shot graph classification are evaluated. Without varying shot counts (e.g., 1, 3, 10, 20), it is unclear how method performance scales with label availability.

### Trivial
- None.

## Nice-to-Haves

- Ablation study isolating per-edge attention from the edge-level placement (e.g., a "NodePrompt+" variant with anchor prompts and per-node-neighbor attention on node features) to clearly attribute gains.
- Visualization or analysis of what learned edge prompts encode (e.g., whether inter-class edges receive different prompts than intra-class edges).
- Experiments on graphs with pre-existing edge attributes to validate compatibility claims.
- Experiments with varying shot counts to characterize regime-dependent performance.

## Removed Points

- **Fine-tuning baselines missing (from Spark):** The paper scopes itself as comparing prompt tuning methods against each other, not against fine-tuning. Adding fine-tuning would be informative but is outside the stated scope; the paper's framing and contribution are about graph prompt tuning methods, and the "Classifier Only" baseline already provides a no-prompt reference point.

- **No experiments on graphs with pre-existing edge attributes (from Spark):** While potentially interesting, evaluating on graphs with edge attributes would test a different scenario (interference between prompt vectors and existing attributes) rather than the core claim of edge-level prompting. This is a nice-to-have rather than a weakness.

- **Convergence plots lack standard deviation bands (from Spark):** These are supplementary analysis plots, not main results. Single-run convergence curves are standard in the field.

- **Formatting/notation nitpicks (from Harsh Critic notes about Eq. 1, Table 1 MultiGPrompt column):** These are minor presentation issues that do not affect substance.

- **"Existing methods are unable to enhance pre-trained GNN models in capturing complex graph structural information" is overstated (from Harsh Critic):** Node prompts do propagate through edges via message passing, but the paper's central argument is that this propagation is uniform and cannot differentiate between neighbors—a valid point for GCN-like architectures. The phrasing is somewhat strong but the underlying argument is reasonable.

- **Theorem 2 existence result criticism (from Harsh Critic):** While the theorem is indeed an existence result, GPF's universality theorem (Fang et al. 2023) is also an existence result of similar type. The comparison is therefore not misleading in the context of the literature. The weakness is properly captured in the "Major" section above about overclaiming generality.

## Novel Insights

The key insight from combining the reviews is that EdgePrompt may be better understood as a *per-message* prompt mechanism rather than a fundamentally "edge-level" concept. The empirical evidence shows that almost all gains come from EdgePrompt+ (with per-edge attention), while the purely edge-level EdgePrompt (single global vector per layer) is comparable to GPF. This suggests the contribution is primarily about richer per-message conditioning rather than the structural distinction between edges and nodes per se—a nuance the paper does not surface.

## Suggestions

- **Add a NodePrompt+ ablation:** The single most impactful experiment would be a node-level version of the anchor-and-attention design, to determine whether the gains come from edge placement or from expressivity.
- **Tone down theoretical claims:** Rewrite Theorem 1's conclusion to accurately reflect its scope (e.g., "under a two-class CSBM, there exist edge prompt configurations that improve class separability"), and acknowledge that Theorem 2 is an existence result analogous to GPF's universality theorem rather than a stronger result.
- **Report parameter counts and training time:** Even a simple table comparing EdgePrompt+ vs. GPF-plus on parameter count and wall-clock time would address practical concerns.

---

**Score calibration:**
- IA-GPL (similar domain, marginal improvements, overparameterized): avg 5.25, withdrawn/rejected
- GPromptShield (graph prompt, accepted poster): avg 6.0
- Edge embeddings in GNNs (edge-level method, marginal empirical gains): avg 5.25, rejected
- "Does Graph Prompt Work?" (weak theory for graph prompts): avg 4.7, withdrawn/rejected

EdgePrompt has stronger empirical results than the edge-embedding and IA-GPL papers (10 datasets, 4 pre-training strategies, consistently competitive), but shares their attribution problem (gains may come from expressivity rather than the conceptual contribution). Its theoretical contribution is overstated. The paper is borderline: the empirical work is solid and the idea is natural, but the core narrative (edges vs. nodes) is not cleanly substantiated, and the theory doesn't support the claimed generality. It sits above the clearly rejected papers but below the accepted GPromptShield (which had a more novel contribution angle in robustness).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>