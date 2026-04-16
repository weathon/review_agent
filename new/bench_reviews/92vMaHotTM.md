## Summary

EdgePrompt introduces graph prompt tuning from the perspective of edges, proposing edge-level learnable prompt vectors that are aggregated through the message-passing mechanism of pre-trained GNN models. The basic version (EdgePrompt) uses a shared global prompt vector per layer, while EdgePrompt+ computes per-edge prompts via attention-weighted anchor prompts. The method is evaluated on 10 datasets under 4 pre-training strategies for both node and graph classification, with theoretical analysis under the CSBM model and a representation equivalence theorem.

## Strengths

- **Intuitive and novel perspective:** Moving prompts from nodes to edges is a natural and well-motivated idea. The "uniform message passing" argument (Figure 1) — that node-level prompts propagate the same vector to all neighbors — clearly identifies a structural limitation of prior methods and provides a clean conceptual justification for edge-level intervention.

- **Comprehensive experimental coverage:** Testing across 10 datasets, 4 pre-training strategies (2 contrastive + 2 generative), and both node and graph classification provides broad empirical evidence. The consistent top-or-runner-up performance of EdgePrompt+ across most settings is notable.

- **Plug-in compatibility with existing GNN architectures:** By injecting prompts into the message-passing aggregation rather than requiring architecture modifications, EdgePrompt works with architectures like GCN and GIN that lack native edge attribute support. This is a practical design advantage.

- **Clear theoretical effort:** Unlike many prompt tuning papers that are purely empirical, this work attempts formal analysis (Theorems 1 and 2), which adds methodological depth even if the results have limitations (discussed below).

## Weaknesses

### Major:

- **Theoretical results are overclaimed relative to their actual strength.** Theorem 2 states that for any graph transformation and any pre-trained GNN, there exist edge prompts that can reproduce the transformed graph's representation. However, this is an unconstrained existence statement — it does not specify how prompts enter the forward pass for each architecture, places no restrictions on prompt dimensionality or capacity, and says nothing about whether gradient-based optimization with few labels can find such prompts. Using this to claim "comparable universal capability with GPF" (Section 4.4) conflates representational capacity under idealized conditions with practical universality. Similarly, Theorem 1 is an existence result under a restricted 2-class CSBM with a single GCN layer, yet the paper leaps to "EdgePrompt+ benefits pre-trained GNN models for node classification" in general. These are existence proofs of expressivity, not guarantees of practical benefit under realistic training conditions.

- **Modest empirical gains over strong baselines, especially given added complexity.** EdgePrompt and GPF show gaps consistently below 1.8% (acknowledged in the paper). EdgePrompt+ improvements over GPF/GPF-plus are often in the 1–4% range, with many results within one standard deviation. On graph classification (Table 3), GPF-plus sometimes matches or exceeds EdgePrompt+ (e.g., DD under EP-GPPT and EP-GraphPrompt, Mutagenicity under SimGRACE). Meanwhile, EdgePrompt+ introduces anchor prompts per layer, per-edge attention scores, and weight matrices — substantially more parameters than simpler baselines. Without parameter count comparisons, it is unclear whether gains come from the edge-prompting mechanism or simply from added model capacity.

- **Scalability and computational overhead not addressed.** EdgePrompt+ computes attention scores for every edge (Eq. 6), requiring O(|E| · M_l) operations per layer where M_l is the number of anchor prompts. For large, dense graphs, this is non-trivial. The paper provides no runtime, memory, or parameter count analysis, which is a significant gap for a method claiming "simple yet effective" design and evaluated on ogbn-arxiv (a graph with ~1.2M edges).

- **Limited backbone diversity for "compatibility" claims.** The abstract and introduction claim the method is "compatible with prevalent GNN architectures," but only GCN (2-layer) and GIN (5-layer) are tested. Architectures like GAT, Graph Transformer, or models that natively support edge features are not evaluated. This is especially relevant because the "uniform message passing" motivation specifically targets GCN-style aggregation — attention-based models already modulate per-edge importance, which may reduce EdgePrompt+'s advantage. The compatibility claim is therefore only weakly supported.

### Minor:

- **Missing ablation on score function φ:** The paper acknowledges (Section 4.2) that "many typical formulations can be used" for φ but reserves exploration for future work. The choice of LeakyReLU-based attention (Eq. 6) is a key design decision that directly affects how edge prompts are customized, yet no alternatives are tested. This leaves the contribution of the attention mechanism vs. the edge-level prompting concept confounded.

- **Inconsistency between Theorem 1 implications and experimental choice of datasets:** The bound T ∈ (1, 1 + p/|p−q|] suggests the largest theoretical benefit when p ≈ q (heterophilic graphs), yet all evaluation datasets are homophilic (Cora, CiteSeer, etc. where p ≫ q). Testing on heterophilic benchmarks would better align theory and experiments and clarify where edge prompts are most beneficial.

- **MultiGPrompt omitted from experiments despite being listed in Table 1.** MultiGPrompt appears in the method comparison table as a related approach but is not included in experimental comparisons. Given it also inserts prompts into hidden representations, it is a more directly comparable baseline than some included methods.

## Nice-to-Haves

- A fine-tuning baseline would contextualize whether prompt tuning (edge-based or otherwise) is competitive with the most natural alternative, though this is outside the paper's stated scope of comparing prompt tuning methods.
- Experiments on heterophilic graph benchmarks (e.g., Chameleon, Squirrel) would connect the theoretical analysis to a setting where edge prompts might shine.
- Visualization of learned edge prompt vectors (e.g., t-SNE colored by intra-class vs. inter-class edges) would provide intuitive evidence that edge prompts learn structurally meaningful distinctions.

## Removed Points

- **Claim that GPF-plus makes neighboring nodes receive identical aggregated messages (Spark reviewer #1):** This misreads the paper. EdgePrompt's argument is that node v_1's prompt p_1 is sent identically to *all* neighbors, not that neighbors receive the same total aggregated message. The distinction is per-edge customization vs. per-node prompt propagation — this is a legitimate point, though it only applies to GCN-style aggregation.

- **Demand for fine-tuning comparison as a "fatal" omission (Spark reviewer #1):** The paper's stated scope is graph prompt tuning, and all baselines are prompt tuning methods. Fine-tuning is a different paradigm. This is a nice-to-have, not a core flaw.

- **Formatting/style nitpicks (removed per instructions).**

- **Claim that Theorem 2 is "vacuous" (Harsh Critic):** While the theorem is indeed an unconstrained existence statement, calling it "vacuous" overstates the issue. The result does establish a minimum expressivity guarantee for EdgePrompt — that it can simulate at least as many transformations as GPF. The real issue is the *leap* from this to practical universality claims, not that the theorem itself is meaningless. Downgraded from "fatal" to "major" — the mathematical result has content, but its interpretation is overreaching.

## Novel Insights

The observation that node-level prompts in GCN-style architectures propagate identically across all outgoing edges — while edge-level prompts enable per-neighbor customization — is a clean and valid insight. However, the extent to which this translates to practical gains depends heavily on the architecture: attention-based GNNs already modulate per-edge importance, potentially diminishing EdgePrompt+'s advantage. The empirical results largely confirm this — gains are moderate and sometimes within noise margins, suggesting the edge-prompting concept is useful but not transformative in current settings.

## Suggestions

1. Report parameter counts and wall-clock time for all methods to clarify whether EdgePrompt+'s gains come from the prompting mechanism or added capacity.
2. Ablate the score function φ (e.g., compare attention, inner product, linear mapping) to disentangle the contribution of the anchoring/attention mechanism from the edge-level prompting concept.
3. Test on at least one attention-based GNN backbone (e.g., GAT) and one heterophilic benchmark to evaluate the generality and boundary conditions of the method.
4. Moderate the universality claims in Theorem 2's discussion — frame it as an expressivity result rather than a practical guarantee.

## Score and Decision

**Calibration:** I compared against: (1) IA-GPL (scores 5–6, reject) — similar domain, moderate novelty, marginal gains over GPF-plus; (2) Edge embedding papers in GNNs (scores 5–6, reject) — edge-level representations with scalability concerns; (3) "Does Graph Prompt Work?" (scores 3–6, rejected/withdrawn) — theoretical analysis of graph prompts; (4) CSBM-based analysis papers (scores 3–5, reject) — restricted theoretical settings. This paper has a cleaner experimental setup than many rejected papers and a genuinely intuitive idea, but suffers from overclaimed theory, modest empirical margins, and missing efficiency analysis. It is stronger than papers with fundamental errors (scores 3–4) but not as strong as papers with clear, substantial improvements and tight theory (7+). The best comparison point is IA-GPL, which had similar strengths/weaknesses and received 5–6.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>