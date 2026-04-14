## Summary
EdgePrompt proposes graph prompt tuning from an edge-level perspective, in contrast to all prior work that operates on node features or hidden representations. The method learns shared (EdgePrompt) or attention-weighted per-edge (EdgePrompt+) prompt vectors injected during message passing, motivated by the observation that node-level prompts propagate uniformly to all neighbors while edge-level prompts enable neighbor-specific adaptation. Theoretical justification is provided via a CSBM-based separability analysis (Theorem 1) and a universality existence result (Theorem 2), with experiments spanning 10 datasets, 4 pre-training strategies, and 6 baselines for node and graph classification.

---

## Strengths

- **The "uniform message passing" diagnosis is the paper's most concrete intellectual contribution.** The observation that node-level prompts are indiscriminately broadcast to all neighbors through the GCN aggregation operator — and that edge-level prompts directly address this by conditioning prompt content on the specific sender-receiver pair — is crisply articulated in Section 4.3 and directly motivates the design. This framing cleanly distinguishes EdgePrompt from all prior graph prompt methods.

- **Empirical coverage is genuinely broad and results are consistently strong.** 5 node-classification + 5 graph-classification datasets, 4 distinct pre-training strategies (2 contrastive, 2 generative), and 6 baselines produce 40 evaluation settings. EdgePrompt+ achieves first or second place in the vast majority of these settings, with particularly striking gains under EP-GPPT pre-training (e.g., Cora node classification: 56.41±3.62 vs. best baseline 41.28±6.92; Table 2) — a margin far exceeding noise.

- **The anchor-prompt design in EdgePrompt+ solves a real practical challenge.** Learning |E| independent prompt vectors is infeasible under few-shot supervision because most edges receive no gradient signal. The weighted-average-over-anchor-prompts construction (Eq. 4–6), where scores are computed from node-pair representations, is a principled and parameter-efficient resolution of this constraint.

- **Code is publicly available** and implementation details (optimizer, learning rate, epoch count, anchor prompt defaults) are clearly stated, supporting reproducibility.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing parameter budget analysis — this is the most pressing concern.** EdgePrompt+ introduces W^(l) ∈ ℝ^{2D_{l-1}×M_l} plus M_l anchor prompt vectors per GNN layer, substantially more than GPF (M_l vectors per layer) or GPF-plus (one weight matrix applied to node features). No parameter counts are reported, and no equal-budget comparison is performed. Prompt tuning is explicitly motivated by *parameter efficiency*; without this accounting, it is impossible to attribute EdgePrompt+'s gains to the edge-level inductive bias rather than to having more tunable capacity. This is the central missing control in the paper.

- **Theorem 1 is existential, not constructive, severing the link between theory and practice.** The theorem guarantees "there always exist" anchor prompts achieving a separation improvement factor T ∈ (1, 1 + p/|p−q|], but provides no guarantee that the optimization in Eq. (7) with the specific parameterization of Eqs. (4)–(6) can find such prompts. The theorem does not explain *when* improvement is expected, only that an improvement *can* exist. As a result, the theory provides useful intuition but does not constitute a mechanistic explanation of the empirical gains.

- **Theorem 2's universality claim is potentially implausible as stated.** The theorem asserts that for *any* pre-trained GNN f and *any* graph transformation T, the shared EdgePrompt (only L learnable vectors total) can match f(X', A'). Matching arbitrary transformations — including arbitrary feature and topology changes — with only a shared vector per layer would require very restrictive assumptions on f or the prompt's integration mechanism. The proof is in the appendix (unavailable here), but the main text states no assumptions, making the claim difficult to assess or believe. At minimum, the main text must state the key conditions under which this result holds.

### Minor

- **Method implementation underspecified for concrete backbones.** Eq. (2) gives the abstract extension of message passing, but neither the main text nor the setup section concretely states how AGG combines node messages and edge prompts for GCN (additive before normalization? additive after? concatenation + projection?) and for GIN (before or after the MLP?). Given that both backbones nominally do not support edge attributes, these details are non-trivial and are essential for reproduction without code.

- **Compatibility claim is weaker than advertised.** The paper advertises compatibility with GNNs that "cannot accommodate edge attributes," but achieves this by modifying the message-passing aggregation operator's forward computation — an architectural wrapper, not purely input-space prompting. Section 4.2 acknowledges the challenge but does not disclose that this changes the backbone's computation graph. The binary ✓ in Table 1 for "PT Compatibility" is therefore somewhat misleading.

- **Backbone diversity insufficient relative to compatibility claims.** All experiments use only 2-layer GCN (node) and 5-layer GIN (graph). The claim of compatibility with "prevalent GNN architectures" is not validated for attention-based models (GAT), heterogeneous aggregators (GraphSAGE), or edge-aware architectures. It is unknown whether the aggregation modification yields consistent behavior across backbone families.

- **Homophily assumption embedded in Theorem 1 is unacknowledged.** The benefit of edge prompts in Theorem 1 scales as p/|p−q|, which vanishes as p → q (weakly structured or heterophilic graphs). All experimental datasets are broadly homophilic (citation networks, TU biochemical graphs). The paper makes no mention of this limitation, and heterophilic settings — where the "uniform message passing" problem is arguably *most* damaging — are entirely untested.

- **No ablation isolating the attention mechanism's contribution.** EdgePrompt+ gains over EdgePrompt could stem from (a) the edge-level placement of prompts or (b) the learned per-edge scoring function. Without a baseline using random or fixed (non-learned) edge weights, the necessity of the attention component is unverified. This is a low-cost but important experiment.

- **No explicit limitations section.** ICLR submissions are expected to discuss failure modes. Minimum required: scalability on dense graphs (O(|E|) per-edge scoring vs. O(|V|) for node prompts), dependence on access to message-passing internals, potential for attention overfitting in few-shot settings, and restricted theoretical guarantees under heterophily.

### Tiny

- **Convergence analysis (Figure 2) is epoch-based only.** EdgePrompt+ adds per-edge scoring at each layer (O(|E|) overhead), so faster convergence in epochs could be offset by longer per-epoch wall-clock time. A brief runtime comparison (e.g., seconds per epoch or total training time) is needed to make the convergence claim meaningful.

- **Some improvements are within one standard deviation.** For example, under GraphCL/Flickr, GraphPrompt (26.08±3.44) outperforms EdgePrompt+ (25.57±3.04); under EP-GraphPrompt/ogbn-arxiv, EdgePrompt (32.67±1.83) outperforms EdgePrompt+ (31.41±1.88). The framing of "consistent superiority" slightly overstates what the tables show.

- **The "GPF-plus as special case of EdgePrompt+" claim (Section 4.2) is asserted without derivation.** GPF-plus adds prompts to node features, not edges; the structural equivalence claimed requires a non-trivial argument that is not provided even as a proof sketch.

---

## Nice-to-Haves

- **Node + edge combined baseline.** Testing GPF-plus + EdgePrompt jointly would reveal whether node and edge prompts are complementary or substitutable, which is the most natural follow-up experiment given the paper's framing.
- **Varying few-shot sizes** (1, 3, 10, full-shot) would characterize how the method's advantage changes with data availability; Theorem 1's T factor may interact with label density in ways worth understanding.
- **Visualization of learned edge prompt scores** (e.g., comparing intra-class vs. inter-class edge scores) would provide direct interpretable evidence for the core uniform-message-passing intuition.
- **Experiments on at least one heterophilic graph** (e.g., Actor, Chameleon) to empirically characterize where the method succeeds or fails when the homophily assumption underlying Theorem 1 does not hold.
- **Full fine-tuning comparison** (even just last-layer tuning) to situate prompt tuning in the broader adaptation landscape.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: Related work must discuss edge-conditioned convolution, relation-aware GNNs, graph rewiring.** The paper's contribution is within the graph prompt tuning paradigm. Demanding engagement with the full GNN adaptation literature is scope creep; the positioning relative to prompt tuning methods (Table 1) is appropriate.
- **Harsh critic: Table 1's binary checkmarks are too favorable.** This is a standard positioning device in this literature and is not a scientific weakness.
- **Harsh critic: No fine-tuning baseline.** Prompt tuning papers in this subfield standardly compare within the prompt tuning paradigm. Fine-tuning represents a different adaptation family with a different parameter budget; its exclusion is consistent with field norms.
- **Spark/Harsh: No experiments on graphs with existing edge features (e.g., molecular bond attributes).** The paper explicitly scopes itself to GNNs that *do not* accommodate edge attributes. Demanding experiments on edge-attributed settings is outside the stated scope.
- **Harsh critic: "NCII" and "NCII09" are not real dataset names.** These are PDF parsing artifacts for NCI1 and NCI109 from TUDataset. Not a real scientific issue.
- **Harsh critic: The claim that prior methods ignore edges is "too narrow to justify significance."** The claim is accurate within the prompt-tuning literature, which is the paper's stated contribution domain. The shift to asking "does it advance understanding beyond all GNN adaptation mechanisms?" is a different and unfairly elevated standard.
- **Strength removed: "The paper is well-written."** Generic, applies to any readable paper.
- **Strength removed: "The topic is important."** Generic.

---

## Novel Insights

The most underexplored connection across all three reviews is the structural alignment between EdgePrompt+'s mechanism and class-conditional edge gating. Node-level prompts, when added to node features, act as a uniform additive bias that gets aggregated across *all* neighbors regardless of class — functionally equivalent to injecting class-blind noise into cross-class edges. EdgePrompt+ implicitly implements a learned filter over the edge set in the prompt space, which is structurally analogous to what attention-based GNNs do to suppress noisy edges. This reframing explains why the gains are especially pronounced under EP-GPPT (an *edge prediction* pre-training strategy, Table 2, e.g., Cora +15 points): the pre-training task has already encoded structural patterns in the representation that edge-level prompts can selectively modulate, whereas node-level prompts cannot exploit this alignment. This hypothesis — that the benefit of edge prompts scales with the structural alignment of the pre-training objective — is not analyzed in the paper and represents a productive theoretical direction.

---

## Suggestions

1. **Report parameter counts for all methods in a table** (number of trainable prompt parameters for each method at M_l = 10), and include at least one equal-budget comparison where GPF-plus is given matching parameters.
2. **Add a concrete implementation box** for Eq. (2) applied to GCN (e.g., h_i^(l) = σ(Â · (H^(l-1) + E^(l)) · Θ^(l))) and GIN, so the method is reproducible without inspecting code.
3. **State Theorem 2's key assumptions in the main text** — at minimum, what class of transformations T is covered, and whether f must be linear or satisfy any architectural constraint.
4. **Add an ablation with fixed/random edge weights** (no learned scoring) alongside EdgePrompt and EdgePrompt+ to isolate how much of the gain comes from edge-level placement vs. the attention module.
5. **Add an explicit limitations section** covering scalability on dense graphs, modification of backbone internals, few-shot overfitting risk, and heterophilic settings.
6. **Test on at least one heterophilic benchmark** to characterize where the theoretical benefit disappears (p ≈ q regime).
7. **Fix the convergence analysis** to report wall-clock time or per-epoch runtime alongside accuracy curves.

---

**Evaluation summary:** The paper makes a clear, well-motivated, and genuinely novel contribution to the graph prompt tuning literature. The empirical results are convincingly broad, and the anchor-prompt construction is a practically useful design. However, the paper currently cannot definitively attribute its gains to the edge-level mechanism because parameter budgets are unreported. The theoretical results, while providing useful intuition, are both existential and in Theorem 2 potentially too broad to be credible without visible assumptions. These are correctable issues, and the core contribution is solid, but the paper in its current form falls short of the evidence standard for a definitive ICLR acceptance.