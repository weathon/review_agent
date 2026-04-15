Now let me read some human review files for calibration.Now I have enough to produce the final consolidated review. Let me synthesize everything.

---

## Summary
EdgePrompt introduces edge-level graph prompt tuning: instead of adding learnable prompt vectors to node features (as in GPF, All-in-one), learnable vectors are injected per-edge into the message-passing aggregation of a frozen pre-trained GNN. A simpler global variant (EdgePrompt) and an attention-based per-edge variant (EdgePrompt+) are proposed and evaluated on 10 datasets across 4 pre-training strategies. Theoretical support includes Theorem 1 (showing existence of anchor prompts that improve class-centroid separation in a 2-class CSBM under GCN) and Theorem 2 (showing EdgePrompt can replicate any node-feature transformation for graph classification). Results show EdgePrompt+ is frequently best or competitive among six baselines.

---

## Strengths

- **Novel and well-motivated perspective.** The paper identifies a concrete and previously unexploited design space: node-level prompts cause *uniform* message-passing of the prompt to all neighbors, whereas edge-level prompts allow differentiated aggregation. This insight is crisp, illustrated clearly in Figure 1, and the design flows naturally from it.

- **Comprehensive empirical evaluation.** Ten datasets, four pre-training strategies, both node and graph classification — this is a genuinely broad experimental sweep for graph prompt tuning. The EP-GPPT results in particular are large and consistent (e.g., 56.41 vs 41.28 on Cora, 43.49 vs 35.32 on CiteSeer), lending confidence that EdgePrompt+ delivers real gains in at least some regimes.

- **Practical architecture compatibility via re-aggregation.** The clever solution to apply edge prompts via the AGG function (Eq. 2) allows the method to work on GNNs that don't natively support edge features (e.g., GCN), which is a concrete engineering contribution.

- **Theoretical grounding comparable to prior work.** Theorem 2 is an existence proof of representational universality analogous to Theorem 1 in GPF (Fang et al., 2023), placing EdgePrompt on equal theoretical footing with the strongest prior node-prompt method.

---

## Weaknesses

### Fatal
*(None identified — the core contribution and supporting evidence are not invalidated by any of the issues below.)*

### Major

- **Architecture compatibility is overstated relative to evaluation.** The Abstract and Introduction claim EdgePrompt is "compatible with prevalent GNN architectures pre-trained under various pre-training strategies," but experiments evaluate only a 2-layer GCN (node classification) and a 5-layer GIN (graph classification). Four pre-training strategies are tested, but only two backbone architectures. Until compatibility is demonstrated on at least GAT or GraphSAGE — which have different aggregation structures — the "prevalent architectures" claim is unsupported by evidence.

- **Aggregation mechanism underspecified.** Eq. (2) abstractly says the GNN aggregates both $h_j^{(l-1)}$ and $e_{ij}^{(l)}$, but does not specify whether this is additive ($h_j + e_{ij}$), concatenation-then-project, or another operator. For GCN (which uses normalized-sum aggregation) and GIN (which uses summation followed by MLP), the concrete implementation of Eq. (2) likely differs, yet no architecture-specific implementation details are given. This significantly impacts reproducibility.

- **Theorem 1 oversold in the narrative.** Theorem 1 proves existence of anchor prompts that increase class-centroid separation in a 2-class CSBM under a 1-layer GCN. The paper's conclusion (Sec. 4.3) — "we can conclude that our proposed EdgePrompt+ benefits pre-trained GNN models for node classification" — is a much stronger general claim. The theorem establishes neither that the gradient-based optimization in Eq. (7) will find such prompts, nor that the result extends to multi-class tasks, heterogeneous datasets, or other backbone architectures. The theory provides supporting intuition, not a guarantee.

- **No parameter efficiency comparison.** EdgePrompt+ introduces $M_l$ anchor prompts plus a $2D_{l-1} \times M_l$ weight matrix per layer. With default $M=10$ and $D=128$, this adds $\sim 327$K trainable parameters on a 2-layer GCN — potentially comparable to or exceeding GPF-plus. Without reporting parameter counts per method, it is impossible to determine whether gains come from edge-specific design or simply from additional capacity.

### Minor

- **Missing MultiGPrompt in experiments.** Table 1 explicitly lists MultiGPrompt as a relevant baseline, yet it is excluded from Tables 2 and 3. The paper marks MultiGPrompt with ✗ in PT Compatibility, which offers a partial implicit justification, but no explicit explanation is given. Since some experiments use EP-GPPT and EP-GraphPrompt which are compatible strategies, clarification is needed.

- **Convergence analysis is weak.** Fig. 2 shows accuracy-vs-epoch plots and asserts faster convergence for EdgePrompt+, but per-epoch wall-clock times are not reported. If EdgePrompt+ computes per-edge attention over all edges per layer, it may be significantly slower per epoch than node-prompt methods, making raw epoch counts misleading.

- **Some margins are within variance of baselines.** In Table 2, e.g., Pubmed/GraphCL: EdgePrompt+ (67.41±5.25) vs GPF (67.67±3.14); Flickr/GraphCL: EdgePrompt+ (25.57±3.04) vs GraphPrompt (26.08±3.44). The paper's "superiority" framing is too strong; "frequently competitive and often best" is more accurate.

- **Score function ablation is absent.** The paper notes that "in-depth investigations into different variants of the score function φ will be reserved for future work," but the attention-based φ in Eq. (6) is a non-trivial design choice. A basic ablation against a simpler score function (e.g., dot product or constant) would validate whether the attention mechanism matters or whether simple anchor averaging suffices.

### Trivial

- **EdgePrompt+ is the headline method but Table 1 only lists "EdgePrompt+ (Ours)"** without positioning both EdgePrompt and EdgePrompt+ clearly as co-contributions. Minor positioning ambiguity.

---

## Nice-to-Haves

- **Comparison with fine-tuning methods.** Including at least a linear probe and a full fine-tune baseline would situate prompt tuning's practical utility; without this, it is hard to assess whether any prompt method is competitive with direct adaptation.
- **Testing on at least one heterophilic graph** (e.g., Chameleon or Actor) would directly test the paper's core structural claim: that edge prompts matter more when edges carry intra/inter-class distinction signals.
- **Visualization of learned edge prompts.** A heatmap or analysis showing whether $e_{ij}$ differs systematically between intra-class and inter-class edges would validate the mechanism qualitatively.
- **Multiple few-shot sizes.** Reporting results at 1-shot, 5-shot, and 10-shot would show whether EdgePrompt+'s advantage is robust to label quantity.

---

## Removed Points
> *These points are flagged to be removed; treat them with caution.*

**Harsh Critic — Point 1 (backbone not truly frozen):** The critic argues that Eq. (2) "alters the forward computation of the frozen backbone." However, in standard graph prompt tuning, "frozen" means the backbone's learned weights are not updated, not that the input to every module is identical. Adding edge prompts to the aggregation is the intended contribution — it is analogous to how node prompt methods add vectors to node features without being accused of "modifying the backbone." This criticism is a mischaracterization of the design philosophy and is removed.

**Harsh Critic — Point 2 (Theorem 2 "extraordinarily broad"):** The critic calls Theorem 2 implausible. However, Theorem 2 is a representational existence result directly analogous to Theorem 1 in GPF (Fang et al., 2023). The paper correctly acknowledges it as an existence claim and uses it to argue comparable universality to GPF. The theorem is appropriate in scope and context; the criticism was overstated.

**Harsh Critic — GPF-plus as special case claim:** The paper states GPF-plus is a special case with "the score function as a linear mapping of x_i." This is an informal connection rather than a proof, but it is a plausible structural observation and not clearly wrong. Characterizing this as a "strong equivalence claim" that undermines the paper is excessive.

**Spark — No fine-tuning baselines:** The paper explicitly studies graph prompt tuning and scopes its comparison to prompt tuning methods. Demanding fine-tuning baselines is scope creep.

**Spark — No heterophilic benchmarks:** The paper's stated scope is graph prompt tuning broadly; testing on heterophilic graphs is a good suggestion (moved to Nice-to-Haves) but not a validity-threatening omission.

**Spark — No edge-aware GNN (GINE) comparison:** Comparing to a GNN that natively supports edge features is outside the stated scope of prompt tuning methods. EdgePrompt is specifically designed for models that do *not* support edge features. Moved to Nice-to-Have.

---

## Novel Insights

The core novel insight — that node-level prompts suffer from **uniform message passing** (every neighbor of a prompted node receives the same prompt vector), and that edge-level prompts precisely fix this by making each directed message carry its own learnable perturbation — is a genuinely clean and under-exploited observation. The anchor-prompt parameterization, which allows per-edge customization even under few-shot label sparsity (since most edges won't be involved in the direct computation of labeled nodes), is an elegant solution to an otherwise thorny engineering problem. Together, these observations meaningfully advance the vocabulary of graph prompt design beyond feature-space manipulations.

---

## Suggestions

1. **Specify exactly how $e_{ij}$ is merged in Eq. (2) for each tested backbone.** For GCN: does the message for edge $(j \rightarrow i)$ become $\hat{A}_{ij}(h_j^{(l-1)} + e_{ij}^{(l)})W$? For GIN: $(h_j^{(l-1)} + e_{ij}^{(l)})$? State these explicitly in Section 4.2 or Appendix.

2. **Report trainable parameter counts** for EdgePrompt, EdgePrompt+, GPF, GPF-plus, and All-in-one in a supplementary table to address fairness concerns.

3. **Add at least one additional backbone** (e.g., GAT or GraphSAGE) to validate the "compatible with prevalent GNN architectures" claim.

4. **Tone down the universality/compatibility framing** in the Abstract and Introduction to "compatible with standard message-passing GNN architectures" and "demonstrated on GCN and GIN across four pre-training strategies."

5. **Include a brief ablation on the score function** (e.g., single-node vs. pairwise vs. constant score) to show that the dyadic attention in Eq. (6) is responsible for gains, not just the anchor-prompt capacity.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores | Notes |
|---|---|---|---|
| GPromptShield (yCN4yI6zhH) | Accept (Poster) | 6,6,6 | Graph prompt + theory, novel focus (robustness), similar depth |
| IA-GPL (VBeLiRkZMP) | Withdrawn | 6,5,5,5 | Instance-aware graph prompts, similar novelty and empirical scope |
| Does Graph Prompt Work? (C1wSR50nYf) | Withdrawn | 3,5,6 | More theoretical, weaker execution |
| Multi-modal Graph Prompt (ax4ZOytBV2) | Withdrawn | 5,3,5,5 | Similar scope but more ambitious with execution gaps |
| Edge embeddings in GNNs (XrtFVM1f6w) | Reject | 5,5,5,6 | Edge-based GNN theoretical analysis, comparable depth |

EdgePrompt is most comparable to GPromptShield (accepted, 6s) and IA-GPL (withdrawn, 6/5/5/5). EdgePrompt has:
- A cleaner, more principled novel contribution than GPromptShield (which is more engineering-oriented)
- Broader empirical evaluation than most calibration papers
- Comparable theoretical depth

Against it:
- Architecture compatibility overclaimed (only 2 backbones vs. "prevalent architectures")
- Aggregation mechanism underspecified
- No parameter efficiency comparison
- Some theoretical overclaiming in the narrative

These are **major but addressable weaknesses** — they don't invalidate the contribution but leave the paper short of a clean acceptance. Positioning slightly below GPromptShield (accepted 6s) and near IA-GPL (borderline), accounting for EdgePrompt's broader evaluation but weaker architectural generalization support, I arrive at **5.5**.

**Evaluation axes:**
- *Originality*: Good — edge-level prompting is a natural but unexploited idea, executed cleanly.
- *Importance of research question*: Moderate-to-good — graph prompt tuning is active and this fills a real gap.
- *Claims well-supported*: Partial — empirical breadth is good, but compatibility/universality claims exceed what the experiments can support.
- *Soundness of experiments*: Adequate but incomplete — broad coverage, but missing parameter counts and limited to 2 backbones.
- *Clarity of writing*: Good overall, with one significant gap in specifying the aggregation mechanism.
- *Value to the research community*: Meaningful — the edge-prompting idea is transferable and the implementation insight (anchor prompts to handle label sparsity) is practically useful.

**Decision: Borderline Reject.** The paper makes a genuine and well-motivated contribution, but needs to substantiate its compatibility/universality claims across more architectures, specify its aggregation mechanism, and report parameter counts before making strong comparative claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>