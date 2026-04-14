## Summary
EdgePrompt and EdgePrompt+ propose to adapt frozen pre-trained GNNs for downstream tasks by placing learnable prompt vectors on edges rather than node features. The core motivation is that node-level prompts propagate uniformly through message passing, whereas edge-level prompts allow differentiated information flow per edge. EdgePrompt learns a single global prompt vector per layer; EdgePrompt+ learns per-edge customized prompts as a weighted mixture of anchor vectors. The paper provides theoretical analyses via CSBM (Theorem 1) and a universality claim (Theorem 2), and evaluates on ten datasets under four pre-training strategies against six baselines.

---

## Strengths

- **Identifying the uniform message-passing problem as a limitation of node prompts.** Section 4.3 and Figure 1 clearly articulate that node-level prompts are passed identically to all neighbors of a node, which can mix cross-class information. This is a concrete and non-trivial structural insight that goes beyond prior node-prompt framing, and the subsequent CSBM analysis formalizes it.

- **Anchor prompt parameterization for few-shot settings.** The design of EdgePrompt+ as a weighted mixture over a small set of shared anchor prompts is a practically motivated solution to the key challenge that most edges in few-shot settings receive no direct supervision. The parameter count scales with the number of anchors rather than graph size, which is a principled design choice.

- **Consistent and broad empirical validation.** Results across 10 datasets, 4 pre-training strategies (contrastive and generative), and 6 baselines are considerably broader than most competing prompt-tuning papers. EdgePrompt+ achieves best or runner-up in the overwhelming majority of conditions, particularly on node classification. The gains under the weaker EP-GPPT pre-training strategy (e.g., Cora: 56.41 vs. the next best 41.28 for node classification) suggest the method adds the most value when the backbone is weakly adapted.

---

## Weaknesses

### Fatal
None identified.

### Major

- **"Compatibility" claim is overstated relative to what the method actually requires.** The paper claims EdgePrompt is "compatible with prevalent GNN architectures pre-trained under various pre-training strategies, especially with those that cannot accommodate edge attributes." However, Equation (2) explicitly reformulates the forward pass by injecting edge prompts into the AGG operator at every layer. This requires modifying internal message-passing code — it is not a pure input-space transformation. For off-the-shelf pre-trained GNN checkpoints, one must re-implement the layer forward function to accept and aggregate edge vectors. This is a meaningfully different operation from methods such as GPF/GPF-plus that only alter input features before passing them through an unmodified model. The paper should clearly state that EdgePrompt requires access to — and modification of — the GNN's layer-by-layer aggregation, and reframe the "compatibility" claim accordingly.

- **The concrete aggregation rule for GCN and GIN is never given.** Equation (2) shows AGG takes edge prompts as an additional input but leaves the specific form abstract. For GCN, the normalized neighborhood sum has a specific normalization degree matrix; where exactly is $e_{ij}^{(l)}$ inserted? Added to $h_j^{(l-1)}$ before normalization? After? Scaled differently? For GIN, does the prompt enter before or after the MLP? These choices materially affect the method's behavior and whether any pre-trained normalization statistics remain consistent. Code is available, but a methods paper at ICLR should provide these formulas in the main text for reproducibility and transparent evaluation.

- **Theorem 2 appears too strong for its parameterization and lacks accessible justification.** The theorem states that a set of globally shared per-layer prompt vectors in EdgePrompt can satisfy $f(\mathbf{X}, \mathbf{A}, \{\mathbf{p}^{(1)},\ldots,\mathbf{p}^{(L)}\}) = f(\mathbf{X}', \mathbf{A}')$ for ANY graph transformation $\mathcal{T}$ and ANY pre-trained GNN $f$. Yet EdgePrompt uses one shared vector per layer — it does not modify topology, cannot add or remove edges, and applies the same perturbation uniformly across all edges. Claiming that this restricted parameterization can replicate arbitrary topology changes for any $f$ requires a very careful set of assumptions about $f$. With the appendix unavailable for review, the main text provides no sketch, no assumptions, and no intuition for why this is not vacuous or degenerate. Given the strength of the claim, the paper must include at minimum the key assumptions and proof sketch in the main text.

- **No computational efficiency analysis.** EdgePrompt+ computes an attention-based score vector (Equations 5–6) for every edge at every layer, adding $O(|\mathcal{E}| \cdot L)$ parameter-dependent operations. For ogbn-arxiv (~1.17M edges) this is a non-trivial cost. The paper makes no report of per-run training time, GPU memory usage, or parameter counts relative to baselines. For a practical method competing on parameter efficiency grounds, this is a significant omission that leaves the claimed utility of the approach incompletely characterized.

### Minor

- **Backbone diversity is limited.** Only a 2-layer GCN (node classification) and a 5-layer GIN (graph classification) are used, despite claiming compatibility across "prevalent GNN architectures." Adding at least one attention-based or edge-aware architecture (e.g., GAT or GraphSAGE) would substantiate the compatibility claim empirically and test whether edge prompts add value beyond architectures that natively handle edges.

- **No fine-tuning baseline.** The introduction frames prompt tuning as an alternative to fine-tuning. Without a fine-tuning reference point, the practical standing of EdgePrompt+ — specifically how much it closes the gap relative to full adaptation — remains uncharacterized. This is important for understanding whether prompt tuning is an effective substitute in this setting.

- **Theorem 1 is an existence result only.** The theorem guarantees there always *exist* anchor prompts and score vectors achieving an improvement factor $T$, but provides no bound on whether gradient-based optimization will find them under limited supervision. This gap between theoretical representability and practical optimization should be acknowledged in the discussion rather than implied as empirical justification.

- **Theorem 2 does not theoretically differentiate EdgePrompt from GPF.** The paper invokes Fang et al.'s (2023) Theorem 1 to conclude "comparable universal capability with GPF." This undercuts the theoretical argument for edges over nodes: both methods are shown to have equivalent universality, so theory does not favor the new method.

- **Ablation study is limited to anchor count.** Section 5.4 studies only the number of anchor prompts. Critical design questions go unstudied: prompting at one layer vs. all layers, the global EdgePrompt vs. the adaptive EdgePrompt+, and the contribution of the attention-based score function vs. a simpler edge gate. Without these ablations, it is unclear how much of the gain comes from edge prompting specifically versus the added trainable per-edge scoring mechanism.

- **Some performance margins are within noise.** Under EP-GraphPrompt, several EdgePrompt+ results (e.g., Pubmed: 73.72±5.10 vs. GPF 73.62±6.42; ogbn-arxiv: 31.41±1.88 vs. EdgePrompt 32.67±1.83) are within or near one standard deviation. The paper's language of "consistent superiority" should be qualified for such cases.

### Tiny

- **Equation (7) is typographically malformed.** The optimization variables and the minimization are not properly structured in the displayed formula; the subscript/superscript rendering conflates the objective symbol and argument.

- **Convergence analysis covers only two of four pre-training strategies.** Figure 2 shows SimGRACE and EP-GPPT; GraphCL and EP-GraphPrompt are omitted without explanation.

- **No limitations or broader impact discussion** is provided in the main text. Even a brief paragraph on scalability to dense graphs and the requirement for layer-level code access would improve transparency.

---

## Nice-to-Haves

- **Visualization of learned edge prompt weights.** A heatmap of $\mathbf{b}_{ij}^{(l)}$ for a small graph, colored by intra- vs. inter-class edges, would directly verify whether EdgePrompt+ learns to differentiate edge types as claimed, or simply acts as a global bias.

- **t-SNE/UMAP of representations with vs. without edge prompts** would empirically validate the linear separability improvement implied by Theorem 1.

- **A combined node + edge prompt experiment.** Testing whether GPF-plus and EdgePrompt+ together yield further gains would clarify whether edge and node prompts capture orthogonal information.

- **Expanded shot settings** (1-shot, 3-shot, 10-shot, full-label) beyond 5-shot and 50-shot to characterize how the few-shot advantage evolves.

- **Discussion of the "Classifier Only" baseline when it is competitive.** Under EP-GraphPrompt (e.g., Pubmed: 72.09 classifier only vs. 73.72 EdgePrompt+), the pre-trained backbone with a linear probe is already near-optimal. Understanding when prompting matters vs. when the backbone suffices is a useful practical insight.

---

## Removed Points
*These points were flagged for removal; treat with caution.*

- **"Uniform message passing of node prompts cannot capture structure" is categorically false (Harsh, §1).** The paper explicitly acknowledges a specific failure mode (uniform propagation to all neighbors from one source node), not a blanket impossibility. The criticism misreads the paper's narrower argument. REMOVED.

- **Table 1 taxonomy is simplistic (Harsh, §2).** This is a formatting/style criticism with no bearing on the method's validity. REMOVED.

- **Notational inconsistency between bold and non-bold (Harsh, §3).** Pure typographic nitpick. REMOVED.

- **Potential unfair comparison due to "backbone adaptation through prompts" (Harsh, §4.2).** The backbone is frozen throughout; EdgePrompt+ introduces trainable parameters only for the score function and anchor prompts, not for backbone weights. The concern that this constitutes "leakage" is a misreading. REMOVED.

- **Theorem 2 lacks justification because it might rely on degenerate assumptions for GCN/GIN (Harsh, §4.4).** The actual concern — that the theorem seems too strong — is retained as a major weakness above, but the specific accusation of vacuity for GCN/GIN without seeing the proof is speculative. The concern is preserved at the level of "the claim requires accessible proof sketch in the main text." REMOVED as stated; subsumed into Major weakness above.

- **EdgePrompt+ resembles a GAT-like attention (Harsh, §4.2).** The paper does use attention-style scores (citing Veličković et al., 2018 explicitly in Equation 6). This design choice is transparent and appropriate; using a standard mechanism as the score function is not a weakness. REMOVED.

- **Demanding proofs of scalability, heterogeneous graphs, or dynamic graphs (Harsh, Conclusion).** These are outside the paper's stated scope. REMOVED as scope creep; scalability concern retained only for the efficiency analysis gap.

- **Requesting multiple-run significance tests beyond the standard 5-run protocol (Harsh, §5.2).** Single-run-per-seed with 5 seeds and ± std is standard in graph learning benchmarking. REMOVED.

- **Requesting broader architectural validation (GAT, GraphSAGE, MPNN) as a major weakness (Harsh, §5.1).** Retained only as a minor weakness; demanding extensive architecture coverage for a methods paper is partially scope creep. WEAKENED.

---

## Novel Insights

The most genuinely novel framing this paper contributes — beyond the method itself — is making explicit the *uniform message-passing problem* for node-level prompts: a node's prompt vector propagates identically to all its neighbors, which is structurally inappropriate when those neighbors belong to different classes. This observation is not just a motivation device; it identifies a concrete failure mode of the dominant paradigm in graph prompt tuning that prior work has not named. The anchor-based parameterization for EdgePrompt+ further introduces a principled solution to the supervision sparsity problem in few-shot prompt learning on graphs. The empirical evidence that EdgePrompt+ gains are largest under weak pre-training (EP-GPPT) — where backbone representations are least discriminative — offers a practical insight: structural edge-level adaptation matters most when the backbone cannot resolve class boundaries on its own.

---

## Suggestions

1. **Provide explicit GCN/GIN aggregation formulas** for how $e_{ij}^{(l)}$ is incorporated — at minimum in a footnote or appendix subsection with GCN: $\hat{h}_j^{(l-1)} = h_j^{(l-1)} + e_{ij}^{(l)}$ before normalized summation (or whatever the actual implementation is). This is the single most impactful clarity fix.

2. **Reframe "compatibility."** Clearly state in Section 4.2 that EdgePrompt requires modifying the GNN's layer-level forward pass (rather than only the input) and explain that "compatibility" means the method applies to pre-trained models regardless of their original pre-training strategy, not that it is a black-box wrapper.

3. **Add a proof sketch for Theorem 2 in the main text**, including the key assumptions about $f$ and a one-paragraph intuition for why a global shared per-layer prompt can replicate arbitrary graph transformations. If the result holds only under specific conditions on $f$, state them explicitly.

4. **Include at least one efficiency table**: number of additional trainable parameters and training time per epoch for EdgePrompt, EdgePrompt+, GPF, and GPF-plus on ogbn-arxiv.

5. **Add ablation experiments**: at minimum, (a) single-layer vs. all-layer edge prompts and (b) attention-based score function vs. a simpler edge gate or learnable scalar, to isolate what drives EdgePrompt+'s gains.

6. **Add one additional GNN backbone** (e.g., GAT) to substantiate the "prevalent architectures" claim empirically, even on a subset of datasets.