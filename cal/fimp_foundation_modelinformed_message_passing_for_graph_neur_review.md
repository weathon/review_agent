=== CALIBRATION EXAMPLE 32 ===

# Final Consolidated Review
Now I have a thorough understanding of the paper. Let me synthesize the three reviews.

---

## Summary
FIMP (Foundation-Informed Message Passing) repurposes pretrained non-textual foundation models (ViT, scGPT, BrainLM) for graph-based tasks by adapting their self-attention layers into cross-node attention message creators: each node is represented as a sequence of feature tokens, and messages from source node $j$ to destination node $i$ are generated via cross-attention where Q comes from $i$ and K/V from $j$. The pretrained Q/K/V weight matrices can be transferred directly from foundation model checkpoints, enabling zero-shot or fine-tuned deployment. FIMP is evaluated across three distinct real-world domains—spatial transcriptomics, street-view image classification, and fMRI reconstruction—demonstrating consistent improvements over GNN baselines and a notable zero-shot capability.

---

## Strengths

- **Mechanistically distinct cross-node token-level attention.** Prior attention-based GNNs (GAT, Graph Transformer, GPS) operate at the node-embedding level, computing scalar attention coefficients between single embedding vectors. FIMP instead computes cross-attention between full token sequences of neighboring nodes, which is genuinely different and allows the pretrained feature-interaction patterns of foundation models to be directly transferred. The paper articulates this distinction clearly in Sections 2.2 and 3.2, and the ablation in Table 5 (GPS + ViT embeddings = 50.0% vs. FIMP-ViT = 63.2%) provides direct empirical evidence that the FIMP message-passing mechanism adds value beyond simply upgrading node features.

- **Strong and consistent gains in image classification.** FIMP-ViT achieves 63.2% accuracy on Mapillary vs. the strongest GNN baseline GPS at 27.4% (with pixel inputs). Even controlling for richer inputs, GPS + ViT embeddings reaches only 50.0%, confirming the architectural contribution rather than merely better feature engineering. This is a large, reproducible gap across 5 runs with low variance (±0.764).

- **Zero-shot graph embedding that outperforms trained GNNs.** FIMP-ViT achieves 40.6% accuracy on Mapillary in zero-shot mode (no graph-specific training), exceeding the best *fine-tuned* GNN baseline (GPS, 27.4%) and the standalone ViT (34.0%). This is a non-trivial demonstration that pretrained self-attention weights already encode relational structure useful for graph tasks.

- **Spatial transcriptomics: large gains with domain-matched pretraining.** On the human heart dataset, FIMP-scGPT achieves R²=0.8119 versus the best GNN (GIN at R²=0.6625), and versus the standalone scGPT (R²=0.0229)—showing that graph structure is necessary and that FIMP effectively combines both. The negative finding that ViT-based FIMP hurts performance on gene data (R²=0.3506 vs. FIMP-base R²=0.3815) is informative and honestly reported; the authors correctly attribute it to domain misalignment rather than inflating it as a general positive.

- **Multi-domain scope with real-world, large-scale datasets.** The paper evaluates on datasets of up to 750K images, 76K fMRI recordings, and tens of thousands of cells. This is not a toy-benchmark paper; the scale and diversity of the experimental validation are meaningful.

---

## Weaknesses

### Fatal
None. The core claims are broadly supported, though individual result validity has issues noted below.

### Major

- **fMRI Table 4: non-comparable masking strategies undermine the reported ~25% improvement.** GNN baselines are evaluated under three masking strategies (replace noise, fill in mean, linear interpolation), while FIMP uses a fundamentally different strategy ("Tokenization + PE"). The "Tokenization + PE" approach likely treats reconstruction of masked token windows as the objective, which could be an inherently easier or differently structured task. The paper does not justify why GNN baselines cannot use "Tokenization + PE" (or conversely, why FIMP was not evaluated under the same noise/mean/interpolation conditions). The best GNN result (GraphSAGE, fill-in-mean, R²=0.320) is compared to FIMP-base (R²=0.578)—a gap of 0.258—but it is unclear how much of this reflects architectural superiority versus masking-strategy choice. The authors need to either (a) evaluate FIMP under the same masking strategy as GNNs, or (b) evaluate GNNs under the "Tokenization + PE" strategy, to make a clean comparison.

- **FIMP-base underperforms standard GNNs on mouse hippocampus cell type classification (Table 2), with no explanation.** FIMP-base scores 49.04% vs. GraphSAGE 51.81%, GIN 52.71%, GPS 52.89%. This is not a small-margin overlap—FIMP-base is the second-worst method on this task (only GCN and GAT are worse). The paper states "FIMP outperforms baseline models at predicting cell types" (Section 4.3), which is only true when using the foundation model variant (FIMP-scGPT). The abstract's unqualified claim is therefore misleading for this task. More importantly, this failure mode of the base architecture is not analyzed: is it due to the cross-attention inductive bias, harder optimization, the tokenization, or a mismatch between the token-level architecture and this specific task? This gap is a real limitation of the method that the paper does not engage with.

- **Missing ablation: architecture benefit vs. pretraining benefit across all domains.** The paper provides FIMP-base (random init) and FIMP+FM results, but this ablation is not presented systematically across all three domains. For fMRI, there is no FIMP with randomly initialized weights (only FIMP-base and FIMP-BrainLM appear in Table 4, but these differ in both initialization and possibly model depth/capacity). Without a consistent paired comparison, the claimed "Foundation-Informed" benefit cannot be cleanly isolated from architectural effects.

### Minor

- **Zero-shot protocol is ambiguous.** Section 4.3 says zero-shot FIMP-ViT performs embedding "without any graph-specific training," but it is not specified which parameters (if any) are updated. If the Q/K/V projection weights are frozen (fully from the ViT checkpoint) and no parameters are updated at all, the zero-shot claim is clean. If any components—readout heads, tokenization projections—are updated, this should be stated explicitly. This matters because the zero-shot result is presented as a key contribution.

- **Tokenization details should be elevated.** The tokenization function $\tau$ is central to the "Foundation-Informed" design—it is what aligns GNN inputs with FM inputs and reduces distribution shift. Yet the domain-specific instantiations (for scRNAseq and fMRI) are entirely deferred to Appendix B. A paper centered on tokenization alignment should describe the gene embedding table and time-window-based fMRI tokenization in the main text, as they directly bear on why domain-matched pretraining helps.

- **FIMP-base substantially outperforms GNNs on image classification (38.6% vs. best GNN 27.4%) — this deserves explanation.** The authors attribute this to "improved gene tokenization" in the context of spatial transcriptomics, but for images the gain from FIMP-base over GPS is large (11+ points) using only randomly initialized cross-attention. Understanding whether this comes from patch-level tokenization, the cross-attention architecture, or optimization dynamics would clarify the independent contribution of the FM pretraining.

- **Scalability analysis is not quantitative.** Cross-node attention complexity is $O(|E| \cdot f^2 \cdot d)$ per layer, where $f$ is the token sequence length. Flash Attention reduces memory but not asymptotic complexity. The paper mentions a training time comparison in Appendix F, but the main text gives no runtime or memory numbers relative to baselines. Practitioners evaluating FIMP for deployment need this information.

- **The novelty claim ("first method to use feature-based cross-node attention") is stated in the main text but supported only in the appendix.** Appendix D presumably differentiates FIMP from heterogeneous graph attention networks and other architectures, but readers evaluating this claim cannot assess it from the main paper. This is a structural weakness in how the contribution is argued.

### Tiny

- **Algorithm 1 does not specify the AGGREGATE and COMBINE functions used in practice.** Mean pooling? Summation? These choices affect representational power and should be specified explicitly.

- **Figure 3 bar chart description** mentions the zero-shot image classification baseline is "slightly higher" than FIMP, which is inconsistent with Table 3 (FIMP-ViT zero-shot 40.6% > ViT 34.0% > GraphSAGE 23.6%). This should be corrected to avoid reader confusion.

---

## Nice-to-Haves

- **Add a mean-aggregation-of-neighbor-ViT-embeddings zero-shot baseline.** In the zero-shot setting, a simple baseline of averaging pretrained ViT embeddings of neighbors (without any training) would strengthen the claim that FIMP's zero-shot benefit comes from the cross-attention mechanism rather than trivially from ViT features being present in the neighborhood.

- **Sensitivity analysis on token sequence length.** It is unclear whether using all $f$ tokens is necessary or whether diminishing returns set in quickly. A plot of performance vs. number of tokens would validate the tokenization design choice and guide practitioners.

- **Investigate the GraphMAE baseline configuration.** GraphMAE scores 15.8% on image classification (Table 3), substantially below GCN (23.9%) and inconsistent with its typical performance profile. Either there is a misconfiguration, or this is a genuine failure mode worth documenting.

- **Visualize cross-node attention weights.** Heatmaps showing which token-pairs between neighboring nodes receive high attention (e.g., which genes, which patches) would provide mechanistic interpretability and strengthen the argument that FIMP is learning meaningful cross-node feature interactions.

- **Evaluate under standard graph benchmarks (e.g., OGB)** to establish generalizability beyond the three specialized scientific domains. This is not a core requirement but would broaden the audience and provide a familiar reference point for the GNN community.

- **Provide deeper guidance on when domain-matched pretraining is necessary.** The paper notes that out-of-domain ViT hurts scRNAseq performance, but does not synthesize this into actionable guidance. When is FIMP-base preferable to FIMP+FM? This would be of immediate practical value.

---

## Removed Points

*These points are flagged for removal — treat with caution, they are included for transparency:*

- **"Zero-shot claim is misleading because the ViT is heavily pretrained"** (Harsh Critic): The zero-shot claim in context clearly refers to zero *graph-specific* training, which is standard usage. The ViT being pretrained on images is a feature of the method, not a misrepresentation. Removed as a factual misreading.

- **"Cross-attention Q/K/V from same-sequence optimized weights may not transfer to cross-sequence"** (Harsh Critic): While a real theoretical concern in principle, the paper acknowledges this and the empirical results show that the transfer works well. Demanding a freeze-vs-finetune ablation here goes beyond reasonable scope for an empirical systems paper. Moved to implicit acknowledgment.

- **"Statistical significance tests (Wilcoxon, t-test) should be reported"** (Harsh Critic): Single-run or 5-run evaluations without formal significance tests are the norm on large-scale graph benchmarks. Requiring formal hypothesis testing here is a non-standard methodological demand. Removed.

- **"Contribution #3 overstates zero-shot performance at ~6% gain"** (Harsh Critic): The zero-shot gain over standalone ViT (34% → 40.6%) plus surpassing *finetuned* GNNs (best: 27.4%) is a genuinely strong result. The critic's framing as "only 6% over ViT" omits the comparison to trained GNNs. Removed as misleading framing.

- **"Related work should be in the main text" (style nitpick):** This is a paper organizational choice, not a substantive flaw. Removed as a formatting/structure nitpick.

---

## Novel Insights

The most genuinely novel observation that emerges from across the three reviews — one not foregrounded in the paper itself — is the following: FIMP-base (random initialization) already substantially outperforms all GNN baselines on image classification (38.6% vs. 27.4%) and gene expression prediction (R²=0.38 vs. 0.30), suggesting that the *token-sequence representation of nodes* combined with cross-node cross-attention is a powerful inductive bias *independently* of pretrained weights. Yet the same architecture underperforms GNNs on cell-type classification. This asymmetry suggests FIMP's architectural prior is well-matched to continuous-value prediction tasks (regression, reconstruction) but potentially mismatched to discriminative classification tasks where graph topology matters more than fine-grained feature interactions. Systematically characterizing this task-type dependency — rather than treating FIMP as a uniformly better architecture — would be a significant contribution to understanding when and why token-level message passing helps.

---

## Suggestions

1. **Resolve the fMRI masking confound (Table 4):** Run the GNN baselines with the "Tokenization + PE" masking strategy, or run FIMP with the same strategies as GNNs. Even if the gap remains large, the comparison will be credible.

2. **Explain and analyze FIMP-base regression on mouse hippocampus cell type classification:** Profile loss curves, attention weight entropy, and learned token representations. Determine whether this reflects optimization difficulty, tokenization mismatch, or an architectural limitation of cross-attention for classification.

3. **Add a paired architecture-vs-pretraining ablation table:** For every domain, report (a) best GNN, (b) FIMP-base (same architecture, random init), and (c) FIMP+FM (same architecture, pretrained). This cleanly separates the three sources of gain: graph structure, token-level architecture, and pretrained knowledge.

4. **Clarify zero-shot protocol precisely:** State explicitly which parameter groups (Q/K/V projections, tokenization projections, readout heads) are frozen versus trained in the zero-shot evaluation, with a one-sentence description in the main text.

5. **Move domain-specific tokenization descriptions (scRNAseq gene embedding table, fMRI time-window tokenization) to Section 3.1:** This is the centerpiece of FIMP's design and should not require an appendix to understand.

---

**Axis evaluations:**

- **Novelty:** Genuinely high. Token-sequence-level cross-node attention as a message-passing primitive, specifically designed to accept pretrained non-textual FM weights, is a new contribution with a clear mechanistic distinction from prior work.
- **Technical soundness:** Moderate. The core mechanism is sound and well-formulated. However, the fMRI comparison methodology is flawed, and the FIMP-base regression on classification is unexplained, leaving some claims on uncertain footing.
- **Empirical support:** Moderate-to-strong. Results on image classification and gene expression are compelling and ablated. fMRI results need the masking confound resolved. Spatial transcriptomics classification results show a weakness the paper does not adequately discuss.
- **Significance:** High. Extending pretrained non-textual FMs to graph-structured data is an important and underexplored direction, and the zero-shot results are particularly exciting.
- **Clarity:** Good overall, with the main gap being tokenization details deferred to the appendix and ambiguity in the zero-shot protocol.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 3.0, 6.0]
Average score: 4.8
Binary outcome: Reject
