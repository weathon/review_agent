## Summary
This paper proposes HG-Adapter, a framework for improving pre-trained heterogeneous graph neural networks through dual structure-aware adapters (homogeneous and heterogeneous) and potential labeled data extension via label propagation. The method is grounded in a derived generalization error bound and evaluated across multiple backbones and datasets.

## Strengths
- **Dual structure-aware adapter design**: The explicit separation of homogeneous and heterogeneous structural adaptation (Section 2.3, Eq. 5-6, 9) is well-motivated for heterogeneous graphs. Figure 2(a) provides empirical validation showing the homophily ratio of the learned homogeneous structure increasing during training.
- **Consistent empirical improvement across backbones**: Table 1 demonstrates HG-Adapter improves performance across three pre-trained HGNN backbones (HDMI, HeCo, HERO) on four datasets (ACM, Yelp, DBLP, Aminer), suggesting the method is not tied to a single architecture.
- **Ablation validates loss components**: Table 2 shows removing individual loss components ($\mathcal{L}_{con}$, $\mathcal{L}_{rec}$, $\mathcal{L}_{mar}$) causes measurable performance drops (e.g., removing $\mathcal{L}_{mar}$ drops ACM Macro-F1 from 92.7% to 90.1%), supporting the design choices.

## Weaknesses

### Fatal
None identified. The potential data leakage concern (see Major) requires clarification but does not definitively invalidate the paper if the evaluation protocol is sound.

### Major
- **Ambiguous evaluation protocol regarding test node exclusion**: Section 2.4 states the method uses "all unlabeled nodes" for label propagation (Eq. 11) and includes them in the contrastive loss (Eq. 12, sum over $\tilde{\mathbf{Y}}_{UL}$). In standard transductive benchmarks (ACM, DBLP, etc.), the full graph structure including test nodes is available during training. The paper does not explicitly clarify whether test nodes are excluded from the propagation matrix $\mathbf{A}$ and loss calculation. If test nodes are included, their pseudo-labels would contribute to training, constituting data leakage that invalidates the generalization claims. This ambiguity is critical given the paper's central focus on "generalization ability."
- **Theoretical claim of lower bound lacks rigorous justification**: Theorem 2.3 presents a standard generalization bound form ($\text{Error} \le \text{Empirical} + O(\sqrt{|\mathcal{P}_M|/n_M})$). The paper claims HG-Adapter achieves a *lower* bound than prompt-tuning methods because adapters "approach optimal parameters" (Section 2.3). However, adapters introduce more trainable parameters than prompt vectors, increasing the complexity term. For the bound to be strictly lower, the reduction in training error must provably outweigh the complexity increase. The main text asserts this occurs but provides no rigorous derivation—only a circular argument that adapters approach optimal parameters *because* they lower the bound. The proof is deferred to the appendix.
- **Marginal improvements without statistical significance testing**: Table 1 shows improvements over the strongest fine-tuning baseline (HERO) are small (e.g., ACM Macro-F1: 92.2±0.5 vs. 92.7±0.4). These differences fall within reported standard deviations, suggesting they may not be statistically significant. Without paired t-tests or similar analysis, claiming "superior effectiveness" over strong baselines is not well-supported, especially for parameter-efficient methods where some performance trade-off is expected.

### Minor
- **Missing parameter efficiency reporting**: A key selling point of adapter/prompt tuning is parameter efficiency, yet the paper does not report trainable parameter counts for HG-Adapter vs. HetGPT vs. full fine-tuning. If HG-Adapter has comparable parameters to fine-tuning, the "efficiency" argument is weakened. If it has more than HetGPT, the performance gain might reflect increased capacity rather than better inductive bias.
- **Theorem 2.3 is generic, not graph-specific**: The generalization bound (Eq. 2-3) is a standard learning-theoretic form that does not contain terms specific to graph structure or adapters. The transition from this generic bound to the specific adapter design is abrupt, without deriving context-specific insights (e.g., how graph homophily affects the bound).

### Trivial
None identified beyond the above.

## Nice-to-Haves
- Report training/inference time comparisons to contextualize efficiency claims.
- Visualize the learned adjacency matrix $\mathbf{A}$ compared to original meta-path structures to show where the adapter re-weighted edges.
- Analyze sensitivity to noisy pseudo-labels when initial labeled data is very scarce.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Critic's claim about "unfair comparison" if asymmetry favors baseline**: Not applicable—no such criticism was raised.
- **Critic's claim about missing related works (GraphPrompt)**: Removed per hard rules—cannot verify external works. The paper does cite HGPrompt and HetGPT appropriately.
- **Parser artifacts (typos, formatting)**: Removed per hard rules—these are not author errors.
- **Strength Finder's claim about "first unified theoretical framework"**: Weakened—the bound is generic and not graph-specific, so this is overstated.
- **Critic's claim about appendix-deferred proofs**: Removed per hard rules—appendix exists in original submission.

## Novel Insights
The paper's core insight—that structural adaptation during tuning (not just feature modification) can improve generalization for heterogeneous graphs—is relevant and aligns with current literature gaps. However, the theoretical framework (Theorem 2.3) does not provide genuinely novel graph-specific insights beyond standard learning bounds. The dual-adapter design is a sensible architectural choice but not fundamentally novel given existing adapter literature.

## Suggestions
1. **Clarify evaluation protocol explicitly**: State whether test nodes are excluded from the label propagation matrix $\mathbf{A}$ and contrastive loss $\mathcal{L}_{con}$. If excluded, describe the masking mechanism. If included, acknowledge this as a transductive semi-supervised setting and adjust claims accordingly.
2. **Add statistical significance tests**: Report paired t-tests or Wilcoxon signed-rank tests comparing HG-Adapter against HERO and HetGPT to substantiate performance claims.
3. **Report parameter counts**: Include a table showing trainable parameters for HG-Adapter, HetGPT, and full fine-tuning to validate efficiency claims.
4. **Strengthen theoretical justification**: Either provide a rigorous proof in the main text showing the training error reduction quantitatively offsets the complexity increase, or temper the claim to state the bound is *expected* to be lower based on empirical observations.

## Score and Decision

**Calibration anchors retrieved:**
- `/home/wg25r/review_agent/human_reviews_2026/FoTtvLkkfU.md` (avg 5.50, Accept Poster): Adapter tuning for GNNs with empirical improvements across 8 datasets. Similar empirical strength but clearer methodology.
- `/home/wg25r/review_agent/human_reviews_2026/cDc95lucVL.md` (avg 6.00, Accept Oral): Heterogeneous graph in-context learning with lightweight adaptation. Stronger theoretical grounding and clearer evaluation.
- `/home/wg25r/review_agent/human_reviews_2026/MynAEqF9Nc.md` (avg 4.00, Reject): Adapter-based GNN fine-tuning with strong experiments but outdated framing and missing comparisons.
- `/home/wg25r/review_agent/human_reviews_2026/qQvNNZrPqw.md` (avg 3.50, Reject): Generalization bounds for GNNs criticized for lacking GNN-specific insights and tightness validation.
- `/home/wg25r/review_agent/human_reviews_2026/n2J1NCtN6T.md` (avg 4.50, Reject): Heterogeneous graph prompting with good experiments but missing scalability analysis and metric alignment issues.

**Comparison:** This paper has stronger empirical consistency than MynAEqF9Nc (4.00) and n2J1NCtN6T (4.50), with improvements across multiple backbones. However, it shares the theoretical weakness of qQvNNZrPqw (3.50)—generic bounds without graph-specific insights—and has a more serious evaluation ambiguity than FoTtvLkkfU (5.50) or cDc95lucVL (6.00). The data leakage ambiguity is a significant concern that prevents scoring above 5.0, as it directly threatens the core generalization claim. The paper is positioned between the 4.50-5.50 anchors, leaning toward 5.0 given the empirical strength but theoretical and evaluation gaps.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>