=== CALIBRATION EXAMPLE 8 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is engaging and reflects the core idea of replacing greedy heuristics with a relaxation-based approach. The abstract clearly states the problem, proposed solution, and contributions. However, there is an inconsistency in the claimed improvement: the abstract states "reduces the per-layer pruning error by up to 80%" while the contributions list says "up to 70%." This discrepancy should be corrected for accuracy.

### Introduction & Motivation
The introduction effectively motivates the problem of post-training pruning for LLMs, highlighting the limitations of greedy methods and the intractability of the combinatorial mask selection problem. The transition to a convex relaxation solved via Frank-Wolfe is well motivated. The contributions are clearly outlined and align with the rest of the paper.

### Methodology (Section 2)
The section provides a clear derivation of how existing methods (SparseGPT, Wanda, RIA) can be viewed as greedy approximations. The convex relaxation and Frank-Wolfe algorithm are introduced appropriately. The gradient computation and precomputation of \(G = XX^\top\) and \(H = WG\) for efficiency are well explained.

**Major concern:** Algorithm 1 in the main text is incomplete and potentially misleading. The critical detail that a fraction \(\alpha\) of high-saliency weights (from the warmstart) is fixed and not optimized is only mentioned in the text and fully described in Algorithm 2 (Appendix B). This design choice is essential for achieving good perplexity, as the authors note that without it (\(\alpha=0.0\)), performance degrades. This should be explicitly integrated into the main algorithm description and discussed as a key component of the method. The method is not simply solving the relaxed problem from scratch; it is a *refinement* procedure that heavily relies on the warmstart to identify which weights to preserve.

### Experiments & Results (Section 3)
The experimental setup is comprehensive, covering multiple models, sparsity regimes, and metrics (perplexity, zero-shot accuracy). Results generally show improvements, especially for zero-shot accuracy and higher sparsity levels.

**Key concerns:**
1. **Inconsistent perplexity improvements:** In Table 1, some entries show SparseFW underperforming the baseline (e.g., DeepSeek-7B at 60% sparsity: Wanda 11.44 vs. SparseFW(Wanda) 11.99). The authors should discuss these cases to provide a balanced view of when the method does not help.
2. **Dependence on hyperparameter \(\alpha\):** Table 2 (Appendix C) shows that performance is highly sensitive to the fraction \(\alpha\) of fixed weights. The best results are often at \(\alpha=0.9\), meaning 90% of the mask is determined by the warmstart (Wanda) and only 10% is optimized. This raises questions about the method's novelty and independence. It is effectively a post-hoc refinement of a greedy mask, not a fully independent optimization. This should be discussed transparently in the main text.
3. **Computational cost:** SparseFW requires 2000 iterations per layer, which is substantially more expensive than one-pass methods like Wanda. While the authors argue this is acceptable for a one-time pruning, a runtime comparison or discussion of trade-offs is missing. Figure 3 shows diminishing returns after many iterations, but the chosen 2000 iterations may still be costly.
4. **Objective mismatch:** The paper notes that reducing the per-layer pruning error does not always translate to better perplexity, necessitating the fixing of weights. This local-global mismatch is a fundamental limitation that should be highlighted more prominently.

### Theoretical Results (Section 4)
The theoretical analysis provides an approximation guarantee combining optimization error (from FW) and thresholding error (from rounding). This is a solid contribution that differentiates the method from purely heuristic approaches. However, the bound depends on \(\lambda_{\max}(Q)\), which may be large in practice, limiting its tightness. The practical utility of the bound could be discussed briefly.

### Conclusion & Limitations
The conclusion summarizes the work adequately. Limitations are acknowledged, including the local-global objective mismatch and the need to fix weights. However, the discussion should be expanded to include:
- The method's heavy reliance on a warmstart (making it a refinement step rather than a standalone solver).
- The computational overhead compared to baselines.
- The sensitivity to the hyperparameter \(\alpha\).

### Writing & Clarity
The paper is generally well-written. However, a few clarifications are needed:
- Algorithm 1 should be updated to reflect the fixing of weights, or at least clearly refer to Algorithm 2.
- The inconsistency in the abstract regarding error reduction (80% vs. 70%) must be resolved.
- Some references to figures (e.g., Figure 1) are not visible in the extracted text, but we assume they are present in the original. No major clarity issues were noted.

### Overall Assessment
The paper proposes a novel approach to LLM pruning by relaxing the combinatorial mask selection problem and solving it with Frank-Wolfe. The method, SparseFW, demonstrates empirical improvements, particularly in zero-shot accuracy and at higher sparsity levels, and is backed by theoretical guarantees. However, the contribution is tempered by the method's heavy dependence on a warmstart mask (fixing 90% of weights based on Wanda saliency) and increased computational cost. The core innovation—using Frank-Wolfe to optimize a convex relaxation—is sound, but the practical implementation is essentially a refinement of existing greedy masks. For ICLR, the paper would be stronger if the authors more transparently address these limitations, clarify the method's reliance on the warmstart, and provide a more nuanced analysis of when the optimization leads to tangible gains. With revisions to highlight and discuss these points, the paper presents a valuable contribution to the pruning literature.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SparseFW, a novel method for post-training pruning of Large Language Models (LLMs). It frames the layer-wise mask selection problem as a convex relaxation of the hard combinatorial problem and solves it using the Frank-Wolfe (FW) algorithm. The core idea is to optimize over the convex hull of binary masks, explicitly accounting for weight interactions that greedy state-of-the-art methods (e.g., Wanda, RIA) ignore. The method includes a practical modification that fixes a fraction of high-saliency weights from a warm-start mask (e.g., from Wanda) before applying FW. SparseFW demonstrates significant reductions in per-layer reconstruction error and consistent improvements in final perplexity and zero-shot accuracy across multiple modern GPT architectures at various sparsity levels.

### Strengths
1.  **Strong Empirical Performance:** The paper provides comprehensive experiments on five state-of-the-art models (LLaMA 3.1, Gemma 2, Yi 1.5, DeepSeek, Qwen2.5) across multiple sparsity regimes (50%, 60% unstructured, 2:4 semi-structured). SparseFW consistently outperforms strong baselines (Wanda, RIA) in zero-shot accuracy and often in perplexity, especially at higher sparsities. Evidence includes Table 1, where SparseFW shows gains (e.g., +1.84% to +3.84% accuracy on LLaMA-3.1-8B at 60% sparsity) and Figure 2 showing up to 80% reduction in per-layer pruning error.
2.  **Theoretical Foundation:** A key differentiator from heuristic methods is the provided approximation guarantee (Lemma 1, elaborated in Appendix E). This formally connects the quality of the thresholded solution from the relaxed problem to the optimal combinatorial solution, bounding the error by optimization error (controlled via FW iterations) and thresholding error. This theoretical rigor is aligned with ICLR's expectations.
3.  **Efficient and Scalable Design:** The method is carefully engineered for LLM scale. It precomputes \(G = XX^\top\) and \(H = WG\), making the per-iteration cost independent of calibration sequence length and batch size (Section 2.3). The Linear Minimization Oracle (LMO) for FW is a simple top-k selection (Eq. 12), and the algorithm naturally supports various sparsity patterns (Appendix D). The memory footprint is manageable, as highlighted in the contributions.
4.  **Rigorous Ablation and Analysis:** The paper includes insightful ablation studies. Figure 3 analyzes iteration and sample efficiency, showing diminishing returns beyond ~2000 iterations but continued gains with more samples. Table 2 (Appendix C) crucially ablates the fraction \(\alpha\) of fixed weights, demonstrating that even a small \(\alpha\) (0.1) helps, with \(\alpha=0.9\) often being optimal, while \(\alpha=0.0\) (pure FW) fails. This honest analysis of a core heuristic component strengthens the work.

### Weaknesses
1.  **Reliance on Heuristic Warm-Start for Good Perplexity:** The paper openly states (Sections 2.3, 5) that vanilla FW (optimizing the full mask) often reduces local pruning error but leads to worse final perplexity. The necessity to "fix" a large fraction (90% in best configurations) of weights based on a simple saliency score (Wanda) is a significant limitation. It suggests that the local layer-wise objective, even when optimized globally, does not align perfectly with the global model performance, and the method still leans heavily on the inductive bias of the warm-start heuristic. This somewhat undermines the claim of fully accounting for weight interactions.
2.  **Increased Computational Cost:** While memory efficient, SparseFW is computationally more expensive than single-pass baselines like Wanda. It requires thousands of FW iterations per layer (Figure 3 suggests ~2000). Although the authors argue the cost is justified for a one-time pruning of a deployed model, this trade-off is not quantified in terms of total wall-clock time or energy, which is important for practical adoption. The method's scalability to the largest models (e.g., 70B+ parameters) is asserted but not demonstrated.
3.  **Marginal Gains in Some Settings and Inconsistent Metrics:** In Table 1, some improvements in perplexity are modest (e.g., for DeepSeek-7B at 50% sparsity, SparseFW is slightly worse than Wanda). The gains, while consistent in accuracy, are sometimes small in perplexity, which is the primary language modeling metric. This raises questions about the practical significance of the improvement relative to the added complexity. Furthermore, the zero-shot accuracy gains, while positive, are evaluated on a limited set of tasks (only the aggregate score from the EleutherAI suite is reported).
4.  **Theoretical-Practical Gap:** The provided theoretical guarantee, while valuable, depends on the maximum eigenvalue \(\lambda_{\max}(Q)\) of the Hessian, which could be very large in practice, making the bound potentially loose. The paper does not discuss the magnitude of this term empirically or how tight the bound is likely to be, limiting the practical utility of the guarantee.

### Novelty & Significance
**Novelty:** The core novelty is the application of a convex relaxation paired with the Frank-Wolfe algorithm to the LLM pruning mask selection problem. While FW has been used in ML and even for inducing sparsity during training, its application to one-shot, post-training pruning of LLMs is novel. The formulation of the mask selection as a convex program over the \(L_1\)-ball is a clear and insightful departure from the greedy, weight-by-weight approaches that dominate the field.

**Significance:** The work is significant as it introduces a principled, optimization-based perspective to a problem typically tackled with heuristics. It demonstrates that even with a necessary heuristic modification (fixing weights), this approach can yield measurable improvements over state-of-the-art methods. It successfully bridges classical optimization theory with a pressing practical problem in efficient LLM deployment. The work meets ICLR's bar for a solid incremental contribution that advances the understanding and performance of a key technique.

### Suggestions for Improvement
1.  **Investigate the Objective Mismatch:** A deeper analysis of why optimizing the exact layer-wise reconstruction objective (even with FW) can hurt global perplexity is needed. Is it an issue of the calibration data? The sequential layer-wise pruning setup? Proposing and evaluating a modified objective or constraint that better correlates with final performance would greatly strengthen the method's foundation and potentially reduce reliance on the warm-start heuristic.
2.  **Provide a Comprehensive Cost-Benefit Analysis:** Include a clear comparison of the total pruning runtime (including calibration data processing, precomputation, and FW iterations) against baselines for at least one standard model. Discuss the trade-off explicitly: how much extra compute (e.g., GPU hours) yields how much improvement in perplexity/accuracy? This is critical for practitioners to assess the method's value.
3.  **Expand Evaluation and Analysis:** Report zero-shot accuracy breakdown per task (e.g., from the 6-8 tasks in the EleutherAI harness) to show if improvements are broad or concentrated. Also, analyze the characteristics of the masks found by SparseFW versus Wanda/RIA (e.g., weight distribution, correlation with activation patterns) to provide more intuition about *why* it works.
4.  **Clarify and Motivate the Warm-Start Strategy:** The choice to fix weights based on the *Wanda* saliency specifically (rather than, say, magnitude or a random subset) should be better motivated. An ablation showing the performance when fixing weights based on different criteria could be insightful. The description of this mechanism in the main algorithm (Algorithm 1) is vague; explicitly incorporating the fix from Algorithm 2 into the main text's pseudocode would improve clarity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to SparseGPT (and other reconstruction methods).** The paper only compares to Wanda and RIA, but SparseGPT is a seminal, strong baseline that also minimizes pruning error and includes weight reconstruction. Without this comparison, the claim of outperforming state-of-the-art methods is incomplete and potentially misleading.
2. **Ablate the warm-start strategy with random or no warm-start.** The method’s success critically depends on fixing a fraction of high-saliency weights from Wanda/RIA. To isolate the contribution of the Frank-Wolfe optimization itself, show results when warm-starting from random masks or when no weights are fixed (α=0.0). This is necessary to confirm that the improvement is due to the optimization and not merely from preserving the same important weights.
3. **Report runtime and memory overhead compared to baselines.** The paper claims memory efficiency and scalability but provides no concrete numbers on runtime or memory usage. For a practical pruning method, these costs are essential to evaluate, especially since SparseFW requires multiple iterations and gradient computations.
4. **Test on larger models (e.g., 70B parameters).** The experiments only go up to 14B parameters. To substantiate claims of scalability, results on models with tens or hundreds of billions of parameters are needed.
5. **Evaluate more sparsity patterns (e.g., 4:8, structured pruning).** The paper only shows results for unstructured and 2:4 semi-structured sparsity. To demonstrate generality, include other common patterns (e.g., 4:8) and structured sparsity (e.g., row/column pruning).

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze why fixing weights is necessary and which weights FW prunes differently.** The paper notes that without fixing high-saliency weights, perplexity degrades. A detailed comparison of masks (which weights FW prunes vs. Wanda) and their impact on layer outputs would reveal the mismatch between the local pruning objective and global performance, which is critical for understanding the method’s limitations.
2. **Evaluate the practical tightness of the theoretical bound.** The bound depends on λ_max(Q); compute this eigenvalue for typical layers and assess whether the bound is non-vacuous. This would strengthen the theoretical contribution by showing it provides meaningful guarantees in practice.
3. **Provide convergence analysis (e.g., duality gap vs. iterations).** The paper uses a fixed 2000 iterations without justification. Plotting the duality gap or optimization error over iterations would show how many iterations are sufficient and whether the chosen number is appropriate.
4. **Assess sensitivity to calibration data domain and size.** The paper shows improvement with more samples but does not test different calibration datasets (e.g., domain-specific data). This is important for real-world applicability where calibration data may be limited or mismatched.

### Visualizations & Case Studies
1. **Visualize the evolution of the mask during Frank-Wolfe iterations for a few layers.** Show how the mask entries change over iterations (e.g., heatmaps) to illustrate whether FW makes meaningful adjustments or only minor tweaks. This would help validate that FW is effectively exploring the feasible set.
2. **Plot the distribution of entries in the relaxed mask before thresholding.** This would reveal how fractional the solution is; if entries are near 0/1, thresholding is accurate, but if they are spread out, thresholding introduces significant error, explaining the gap between continuous and thresholded performance in Figure 4.
3. **Case study on layers with high vs. low error reduction.** Select layers where SparseFW significantly reduces pruning error and others where it does not. Analyze the weight and activation patterns to understand when the method works best and why.

### Obvious Next Steps
1. **Integrate weight reconstruction (like SparseGPT) into the framework.** Since SparseFW only optimizes the mask, combining it with a weight reconstruction step could further reduce pruning error and improve end performance. This is a natural extension and should be discussed.
2. **Extend to adaptive sparsity allocation across layers.** The paper uses uniform sparsity. Allowing layer-wise sparsity budgets (based on sensitivity) within the FW framework could lead to better global performance.
3. **Consider a global pruning objective that accounts for inter-layer interactions.** The layerwise approach ignores dependencies between layers. Formulating a global problem (even approximately) could be a significant advance.
4. **Optimize the Linear Minimization Oracle (LMO) for extreme scale.** For models with hundreds of billions of parameters, the top-k operation in the LMO may become a bottleneck. Discuss or implement approximate LMOs or distributed versions to maintain efficiency.

# Final Consolidated Review
## Summary
This paper proposes SparseFW, a method for post-training pruning of Large Language Models (LLMs). It frames the combinatorial mask selection problem as a convex relaxation and solves it using the Frank-Wolfe algorithm. The method significantly reduces per-layer reconstruction error and demonstrates consistent improvements in zero-shot accuracy across multiple modern GPT architectures at various sparsity levels, backed by theoretical approximation guarantees.

## Strengths
- **Strong and Consistent Empirical Gains:** SparseFW demonstrates consistent and often significant improvements over strong baselines (Wanda, RIA) in zero-shot accuracy across five state-of-the-art models (e.g., LLaMA 3.1, Gemma 2) and multiple sparsity regimes (50%, 60%, 2:4). Evidence from Table 1 shows accuracy gains of up to +3.84% on LLaMA-3.1-8B at 60% sparsity. It also drastically reduces the local pruning objective, with per-layer error reductions of up to 80% (Figure 2).
- **Principled Optimization Foundation with Theoretical Guarantees:** The core contribution is a novel application of convex optimization (Frank-Wolfe) to the LLM pruning problem. This is differentiated from purely heuristic methods by providing a theoretical approximation guarantee (Lemma 1, Appendix E) that bounds the error of the final binary mask relative to the optimal combinatorial solution, linking optimization and thresholding error.
- **Efficient and Scalable Design for LLMs:** The method is carefully engineered for scale. It precomputes \(G = XX^\top\) and \(H = WG\), making the per-iteration cost independent of sequence length and batch size (Sec. 2.3). The Frank-Wolfe Linear Minimization Oracle is a simple top-k selection, and the algorithm naturally supports unstructured and semi-structured sparsity patterns (Appendix D), maintaining a manageable memory footprint.

## Weaknesses
- **Heavy Reliance on a Heuristic Warm-Start for Good Global Performance:** A critical limitation is that optimizing the pure relaxed objective (\(\alpha = 0.0\)) often reduces local error but hurts final perplexity. To achieve gains, the method must "fix" a large fraction (e.g., 90%) of high-saliency weights from a warm-start mask (e.g., Wanda) before applying Frank-Wolfe. This reveals a persistent mismatch between the local layer-wise objective and global model performance and means SparseFW functions primarily as a refinement of an existing heuristic mask, not a fully standalone solver.
- **Substantial Increase in Computational Cost:** SparseFW requires thousands of Frank-Wolfe iterations per layer (Figure 3 suggests ~2000 for convergence), making it significantly more computationally expensive than single-pass baselines like Wanda. While argued as a one-time cost, this trade-off is not quantified (e.g., wall-clock time or GPU hours), which is essential for practical evaluation and adoption.
- **Inconsistent and Sometimes Marginal Perplexity Improvements:** While zero-shot accuracy improvements are consistent, gains in perplexity—the primary language modeling metric—are more mixed. In Table 1, SparseFW sometimes underperforms the baseline perplexity (e.g., DeepSeek-7B at 60% sparsity: Wanda 11.44 vs. SparseFW(Wanda) 11.99). The paper does not analyze these cases, leaving the performance profile partially unclear.

## Nice-to-Haves
- Comparison to SparseGPT, a strong baseline that includes weight reconstruction, would provide a more complete picture of the state-of-the-art.
- A runtime and memory usage comparison against baselines would help practitioners evaluate the compute/performance trade-off more concretely.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Provide a deeper analysis of the local-global objective mismatch. Investigate the characteristics of the masks SparseFW adjusts versus those it fixes (e.g., correlation with activation patterns) to better understand why preserving heuristic saliency is necessary.
- Include a breakdown of zero-shot accuracy gains by individual task (e.g., from the EleutherAI harness) to show whether improvements are broad or task-specific.
- Clarify the main algorithm description. Algorithm 1 should explicitly reference the critical weight-fixing step detailed in Appendix B (Algorithm 2) to avoid misleading readers about the method's standalone nature.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
