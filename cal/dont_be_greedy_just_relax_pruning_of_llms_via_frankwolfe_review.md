=== CALIBRATION EXAMPLE 7 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is engaging and accurately reflects the core contribution: replacing greedy mask selection with a convex relaxation solved via Frank-Wolfe. The abstract clearly states the problem, limitations of existing methods, and summarizes the three contributions (method, empirical gains, theory). However, there is a minor inconsistency: the abstract claims "up to 80%" reduction in per-layer pruning error, while the body text mentions "up to 70%" in the contributions list and "up to 80%" in Figure 2. This should be corrected for consistency.

### Introduction & Motivation
The introduction effectively motivates the challenge of pruning LLMs without retraining, highlights the combinatorial hardness of mask selection, and positions the convex relaxation + Frank-Wolfe as a novel alternative to greedy heuristics. Contributions are clearly listed. The related work section is appropriate but could be more detailed in distinguishing between mask-selection methods (Wanda, RIA) and reconstruction-inclusive methods (SparseGPT). The decision to compare only to pure mask-selection methods is justified but should be explicitly stated.

### Method / Approach
Section 2 provides a clear unification of SparseGPT, Wanda, and RIA as greedy approximations, which is insightful. The derivation of the convex relaxation and the Frank-Wolfe algorithm is well-presented. The Linear Minimization Oracle (LMO) is efficiently described for unstructured and semi-structured sparsity. However, the actual algorithm used (SparseFW) includes a critical caveat not fully captured in Algorithm 1: to achieve good perplexity, a fraction \(\alpha\) of weights with highest saliency (e.g., from Wanda) must be fixed as unprunable, and FW optimizes only over the remaining weights. Algorithm 2 in the appendix provides the full details, but this hybrid nature should be more prominent in the main text. This design choice implies that SparseFW does not fully optimize over all weights; rather, it refines the less salient subset. The ablation in Table 2 shows that best results require fixing 90% of weights (\(\alpha=0.9\)), which raises questions about the extent to which weight interactions are actually being exploited for the most important decisions. The method also requires a warm-start mask (from Wanda or RIA), which should be explicitly stated in the algorithm description.

### Experiments & Results
The experimental setup is standard and uses a range of modern GPT models and sparsity regimes. Table 1 shows that SparseFW generally matches or improves perplexity and zero-shot accuracy over baselines, with gains more pronounced at higher sparsities (60%, 2:4). However, the omission of standard deviations or confidence intervals makes it difficult to assess statistical significance, especially for small improvements (e.g., accuracy changes of ~0.5%). Figure 2 demonstrates strong reductions in per-layer pruning error (up to 80%), supporting the claim that SparseFW better optimizes the local objective. Figure 3 provides useful ablations on iteration and sample efficiency. Key concerns:
1. **Comparison to SparseGPT**: While the focus on mask-selection methods is reasonable, a comparison to SparseGPT (which includes reconstruction) would provide a more complete picture of state-of-the-art.
2. **Ablations**: The ablation on \(\alpha\) (Table 2, appendix) is crucial but should be discussed more deeply in the main text. Why does \(\alpha=0.9\) work best? Does this imply that Wanda’s saliency is very reliable for the most important weights, and FW only refines the “tail”?
3. **Computational cost**: SparseFW is iterative and more expensive than Wanda/RIA. A runtime comparison or discussion of trade-offs (e.g., time vs. performance gain) would be valuable for practitioners.
4. **Reproducibility**: Details are sufficient, but hyperparameters (\(\alpha=0.9\), \(T=2000\), 256 samples) should be clearly stated in the main text. The promise to release code is positive.

### Theoretical Results
Section 4 provides a theoretical guarantee linking the relaxed solution to the original combinatorial problem. Lemma 2 gives a bound on the suboptimality after rounding, accounting for optimization error and thresholding error. This is a solid contribution that differentiates SparseFW from purely heuristic methods. However, the bound depends on \(\lambda_{\max}(Q)\), which is data-dependent, and includes a term scaling with \(\sqrt{d_{\text{in}} d_{\text{out}} k}\). A discussion of the bound’s tightness or empirical behavior would strengthen the section.

### Writing & Clarity
The paper is generally well-written and logically structured. Some improvements:
- Algorithm 1 should either be replaced by Algorithm 2 or clearly reference the full version with fixed weights.
- The caveat about fixing weights (Section 2.3) is critical and should be more prominently integrated into the method description.
- Figure 1 is referenced but not visible in the provided text (likely a parsing artifact); assuming it is clear in the original.
- Minor inconsistencies in error reduction percentages (abstract vs. body) should be fixed.

### Limitations & Broader Impact
The conclusion honestly acknowledges key limitations: the mismatch between local pruning error and global perplexity, and the need to fix high-saliency weights to avoid degradation. This local-global gap is an important insight. Broader impact is not discussed; while pruning generally reduces resource consumption, a brief statement on potential societal impacts would be appropriate.

### Overall Assessment
This paper presents a novel and theoretically grounded approach to LLM pruning by solving a convex relaxation of the mask selection problem via Frank-Wolfe. The method demonstrates significant reductions in per-layer pruning error and consistent, though sometimes modest, improvements in perplexity and zero-shot accuracy, especially at higher sparsities. The hybrid design (fixing a large fraction of weights based on greedy saliency) is pragmatic but somewhat diminishes the claim of fully accounting for weight interactions. The work is solid and could be of interest to the ICLR community, but acceptance would benefit from addressing the following: providing statistical significance for results, more thorough ablation studies (e.g., varying warm-start strategies, computational cost analysis), and a clearer discussion of the implications of the \(\alpha=0.9\) finding. With these revisions, the paper could meet ICLR’s bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SparseFW, a layerwise pruning method for Large Language Models (LLMs) that formulates mask selection as a convex optimization problem over the convex hull of binary masks and solves it using the Frank-Wolfe (FW) algorithm. The method aims to account for weight interactions that are ignored by greedy heuristics like Wanda and RIA. The authors demonstrate reduced per-layer pruning error and improved final perplexity and zero-shot accuracy across several modern GPT architectures at various sparsity levels.

### Strengths
1. **Novel Optimization Formulation**: The paper provides a principled approach by relaxing the combinatorial mask selection problem to a convex program and employing the Frank-Wolfe algorithm. This is a clear departure from greedy, interaction-ignoring heuristics and is well-motivated (Sections 1, 2.2).
2. **Strong Empirical Evaluation**: The paper includes extensive experiments across five state-of-the-art models (LLaMA-3.1, Gemma-2, Yi-1.5, DeepSeek, Qwen2.5) and multiple sparsity regimes (50%, 60% unstructured, 2:4 semi-structured). Results show consistent, though sometimes modest, improvements in WikiText perplexity and more robust gains in zero-shot accuracy over strong baselines (Table 1, Section 3).
3. **Theoretical Grounding**: The authors provide an approximation guarantee (Lemma 1, expanded in Appendix E) linking the solution of the relaxed problem to the original combinatorial problem after rounding, a formal benefit not offered by baseline heuristics.
4. **Ablation and Analysis**: The paper includes useful ablations on the critical hyperparameter α (the fraction of weights fixed from the warmstart) and on the sample/iteration efficiency of SparseFW, providing insights into its behavior and practical setup (Figure 3, Table 2/Appendix C).

### Weaknesses
1. **Dependence on Warmstart Fixation**: The method's strong empirical performance crucially relies on fixing a large fraction (α=0.9) of high-saliency weights from the Wanda/RIA warmstart. The paper notes that vanilla FW (α=0.0) often performs worse than baselines (Section 2.3, Table 2). This undermines the claim that the optimization alone is sufficient and suggests the method is more of a refinement of existing saliency scores rather than a fully standalone solution.
2. **Limited Comparison Scope**: The primary comparison is against mask-selection methods Wanda and RIA. A direct comparison with SparseGPT—which includes a weight reconstruction step and is a very strong, widely-used baseline—is notably absent, making it difficult to assess SparseFW's position in the broader pruning landscape.
3. **Increased Computational Cost**: SparseFW requires iterative optimization (e.g., 2000 iterations per layer) and more calibration samples for best performance, making it significantly more expensive than single-pass methods like Wanda. While the paper argues this cost is worthwhile for a deployed model, the computational trade-off is not quantified (e.g., total wall-clock time or FLOPs) (Figure 3).
4. **Unresolved Objective Mismatch**: The paper acknowledges but does not fully address the "mismatch between local and global objectives." The need to fix weights indicates that minimizing the local layerwise reconstruction error does not guarantee better global model performance, which is a fundamental limitation of the layerwise pruning framework adopted by all compared methods (Section 5).

### Novelty & Significance
The core novelty lies in applying the classical Frank-Wolfe algorithm to the relaxed pruning mask problem for LLMs, providing a theoretically-justified alternative to greedy solvers. The significance is moderate: the empirical gains are consistent but often incremental, and the method's reliance on warmstart fixation tempers the novelty of the optimization core. The work demonstrates that more sophisticated optimization can improve upon greedy criteria, which is a valuable proof of concept for the community. However, the practical impact may be limited by the increased compute cost and the lack of a comparison to the stronger SparseGPT baseline.

### Suggestions for Improvement
1. **Compare with SparseGPT**: A direct comparison with SparseGPT (and potentially other methods with weight reconstruction) is essential to properly situate the contribution. This should include performance, speed, and memory metrics.
2. **Deeper Analysis of the α Hyperparameter**: The paper should investigate why fixing 90% of the weights is optimal. Is this revealing a fundamental limitation of the local objective? An analysis of the correlation between per-layer error reduction and final perplexity change across layers could be insightful.
3. **Quantify Computational Overhead**: Provide a clear analysis of the total computational cost (time, memory) of SparseFW versus baselines, especially in relation to the achieved performance gains. This is critical for assessing practical utility.
4. **Explore Adaptive Spararsity**: The uniform sparsity allocation across layers is a suboptimal but common practice. Leveraging the FW framework to potentially allocate sparsity budgets across layers in a coordinated, non-uniform way could be a promising direction mentioned but not explored.
5. **Clarify the Objective Mismatch**: The discussion around the need for weight fixation (α) should be moved from the appendix into the main body and framed as a key finding and limitation of the layerwise approach. Suggesting ways to better align the local objective with global performance would strengthen the paper.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison to SparseGPT.** The paper explicitly avoids comparing to SparseGPT (a core baseline) because it involves weight reconstruction. This is a critical omission. The claim of outperforming "state-of-the-art LLM pruning approaches" is unsupported without this comparison, as SparseGPT is a standard and often superior baseline for one-shot pruning.
2. **Comprehensive efficiency analysis.** The method is more compute-intensive. A rigorous comparison of runtime, memory, and performance trade-offs against Wanda, RIA, and SparseGPT is essential to judge its practical utility. The paper mentions compute cost is "worthwhile" but provides no data to support this claim.
3. **Ablation on the warm-start dependency.** Performance heavily relies on fixing ~90% of weights using Wanda/RIA saliency. A systematic ablation is needed to show what happens when this warm-start comes from other methods (e.g., random, magnitude) or if the ratio is tuned per layer/model. The current approach risks being a minor refinement of Wanda rather than a standalone method.
4. **Evaluation on a broader suite of tasks.** Reliance on WikiText perplexity and a single zero-shot set (EleutherAI) is insufficient for ICLR. Results on popular benchmarks (e.g., MMLU, HellaSwag, ARC) are needed to convincingly demonstrate preserved model capability.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of mask differences.** The claim that SparseFW "accounts for weight interactions" needs validation. A layer-wise analysis comparing which specific weights are pruned differently by SparseFW versus Wanda/RIA is required. Without this, the mechanism of improvement is unclear.
2. **Investigation of the local vs. global objective mismatch.** The paper notes that optimizing the local pruning error does not guarantee better perplexity, requiring a fix from Wanda. A deeper analysis is needed to understand why this happens—e.g., does optimizing the local objective lead to catastrophic pruning of a few critical weights? This is central to the method's limitations.
3. **Examination of the theoretical bound's practical tightness.** The guarantee depends on λ_max(Q). An empirical analysis of this eigenvalue's magnitude across layers and models is needed to show whether the bound is meaningful or vacuous in practice.

### Visualizations & Case Studies
1. **Visual comparison of learned masks.** Heatmaps visualizing the continuous mask values before thresholding and the final binary masks for SparseFW versus baselines would reveal if the method finds meaningfully different sparsity patterns or merely perturbs the warm-start mask.
2. **Case studies of failure modes.** The paper should identify and visualize specific layers or matrices where SparseFW (with α=0.0) reduces pruning error but drastically harms perplexity. This would concretely illustrate the local-global mismatch and justify the need for the warm-start fix.

### Obvious Next Steps
1. **Incorporate SparseGPT as a primary baseline.** Any pruning paper submitted to ICLR must position itself against the current one-shot pruning standard (SparseGPT), not just the simpler saliency-based methods (Wanda, RIA).
2. **Develop an adaptive strategy for the fixed-weight ratio (α).** Using a fixed α=0.9 is a major heuristic. The paper should propose and test a principled, perhaps layer-adaptive, method to determine which weights to fix, moving beyond reliance on the warm-start method's saliency scores.
3. **Explore integration with a light reconstruction step.** Since the method focuses on mask selection, a natural next step is to combine it with a fast, closed-form weight reconstruction (like SparseGPT's) to see if gains are complementary. The current isolation from reconstruction is a limitation.

# Final Consolidated Review
## Summary
This paper introduces SparseFW, a layerwise pruning method for Large Language Models (LLMs) that formulates mask selection as a convex optimization problem over the convex hull of binary masks, solved via the Frank-Wolfe algorithm. The method reduces per-layer pruning error and shows consistent improvements in perplexity and zero-shot accuracy over greedy baselines across several GPT architectures at various sparsity levels.

## Strengths
- **Novel optimization framework**: The paper provides a principled departure from greedy heuristics by relaxing the combinatorial mask selection problem to a convex program and employing Frank-Wolfe, which explicitly accounts for weight interactions and supports theoretical analysis.
- **Extensive empirical validation**: Experiments across five modern LLMs (e.g., LLaMA-3.1, Gemma-2) and multiple sparsity regimes (50%, 60% unstructured, 2:4 semi-structured) demonstrate reduced per-layer pruning error (up to 80%) and consistent gains in final perplexity and zero-shot accuracy over strong mask-selection baselines (Wanda, RIA).
- **Theoretical grounding**: Approximation guarantees link the relaxed solution after rounding to the original combinatorial problem, offering a formal advantage over heuristic methods.

## Weaknesses
- **Heavy reliance on warmstart fixation**: The method's strong performance crucially depends on fixing a large fraction (α=0.9) of high-saliency weights from greedy warmstarts (Wanda/RIA). Vanilla Frank-Wolfe (α=0.0) often underperforms baselines, undermining the claim that optimization alone suffices and revealing that SparseFW acts more as a refinement of existing saliency scores than a standalone solver.
- **Unquantified computational overhead**: SparseFW requires iterative optimization (e.g., 2000 iterations per layer) and more calibration samples for best results, making it significantly more expensive than single-pass methods. The paper argues this cost is worthwhile for deployment but provides no analysis of runtime or memory trade-offs, limiting practical assessment.
- **Persistent local-global objective mismatch**: The paper acknowledges that minimizing per-layer pruning error does not reliably improve global perplexity, necessitating the weight-fixation heuristic. This fundamental limitation of the layerwise pruning framework is not resolved, and the need for α highlights that the local objective remains misaligned with model performance.

## Nice-to-Haves
- Direct comparison to reconstruction-inclusive methods like SparseGPT would help situate the contribution within the broader pruning landscape.
- Analysis of mask differences (e.g., which weights are pruned differently) could elucidate how weight interactions are exploited in practice.
- Empirical examination of the theoretical bound's tightness, particularly the data-dependent eigenvalue λ_max(Q), would strengthen the theoretical discussion.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Inconsistency in error reduction percentages**: The abstract cites "up to 80%" while the contributions list "up to 70%"; this is a minor typo that does not affect the core claims.
- **Lack of statistical significance**: Table 1 does not report standard deviations, but single-run evaluation is common in large-scale LLM pruning due to computational cost, and the paper uses multiple seeds for ablation plots (Figure 3).
- **Request for broader task evaluation**: The paper uses WikiText perplexity and EleutherAI zero-shot accuracy, which are standard benchmarks in prior work (e.g., Wanda, RIA); demanding additional benchmarks is scope creep without evidence that current evaluation is insufficient.

## Novel Insights
The paper demonstrates that classical constrained optimization (Frank-Wolfe) can effectively improve upon greedy heuristics for LLM mask selection, reducing per-layer error substantially. However, it also surfaces a critical insight: even with better local optimization, the layerwise objective fails to guarantee global performance improvements, as evidenced by the need to fix most weights based on greedy saliency. This underscores a fundamental challenge in decoupled pruning approaches.

## Suggestions
- Quantify the computational overhead of SparseFW (e.g., wall-clock time, memory usage) compared to baselines to clarify the trade-off between cost and performance gain.
- Investigate adaptive or principled methods for determining the fixed-weight ratio α, potentially per layer, to reduce heuristic dependency and improve robustness.
- Explore integrating the mask selection with a lightweight weight reconstruction step (e.g., akin to SparseGPT) to see if gains are complementary and address the local-global mismatch more effectively.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
