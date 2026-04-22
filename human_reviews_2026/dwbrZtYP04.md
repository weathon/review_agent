# SparseD: Sparse Attention for Diffusion Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
While diffusion language models (DLMs) offer a promising alternative to autoregressive models (ARs), existing open-source DLMs suffer from high inference latency. This bottleneck is mainly due to the attention’s quadratic complexity with respect to context length in computing all query–key pairs.
Intuitively, to reduce this complexity, a natural strategy is to restrict attention to sparse patterns that retain only the most relevant connections. 
Such approaches are well-established in ARs, where attention follows fixed and clearly defined sparse patterns.
However, in DLMs, we observe distinct sparsity behaviors: (1) attention patterns vary across heads, (2) attention patterns in each head remain highly similar across denoising steps, and (3) early denoising steps are critical for generation.
These findings render sparse attention methods designed for ARs largely incompatible with DLMs, as they fail to capture head-specific structures and risk degrading generation when applied in early denoising steps.
To address these challenges, we propose **SparseD**, a novel sparse attention method for DLMs. 
Leveraging the observations, SparseD only requires pre-computing head-specific sparse patterns one time, and reuses them across all steps. This prevents recomputing sparse patterns at each denoising step.
Meanwhile, SparseD uses full attention in the early steps, then switches to sparse attention later to maintain generation quality. 
Together, these establish SparseD as a practical and efficient solution for deploying DLMs in long-context applications. Experimental results demonstrate that SparseD  achieves lossless acceleration, delivering up to $1.50\times$ speedup over FlashAttention at a 64k context length with 1,024 denoising steps. Code is available at https://github.com/INV-WZQ/SparseD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the high inference latency of diffusion language models (DLMs), which stems from the quadratic complexity of the attention mechanism applied over all tokens at each denoising step. The authors propose SparseD, a sparse attention method tailored for DLMs. The method is motivated by three key empirical observations: (1) DLM attention patterns are "head-specific," meaning they vary significantly across different attention heads; (2) patterns *within* a given head are "highly similar" across all denoising steps; and (3) the early denoising steps are "critical" for generation quality, and applying sparsity too early causes performance degradation.

Based on these findings, SparseD's design is threefold:
1.  **Skipping Sparse:** It applies full (dense) attention for an initial percentage of denoising steps (e.g., first 20%) to preserve generation quality.
2.  **Isolated Selection:** At the end of this initial phase, it computes the full attention matrix *once* and selects a head-specific, block-wise top-$\rho\%$ sparse pattern. This selection is done "isolating" prefill and generation tokens to ensure both are adequately represented.
3.  **Sparse Reusing:** This single, pre-computed head-specific sparse pattern is then cached and reused for all remaining denoising steps, leveraging the observed temporal stability.

Experiments on models like Dream-7B and LLaDA-1.5 show that SparseD achieves "lossless" accuracy on various benchmarks (MMLU, RULER) while delivering significant speedups over standard FlashAttention (e.g., up to 1.50x at 64k context with 1024 steps), as the one-time pre-computation cost is amortized.

### Strengths
1.  **Novel Empirical Insights:** The paper's main strength is its clear identification and empirical validation of three specific attention behaviors in DLMs: (1) head-specific patterns, (2) temporal stability of patterns, and (3) high sensitivity of early denoising steps. This analysis is novel and provides a solid foundation for future work on DLM efficiency.
2.  **Well-Motivated Design:** The SparseD method is a strong example of principled system design. Each of its three components (Skipping Sparse, Isolated Selection, Sparse Reusing) is a direct and logical solution to one of the three identified empirical observations.
3.  **Strong Experimental Validation:** The evaluation is comprehensive. The authors demonstrate that SparseD achieves its "lossless" claim by benchmarking against the original models. Crucially, they also show its superiority to two distinct and relevant classes of baselines: (a) AR-based sparse attention (which fails on accuracy) and (b) other DLM acceleration methods (which degrade on long-context tasks).
4.  **Good Ablation Study:** The ablation in Table 2 is a model of clarity. It individually removes each component of SparseD and shows the precise, and severe, negative impact: removing "Skipping Sparse" destroys accuracy (Observation 3 validated), removing "Sparse Reusing" explodes latency (Observation 2 validated), and removing "Isolated Selection" hurts accuracy (heuristic validated).
5.  **Practicality and Scalability:** The method provides a practical speedup (up to 1.5x) that compellingly scales with the number of denoising steps $T$ (Figure 5). This makes it particularly attractive for high-quality generation, which often requires many steps.

### Weaknesses
1.  **Lack of Deeper Analysis:** The paper is purely empirical. It does not offer any hypothesis or theoretical exploration for *why* these attention patterns are temporally stable. The denoising process is defined by $T$ steps of iterative refinement, so the $Q$, $K$, and $V$ representations are constantly changing. It is non-obvious and fascinating why the resulting $QK^T$ patterns should remain static. While a full theoretical proof is not required, some discussion or preliminary analysis (e.g., measuring the cosine similarity of $Q$/$K$ vectors over time) would have strengthened the paper.
2.  **Scalability of the $O(N^2)$ Pre-computation Step:** The method requires a single, full $O(N^2)$ attention computation at step $T \times \text{skip}\%$ to build the sparse pattern. While the authors use a memory-efficient block-wise implementation, the *computational cost* of this step is still $O(N^2)$. The experiments stop at 64k context. For truly massive contexts (e.g., 256k, 1M), this single pre-computation step could become the new latency bottleneck, potentially dominating the cost of all subsequent sparse steps. The paper does not analyze the scaling of this specific step or discuss this potential limitation.

### Questions
1.  Could the authors provide a more quantitative analysis of "Attention Similarity Across Time" (Observation 2)? For example, by plotting the Jaccard similarity of the selected top-$\rho\%$ blocks between the pattern computed at step $T \times \text{skip}\%$ and the "ground truth" top-$\rho\%$ blocks at each subsequent step $t$? This would be much stronger evidence than the current heatmaps.
2.  How robust is the "lossless" claim to the temporal similarity assumption? What is the performance impact (e.g., on RULER) if the sparse pattern from one prompt is (incorrectly) reused for a different, unseen prompt? This would stress-test the model's reliance on the exact pattern.
3.  Regarding the $O(N^2)$ pre-computation step: How does the wall-clock latency of this *single step* scale with context length (e.g., from 4k to 64k)? At what context length $N$ (assuming $T=1024$) would the authors project this one-time $O(N^2)$ cost to become larger than the cumulative $O(T \cdot N \cdot \rho N)$ cost of the remaining sparse steps?
4.  The "Isolated Selection" heuristic uses the same selection ratio $\rho\%$ for both prefill and generation tokens. Is this an optimal choice? Have the authors experimented with different ratios (e.g., $\rho_{\text{prefill}}$ and $\rho_{\text{gen}}$)? One might hypothesize that static prefill tokens and dynamic generation tokens could benefit from different sparsity levels.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a sparse attention mechanism designed to accelerate Diffusion Language Model (DLM) inference, particularly in long-context, high-step settings. While it is the first or one of the first approaches to apply sparse attention for DLMs, it is based on mainly *empirical* observations and shows only incremental improvements.

### Strengths
- Empirical Validation of DLM Dynamics: The analysis highlighting the temporal consistency and the importance of early denoising steps is highly insightful and provides foundation for a DLM-specific acceleration technique. The ablation study confirming the necessity of the "Skipping Sparse" component is well-executed and validates the central hypothesis regarding diffusion step sensitivity. 

- State-of-the-Art Accuracy: the method achieves an almost lossless performance, competitive with other SOTA methods.

### Weaknesses
- *(General)* The paper is based mainly on very empirical observations concerning the distribution of attention scores. There is no validation of these properties through a quantitative analysis. 
- *(General)* The method per-se is not very novel, as it applies principles that are well known already in ARMs (top-k selection based on attention scores in blocks) to DLMs.
- *(General)* The observation that attention patterns are head-specific is not a new empirical finding in the literature, and was already noticed in ARMs (cf. "Duo Attention", ICLR 2025 for KV Cache compression ARMs). 
- *(Results)* The comparison against StreamingLLM looks ill-posed. Streaming LLM was developed for ARMs and is based on attention sinks in ARMs, but other work shows sinks in DLMs behave very differently (e.g. "Attention Sinks in Diffusion Language Models" Arxiv Preprint) so it is not surprising that it underperforms greatly in this scenario.
- *(Results)* The comparison with other baselines does not look very clear. Sparse methods are strongly dependent on the sparsity threshold, but the main results are displayed with a fixed chosen threshold for each method, making it difficult to understand the effective trade-off between accuracy and sparsity. A proper comparison should include different sparsity levels, which imply different FLOPs and memory footprint tradeoffs.
- *(Results)* Accuracy and Latency are evaluated against different sets of baselines which makes the results part a bit confusing.

### Questions
-  I think the choice of the baselines is not clear. This is a sparse attention method but the chosen baselines for efficiency include KVCaching methods for DLMs. However, KVCaching methods mainly aim at storing pre-computed values, while sparse attention aims at reducing computations. I think the paper would benefit from a clearer discussion of these two different aspects. Are they complementary, orthogonal ? How do they relate ? 
-  For the same reason, it is not clear to me if this method is reducing memory footprint other than improving latency ? In line 294 the authors claim that the method reduced memory footprint but not experiments or comparisons are shown in this direction.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SparseD, a sparse-attention framework specifically designed for diffusion language models (DLMs).  
The authors first present three empirical observations:  
1. Attention patterns are head-specific rather than uniform across heads.  
2. Within each head, the attention pattern is stable across denoising steps.  
3. Early denoising steps are critical, and applying sparsity too early degrades quality.  

Based on these insights, SparseD adopts a simple and practical design:  
- Skip sparsity during the early diffusion steps.  
- At a representative step, compute full attention once, select block-wise Top-ρ attention regions per head, and store these as reusable sparse masks.  
- Reuse the same mask across later steps, with isolated selection for prefill and generation tokens to better handle different distributions.  

Experiments on Dream-7B-Instruct and LLaDA-1.5 show that SparseD maintains almost identical accuracy  while achieving significant speedup over FlashAttention at 64 k context and up to 1024 diffusion steps.  
Ablation studies confirm that skipping early steps and reusing fixed patterns are both necessary for preserving model quality.  

Overall, SparseD offers a simple yet effective mechanism to accelerate DLM inference while retaining quality, contributing a valuable empirical and practical perspective to efficient diffusion-based text generation.

### Strengths
- Novel DLM-specific insights: Identifies and validates three key empirical properties (head-specific, step-stable, early-step-sensitive) that distinguish DLM attention from AR models.  
- Simplicity and practicality: The “skip-early + reuse mask + isolated selection” design is lightweight, easily implementable on top of Flash/FlexAttention, and hardware-friendly.  
- Strong empirical results: Demonstrates consistent latency improvements with negligible performance drop across multiple DLMs and tasks (MMLU, GSM8K, HumanEval, RULER).  
- Comprehensive ablations: Validates each component’s necessity and explores sensitivity to skip ratio and sparsity rate ρ.  
- Clear presentation: Includes concise pseudo-code and diagrams explaining mask generation and reuse; writing is crisp and reproducible.

### Weaknesses
1. Overhead accounting clarity.  The paper amortizes sparse-pattern pre-computation but does not quantify the actual wall-clock fraction of that cost across steps.  

2. Storage footprint not quantified.  The memory cost for storing per-head, per-layer sparse masks (after block-wise Top-ρ selection) is not reported, leaving uncertainty about scalability under 64 k contexts and many layers.  

3. Comparison to dynamic sparsity baselines.  The paper argues autoregressive (AR) sparsity patterns do not transfer to diffusion models, but does not include an adapted dynamic head-wise Top-ρ baseline for DLMs to confirm the advantage of static reuse.

### Questions
1. What is the per-layer/head memory footprint of the stored sparse masks under your standard setting?  
2. Have you tried computing masks at multiple checkpoints (e.g., twice in the diffusion process) and reusing them thereafter?  
3. If the prompt distribution shifts (e.g., code → narrative), do the pre-computed masks still perform well?  
4. How do latency and accuracy trade off with smaller or larger block sizes for a fixed ρ?  
5. Can heads with low cross-step similarity be selectively excluded from mask reuse to improve robustness?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **SparseD**, a **training-free, inference-time** sparse attention method for **Diffusion Language Models (DLMs)**. The key observations are: (1) **head-specific** attention patterns, (2) **high step-wise consistency** within a head, and (3) **early steps are sensitive** to sparsification. SparseD thus applies **full attention in early steps**, then **pre-computes head-specific sparse patterns once** using **block-wise top-ρ% selection with isolated selection for prefill vs. generation tokens**, and **reuses** the pattern thereafter. Experiments on Dream-7B-Instruct and LLaDA-1.5 show **up to 1.50× speedup** over FlashAttention at 64k context and 1,024 steps with negligible accuracy loss.

### Strengths
**Practical & training-free:** No retraining; one-shot sparse mask + reuse; early-step full attention preserves quality.  
- **Simple, hardware-friendly design:** Block-wise avg-pool → top-ρ% selection; **isolated selection** for prefill/generation.  
- **Consistent wins at scale:** Clear latency gains at long contexts; speedups increase with more steps due to reuse.

### Weaknesses
## Major
- **Theoretical under-pinning is light.** No bounds/guarantees on error from mask reuse or head-specific sparsity; argument is largely empirical.  
- **Model diversity is limited.** Only two DLMs (Dream-7B, LLaDA-1.5); generality to other DLM families or hybrid architectures is unproven.  
- **Early-step sensitivity.** Skip ratio is tuned (20–30%) rather than principled; transferability across schedules/steps remains unclear.

## Minor
- **Amortized precompute cost** is not fully quantified in wall-clock breakdowns, though reuse intuition is sound.  
- **Context vs. cache baselines.** A deeper head-to-head with cache-based methods across sequence lengths would help.  
- **AR-sparse comparisons.** Positioning vs. structured AR patterns (e.g., Longformer/BigBird) could be elaborated.

### Questions
1. **Skip ratio sensitivity:** How does accuracy/speed trade-off vary beyond 20–30% and across different total steps (e.g., 256 vs. 2048)? Any adaptive criterion?  
2. **Dynamic prompts/streaming:** If prefill/generation boundaries change mid-run, can the mask be updated without losing reuse benefits?  
3. **Precompute overhead:** What is the exact one-time cost for mask construction at 64k and how does it scale with heads/layers? (A timing table would help.)  
4. **Composability with caches:** Any synergy or redundancy when combining SparseD with dKV-Cache/Fast-dLLM for short vs. long contexts?  
5. **Very long contexts (128k+):** Does block granularity dominate selection quality; how does isolated selection behave there?

### Soundness
3

### Presentation
3

### Contribution
2
