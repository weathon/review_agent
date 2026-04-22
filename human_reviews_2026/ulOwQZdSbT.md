# Hierarchical Adaptive Eviction for KV Cache Management in Multimodal Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
The integration of visual information into Large Language Models (LLMs) has enabled Multimodal LLMs (MLLMs), but the quadratic memory and computational costs of Transformer architectures remain a bottleneck. Existing KV cache eviction strategies fail to address the heterogeneous attention distributions between visual and text tokens, leading to suboptimal efficiency or degraded performance.
In this paper, we propose Hierarchical Adaptive Eviction (HAE), a KV cache eviction framework that optimizes text-visual token interaction in MLLMs by implementing Dual-Attention Pruning during pre-filling (leveraging visual token sparsity and attention variance) and a Dynamic Decoding Eviction Strategy (inspired by OS Recycle Bins) during decoding. 
HAE minimizes KV cache usage across layers, reduces computational overhead via index broadcasting, and theoretically ensures superior information integrity and lower error bounds compared to greedy strategies, enhancing efficiency in both comprehension and generation tasks. 
Empirically, HAE reduces KV-Cache memory by 41\% with minimal accuracy loss (0.3\% drop) in image understanding tasks and accelerates story generation inference by 1.5× while maintaining output quality on Phi3.5-Vision-Instruct model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Hierarchical Adaptive Eviction (HAE), a two-stage KV cache management framework for multimodal LLMs. HAE combines Dual-Attention Pruning (DAP) during the pre-filling stage and a  Dynamic Decoding Eviction Strategy (DDES) during decoding to reduce KV  redundancy while preserving accuracy. Experiments on LLaVA- and Phi-based models report about 40% KV-cache  reduction and 1.5× inference speed-up with minimal accuracy loss.

### Strengths
[1] Addresses a meaningful efficiency problem in multimodal LLMs.   
[2]  The hierarchical (prefill + decoding) strategy is conceptually intuitive  and simple to implement.   
[3] The method is training-free and demonstrates clear empirical   efficiency gains.

### Weaknesses
[1] The proposed Dual-Attention Pruning appears almost identical in  formulation and wording to the Dual-Attention Filtering in MustDrop (Liu et al., 2024b)—Equations (1)–(3) and the surrounding descriptions are very similar. This overlap raises serious questions about the originality and unique  contribution of the submission.    
[2] Fig. 2: It is unclear whether the variance is computed per layer or  accumulated across all layers. If it is the latter, could the higher variance of visual tokens simply  result from token mismatch across layers?   
[3] Equation (2): The equation selects visual tokens with smaller A_j,  yet the text says these tokens are retained. Shouldn’t low-attention tokens be evicted instead, or is the  inequality direction reversed?   
[4] Definition 2: The notation |V^p| = |V| - |C| introduces a set C that  has not been clearly defined. Is it a separate set used during decoding?  
[5] What does the summation over t represent in practice? Additionally, the symbol d appears both in the attention scaling  term (\sqrt{d}) and as the decoding buffer size in Definition 2. Are these referring to the same quantity, or should they be  distinct?
[6] For parameter analysis, only three thresholds (0.001, 0.0012,  0.0015) are tested, and the claim that 0.0015 is optimal is not  convincing.

### Questions
Section 2.1.1: “As shown in Figure 1” → should be “Figure 2.”   
Section 4.2: Missing or malformed references for TGIF, MSVD, and  MSRVT benchmarks   
Section 2.2: Inconsistent terminology — both “dual-attention” and  “double-attention” appear for DAP.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a critical bottleneck in Multimodal LLMs (MLLMs), the high memory/computational cost of KV caches, by proposing the Hierarchical Adaptive Eviction (HAE) framework. Its core strength lies in targeting the overlooked heterogeneous attention distributions between visual and text tokens, which existing single-modal eviction strategies fail to handle.

### Strengths
- HAE’s two-stage design (Dual-Attention Pruning for pre-filling, Dynamic Decoding Eviction Strategy for decoding) is logically motivated. The pre-filling stage leverages visual token sparsity in the first layer and broadcasts eviction indices to other layers, reducing redundant computations. The decoding stage uses an OS-inspired ``recycling bin'' to avoid hasty greedy eviction, balancing efficiency and information retention. 
- Theoretical analyses (Theorem 2.1 on cache integrity, Corollary 2.1 on error bounds) provide a basic mathematical foundation, and empirical results are compelling: 41% KV cache reduction with only 0.3% accuracy loss in image understanding, and 1.5× faster inference in story generation on Phi3.5-Vision-Instruct. Comparisons with baselines (e.g., MustDrop, H2O) in Tables 1–4 further validate HAE’s superiority in both performance and efficiency. Overall it's a sound paper.

### Weaknesses
First, the ablation study (Table 3) is relatively shallow—more experiments on how hyperparameters (e.g., threshold r, recycling bin size) affect performance would strengthen robustness. Second, while HAE outperforms training-free baselines, its comparison with trainable methods (e.g., Dynamic-LLaVA) is limited; the gap in MMB (64.0 vs. 65.4) needs more analysis on why trainable methods still have minor advantages. Third, the case demonstration in Appendix A.3.5 lacks qualitative details—clearer examples of how HAE preserves cross-modal coherence (vs. H2O/MustDrop) would improve readability.

Despite these flaws, HAE makes a valuable practical contribution to efficient MLLM inference.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Hierarchical Adaptive Eviction (HAE), a novel framework for efficient KV cache management in multimodal large language models (MLLMs). HAE consists of two key components: Dual-Attention KV Pruning, which identifies and shares important visual tokens across layers based on first-layer attention, and a Dynamic Decoding Eviction Strategy, which balances decoding latency and accuracy. Empirical results demonstrate that HAE accelerates inference while maintaining output quality on mainstream MLLMs.

### Strengths
1. Significant reduction in KV cache memory (up to 41%) with negligible accuracy loss for MLLMs.
2. The method is training-free, making it easy to adopt for existing models.
3. This paper is well-structured and the writing is easy to flow.

### Weaknesses
1. The paper would benefit from a brief explanation of MLLMs architecture, such as the Phi-3.5 Vision-Instruct model, to help readers better understand how these models compare to pure LLMs.
2. In section 2.1 observation, the paper lacks clear definitions for *sparsity rate* and *variance*. Meanwhile, the figure quality should be improved. 
3. Previous work [VLCache](https://arxiv.org/pdf/2410.23317) provides a comprehensive, layer-wise attention sparsity analysis for MLLMs. The authors appear to draw similar conclusions, so the novelty here should be clarified.
4. For the figure 1, the overall framework should be greatly improved to help understanding. 
5. Lack of end-to-end latency, prefill time, decoding time analysis, and additional overhead. 
6. In section 4.3 Ablation study, it would be better to profile the H2O to demonstrate the performance bottle neck and analyze where you harvest the performance benefits based on HAE. 
7. H2O is not a particularly strong baseline. Recent advanced methods, such as [SnapKV](https://arxiv.org/abs/2404.14469), [AdaKV](https://arxiv.org/abs/2407.11550) should be included for comparison.

### Questions
None

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Hierarchical Adaptive Eviction (HAE), a training-free framework for efficient KV cache management in Multimodal Large Language Models (MLLMs) that addresses the memory bottleneck caused by processing both visual and textual tokens. The key innovation lies in recognizing that visual and text tokens exhibit different attention distribution patterns, leading to a two-stage approach: Dual-Attention Pruning (DAP) during pre-filling that identifies and removes redundant visual tokens in the first layer then broadcasts these decisions across all layers, and Dynamic Decoding Eviction Strategy (DDES) during generation that uses a "recycling bin" mechanism to defer eviction decisions rather than greedily removing tokens immediately. The authors provide theoretical guarantees for information integrity and bounded error propagation, demonstrating that their buffered approach achieves tighter error bounds than greedy methods. Empirically, HAE reduces KV cache memory by 41-47% with minimal accuracy loss (0.3% drop) on image understanding benchmarks like GQA and ScienceQA using LLaVA-1.5-7B and Phi3.5-Vision-Instruct models, while achieving 1.5× speedup on image-based story generation tasks, outperforming existing methods like H2O, MustDrop, and SparseVLM that either focus only on single modalities or require additional training.

### Strengths
1. The paper introduces a hierarchical approach that explicitly addresses the heterogeneous attention patterns between visual and textual tokens in MLLMs. The observation that visual tokens exhibit higher sparsity than text tokens, particularly in early layers, provides empirical justification for developing different methods. The Dual-Attention Pruning mechanism exploits this by computing eviction decisions only in the first layer and broadcasting them across the network, reducing both memory footprint and computational overhead. 

2. The paper also provides theoretical analysis through Theorem 2.1 and Corollary 2.1, establishing bounds on information loss. This demonstrates that the Dynamic Decoding Eviction Strategy achieves tighter error bounds compared to greedy eviction methods. This theoretical result distinguishes HAE from heuristic-based approaches and provides confidence in its robustness across different scenarios.

3. The evaluation covers both multimodal understanding tasks (GQA, MMB, MMMU, ScienceQA) and generation tasks (Seed-Story), demonstrating the method's versatility, using wide range of MLLMs e.g., LLaVA-1.5-7B, Phi3.5-Vision-Instruct, Video-LLaVA. The paper also compares against several baselines including both training-free (H2O, MustDrop, SparseVLM) and trainable methods (Dynamic-LLaVA, VoCo-LLaMA). HAE achieves competitive or better performance to trainable methods while being training-free. Additionally, the ablation studies systematically isolate the contributions of pre-filling versus decoding stage optimizations.

### Weaknesses
1. Theorem 2.1 assumes exponential decay with constant rate $\lambda$ (line 687), which may not reflect actual attention dynamics in transformers. The worst-case analysis provides loose bounds that may not be tight in practice. The proof relies on geometric series summation that assumes independent evictions, but KV eviction decisions are sequentially dependent. The gap between theoretical guarantees and empirical performance is not discussed.

2. Broadcasting first-layer eviction decisions wto all subsequent layers assumes uniform redundancy patterns across the network, and this seems overly simplistic. Figure 5 shows 80-90% overlap, and this still means 10-20% of tokens are incorrectly handled in deeper layers. It would be better if the paper could explore layer-specific or layer-group-specific strategies. Different layers may focus on different visual aspects (edges vs. semantics), making uniform eviction suboptimal.

3. The proposed hierarchical cache management system and a Dual-Attention Pruning (DAP) introduce additional overheads. The contribution of the paper would be stronger if a detailed, worst-case analysis of the overheads (e.g., dual-attention score computation, index broadcasting latency, recycling bin maintenance cost, and memory access patterns for non-contiguous KV cache operations) is provided.

### Questions
Minor typos:
- line 403, references are missing: “TGIF ??, MSVD ??, and MSRVT ??.”
- line 712, “log(1 − ϵ) < 0” -> “log(1 − \lambda) < 0”

### Soundness
2

### Presentation
1

### Contribution
2
