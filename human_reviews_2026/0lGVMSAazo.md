# One Stone Three Birds: Training-free Core-context-aware Attention for Efficient LLM Prefilling, Decoding, and KV Caching

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
The quadratic computational complexity of self-attention poses a critical bottleneck for large language models (LLMs) processing ultra-long contexts. While various sparse attention and KV cache compression methods have been proposed to improve efficiency, they often suffer from limitations such as reliance on fixed patterns, inability to handle both prefilling and decoding stages, or the requirement for additional training. In this paper, we propose Training-free Core-context-aware Attention (TFCA-Attention), a training-free sparse attention mechanism that achieves “one stone three birds”: it unifies acceleration for prefilling, decoding, and KV cache reduction through a consistent sparsity mechanism. TFCA- Attention features an offline calibration phase that determines head-specific sparsity budgets and an online token selection phase that adaptively retains core context tokens using a lightweight redundancy metric. Theoretically, we provide a bounded approximation error guarantee, ensuring long context modeling accuracy. Extensive experiments demonstrate that TFCA-Attention achieves a 2.8× speedup and reduces KV cache by 61% at 128K context length while maintaining performance comparable to full attention across various benchmarks, offering a practical plug-and-play solution for efficient long-context inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TFCA-Attention, a training-free dynamic sparse attention mechanism designed to solve the $O(L^2)$ computational bottleneck for long-context LLMs. The authors identify that existing methods are "siloed," accelerating either prefilling or decoding, but not both. TFCA-Attention proposes a unified "one stone three birds" solution to simultaneously accelerate prefilling, speed up decoding, and reduce KV cache using a single, consistent sparsity mechanism. This is achieved through a two-phase approach: a one-time Offline Calibration determines head-specific sparsity budgets based on each head's redundancy level, and an Online Selection phase uses a lightweight metric to dynamically select "core context" tokens, adapting to the input. Experiments demonstrate that at 128K context, this method achieves a 2.8x speedup and a 61% KV cache reduction while maintaining performance comparable to full attention.

### Strengths
1. The proposed method is "plug-and-play", requiring no architectural modifications or retraining, making it easy to apply to existing LLMs.
2. The paper demonstrates significant gains, including a 2.8x prefill speedup, a 2.1x decoding speedup, and a 61% KV cache reduction at 128K context length.
3. The proposed method maintains performance comparable to the original full-attention models across diverse long-context benchmarks, such as LongBench-E and RULER.
4. The approach is supported by a theoretical guarantee proving that its approximation error is bounded and controllable.

### Weaknesses
1. The claim that "current approaches are siloed, accelerating only one stage" is an overstatement. Several existing works [1,2,3] have demonstrated acceleration in both the prefill and decoding stages. This overstatement should be revised, and appropriate citations should be added.

2. The ablation or comparison against dense attention kernels seems to be missing, (perhaps the “vanilla self-attention” in Fig. 3 may correspond to the dense kernel -- e.g., FlashAttention2). If not, could you report the speedup ratio of the proposed method compared with FlashAttention-2 across different sparsity ratios? 

3. [Minor] The legend in Figure 3 labels the method as "READ-LLM (Ours)", which is inconsistent with the "TFCA-Attention" name used throughout the paper

### Reference
[1] https://arxiv.org/pdf/2502.11089

[2] https://arxiv.org/abs/2502.14866

[3] https://arxiv.org/abs/2410.13276 & https://arxiv.org/pdf/2506.08889

### Questions
Please refer to the Weaknesses section.

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
The paper proposes TFCA-Attention, a training-free sparse attention mechanism that accelerates prefill, reduces decoding latency, and reduces KV cache size in a unified way. The method consists of two stages: an offline calibration stage that estimates a token budget for each head, and an inference stage that dynamically selects core tokens based on a lightweight importance score computed from the last-token query. The approach assigns different budgets to different heads, preserves a fixed local window for short-range dependencies, and prunes the remaining KV tokens to stay within those budgets. This method achieves large speedups on long-context benchmarks while maintaining comparable accuracy to full attention.

### Strengths
1. Instead of using a single global sparsity pattern, the method adapts budgets at the granularity of attention heads and context blocks. This is reasonable: different heads specialize differently, so a non-uniform allocation can preserve important heads and tokens better than a uniform top-k policy.
2. The method does not require model fine-tuning or architectural modifications, and the offline calibration step is lightweight.
3. The paper reports latency, memory, and accuracy on a wide range of long-context tasks. The evaluations are extensive and demonstrate consistent improvements over dense attention under various sequence lengths.

### Weaknesses
1. **Novelty seems limited:** 
- The paper claims to be the “first unified method” to optimize prefill, decoding, and KV cache usage, but this claim overlooks DuoAttention, which already addresses the same three aspects within a single sparsity design. Like TFCA, DuoAttention also uses an offline calibration process to determine per-head retention behavior and applies that configuration during both prefill and decoding.
- The main difference with DuoAttention is the granularity: DuoAttention uses coarser budget levels (two types of heads), while TFCA supports multi-level discretization per head. This refinement is worthwhile, but it does not constitute a fundamentally new paradigm. The paper should acknowledge this prior work and clearly differentiate itself. Without such a comparison, the contribution risks being incremental.

2. **Latency evaluation:**
- It would also be appropriate to include Duo Attention’s latency (prefill and decode) as a baseline.
- The latency tables (e.g., Table 2) compare TFCA vs. full attention, but it is unclear what full attention implementation is being used. Is the dense baseline using highly optimized kernels such as FlashAttention / FlashInfer? These kernels are now standard practice for speeding up attention on modern GPUs. If FlashAttention / FlashInfer is used, that should be clearly stated in the paper so readers can judge fairness. If it is not used, then a FlashAttention/FlashInfer timing baseline should be added.

### Questions
1. Could the authors clarify how TFCA compares to Duo Attention in terms of accuracy, latency (prefill and decoding), and memory usage?
2. Is the “full attention” baseline in the latency tables implemented with optimized kernels such as FlashAttention or FlashInfer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TFCA-Attention, a training-free dynamic sparse attention mechanism that unifies acceleration for prefilling, decoding, and KV cache compression. The method determines head-specific sparsity configurations offline and performs lightweight context-aware token selection online. The approach provides a theoretical error bound and demonstrates up to 2.8× speedup and 61% KV cache reduction on long-context benchmarks without sacrificing accuracy.

### Strengths
1. The proposed method is training-free, facilitating straightforward integration with existing models.
2. The analysis of attention scores presented in Figure 1 is insightful and clearly articulated.
3. The experimental evaluation is comprehensive, encompassing multiple large language models (LLMs), such as LLaMA3.1 and Qwen2.5, across both long- and short-context benchmarks.
4. The paper is well-written, with clear figures and thorough documentation of reproducibility details.

### Weaknesses
1. The claim that prior work accelerates either only the prefilling or only the decoding stage is an overstatement. Several existing methods, including MoBA (Mixture of Block Attention for Long-Context LLMs), Native Sparse Attention, DuoAttention, LServe, and RocketKV, accelerate both stages to varying extents. This body of work should be acknowledged.
2. The explanation provided for point (6)—that "it increases as attention becomes more concentrated on a few tokens"—is inaccurate and requires revision.
3. The token-selection strategy relies exclusively on the final query token. This dependency could degrade performance in multi-turn conversational contexts, a potential limitation that is not addressed in the paper.

### Questions
1. How can the proposed method be integrated with Grouped-Query Attention (GQA) or Multi-Query Attention (MQA)? Can you give a detailed explanation?
2. What is the method's performance on multi-turn conversation benchmarks?
3. Could you provide a detailed cost analysis of the computational overhead, separating the token selection and attention calculation costs? This breakdown would help clarify how the end-to-end speedup is influenced by the generation length during the decoding stage.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles the well-known inference bottleneck in long-context LLMs. The authors' main premise is that current methods are fragmented, only speeding up either prefill (via sparse attention) or decoding (via KV cache compression). They propose TFCA-Attention as a "one stone three birds" solution to do all three in a unified, training-free way. The method works in two stages: an offline calibration to set head-specific sparsity budgets, and an online phase that uses these budgets to dynamically select a 'core' set of tokens. The authors claim significant speedups (2.8x) and memory reduction (61%) at 128K context with no performance loss.

### Strengths
The paper is well-motivated and tackles a very important, practical problem. The goal of a unified, training-free framework for prefill, decoding, and KV cache reduction is highly significant. The "plug-and-play" aspect is a clear strength, as it avoids any need for expensive model retraining.

### Weaknesses
Despite the promising goal, the paper suffers from several major weaknesses, one of which I believe is a fundamental flaw in the method itself.

My primary issue is with the online methodology. The entire 'context-aware' global token selection for the prefill stage hinges on an importance score $s$. This score is calculated using only the query vector of the very last token ($Q_{L,:}$) to score all keys in the sequence. The justification for this—that the last token 'encapsulates summarizing information'—is a very weak heuristic. It's difficult to believe that this single query can act as a reliable proxy for the attention needs of all $L$ tokens in the sequence. This assumption seems fundamentally unsound and undermines the 'core-context-aware' claim.

Second, the 'unified' contribution is, in my view, significantly overstated. The paper criticizes 'ad-hoc integration' of separate prefill and decode-only methods. Yet, the proposed method is an ad-hoc integration. Section 4.2 describes a sparse attention method for prefill, and the last paragraph of that section describes a separate block-wise KV eviction strategy for decoding. The only 'unification' is the re-use of the offline sparsity budget $P^*$. This is not the deep, novel integration the paper claims, and it doesn't seem to solve any 'compatibility complexity' that wouldn't be present in other combined approaches.

This leads to the most critical experimental omission. The paper claims its unified approach is superior to combining existing methods (e.g., MInference + SnapKV). But Table 5 only shows an accuracy comparison, where TFCA-Attention is only marginally better (e.g., 52.44 vs. 52.23). The paper completely fails to provide the end-to-end latency, speedup, and memory benchmarks for these combined methods. This is the single most important experiment needed to back up the central 'one stone three birds' claim. Without it, we have no evidence that TFCA-Attention is actually better; the ad-hoc combinations might be much more efficient.

Finally, on a presentation note, the paper is confusing. The method is called 'TFCA-Attention' throughout the entire text, but in Figure 3—the main results graph—it's labeled 'READ-LLM (Ours)'. This is a sloppy and unexplained inconsistency.

### Questions
1. Please provide a much stronger justification for using only the last token's query ($Q_{L,:}$) to determine global importance for the entire prefill stage. Why should this single query be a good proxy for all $L$ queries?

2. To support the paper's central claim, the authors must provide the end-to-end latency and memory benchmarks (prefill + decode) for the combined methods in Table 5 (MInference + SnapKV, etc.). Why was this critical comparison omitted?

3. What is READ-LLM in Figure 3? Why is the method named differently in the text (TFCA-Attention) and the main results figure?

4. Can you ablate the choice of the Simpson concentration index for the redundancy metric $h_j$? What is its advantage over simpler, more common metrics like block-wise entropy?

### Soundness
2

### Presentation
2

### Contribution
2
