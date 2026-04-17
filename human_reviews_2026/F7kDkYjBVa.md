# RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) have achieved remarkable progress in long-context generation. As the context length increases, the Key--Value (KV) cache requires the GPU memory with a linear growth rate. KV cache compression is treated as a promising method to reduce the memory usage by permanently discarding a large portion of unimportant KV pairs, but at the expense of inference accuracy. On the other hand, retrieval-based methods employ the CPU memory to store the full KV cache and compute the attention via expensive CPU-GPU I/O, which keeps the accuracy but suffers from huge inference latency. To address these issues, we propose a new inference framework called RACC, which combines both compression based methods and retrieval based methods. To be specific, we employ the KV cache compression method to maintain a high-quality KV cache in the GPU memory, while sotring all the KV pairs evicted by the compression method. In addition, efficient and accurate retrieval conducted on the CPU side finds out important tokens for the one being generated, which is then concatenated with those KV cached in the GPU memory for accurate generation. Extensive experiments demonstrate that RACC achieves near-lossless inference while using only 15\% of the original KV cache. Moreover, its combination with prefill-only compression methods improves generation accuracy by 3--10\%. Our code is publicly available at \url{https://anonymous.4open.science/r/CDKEY/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation proposes a hybrid framework to address the trade-off between accuracy and efficiency in long-context LLM inference. RACC maintains a compressed subset of important KV pairs on GPU while asynchronously retrieving potentially useful evicted tokens from CPU memory using maximum inner product search. Experimental results show that RACC achieves near-lossless accuracy with only 15% of the original KV cache, reducing latency to near full-cache levels and improving accuracy by 3–10% when combined with existing compression methods.

### Strengths
1. Clear motivation and solid problem framing: The paper addresses a critical and practical bottleneck in long-context LLM inference. The motivation is well grounded and directly linked to real-world deployment constraints.
2. Strong empirical performance: RACC consistently achieves near-lossless generation while using only ~15% of the original KV cache, with accuracy improvements of 3–10% when integrated with existing compression methods.
3. Theoretical grounding: The reduction of MIPS retrieval to nearest-neighbor search provides a clear mathematical foundation for the retrieval design, and the analysis of different retrieval indices adds depth

### Weaknesses
1. Limited novelty beyond integration: The core idea is conceptually straightforward and has been implicitly discussed in prior works (e.g., PQCache with approximate retrieval, SnapKV with selective KV maintenance)
2. Unclear retrieval efficiency vs. accuracy trade-off: The retrieval module relies on approximate MIPS, but the paper does not quantify how retrieval accuracy impacts generation quality.
3. Lack of principled design for hybrid coordination: The interaction between the compression module and retrieval module appears heuristic rather than theoretically justified or learned.

### Questions
1. How does RACC fundamentally differ from simply adding a retrieval buffer on top of an existing compression method?
2. Would the improvement still hold if retrieval results are imperfect or if the vector database grows large?
3. Can the authors clarify how their asynchronous retrieval differs from prefetch or speculative retrieval mechanisms proposed before?
4. Is there any formal analysis or ablation explaining why the specific design heuristics are optimal or robust?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RACC, a hybrid KV-cache management framework, which combines a compression-based method with a retrieval-based method. The core innovation lies in its asynchronous design, leveraging previously generated tokens as queries for retrieval to avoid stalling the GPU, and its scenario-aware compression strategies. The paper demonstrates that RACC can maintain near-lossless accuracy using only 15% of the original KV cache and can be orthogonally applied to other compression methods to boost their performance.

### Strengths
1. The core idea of augmenting a fast-but-lossy compression method with a slow-but-accurate retrieval method is effective and the empirical results support this, showing a superior accuracy-latency trade-off compared to compression-only baselines.
2. The decision to use the previously generated token as a query for retrieval, rather than the current one, is a novel insight. This asynchronous design is the key to hiding the retrieval latency and is a major differentiator from synchronous retrieval-based methods.
3. RACC's retrieval module can be plugged into existing compression methods like SnapKV and PyramidKV to improve their accuracy.
4. The paper provides a solid evaluation of latency, including Time-To-First-Token (TTFT) and Overall Generation Latency (OGL) across different output lengths (Table 1).

### Weaknesses
1. The chosen baselines (Scope, CakeKV, SqueezeAttn) are not the strongest or most relevant points of comparison. The field has recently seen several high-profile works that are directly comparable to RACC. 
2. Section 4.2 discusses MIPS and its transformation to NNS, but it is never explicitly stated which specific retrieval algorithm is used in the main RACC experiments. The extensive evaluation of Flat, IVFPQ, and HNSW in Appendix D lacks full method description. 
3. In Table 4, the metric is simply "Score." Without a definition (e.g., ROUGE, BLEU, a task-specific metric), these numbers are uninterpretable.
4. The paper does not detail the synchronization mechanism between the CPU and GPU. How many tokens ahead is the retrieval? What happens if the retrieval for token t isn't complete by the time token t+1 is being generated? 
5. The paper does not include an ablation study to isolate the contribution of the retrieval component. How much does accuracy improve when adding retrieval to a base compression method? What is the latency overhead of the retrieval module in isolation?
6. The performance of RACC likely depends on the compression ratio α, the grouping frequency n, and the retrieval budget β. A sensitivity analysis on these key hyperparameters is missing.
7. The introduction and related work sections do a poor job of motivating why this specific hybrid approach is needed now. The critique of existing methods is surface-level.

### Questions
The same as the weakness.

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
5

### Summary
This paper introduces RACC, a novel hybrid framework to address the significant KV cache bottleneck in long-context LLM inference. The method combines GPU-side KV cache compression with an efficient, asynchronous retrieval mechanism. RACC evicts KVs to a CPU-managed vector index and retrieves them asynchronously. This retrieval is guided by queries from previously generated tokens, leveraging the observation that adjacent tokens share similar attention patterns. The goal is to achieve the near-lossless accuracy of full-context models while maintaining the low-latency performance of compression-only methods.

### Strengths
1) The combination of compression and retrieval approaches is conceptually sound, addressing the limitations of each individual approach (accuracy degradation in compression methods and latency increase in retrieval methods).

2) The asynchronous CPU-side retrieval mechanism is clever, leveraging the similarity between adjacent tokens' attention distributions to avoid blocking GPU inference, thus maintaining low latency.

3) RACC demonstrates impressive performance, maintaining accuracy close to full-cache methods while using only 15% of the original KV cache.

4) The framework shows good compatibility with existing compression methods like SnapKV and PyramidKV, enhancing their performance when integrated with RACC.

### Weaknesses
1. Inadequate Baseline Comparisons

The paper fails to include critical baselines necessary to substantiate its claims. It positions RACC as a superior hybrid solution over pure compression and pure retrieval methods, yet experimental comparisons (Tables 1–3) only cover compression-based methods (e.g., Scope, CakeKV, SqueezeAttn) and omit direct comparisons with retrieval-based methods (e.g., PQCache) discussed in the related work—leaving the claim that RACC avoids "huge inference latency" of retrieval methods unsubstantiated.

Additionally, RACC is not thoroughly compared against other hybrid memory management approaches for LLMs, such as recent works exploring CPU-GPU collaborative inference strategies, which would serve as more appropriate baselines for its hybrid design.

2. Limited Experimental Design and Generalization

Narrow model scope: Experiments are restricted to only two 8B-scale models (LLAMA-3.1-8B, Mistral-7B-v0.2) with no validation on larger models (e.g., 70B-scale or above) where memory constraints are more severe—undermining claims of generality.

Unjustified compression ratio: A single compression ratio (15%) is used across all experiments, with no justification for this specific value or exploration of performance trends under different compression ratios.

Missing ablation for key components: The paper introduces three adaptive compression strategies (LISO, SILO, LILO) for different tasks but lacks a clear ablation study. It neither specifies which strategy was applied to each benchmark (LongGenBench, LongProc) nor provides comparative analysis to demonstrate why this adaptive approach outperforms a single naive strategy (e.g., using SILO for all tasks)—failing to validate this design choice.

3. Insufficient Methodological Novelty, Details, and Clarity

Limited novelty: RACC essentially combines two existing approaches (compression and retrieval). While its asynchronous retrieval mechanism is clever, it builds directly on established vector search techniques and well-documented attention distribution properties from prior work, offering limited novel contributions.

Missing critical details: Key technical details are insufficiently described, including the CPU-GPU communication mechanism, vector index construction process, and overhead of building/maintaining the vector index (a gap尤为 notable given the paper’s focus on inference performance).

Poor component placement: The "voting mechanism"—a core part of the compression module—is relegated to Appendix B. Given its importance to RACC’s operation, a complete description should be integrated into the main body (Section 4.3) to improve clarity.

4. Incomplete Latency Analysis

The paper’s latency analysis is inadequate, as it does not fully account for potential variability in CPU-GPU communication overhead under different workloads or hardware configurations. While it claims minimal latency impact, it fails to investigate worst-case scenarios or performance degradation under resource contention (e.g., when batch size > 1).

5. Unaddressed Scalability and Applicability Limitations

RACC’s scalability to broader scenarios is not addressed: the paper does not adequately explain how it would perform with longer contexts (>64K tokens) or scale to much larger models—key use cases where memory-efficient solutions are most critical.

### Questions
see weakness part

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents RACC, a hybrid KV cache framework that addresses the accuracy-latency trade-off in long-context inference by combining GPU-side compression with CPU-side retrieval. Its core innovation is an asynchronous retrieval mechanism that uses previously generated tokens as queries, leveraging their attention similarity to current tokens to run retrieval in parallel with GPU decoding and avoid I/O bottlenecks. RACC achieves near-lossless accuracy using only 15% of the original KV cache, demonstrating superior performance over existing compression-only and retrieval-only methods.

### Strengths
1. The core innovation lies in using query vectors from previously generated tokens to conduct asynchronous CPU-side retrieval. This design adeptly circumvents the severe latency bottleneck caused by synchronous I/O, which plagues traditional retrieval-based methods.

2. RACC successfully breaks the longstanding compromise between inference efficiency and generation accuracy. It retains the high efficiency and low memory footprint of compression on the GPU while using asynchronous retrieval to "recall" crucial evicted information, thus maintaining high fidelity.

3. The experimental results are compelling. RACC delivers near-lossless performance with only 15% of the KV cache. Under the same compression ratio, it significantly outperforms SOTA baselines like SqueezeAttn, Scope, and CakeKV in generation accuracy on benchmarks like LongGenBench and LongProc.

### Weaknesses
1. High Dependence on the Core Observation: The method's efficacy hinges entirely on the observation that "adjacent tokens share similar attention patterns". For tasks requiring rapid, sharp context shifts or the precise retrieval of a single, non-redundant fact from the distant past, using a stale query ($q_{t-n}$) may fail to retrieve the exact KV pairs needed by the current token ($q_t$)

2. Increased System Complexity: The framework introduces non-trivial system complexity. It requires maintaining a persistent vector database on the CPU and managing sophisticated, error-free asynchronous communication and data transfer between the CPU and GPU.

3. Unaddressed Dynamic Indexing Costs: For SILO and LILO scenarios, the paper states the CPU-side vector index is "incrementally updated". However, the paper does not analyze the computational overhead of these dynamic index updates or the potential for the MIPS search itself to become a new bottleneck as the index grows to manage millions of evicted tokens.

### Questions
1. Query Lag Sensitivity: What is the exact lag used in the experiments between the query token $q$ and the token $t$ currently being generated (e.g., is it $q_{t-1}$ or $q_{t-n}$)? How sensitive is RACC's performance (both accuracy and latency) to this "query staleness" or lag?

2. Dynamic Index Bottleneck: In LILO scenarios , the CPU index must grow continuously as tokens are evicted. At what point (e.g., after how many generated tokens) does the CPU-side MIPS search and/or incremental index updating become the new performance bottleneck, potentially negating the latency gains from asynchronous execution?

3. Grouping Compression Parameters: The paper introduces a "grouping compression strategy" (compressing every n new tokens) to reduce computational overhead. What value of n was used in the experiments? How was the trade-off between reducing compression frequency (a larger n) and ensuring the GPU cache remains sufficiently relevant (a smaller n) determined?

### Soundness
4

### Presentation
3

### Contribution
3
