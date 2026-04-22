# KVCompose: Efficient Structured KV Cache Compression with Composite Tokens

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Large language models (LLMs) rely on key-value (KV) caches for efficient autoregressive decoding; however, cache size grows linearly with context length and model depth, becoming a major bottleneck in long-context inference. Prior KV cache compression methods either enforce rigid heuristics, disrupt tensor layouts with per-attention-head variability, or require specialized compute kernels.
        
We propose a simple, yet effective, KV cache compression framework based on attention-guided, layer-adaptive composite tokens. Our method aggregates attention scores to estimate token importance, selects head-specific tokens independently, and aligns them into composite tokens that respect the uniform cache structure required by existing inference engines. A global allocation mechanism further adapts retention budgets across layers, assigning more capacity to layers with informative tokens. This approach achieves significant memory reduction while preserving accuracy, consistently outperforming prior structured and semi-structured methods. Crucially, our approach remains fully compatible with standard inference pipelines, offering a practical and scalable solution for efficient long-context LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents KVCompose, a structured KV-cache compression method for LLM inference. The method first aggregates attention across tasks or contexts to score token importance and then creates composite tokens by allowing each KV head to select its own most important original positions, aligning those selections into a shared per-layer sequence dimension, which preserves the standard dense tensor format. A layer-adaptive global budget distributes retention across layers by ranking composite-token scores. Experiments on RULER-4096 with LLaMA-3.1-8B, Qwen2.5-7B-Instruct, and Qwen3-14B show higher AUC than other baselines such as TOVA, SnapKV, and PyramidKV.

### Strengths
* KVCompose is evaluated across various model families and model sizes.
* Ablation studies examine key design choices, such as operator selections for aggregation, Aggtask, Agggroup, and Agghead. These analyses help clarify the contribution of each component.

### Weaknesses
* No runtime latency and memory usage comparison. Reporting compression ratios alone may not fully reflect efficiency improvements. Latency and memory statistics would provide a clearer picture of the practical benefits of KVCompose.

* The range of benchmarks is somewhat limited. KVCompose has been evaluated only on RULER-4096. Evaluation with longer input lengths on RULER and on other widely used benchmarks, such as LongBench, would provide a more comprehensive assessment of KVCompose's performance.

* What is the computational cost of the token selection process? For example, it would be helpful to discuss whether KVCompose remains compatible with Flash-Attention, given that attention computation is explicitly involved. It may be useful to compare or discuss KVCompose in relation to recent structured eviction approaches, such as “LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models” (ICML'25), particularly under accuracy-latency trade-offs.

### Questions
Please refer to the points raised in the Weaknesses section above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces KVCompose. The core innovation is the concept of "composite tokens." Recognizing that while the number of tokens must be uniform across heads in a layer for tensor compatibility, the identity of those tokens need not be, KVCompose allows each head to independently select its most important tokens based on aggregated attention scores.  This aims to combine the flexibility of per-head eviction (unstructured) with the hardware efficiency of dense tensor layouts (structured).

### Strengths
The concept of "composite tokens" is an interesting attempt to bridge the gap between unstructured eviction flexibility and the hardware efficiency required by standard inference engines.

### Weaknesses
**Q1** The central claim is "fully compatible with standard inference pipelines" and "Works without engine changes". However, . Modern LLMs, including Llama and Qwen (used in the experiments), rely heavily on positional embeddings, often Rotary Positional Embeddings (RoPE). RoPE applies rotations based on the absolute position of tokens. KVCompose's core mechanism constructs composite tokens by aligning K/V entries from different original positions across different heads. For example, the k-th composite token might contain the K/V from position 50 for Head 1 and position 1000 for Head 2. Standard inference engines assume that all K/V entries at tensor index k share the same original position, applying a uniform positional embedding for that index. KVCompose does not follow this assumption. 

**Q2** The claim that KVCompose is "Streaming ready" (Table 1) is misleading. The proposed scoring mechanism (Section 3.2) and global budget allocation (Eq. 22, 23) require processing the full context N to calculate attention statistics and determine budgets. This makes the approach suitable only for fixed-context compression (e.g., prompt compression during prefill) but not for true unbounded streaming applications (like StreamingLLM), which must make online eviction decisions as tokens arrive.

**Q3** I think the reorganization of the cache into composite tokens breaks the semantic and positional continuity of the cache structure. This fundamentally complicates or prevents the application of standard system optimizations crucial for efficient serving, such as prefix caching, cache sharing across requests, and speculative decoding. 


**Q4** The evaluation is severely limited in context length, failing to validate the method's effectiveness for long-context inference. Experiments are solely conducted on Ruler-4096. This is insufficient, as KV bottlenecks become critical at much longer sequences. The method must be evaluated on significantly longer contexts (e.g., 32k, 128k+) with limited token budgets 256-512.

**Q5** The evaluation lacks essential metrics and task diversity. Performance on tasks like long-document summarization or complex QA (e.g., LongBench2). The impact of eviction on coherence is critical in multi-turn scenarios, requiring evaluation on benchmarks such as SCBench.

**Q6.** The experimental evaluation is weak as they are ignoring recent work such as Lcache, RocketKV, ShadowKV, and seerAttention.
At least you should compare with RocketKV and ShadowKV in different token budgets and show how far you are from them.

**Q7.** The paper claims efficiency but provides no empirical analysis of the computational overhead. KVCompose introduces significant computation during the prefill phase: attention aggregation, sorting scores for every head (O(L * H_kv * N log N)), and indexed gathers to reorganize the cache. The latency and throughput impact of these operations must be measured and reported. 
Q8 The authors incorrectly state that the "memory required to store KV pairs" grows quadratically with context size. The KV cache memory footprint grows linearly (O(N)), as correctly noted later. The computational cost of prefill attention is quadratic (O(N^2)).
Q9. The memory calculation assumes FP32 (4 bytes). Modern deployments predominantly use FP16/BF16 (2 bytes); the calculation should reflect modern precision standards.

**Q10.** Minor: Conclusion and abstract should be in one paragraph.

### Questions
Please look at the weakness section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces KVCompose, a structured KV cache compression method. It uses attention-guided scoring to identify important tokens , allows each head to select different tokens while maintaining a uniform tensor structure (termed "composite tokens") , and adaptively allocates retention budgets across layers. The method is designed for compatibility with standard inference engines and demonstrates strong performance on the Ruler-4096 benchmark against other structured methods.

### Strengths
- The method intelligently allocates a global retention budget across layers based on token importance scores, rather than using fixed heuristics.

- Consistent performance gain is observed across the board.

### Weaknesses
- The method requires to collect attention scores, adding complexity. 

- Experiments mostly conducted on relatively small models. Would suggest the author conduct experiments on larger models and more challenging problems, like Qwen3-thinking and AIME.

### Questions
1. Can you clarify on the concept of "composite tokens"? What's its difference from conventional kv cache? What benefits it brings by viewing kv cache from different layer as a token? 

2. The idea for consturcting adaptive kv cache sounds very similar to "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes KVCompose, a structured KV cache compression method for long-context LLM inference, by pre-collecting attention distributions, calculating token importance, constructing "composite tokens," and using global sorting to achieve layer-adaptive retention budget allocation, while maintaining each layer's KV cache still as a uniform length, dense tensor. Token selection is fine-grained at the head dimension, yet implemented as structured eviction.

On long-context benchmarks such as Ruler-4096 and kvpress, compared with methods like TOVA, StreamingLLM, SnapKV, PyramidKV, and DuoAttention, KVCompose can achieve higher compression rates at the same accuracy tolerance and significantly outperforms in the AUC metric.

### Strengths
Structured and engineering-friendly

The design of composite tokens is natural and inspiring

Global layer-adaptive budget allocation

The method is conceptually clear and formally complete

### Weaknesses
The evaluation benchmarks are narrow and lack real-world tasks and diverse model settings. Currently, the main evaluations focus on Ruler-4096 / kvpress synthetic long-context tasks. These tasks are sensitive to patterns like needle in a haystack / aggregation, but they are insufficient to demonstrate generalizability to real multi-turn dialogues, code, long-form summarization, and RAG scenarios and robustness across more architectures.

Attention = Importance assumption lacks deeper analysis. The method is entirely based on aggregated attention scores, but related literature has pointed out that attention does not always reliably characterize causal contributions. The paper performs an ablation study on pooling methods, but it lacks comparison with gradient/importance estimation, also lacks discussion of failure cases or scenarios where attention is unreliable. Currently, the impression is that it works empirically on this benchmark, but the theoretical and empirical analysis is somewhat shallow.

Semantic and Interpretability Issues of Composite Tokens. Composite tokens combine different original positions from different heads into the same new position, essentially altering the alignment of each head. Although this does not mathematically violate the attention formula, it can make position-wise distribution interpretation difficult. Also, the behavior under positional encoding or certain structured prompt patterns opaque.

The evidence for being practically deployable and user-friendly is insufficient.  
The article repeatedly emphasizes that there is no need to customize the kernel and that it can directly integrate with vLLM / HF, which is a selling point; however, it lacks actual end-to-end throughput and latency data;  Memory vs QPS curves under real serving settings;  
System-level comparison with existing simple sliding window.

The method relies on a preconstructed task set T to collect attention. If the downstream task distribution changes, do we need to recollect attention and regenerate the mapping? For closed-source or commercial models, how can we efficiently collect enough attention to create a static configuration?  

Currently, it is more of a conceptual implementation compatible claim. We would like to see solid metrics supporting the conclusion of being practical & scalable.

### Questions
It is recommended to briefly explain in the main text the offline overhead involved in generating compressed configurations.

You can add a diagram for comparison traditional structured eviction, this allows readers to immediately see the structural differences. 

In the comparison methods, some, such as SnapKV / KVzip, are reconstruction-based / unstructured, which differs slightly from the goal of this work, which is to only perform structured eviction. 

It is recommended to release the implementation to lower the barrier for reproduction.

### Soundness
2

### Presentation
2

### Contribution
2
