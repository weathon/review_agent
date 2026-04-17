# AdaCache: Adaptive Caching and Context Augmentation for Efficient LLM Serving

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) significantly enhances Large Language Models by integrating external knowledge sources, but at the cost of substantial computational overhead from extended input sequences. 
Current RAG systems exhibit two fundamental inefficiencies: redundant processing of frequently retrieved text chunks across multiple queries, and uniform deep retrieval that over-provisions context regardless of query complexity.
We present AdaCache, an adaptive caching framework that addresses these limitations through dual optimization strategies. 
First, we introduce a cache-aware partial recomputation mechanism that profiles attention patterns to construct selective cache variants, enabling flexible reuse while preserving cross-chunk dependencies. 
Second, we develop adaptive context augmentation that dynamically determines optimal retrieval depth via lightweight confidence estimation, avoiding unnecessary overhead on simple queries.
Comprehensive experiments across diverse datasets and LLMs demonstrate that AdaCache delivers substantial improvements in Time-To-First-Token compared to state-of-the-art RAG caching systems, while preserving generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper points two major inefficiencies in current RAG systems: (1) cross-query context overlap, where identical text chunks from the external knowledge base are repeatedly retrieved across multiple user queries, and (2) a small fraction of text chunks dominate the retrieval requests. Based on these two findings, this paper aims to address both computational inefficiency simultaneously. AdaCache is then proposed to speed up the loading of contextual information. Experimental results on Llama 3 8B, Qwen 3 8B, and Qwen 3 4B show the proposed method is faster than CacheBlend and Prefix Caching in terms of TTFT with close correctness.

### Strengths
* This work is well-motivated and easy to follow. 

* Experiments support the effectiveness of the proposed AdaCache methodology.

### Weaknesses
* The current experiments are based on MMLU, MMLU-Pro, superGPQA, and TriviaQA, where the question can be answered by single factual chunk. Evaluations on more challenging RAG scenarios that require handling long and/or complex contextual information could be added for revealing the limitation of AdaCache. 

* This framework requires the manipulation of KV-cache and attention values, limiting its applications to open-weight models. 

* Comparison with [TurboRAG](https://arxiv.org/abs/2410.07590) could be added.

### Questions
* Can these method be seamlessly integrated with dense retrieval model, where the embedding of each chunk is also the cache? 

* The quality of Figure 3 could be further improved for readability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a efficient LLM Serving strage named AdaCache, which addresses two key inefficiencies of current RAG systems—redundant processing of frequent text chunks and uniform deep retrieval—via two optimization strategies. It uses cache-aware partial recomputation, which profiles attention patterns to build selective cache variants for flexible reuse while preserving cross-chunk dependencies. It also adopts adaptive context augmentation, dynamically determining optimal retrieval depth through lightweight confidence estimation to avoid unnecessary overhead on simple queries. Experiments across datasets and LLMs show AdaCache cuts Time-To-First-Token (TTFT) by 1.4x–5.0x versus state-of-the-art RAG caching systems, while maintaining generation quality.

### Strengths
Easy architecture and good  performance.

- Efficient Resource Utilization: By reusing cached KV states of frequent text chunks and avoiding over-retrieval for simple queries, AdaCache drastically reduces redundant computation. This translates to a 1.4x–5.0x TTFT reduction, greatly improving LLM serving throughput without wasting computational resources.​
- Quality Preservation: Its cache-aware partial recomputation preserves cross-chunk dependencies via attention pattern analysis and selective recomputation, while adaptive context augmentation uses a composite confidence metric to ensure sufficient context for accurate generation. Thus, it maintains generation quality even with efficiency gains.​
- Strong Adaptability: The three-tier cache hierarchy (hard prefix, soft prefix, independent cache) adapts to different prefix match conditions, and adaptive context augmentation adjusts retrieval depth based on query complexity. It works across diverse datasets and LLMs, showing broad applicability.

### Weaknesses
Some points can be optimized.

- Dependence on Attention Pattern Consistency.
The cache-aware recomputation relies on consistent layer-wise attention patterns (e.g., early localized attention, deep attention sinks). If LLMs have irregular attention distributions for certain tasks/domains, the cache variant design may fail, reducing reuse efficiency or accuracy.​
- Increasing System Complexity. 
The three-tier cache management, selective recomputation ratio tuning, and composite confidence metric (KL divergence + entropy) add complexity to system implementation and maintenance. This raises the barrier for integration into existing RAG pipelines, especially for teams with limited engineering resources.​
- Potential Overhead in Incremental Augmentation.
Adaptive context augmentation requires multiple forward passes (adding one chunk at a time) for queries needing deep retrieval. Though cached reuse mitigates this, it may still introduce more latency than static retrieval for complex queries requiring full top-k chunks, offsetting efficiency gains.​
- Reliance on Confidence Metric Tuning.
 The confidence metric (weighted KL divergence and entropy) needs optimization on validation sets. Poor tuning (e.g., misaligned weights) could lead to premature termination (insufficient context, lower accuracy) or unnecessary augmentation (wasted resources), making performance unstable across unseen data.

### Questions
Large models preformance can be provided for 14b/30b/72b and so on.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
AdaCache proposes a smarter, adaptive framework to speed up RAG in LLMs by solving two common system bottlenecks, repeatedly re-processing the same retrieved context chunks for multiple queries, and indiscriminately adding lots of context even when the query doesn’t need it. To fix this, AdaCache introduces a cache-aware partial recomputation system, profiling the attention flows so it only recalculates what’s actually needed and dynamically matches context prefixes and an adaptive context augmentation strategy based on confidence scores for each query, so simpler questions don’t get massive context of just because. In experiments across several standard datasets (MMLU, SuperGPQA, TriviaQA), AdaCache shows big wins in time-to-first-token , achieving up to 5x speedup over some baselines, all while keeping generation quality intact.

### Strengths
First to combine chunk-level attention analytics and hierarchical caching with dynamic, query-by-query adaptive retrieval depth. Demonstrates large and consistent speedups (up to 5x TTFT) across varied tasks, while maintaining or improving answer quality. Empirical coverage multiple models, datasets, real hardware setup, system code/pseudocode details included. Addresses two core RAG bottlenecks context reuse, variable context needs, rather than just one. Well-justified caching logic, tested across different chunk retrieval densities.

### Weaknesses
Technical sections like attention analysis, selective recomputation ratio could use more intuitive explanation for ML domain readers. Adaptive context augmentation relies on confidence measurements that may need more validation for edge-cases/unusual queries. Tested mainly on general QA/benchmark tasks. broader evaluation with multi-turn dialogue, enterprise data, multilinguality would add a lot. Efficiency gains are only as good as the popularity distribution of text chunks. if access is more uniform, the cache hit rates drop. Hardware setup is fairly high-end. results might differ for lower-resourced deployments.

### Questions
Could the adaptive context augmentation framework be extended to multi-hop QA or dialogue, with context dependencies beyond the initial retrieval? How robust is the cache-aware recomputation under rapidly changing corpora e.g., streaming or up-to-date knowledge bases, what is the trade off? Are there edge-case queries where adaptive context addition either under-fetches (missing context) or over-fetches (noise)? Any way you could provide results for wider domain/task evaluation beyond QA/reading comprehension?

### Soundness
3

### Presentation
4

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
The paper proposes AdaCache, a RAG-oriented serving framework that (i) builds a 3-tier, attention-aware KV cache (hard prefix, soft prefix, independent) and selectively recomputes only attention-critical tokens, and (ii) runs an adaptive context augmentation loop that stops adding retrieved chunks once the model is confident. On QA-style RAG benchmarks and several open LLMs, it reports 1.4×–5× TTFT improvements over prior caching systems while keeping accuracy.

### Strengths
- Tackles an actually painful RAG-serving bottleneck and is compatible with existing inference stacks.

- Sensible combination of attention-guided partial recomputation with incremental context — the two parts reinforce each other.

- Empirical results show clear win over prefix-only caching and even over CacheBlend in long-context regimes.

### Weaknesses
- Each core idea has close prior art: RAGCache / PromptCache / CacheBlend on the caching side, and Active / Adaptive / Bandit RAG on the context-length side; the paper currently oversells novelty. 

- Experiments do not ablate how much gain comes from "adaptive context" vs "better cache reuse", and they do not compare to adaptive-retrieval baselines.

- The proposed confidence computation and attention-sink profiling may add non-trivial overhead and may not generalize to other models/tasks.

### Questions
- Can you report an ablation that fixes the cache to CacheBlend-style selective recompute and only adds your ACA loop, to show the incremental benefit?

- How exactly is the soft-prefix cache key defined, and what prevents explosion in cache entries when the sink chunk moves?

- What is the actual extra latency of computing per-layer logits for the confidence score on your hardware, and is it still a win when retrieval is remote / high-latency?

### Soundness
3

### Presentation
3

### Contribution
3
