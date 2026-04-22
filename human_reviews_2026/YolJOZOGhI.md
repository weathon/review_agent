# KVCache-Centric Memory for LLM Agents

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
LLM agents in complex, long-horizon workflows are constrained by the model’s context window. Current plaintext-based memory systems suffer from unstable retrieval accuracy and disrupt prefix caching, harming both performance and efficiency. 
We propose MemArt, a novel memory paradigm that operates directly within the LLM-native format: the key-value (KV) cache. Instead of using plaintext, MemArt stores conversational turns as reusable KV cache blocks and retrieves relevant memories by computing attention scores in latent space. To enable accurate and efficient retrieval, we develop a multi-token aggregation retrieval strategy that uses compressed keys for efficient KV selection and a decoupled position encoding mechanism to ensure retrieved blocks are safely and coherently reused. On the LoCoMo benchmark, MemArt improves accuracy by over 11\% (up to 39.4\%) compared to state-of-the-art plaintext-based memory methods, nearly matching full-context performance. Critically, it achieves this while reducing prefill tokens by over two orders of magnitude (91-135$\times$), representing a significant leap forward for building powerful and efficient long-context agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MemArt, a memory system that stores and retrieves past context directly in KV-cache space instead of plaintext. It introduces: (i) AABB-based key compression to index each KV block with min-max vectors; (ii) multi-token aggregation retrieval that scores blocks using normalized per-token relevance and then aggregates across tokens; and (iii) decoupled positional encoding that strips RoPE at storage time and re-embeds positions at reuse time to avoid positional mismatch.

### Strengths
MemArt’s KV-native retrieval aligns with the attention mechanism and removes prompt concatenation, which can avoid retrieval drift and preserve prefix-caching efficiency. The AABB compression is simple and allows a fast coarse filter before fine attention. The multi-token aggregation is well-motivated and the ablations (Softmax vs reciprocal-rank; Sum vs Max; block size) help isolate what matters. The decoupled positional encoding is clearly described and addresses long-context reuse failure modes.

### Weaknesses
1. Model coverage is limited for a 2025–2026 submission. Results are only on LLaMA-3.1-8B-Instruct and Qwen-2.5-7B-Instruct, with no newer families and no size sweep to show scaling trends.

2. Baseline breadth is narrow. The method is compared to plaintext memory systems (Mem0, Zep), but there is no head-to-head with cache-centric and dynamic sparse attention systems that also select KV blocks (for example Arkvale, InfLLM, Quest, NSA), even though they are discussed.

3. Scope of datasets is narrow. All main results are on LoCoMo; there is no test on other long-horizon agent traces

### Questions
Because MemArt stores and retrieves latent KV-cache tensors instead of text, the retrieved memory is not human-interpretable. This makes it difficult to verify what information is actually being recalled or whether retrieval errors occur. Can you provide any mechanism to improve interpretability — for example, storing lightweight metadata (token spans, summaries, or embeddings) alongside each KV block, or decoding retrieved KV tensors back into approximate text via the model’s unembedding layer? Additionally, can you report any qualitative analysis showing that the retrieved memories correspond to semantically relevant parts of the dialogue? Without such transparency, it is hard to assess whether MemArt retrieves correct information or merely benefits from implicit correlations.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
MemArt reframes agent memory as KVCache-centric rather than plaintext. The paper shows how MemArt stores past turns as reusable KV blocks and retrieves them by computing attention in latent space. This avoids retrieval drift and preserving prefix-caching benefits. 
The system comprises (a) AABB-based key compression for each fixed-size block, (b) multi-token aggregation retrieval that scores blocks against all query tokens, (c) decoupled positional encoding that re-embeds retrieved KV without stale RoPE offsets, and (d) a managed memory pool. Compression represents each block with coordinate-wise minima and maxima, enabling fast coarse filtering. Notably, the relevance for a single token is upper-bounded via the dot-product with those bounds.  For multi-token prompts, scores are first normalized per token across all blocks and then aggregated to select top-k blocks in chronological order. Retrieved blocks are concatenated with the current KV and re-encoded with unified positions, ensuring coherent attention within the current window without exceeding native limits.

### Strengths
MemArt's design is quite interestng. It reframes memory to be KVCache-centric with latent-space retrieval, decoupled positional encoding, and lightweight AABB key compression with multi-token aggregation. This yields a model-agnostic, plug-and-play system.

### Weaknesses
System-wise, memory-pool I/O can add non-trivial latency, and safe reuse critically depends on decoupled positional re-embedding. The issue is that, without it, long-context behavior can be non-performant.

### Questions
1. I am curious, what is the precision and recall trade-off of the AABB block filter on adversarial or highly paraphrased queries?

2. What is the end-to-end latency and memory-traffic breakdown (prefill, retrieval, re-embed), and how would specialized KV hardware change the bottlenecks?

3. How does MemArt compare head-to-head with KV pruning strategies like Keyformer and MorphKV under the same memory budget and latency constraints?

### Soundness
3

### Presentation
2

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
This paper proposes MemArt, a new KV-cache centric memory paradigm for LLM agents that replaces plaintext with direct reuse of latent KV cache blocks. Instead of re-feeding retrieved text into prompts, MemArt stores and retrieves prior computation directly in latent space which dramatically improves both accuracy and efficiency. Specifically, they propose to compress keys via a bounding box, then they compute the attention over KV blocks through normalization and aggregation over the query tokens. Finally, they append these KV blocks after injecting the positional index to start the decoding.

### Strengths
* The proposed KV-cache centric memory paradigm can directly reuse the calculated KV during prefill, which reduces computational overhead.
* The proposed multi-token aggregation does alleviate retrieval overhead by reducing the number of index.
* Their proposed decoupled positional encoding practically solves the issue.

### Weaknesses
* While the proposed method achieves higher accuracy and lower latency, it inevitably involves an ever growing memory size that might cause storage issue. This is due two design choices: 1) the KV cache is represented in float numbers and it scales much faster than plaintext; 2) the memory is linearly growing with no upper bound on the size.
* Another drawback of using KV cache paradigm is the generalization across models. The importance of memory sharing amplifies in multi-agent systems, where one model needs to understand the other model’s memory. It limits the scope of the paper.
* There seems to lack some experimental comparison with KV cache compression literature. I have listed several below for reference.
1. _H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models_
2. _SnapKV: LLM Knows What You are Looking for Before Generation_
3. _A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression_
* Finally, the model size experimented are limited and primarily lies around 7/8B. I believe the work benefit from validating on larger scale models such as 32B or MoE models.

### Questions
* How would you discriminate your work from KV cache compression literature?

### Soundness
2

### Presentation
3

### Contribution
2
