# SWAN: Sparse Winnowed Attention for Reduced Inference Memory via Decompression-Free KV-Cache Compression

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Large Language Models (LLMs) face a significant bottleneck during autoregressive inference due to the massive memory footprint of the Key-Value (KV) cache. Existing compression techniques like token eviction, quantization, or other low-rank methods often risk information loss, have fixed limits, or introduce significant computational overhead from explicit decompression steps. In this work, we introduce SWAN, a novel, fine-tuning-free framework that eliminates this overhead. Our method uses an offline orthogonal matrix to rotate and prune the KV-cache, which is then used directly in the attention computation without any reconstruction. Our extensive experiments demonstrate that SWAN, augmented with a small dense buffer, offers a robust trade-off, maintaining performance close to the uncompressed baseline even at aggressive 50-60\% memory savings per-token on KV-cache. A key advantage is its runtime-tunable compression level, allowing operators to dynamically adjust the memory footprint, a flexibility absent in methods requiring fixed offline configurations. This combination of a decompression-free design, high performance under compression, and adaptability makes SWAN a practical and efficient solution for serving LLMs with long contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes to improve memory footprint of the KV-cache, by using an offline orthogonal matrix to rotate and prune the KV-cache. SWAN (Sparse Winnowed Attention) is introduced to perform attention directly on a compressed, sparse KV-cache without (conventional) reconstruction, called "decompression-free" as indicated in the paper title. Being free of decompression/reconstruction leads to simultaneous memory and compute savings -- this claim and contribution sound promising!

### Strengths
1. SWAN can achieve up to 50-60% memory savings due to the improvement in memory footprint of KV-cache. Being free of decompression/reconstruction leads to simultaneous memory and compute savings. In other words, unlike other existing decomposition-based KV-cache compression methods, SWAN incurs no (or minimal) computation overhead for reconstruction, and thus save compute as well.

### Weaknesses
1. Being free of decompression/reconstruction leads to simultaneous memory and compute savings -- this claim and contribution sound promising! However, if the authors claim to save compute, then they should also demonstrate results of FLOPs (# of floating-point operations), LLM runtime (end-to-end latency), and speed-up (percentage of latency improvement); but these are not thoroughly discussed, nor experimentally analyzed in the paper.
2. Direct comparisons (with experiments/results) against other related works, especially those requiring reconstruction, should be made. My (the reviewer's) point of view is: LLM inference is memory-bound and a certain degree of computation overhead for reconstruction is not harmful and may even be beneficial for runtime/latency because the fact behind computation overhead for reconstruction is the significant relieve/improvement in memory footprint, and the improvement in memory footprint speeds up LLM inference despite computation overhead for reconstruction.
3. The authors did not talk much about the prefilling stage. It is not clear whether the prefilling stage may need to adapt to SWAN.

### Questions
My questions and suggestions are basically from "Weaknesses" as aforementioned.
1. From Weakness 1: Please demonstrate results of FLOPs (# of floating-point operations), LLM runtime (end-to-end latency), and speed-up (percentage of latency improvement).
2. From Weakness 2: Please experimentally compare SWAN against other related works, especially those requiring reconstruction.
3. From Weakness 3: Please address my concern about the adaptation, if any, of the prefilling stage owing to SWAN.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents SWAN (Sparse Winnowed Attention), a fine-tuning-free framework for compressing the KV cache during autoregressive decoding. SWAN first applies an orthogonal rotation (derived from offline SVD on joint Q–K and V–O subspaces) to concentrate information, then prunes each token’s key and value vectors by top-k magnitude, storing them in a sparse format. Attention is computed directly on this hybrid cache (dense recent buffer + sparse history) without decompression. The paper provides a space and compute analysis, including a break-even sequence length for speedups, and evaluates on Llama-3.1-8B-Instruct and OLMoE-1B-7B across GSM8K, MMLU/ARC, and LongBench. Results show that SWAN can maintain full-precision model qu

### Strengths
+ **Decompression-free design:** SWAN allows attention to run directly on a sparse cache, removing the need for reconstruction or merging operations that typically introduce overhead in low-rank or codec approaches.
+ **Clear, implementable mechanism:** Algorithm 1 precisely specifies runtime steps (project, buffer, prune-to-top-k, append to sparse cache, then hybrid attention), and Fig. 1 clarifies the data path.

### Weaknesses
+ **No latency or throughput evaluation:**
Although a theoretical efficiency analysis is provided, no empirical runtime measurements are presented. Wall-clock latency, throughput, or per-step breakdowns (prefill vs. decode) are missing, making it unclear how much real-world speedup SWAN achieves.

+ **No any baseline comparisons:**
The paper does not compare against any prior baseline approach. In particular, recent hidden-dimension compression methods such as Palu (low-rank) and EigenAttention (low-rank) are missing. Post-RoPE activations are commonly higher rank; I recommend comparing with these works to assess trade-offs (e.g., reconstruction cost, accuracy drops, memory savings). Meanwhile, ThinK (channel pruning) is also worth discussing.

+ **Limited evaluation scope (long-context tasks):**
Despite its stated motivation for long-sequence memory efficiency, experiments are largely restricted to short-context benchmarks (e.g., GSM8K, MMLU, ARC). The only long-context coverage is a selected subset of LongBench (e.g., summarization). More comprehensive long-context evaluations (e.g., RULER), ideally with runtime measurements, would better demonstrate SWAN’s effectiveness in its intended regime.

### Questions
+ **Handling irregular sparsity:**
Since each token drops a different subset of channels, how does your implementation manage this irregular sparsity during attention? Do you use per-token CSR-like indexing, and how is it parallelized efficiently on a GPU?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SWAN (Sparse Winnowed Attention), a decompression-free framework for KV-cache compression in large language model inference.
SWAN constructs orthogonal rotation matrices offline using singular value decomposition (SVD) over model activations to project Key and Value tensors into a subspace where information is more concentrated. During inference, it maintains a hybrid cache consisting of a sparse, pruned cache for older tokens and a small dense buffer for recent tokens to preserve accuracy.
The authors provide theoretical analyses of both space and computational complexity, showing that SWAN can yield significant savings in memory and FLOPs once the sequence length exceeds a predictable threshold.
Empirically, they evaluate the approach across mathematical reasoning (GSM8K), commonsense and knowledge benchmarks (MMLU, ARC-Challenge), and long-context understanding (LongBench), demonstrating that SWAN achieves up to 50–60% KV-cache memory reduction with minimal performance degradation.

### Strengths
* The paper is clearly written and well-structured, making it easy to follow.

* It introduces an interesting compression approach that converts the KV-cache into a hybrid sparse–dense representation and performs attention computations directly on the compressed cache without decompression.

* The accuracy evaluation is comprehensive, covering a diverse range of tasks—from mathematical reasoning to commonsense understanding and long-context processing—demonstrating the method’s generality.

### Weaknesses
* The paper lacks a solid system-level implementation to substantiate its claimed efficiency. The computational savings are analyzed only theoretically, without validation through real runtime measurements. Since the method depends on storing pruned tensors in a sparse (CSR) format, which is typically inefficient unless sparsity is extremely high (>99%), it is unclear whether the reported compression ratios (30–50%)—where accuracy is largely preserved—actually yield any practical speedup.

* The effectiveness of the proposed method appears limited. Most of the retained accuracy comes from the 128-token dense buffer, which merely preserves the most recent tokens. Without this buffer, performance degrades sharply and even collapses on long-context benchmarks. This diminishes the overall contribution, as long-context scenarios are precisely where KV-cache bottlenecks are most critical. Furthermore, the paper omits an important baseline comparison against a pure 128-token sliding window attention, which would clarify how much of the gain comes from SWAN itself versus the buffer.

* The runtime overhead of the proposed approach is not thoroughly analyzed. While the paper presents a limited complexity discussion, it overlooks several potential sources of cost, including (1) applying the projection matrix to Key vectors at each decoding step, and (2) performing the top-k pruning required to build the sparse cache. These steps could introduce non-trivial runtime overhead, yet no empirical evidence is provided to demonstrate that these costs are negligible.

### Questions
* What is the baseline accuracy when using only the 128 most recent tokens (i.e., without the proposed sparse cache)?

* How is the compression ratio computed in cases that include a 128-token buffer? To match the overall compression rate of the non-buffered setting, does the method apply more aggressive compression to the older, pruned tokens?

* What is the typical runtime latency introduced by the on-the-fly Key projection and the top-k pruning operations for evicted tokens during inference?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SWAN introduces a decompression free KV cache compression framework for large language model inference. It uses offline SVD based orthogonal rotations to concentrate important information into fewer dimensions and prunes less important components, allowing attention to operate directly on a hybrid cache composed of a sparse historical cache and a small dense buffer for recent tokens. This approach eliminates reconstruction overhead, reduces both memory and computation costs, and supports runtime adjustable compression levels. Experiments on Llama and OLMoE models show that SWAN maintains near baseline accuracy on reasoning and long context benchmarks while achieving around 50 to 60 percent KV cache memory savings.

### Strengths
- eliminates reconstruction overhead by performing attention directly on compressed KV caches.
- combines a sparse historical cache with a small dense buffer for recent tokens, effectively preserving accuracy.

### Weaknesses
- while theoretical compute savings are analyzed, the paper does not provide concrete wall-clock latency or throughput comparisons on modern GPU kernels (e.g., FlashAttention or Triton baselines), leaving practical efficiency uncertain.
- the claimed compute benefits rely on sparse-dense matvec operations, but these are often inefficient on current GPU hardware; implementation feasibility and actual speedups are not validated.
- applying the orthogonal projection to queries and keys at each decoding step introduces extra matrix multiplications, and the paper lacks quantitative analysis of this overhead.

### Questions
- Can the authors provide actual wall-clock latency and throughput measurements on modern GPUs (e.g., A100, H100) comparing SWAN with dense attention and existing compression methods like KVQuant or GEAR?

### Soundness
2

### Presentation
2

### Contribution
2
