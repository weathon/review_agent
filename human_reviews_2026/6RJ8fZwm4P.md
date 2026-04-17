# LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
While Key-Value (KV) cache succeeds in reducing redundant computations in auto-regressive models, it introduces significant memory overhead, limiting its practical deployment in long-sequence scenarios. Existing KV retrieval methods attempt to mitigate this by dynamically retaining only a subset of KV entries on the GPU. However, they still suffer from notable efficiency and accuracy bottlenecks due to per-token retrieval and coarse-grained page-level KV management strategy, especially in long-output reasoning scenarios. With the emergence of large reasoning models, efficiently handling such scenarios has become increasingly important. To address this issue, we present two key observations: (1) critical KVs exhibit strong temporal locality during decoding, and (2) these KVs exhibit distinct distribution patterns across the input prompt and the generated output. Building on these observations, we propose \emph{LouisKV}, an efficient KV cache retrieval framework designed for various long-sequence scenarios. Specifically, LouisKV introduces a semantic-aware retrieval strategy that leverages temporal locality to trigger retrieval only at semantic boundaries, drastically reducing computation and data transfer overhead. LouisKV also designs a decoupled, fine-grained management scheme that tailors differentiated strategies for input and output sequences to create retrieval units that better match the model's attention patterns, thereby enabling the precise identification of critical KVs. Furthermore, to boost system efficiency, LouisKV incorporates several kernel-level optimizations, including custom Triton and CUDA kernels to accelerate the KV clustering and retrieval. Evaluation results show that LouisKV achieves up to 4.7$\times$ speedup over state-of-the-art KV retrieval methods while maintaining near-lossless accuracy across diverse long-sequence tasks, including long-input short-output, short-input long-output, and long-input long-output scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LouisKV, a KV cache management framework for long-context and long-generation inference. The design is motivated from the observation of temporal locality between adjacent tokens and distinct spatial distributions between the long inputs and long outputs. During prefill, LouisKV uses k-means clusters and triggers retrieval during decoding only at semantic boundaries detected by query vector cosine similarity. Offloading is also performed for efficiency, and additional system level optimizations using CUDA kernels is done to enable speed-ups. Experiments on long-input tasks like long bench and complex reasoning tasks like AIME show that LouisKV achieves a speedup and performs better than other retrieval methods like Arkvale and Quest.

### Strengths
1. The decoupled management scheme which applies different strategies for both prefill and decoding stage, along with system level optimizations lead to strong performance compared to the other testes retrieval baselines.
2. Experimental setup is very comprehensive, testing for not only long-context input, which is more common in KV cache literature, but also long-output using reasoning benchmarks like MATH500 and AIME.

### Weaknesses
The contribution of the paper is spread across multiple aspects, making it difficult to recognize the main contribution. LouisKV modifies multiple stages of the LLM inference pipeline: Prefill with k-means clustering and asynchronous offloading. Decoding with semantic-boundary detection, and asynchronous offloading. System level optimizations with triton kernels. While each piece makes sense, it is difficult to assess conceptual novelty.

For instance, the two observations noted in motivation has been reported before and already made use of in prior works. For example, ShadowKV [1] makes use of temporal locality to rebuild only necessary KV pairs. The second observation of distinct patterns has also been noted in SCOPE [2]. Similarly, the novelty of boundary trigger is limited, as FreeKV [3] measures cosine similarity between adjacent tokens' query vectors during long generation, and clustering has also been used during prefill stage (ClusterKV [4]). While the synthesis of these components is novel and the engineering effort is appreciated, it is difficult to isolate or have a view of what the main contribution of the paper is.


[1] Sun, Hanshi, et al. "Shadowkv: Kv cache in shadows for high-throughput long-context llm inference." arXiv preprint arXiv:2410.21465 (2024).

[2] Wu, Jialong, et al. "Scope: Optimizing key-value cache compression in long-context generation." arXiv preprint arXiv:2412.13649 (2024).

[3] Liu, Guangda, et al. "FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference." arXiv preprint arXiv:2505.13109 (2025).

[4] Liu, Guangda, et al. "Clusterkv: Manipulating llm kv cache in semantic space for recallable compression." 2025 62nd ACM/IEEE Design Automation Conference (DAC). IEEE, 2025.

### Questions
1. How sensitive are results to the local buffer W and sink tokens S?

2. How was the number of clusters determined in k-means?

3. The clustering on prefill KVs can be quite expensive. Could you quantify this latency overhead and how it scales with input length?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes **LouisKV**, a KV-cache retrieval framework for long input and long output generation. The core idea is to avoid per-token retrieval by exploiting temporal locality in decoding and to improve precision by managing input and output KV caches differently. Concretely, LouisKV (i) detects semantic boundaries using the cosine similarity of consecutive query vectors and only triggers retrieval at these boundaries, and (ii) applies k-means clustering over prefill keys for inputs while forming temporal segments for generated outputs, each represented by a centroid for fast similarity lookup. The system adds Triton and CUDA kernels and a group-consistent selection strategy to reduce transfer and compute overhead.

### Strengths
1) **Decoding efficiency by fewer retrieval triggers.** Triggering only at semantic boundaries reduces repeated estimation and transfers while preserving accuracy within a segment, which the paper motivates and supports with locality measurements and latency breakdown. 

2) **Covers multiple inference scenarios.** The study includes long-input short-output, short-input long-output, and long-input long-output, showing stable accuracy at fixed budgets and scale-up in batch size without OOM. 

3) **Solid systems work.** Custom Triton and CUDA kernels and group-consistent selection improve throughput in realistic deployments with GQA. The ablation indicates which component contributes most.

### Weaknesses
1) **Motivation scope.** The assumption that decoding naturally forms stable semantic segments holds on math and step-wise tasks but may not on dialog or code completion with frequent topic shifts.

2) **Clustering cost and TTFT.** Input clustering is central, yet the paper reports end-to-end latency without an explicit time-to-first-token under very long prompts. Please add TTFT curves vs prompt length and show the amortization benefit when queries are short. Also report k-means wall time vs input length and number of clusters, and whether the asynchronous pipeline hides this cost under prefill compute. 

3) **Long-context stress tests.** Results include LongReason up to 64K. RULER is a standard stress suite for long-context retrieval and sequence length extrapolation. Please add RULER at 32K-128K with accuracy and latency, and clarify memory usage under those settings.

### Questions
1) **Trigger statistics.** Please report the distribution of semantic segment lengths across datasets, average r at boundaries, and per-layer agreement on boundary detection. This can justify the amortization claim with hard numbers beyond the ablation. 

2) **Generated KV retention in long-reasoning.** In short-input long-output tasks, do you offload all generated KVs to CPU after the local buffer fills so they remain recallable. If yes, what is the incremental CPU-GPU traffic per segment, and how does that compare with page-level Arkvale under the same budget. A per-token bytes-moved metric would help.

3) **Cluster locality in text space.** Within a cluster, are source text spans mostly contiguous or scattered. If clusters often map to contiguous spans, would span-aware grouping or block-sparse layouts reduce transfers further. Any analysis of edit distance between token positions in the same cluster.

4) **Citation coverage.** More semantic-based retrieval work could be added.
[1] Chen, Guoxuan, et al. "Sepllm: Accelerate large language models by compressing one segment into one separator." arXiv preprint arXiv:2412.12094 (2024).
[2] Zhu, Yuxuan, et al. "SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching." arXiv preprint arXiv:2504.00970 (2025).
[3] Hooper, Coleman, et al. "Squeezed attention: Accelerating long context length llm inference." arXiv preprint arXiv:2411.09688 (2024).

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
This paper presents LouisKV, a framework to accelerate long-context LLM inference by optimizing KV cache retrieval.

LouisKV uses a semantic-aware retrieval strategy that triggers KV retrieval only at semantic boundariesr to reduce computation and data transfer.
During the prefill stage, LouisKV performs k-means clustering on the KV cache to group semantically similar tokens, using the centroids for retrieval. In the decode stage, it partitions generated KVs into fixed-size temporal segments, retrieving them as clusters when a semantic boundary is detected.

Experiments demonstrate that LouisKV achieves up to 4.7× speedup over state-of-the-art KV retrieval methods such as Arkvale, while maintaining near-lossless accuracy across diverse long-sequence tasks.

### Strengths
The paper presents comprehensive experiments across diverse benchmarks, models, and task types to demonstrate that LouisKV achieves near-lossless accuracy while improving inference efficiency.

### Weaknesses
This paper exhibits **a severe lack of novelty** and raises potential concerns about overlap with prior work.

In particular, the core KV cache retrieval mechanism in LouisKV appears **nearly identical** to that of ClusterKV \[1\].
Both of them apply k-means clustering to the KV cache during the prefill stage, select retrieval units based on cluster centroids, and manage KV entries generated during decoding separately. 
Furthermore, system-level components such as the asynchronous clustering, offloading pipeline and the CUDA kernel for efficient selection have also been introduced in ClusterKV. 
As a result, LouisKV does **not present clear methodological or engineering innovation** beyond what has been previously published. 
The absence of explicit discussion or experimental comparison against ClusterKV further weakens the paper’s contribution.

The claimed semantic-aware adaptive retrieval also offers **limited novelty**. The notion of exploiting temporal locality in decoding has been explored in prior works \[2, 3, 4\].
For example, the specific strategy of triggering retrieval based on cosine similarity between consecutive query vectors seems essentially the same as in \[3\].
Consequently, the contribution of LouisKV beyond existing literature is unclear.

\[1\] ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression

\[2\] HShare: Fast LLM Decoding by Hierarchical Key-Value Sharing

\[3\] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

\[4\] CaliDrop: KV Cache Compression with Calibration

### Questions
What are the unique contributions and technical novelties of LouisKV compared to prior KV retrieval frameworks, particularly ClusterKV.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors analyze long-sequence inference and discover that critical KVs (i.e. important tokens) demonstrate strong temporal locality during decoding (i.e., critical KVs between neighboring decoding steps are similar), and that critical KVs are sparsely distributed in prefill, but during decoding are locally concentrated within certain reasoning steps. Based on these observations, separate prefill/decoding cache management strategies are leveraged to improve recall from the CPU to the GPU of relevant KV entries. Custom kernels are implemented resulting in 4.7x speedup against baselines with minimal performance drop.

### Strengths
- Long-sequence KV strategies are very focused on long prefill. With the advent of reasoning models, the long decoding phase is now common, and the authors' study of this scenario is timely. 

- The described approach is simple and easy to understand. Mixed CPU/GPU KV management strategies are relatively less common, but are becoming increasingly important as token eviction continues to prove less generalizable. 

 - The method is both fast and maintains high performance on a wide variety of task types. 

 - A variety of model families are tested thus demonstrating architecture generalizability.

### Weaknesses
- As with all mixed CPU/GPU KV cache strategies, the approach requires careful kernel level optimizations to mitigate data transfer latency, thus making this approach non-hardware/software agnostic. 

- Selection of $\tau$ seems to vary by both model and task. Time spent tuning this hyper-parameter limits the overall efficiency of this framework.

- Small sample sizes from the math reasoning benchmarks.

### Questions
- Could you try more questions from MATH500 and GPQA? You do not need to try the entirety of these datasets, but perhaps try 2-3 50 question samples and report the error. 

 - How quickly can $\tau$ can be tuned? That is, could you study how many questions are practically needed per model to determine a reasonable selection of this hyperparameter?

### Soundness
3

### Presentation
3

### Contribution
2
