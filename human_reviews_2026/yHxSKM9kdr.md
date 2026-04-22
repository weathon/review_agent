# IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Key-Value (KV) cache plays a crucial role in accelerating inference in large language models (LLMs) by storing intermediate attention states and avoiding redundant computation during autoregressive generation. However, its memory footprint scales linearly with sequence length, often leading to severe memory bottlenecks on resource-constrained hardware. Prior work has explored offloading KV-cache to the CPU while retaining only a subset on the GPU, but these approaches often rely on imprecise token selection and suffer performance degradation in long-generation tasks such as chain-of-thought reasoning. In this paper, we propose a novel KV-cache management strategy, IceCache, which integrates semantic token clustering with PagedAttention. By organizing semantically related tokens into contiguous memory regions managed by a hierarchical, dynamically updatable data structure, our method enables more efficient token selection and better utilization of memory bandwidth during CPU–GPU transfers. Experimental results on LongBench show that, with a 256-token budget, IceCache maintains 99\% of the original accuracy achieved by the full KV-cache model. Moreover, compared to other offloading-based methods, IceCache attains competitive or even superior latency and accuracy while using only 25\% of the KV-cache token budget, demonstrating its effectiveness in long-sequence scenarios. The code is available on our project website at https://yuzhenmao.github.io/IceCache/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces IceCache, a novel KV-cache management strategy designed to mitigate memory bottlenecks for LLMs processing long sequences. The core idea is to move beyond storing the KV-cache in its original token order. Instead, IceCache clusters tokens based on the semantic similarity of their key embeddings into a hierarchical structure called a DCI-tree. During inference, it employs a fast Approximate Nearest Neighbor (ANN) search algorithm (M-DCI) on the CPU to perform a query-aware, head-specific selection of the most relevant memory pages. These selected pages are then transferred from CPU to GPU for attention computation. The system is optimized with a pipelined workflow that overlaps the CPU-based page selection with GPU computations to hide latency. Experimental results across several benchmarks and models show that IceCache can maintain high accuracy (over 99% in some cases) with a significantly reduced token budget compared to the full KV-cache and outperforms other baseline methods.

### Strengths
- The authors compare IceCache against a strong and sufficient set of six recent state-of-the-art KV-cache optimization methods.

- The paper is well-organized and clearly written. The motivation is well-established, the proposed method is described logically with helpful diagrams (e.g., Figure 1 and 2).

- The application of a hierarchical ANN algorithm (M-DCI) to cluster and retrieve from the high-dimensional key-embedding space of the KV-cache is a novel and interesting approach.

### Weaknesses
- The paper does not provide an analysis of how different parameter choices of M-DCI affect the final model accuracy and inference latency. Furthermore, the specific parameter values used in the experiments are not clearly described, which could hinder reproducibility.

- The latency analysis in Section 5.5 focuses on "Time to the second token" (TT2T) and "Time per output token" (TPOT). While informative, it lacks a direct end-to-end runtime comparison against full attention.

- A central claim is that grouping semantically similar tokens into the same memory pages improves efficiency. However, there is no ablation study to isolate and quantify this specific contribution. Since the index selection happens on the CPU, a key question arises: does this memory layout truly enhance efficiency, or is the performance bottleneck dominated by the CPU search and the CPU-GPU data transfer overhead?

### Questions
1.  Could you provide performance curves showing how the inference speed (e.g., tokens per second) scales with increasing context length?

2.  The ANN search is currently performed on the CPU to overlap with GPU computations. Have you considered the feasibility of further accelerating the M-DCI query step by implementing it on the GPU?

3.  The proposed method seems to be tightly integrated with a CPU offloading strategy. Is it possible to adapt IceCache for a non-offload scenario (i.e., where the full KV-cache fits in GPU memory, but sparsity is desired to accelerate computation)? Or does the method fundamentally rely on the CPU having access to the cache for the DCI-tree query to work?

### Soundness
2

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
IceCache introduces a new approach to memory-efficient KV-cache management for long-sequence LLMs. It uses a hierarchical DCI-tree that clusters key embeddings based on semantic similarity, grouping related tokens into memory pages. IceCache further employs bulk data loading and CPU-GPU pipelining to minimize latency. Tested on models like LLama-3.1-8B, Mistral-7B, Qwen3-32B, LongChat-7B on various tasks. It maintains over 99% accuracy with a 256 token budget.

### Strengths
1. High efficiency and accuracy: while cutting KV memory usage, achieve high accuracy
2. Fine-grained retrieval: per-head-query-aware selection, improve attention focus
3. Efficient pipelining: CPU-GPU overlap computation with data movement to reduce latency

### Weaknesses
1. Limited analysis: missing ablations on index depth, page size and computational cost of clustering and updates
2. Scalability uncertainty: effectiveness and efficiency on extremely long contexts or distributed multi-GPU settings
3. Figure clarity and presentation issues: figure 1 & 4 are not well-explained or visually clear. lack of consistent notation and labeling.

### Questions
1. What is the latency for prefill stage, since you introduce additional clustering and indexing 
2. Did you try on multi-turn conversation where context may change, since I assume the serial workflow in Figure 4 means multi-turn. What is the latency and accuracy.
3. Figure 4 is lack of explanation and notations, hard to understand. 
4. Did IceCache select page for each decoding step or only once for the whole decoding phase.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a KV-cache management method called IceCache, combining semantic token clustering (using a hierarchical DCI-tree) with the PagedAttention mechanism to optimize long-sequence LLM inference. The core idea is to group semantically related tokens into the same memory pages, enabling more accurate and efficient query-aware page selection during decoding, compared to methods that rely on the original token order. The paper demonstrates better results compared with baselines, maintaining over 99% of full-model accuracy with a significantly reduced KV-cache budget (e.g., 256 tokens) across a wide range of long-context benchmarks.

### Strengths
1. Grouping tokens by semantic similarity in key-embedding space improves cache hit rates, which is powerful and clearly explained.
2. Comprehensive evaluation with diverse benchmarks and multiple models with impressive results.
3. Superior accuracy-latency trade-off compared to other high-accuracy, retrieval-based methods.

### Weaknesses
1. Lack quantitative analysis of the computational cost of building and maintaining the DCI-tree on the CPU. How significant is the CPU utilization? Could this become a bottleneck on a system with a less powerful CPU or when running multiple instances?
2. The storage cost of the DCI-tree indices themselves is not discussed. For a context length of 100k tokens per layer and per head, what is the memory footprint of the index on the CPU? 
3. Compared to OmniKV, the overhead is primarily from the DCI-query, the more complex data movement, or both? A direct comparison of these components with a leading baseline like OmniKV would be more informative.
4. The description of M-DCI with P-DCI, but the algorithm in Page 4's pseudocode and Appendix B.1 uses a simpler promotion and parent-assignment scheme, the consistency should be improved.
5. The entire method hinges on the quality of the key embeddings for clustering. How about the assumption of the clustering not hold?
6. The promotion ratio r for the DCI-tree and the number of levels L are crucial hyperparameters. How were they chosen? How sensitive are the results to their values?

### Questions
The same as the weakness.

### Soundness
3

### Presentation
2

### Contribution
3
