# TyphoonMLA: A Mixed Naive-Absorb MLA Kernel For Shared Prefix

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Multi-Head Latent Attention (MLA) is a recent attention mechanism adopted in state-of-the-art LLMs such as DeepSeek-v3 and Kimi K2. Thanks to its novel formulation, MLA allows two functionally equivalent but computationally distinct kernel implementations: naive and absorb. While the naive kernels (e.g., FlashAttention) are typically preferred in training and prefill for their computational efficiency, existing decoding kernels (e.g., FlashMLA) rely on the absorb method to minimize HBM bandwidth usage. However, the compute-bound nature of the absorb implementations prohibits performance benefits from data reuse opportunities in attention calculations, such as shared prefixes. In this work, we introduce TyphoonMLA, a hybrid approach that combines naive and absorb formulations to harness the strengths of both. TyphoonMLA effectively leverages the shared prefix by applying the naive formulation to the compute-bound parts of attention calculations, while reducing the bandwidth requirements for non-shared parts by using the absorb formulation. As a result, TyphoonMLA improves the throughput of attention calculations in MLA architectures by up to 3× and 3.24× on NPU and GPUs, and boosts end-to-end throughput by up to 1.48× in tokens per second, with only a 3\% overhead in HBM size.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TyphoonMLA, a novel hybrid attention kernel designed to accelerate inference with shared prefixes for Large Language Models (LLMs) that utilize the Multi-Head Latent Attention (MLA) architecture, such as DeepSeek-v3 and Kimi K2. The authors identify a key performance bottleneck in existing MLA implementations (e.g., FlashMLA), which exclusively use a compute-bound "absorb" formulation to minimize HBM bandwidth. While effective for single sequences, this approach does not take advantage of data reuse opportunities in common shared-prefix scenarios (e.g., system prompts, tree-based decoding).

The core contribution is a hybrid method that partitions the attention calculation. It applies the compute-efficient "naive" formulation to the shared-prefix portion of the KV-cache, where high data reuse mitigates its higher bandwidth cost. Concurrently, it uses the bandwidth-efficient "absorb" formulation for the non-shared portions. By combining the outputs of these two parallel paths, TyphoonMLA achieves significant throughput gains—up to 3.24x on GPUs and 3x on NPUs—with a negligible HBM footprint overhead of approximately 3%.

### Strengths
1. The idea of creating a hybrid kernel that combines the naive and absorb formulations within a single attention computation is simple and insightful. Prior work has explored prefix sharing for standard MHA/GQA, but those mechanisms are typically memory-bound. This paper clearly identifies that MLA's decoding stage is compute-bound, making the problem fundamentally different and requiring a new approach. The solution is an effective combination of existing concepts tailored to the characteristics of the MLA architecture.
2. The Roofline analysis (Figure 6) clearly motivates the problem and visualizes the performance trade-off between the two formulations. The computational analysis in Table 1 provides a solid mathematical foundation for the expected gains. The experiments are thorough, validating the approach on two different hardware platforms (NVIDIA GPUs, Huawei Ascend NPUs), two state-of-the-art models, and realistic workloads using real-world system prompts of varying lengths. The inclusion of a latency breakdown (Figure 4) is a major strength, as it serves as a microbenchmark that directly validates the core hypothesis—that the speedup mainly comes from accelerating the shared part of the computation.
3. The paper is clearly written and easy to follow. Figure 1 provides a good visual schematic that contrasts the naive, absorb, and the proposed TyphoonMLA data flows, making the core mechanism immediately understandable. The results are presented clearly and support the paper's claims.
4. As LLMs are increasingly deployed with long and complex system prompts, and with the rise of reasoning techniques like Tree-of-Thought, optimizing for shared prefixes is becoming more important. The MLA architecture is used by several open-source models, and this work provides a solution that offers substantial, real-world performance improvements. The fact that it requires no model retraining and can be integrated into existing serving frameworks like vLLM increases its practical relevance.

### Weaknesses
1. The paper addresses the MLA kernel performance in shared prefix scenarios, but in Table 2, the system prompts are sampled from Claude-4, OpenAI/o3, and Grok/Personas, which do not use MLA. However, the models tested are DeepSeek-V3 and Kimi K2, which do use MLA. The experimental design could be improved by using system prompts that are more representative of DeepSeek-V3 and Kimi K2.
2. The paper does not explore how the length of the shared prefix affects acceleration. It is unclear at what length positive gains in TTFT and TGR metrics are achieved.
3. Section 3.1 describes the "Fall-back to Absorb" mechanism, but does not clearly provide the predefined threshold of gbs >= 64 in the main text or experiments; instead, it is discussed in the appendix. It is suggested to clarify this in the main paper in future versions.
4. The overhead of the "CombineLSE" step, which merges the results from the two parallel paths, is reported as small but is not analyzed. The paper would be improved by briefly discussing this step's computational complexity and how it scales with batch size or the number of attention heads, as this is important for understanding limitations in extreme-scale scenarios.
5. The authors claim the method is numerically identical to standard MLA, which is theoretically sound. However, the paper does not provide empirical evidence, such as a validation table confirming identical outputs (e.g., bit-for-bit) or benchmark results compared to the baseline. Such results would confirm the implementation's correctness.
6. The paper claims compatibility with systems like vLLM but does not sufficiently discuss the engineering complexity of deploying this method in dynamic, real-world serving environments. Key challenges are not addressed, such as managing a hybrid KV-cache layout within a memory manager like PagedAttention (which could increase fragmentation), or handling load imbalances due to highly variable non-shared sequence lengths within a batch.

### Questions
1. Table 1 and Section 3.2 do not clearly define what `k` means, for example in $L_s40k + BL_n40k$. Please clarify the meaning of `k`.
2. Figure 6 analyzes the Roofline for different batch sizes, but Figures 2 and 3 only show results for bsz=256, 512, and 1024. Could you add results for batch sizes 1, 8, 32, and 128? If small batch sizes are not relevant in practice, please provide supporting evidence.
3. Figure 3 shows only bsz=1024; can you also show results for multiple batch sizes on the GPU, similar to Figure 2 for the NPU?
4. Can the paper provide end-to-end acceleration comparisons for TTFT and TGR between FlashMLA, FlashInfer, and TyphoonMLA?
5. Could you elaborate on the system's behavior in a dynamic serving environment with heterogeneous requests? Specifically:  
   a) How does the system handle load balancing between the naive and absorb paths when non-shared sequence lengths are highly variable within a batch?  
   b) What are the specific memory management challenges (e.g., fragmentation) when integrating the hybrid KV-cache with continuous batching?
6. The performance gains are shown for a contiguous shared prefix. Have you considered generalizing this hybrid approach to more complex, non-contiguous sharing patterns (e.g., shared data blocks in a RAG context), or the potential applicability of the core "hybrid compute/memory-bound" principle to other architectures like standard MHA/GQA as hardware evolves?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a performance bottleneck in modern Large Language Models (LLMs) that use Multi-Head Latent Attention (MLA), such as DeepSeek-v3. The standard absorb implementation used for decoding is memory-efficient but compute-bound, preventing it from benefiting from data reuse in scenarios with shared prefixes (e.g., system prompts).   

The authors propose TyphoonMLA, a novel hybrid attention kernel that partitions the computation. It applies the computationally cheaper naive formulation to the shared prefix portion of the Key-Value Cache to reduce floating-point operations (FLOPs), while using the memory-bandwidth-efficient absorb formulation for the non-shared parts. This hybrid approach is a bit-exact, drop-in replacement that requires no model retraining.

### Strengths
While prefix sharing optimization isn't new, the paper is the first to identify and address the specific bottleneck it creates for Multi-Head Latent Attention (MLA) models. 

The computational analysis in Section 3.2 (Table 1) is clear and compelling. It precisely breaks down the HBM read and MAC requirements, mathematically demonstrating why the hybrid approach should be superior by reducing MACs in the compute-bound shared region and HBM reads in the memory-bound non-shared region.

Besides, this is not a theoretical optimization for a toy problem. It directly accelerates models that are at the forefront of LLM development (DeepSeek, Kimi). As system prompts and context lengths grow, this bottleneck becomes increasingly important.

### Weaknesses
My main issue with the paper is that the core idea feels less novel than presented. Splitting attention into shared and non-shared parts is a known trick in the MHA/GQA world—we've seen this before in libraries like FlashInfer with its "Cascade Inference" and in Hydragen. The paper doesn't acknowledge this prior art, which makes its contribution seem more like a clever adaptation to MLA's specific compute/memory tradeoffs rather than a brand-new algorithm.

Beyond that, there are a few other points that need tightening up:
1. The paper is a bit hand-wavy about the "predefined threshold" used to switch back to the absorb kernel. It's not clear how this is set or how sensitive the performance is to this magic number.   

2. It claims to handle complex tree-structured prefixes (like in Tree-of-Thought), but doesn't show how. This is a non-trivial problem that other kernels, like FlashForge, were specifically designed to solve.   

3. The NPU results are impressive, but they feel a bit like a black box. Because the implementation relies on proprietary Huawei libraries (CANN/CATLASS), it's hard to judge how portable the approach is. The paper doesn't discuss the underlying principles that would make this work on other NPU architectures

### Questions
You mention that the kernel falls back to an absorb-only implementation below a "predefined threshold". How is this threshold determined? Is it found empirically for each hardware/model pair? It would be helpful to understand how sensitive the overall performance is to this value. An ablation study on this threshold would add a lot of value.

You claim compatibility with tree-structured workloads like Tree-of-Thought. However, the method described seems tailored to a simple linear prefix. How would TyphoonMLA handle a more complex KV cache layout, where there might be a root, several shared branches, and then unique leaves?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a method that accelerates inference for transformers in the presence of shared prefixes, such as system prompts in modern LLMs. The method, called TyphonMLA, is a hybrid kernel that combines regular attention and MLA from DeepSeek. It consists of using regular attention for shared prefixes and MLA for the rest. Moreover, the kernel falls back to standard MLA at small batch sizes. In practice, the hybridization is carried by merging both attention results via a shared softmax denominator. On NPUs and GPUs, with real long system prompts (Claude 4, GPT o3, Grok), TyphoonMLA reaches up to 3x throughput over standard FlashMLA baselines, with just 3% HBM overhead.

### Strengths
There are many things that I liked in this paper and that I believe it will be helpful to others at ICLR:

-  MLA decode today is compute-bound because of absorb, so prefix-sharing kernels from MHA/GQA don't directly help. TyphonMLA fills this gap.

- The idea is very simple but impactful. 

- The paper brings concrete computational analysis. For example, Table 1 shows MAC ops and HBM reads for naive, absorb, and TyphoonMLA; and Figure 4 shows a detailed latency breakdown of TyphonMLA across different stages of the kernel. 

- Since the idea is quite straightforward and the method is exact (same output as original MLA), TyphonMLA works well with standard inference techniques, such as paged KV. Therefore, it can be easily included into frameworks like vLLM and SGLang.

- Strong results on both NPUs and GPUs, with two real MLA models (DeepSeek-v3, Kimi K2) using multiple real system prompts.

### Weaknesses
A few things to be improved in the paper include: 

- Right now Table 1 is overwhelming to read as it contains a lot of symbols. One idea is to organize the table by placing MAC rows first and HBM reads rows later so that methods are more easily comparable. Distributing $L_s$ and $L_n$ for all methods would also improve things. E.g., $BS_q (L_s + L_n) H (2D_l + D_r)$ to $BS_q L_s H(2D_l + D_r) + B S_q L_n H(2D_l + D_r)$.

- The impressive results of 3x speedup happens in shared prefix scenarios, which is not always the case in transformers.

- I believe the bar plots are clean and informative, but they would be much better with error bars / confidence intervals to get an idea of how robust are the results.

### Questions
Can you describe in more details how TyphonMLA kernel is implemented in a pseudocde?

### Soundness
4

### Presentation
3

### Contribution
4
