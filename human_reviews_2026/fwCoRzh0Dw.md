# InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
In modern large language models (LLMs), handling very long context lengths presents significant challenges as it causes slower inference speeds and increased memory costs. Additionally, most existing pre-trained LLMs fail to generalize beyond their original training sequence lengths. To enable efficient and practical long-context utilization, we introduce \textit{InfiniteHiP}, a novel and practical LLM inference framework that accelerates processing by dynamically eliminating irrelevant context tokens through a modular hierarchical token pruning algorithm. Our method also allows generalization to longer sequences by selectively applying various RoPE adjustment methods according to the internal attention patterns within LLMs. Furthermore, we offload the key-value cache to host memory during inference, significantly reducing GPU memory pressure. As a result, InfiniteHiP enables the processing of up to 3 million tokens on a single L40s 48GB GPU -- 3x larger -- without any permanent loss of context information. Our framework achieves an 18.95x speedup in attention decoding for a 1 million token context without requiring additional training. We implement our method in the SGLang framework and demonstrate its effectiveness and practicality through extensive evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
InfiniteHiP improves the KV cache offloading mechanism of HiP Attention (ICLR 2025) by enhancing its cache management policy. The core idea remains the same to manage the KV cache on the unified memory space while keeping a smaller key bank on the GPU memory, which acts as a cache. The use of the Least Recently Used (LRU) policy as the eviction mechanism is incremental.


After reviewing section 3, FROM HIP TO INFINITEHIP, we are certain that this work is incremental. The token pruning is borrowed from H2O; the dynamic RoPE adjustment is a trick; and Least Recently Used (LRU) is incremental. This is an engineering-heavy paper with incremental improvements over existing work, overstated claims, and limited novel insights. To maintain the high standard of the ICLR conference, we tend to reject this paper.

### Strengths
The work integrates sparse attention, offloading, and OOL generalization into one unified system. The training-free design and work integration can lead to better performance.

We believe training-free inference is essential for effective inference, and this paper demonstrates it.

GPU kernels for InfiniteHIP are a good implementation.

### Weaknesses
The experimental benchmark selection is LongBench and ∞Bench to prove the effectiveness of InfiniteHiP. However, the context length of LongBench (32K) and ∞Benc (100k) is much lower than its claim of supporting 3 MILLION TOKENS on a single GPU. That means the extended context length has not been proven effective for extremely long context tasks. We suggest that the authors conduct experiments on LongBench v2 with a longer context length.

In Table 5, the RULER Performance of InfiniteHiP starts to be lower than full attention at 128k (74.99 vs. 76.89). Will this tend to continue to go down for a longer context > 128k? This trend can make the title up to 3 million tokens on a single GPU an overstated claim if the InfiniteHiP can not maintain accuracy for long context.

The RoPE Strategy of sing chunk-indexed RoPE for layers 1-3 and relative RoPE for layers 4-32 is based on observing "sliding window patterns in early layers" (Appendix D). Why exactly layers 1-3? What about layers 1-8 or other setting? An ablation study in other settings would help a lot.

The baseline is also out of date, which compares FA2 instead of FA3 [1] or flashinfer [2]. Other lossy baselines include H2O, StreamingLLM, and InfLLM, from 2023-2024. We recommend a state-of-the-art baseline like [3] or [4]

[1] Ye Z, Chen L, Lai R, et al. Flashinfer: Efficient and customizable attention engine for llm inference serving[J]. arXiv preprint arXiv:2501.01005, 2025.

[2] Shah J, Bikshandi G, Zhang Y, et al. Flashattention-3: Fast and accurate attention with asynchrony and low-precision[J]. Advances in Neural Information Processing Systems, 2024, 37: 68658-68685.

[3] Song W, Jayanthi S M, Ronanki S, et al. Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers[J]. arXiv preprint arXiv:2506.01215, 2025.

[4] Deng W, Yang Y, Du P, et al. HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference[J]. arXiv preprint arXiv:2507.03153, 2025.

### Questions
Analysis of the impact of InfiniteHIP on network reasoning capabilities？

How would chunk size affect the InfiniteHiP performance?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces InfiniteHiP, a training-free inference framework designed to address the challenges of processing extremely long contexts in Large Language Models (LLMs). The work tackles three main issues: the high computational and memory costs of the attention mechanism, the failure of pre-trained models to generalize beyond their training length, and the significant GPU memory pressure from the Key-Value (KV) cache. The core contributions are: 1) A modular multi-stage hierarchical token pruning algorithm that dynamically eliminates irrelevant context to accelerate attention. 2) A dynamic RoPE adjustment method that enables out-of-length generalization without fine-tuning. 3) An efficient KV cache offloading system that uses host memory and an LRU policy to manage the cache on a single GPU. The authors demonstrate that InfiniteHiP can process up to 3 million tokens on a single 48GB GPU, achieving significant speedups and strong performance on long-context benchmarks.

### Strengths
- The paper is evaluated on a comprehensive set of benchmarks, including LongBench, RULER, and ∞Bench.

- The work is substantial, integrating multiple techniques (sparse attention, OOL generalization, and KV cache offloading) into a single, practical framework. The implementation within the SGLang framework and detailed performance analysis show a significant engineering effort.

- The proposed method achieves strong performance.

### Weaknesses
- Crucial details of the proposed method, particularly the complete algorithms for context pruning (Algorithms 1-3), are deferred to the appendix. While this may be due to space constraints, it makes it challenging for the reader to fully grasp the core mechanism without frequently referencing the appendix.

- The heuristic used in the `SelectRep` algorithm is a primary concern. The paper states that when a chunk is divided into two branches, the **first token** of each branch is used as a proxy to decide which branch to discard . This choice seems counter-intuitive. Considering the nature of the causal attention mask, the **last token** of a branch would likely be a more representative summary of the information within that branch. However, even so, the assumption that a single, fixed-position token can reliably represent an entire chunk is not convincingly justified and lacks strong empirical support in the paper.

- The paper could be strengthened by discussing and comparing its KV cache offloading mechanism with other recent works[1,2,3]. 

I am willing to raise my score if my concerns are adequately addressed.

[1] InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management

[2] Arkvale: Efficient generative llm inference with recallable key-value eviction

[3] OmniKV: Dynamic context selection for efficient long-context LLMs

### Questions
1.  A significant contribution of this work is the sophisticated KV cache management system. Given its practicality, do the authors plan to open-source the code to facilitate reproducibility and encourage further research in this area?

2.  Could the author share insights on why the first token was chosen as the representative token?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
InfiniteHiP is a training-free long-context inference framework designed to address three key bottlenecks of LLMs when processing long sequences: computational efficiency, memory consumption, and generalization beyond the pretraining window.
Building upon the original HiP, InfiniteHiP introduces a series of system-level improvements that make long-context inference feasible on a single GPU. The framework consists of three major components: Hierarchical Multi-Stage Pruning; Dynamic RoPE Adjustment, which adapts positional encoding strategies dynamically to enable out-of-length generalization for short-context pretrained models; and Hierarchical KV Offloading with LRU Policy, which manages multi-stage cache refreshing and memory transfer between GPU and host to minimize VRAM pressure. Through the synergy of these mechanisms, InfiniteHiP achieves significant performance improvements within the SGLang inference framework, specifically, a 7.24× end-to-end decoding speedup and an 18.95× acceleration in attention computation on million-token contexts, all without requiring any retraining.

### Strengths
1. The work demonstrates strong practicality and engineering significance. InfiniteHiP can be directly integrated with a variety of existing models, such as LLaMA, Qwen, Gemma, and EXAONE, providing a general and deployment-ready solution for long-context inference on commodity GPUs.
2. Another notable strength lies in its unified and system-oriented design perspective. Instead of focusing on a single optimization aspect, the framework simultaneously tackles the three major challenges of long-context modeling: computation, generalization, and memory through a coherent modular architecture.

### Weaknesses
1. Despite its strong engineering impact, the scope of related work is relatively limited, covering only four prior studies, which may not sufficiently position InfiniteHiP within the broader literature of efficient attention and memory optimization.
2. The main innovations reside at the system level, and the algorithmic novelty is incremental rather than conceptual. Each of the three modules, pruning, RoPE adjustment, and KV management, builds upon previously established ideas, leading to the impression of being “incremental but practical.”
3. Although several ablation experiments are presented, the paper lacks a systematic quantitative analysis that isolates and justifies the independent contribution of each module. Strengthening the analytical rigor and theoretical interpretation of these components would significantly enhance the paper’s scientific depth and persuasive power.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3
