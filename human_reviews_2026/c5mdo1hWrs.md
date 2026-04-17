# FSA: An Alternative Efficient Implementation of Native Sparse Attention Kernel

- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
Recent advance in sparse attention mechanisms has demonstrated strong potential for reducing the computational cost of long-context training and inference in large language models (LLMs). Native Sparse Attention (NSA), one state-of-the-art approach, introduces natively trainable, hardware-aligned sparse attention that delivers substantial system-level performance boost while maintaining accuracy comparable to full attention. However, the kernel implementation of NSA forces a loop order that is only efficient with a relatively large number of query heads in each Grouped Query Attention (GQA) group, whereas existing LLMs widely adopt much smaller number of query heads in each GQA group --- such an inconsistency significantly limits the applicability of this sparse algorithmic advance. In this work, we propose **F**lash **S**parse **A**ttention (**FSA**), an alternative kernel implementation that enables efficient NSA computation across a wide range of popular LLMs with varied smaller number of query heads in each GQA group on modern GPUs. Compared to vanilla NSA kernel implementation, our empirical evaluation demonstrates that FSA achieves (i) up to 3.5x and on average 1.6x kernel-level latency reduction, (ii) up to 1.25x and 1.09x on average end-to-end training speedup on state-of-the-art LLMs, and (iii) up to 1.36x and 1.11x on average for prefill-phase speedup in LLM generative inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The core issue addressed in this paper: NSA's cyclic ordering is only efficient when the number of query heads in the GQA group is large. However,  current attention mechanisms often employ GQA with a small number of attention heads, requiring NSA to use padding to meet hardware instruction requirements. This leads to redundant memory access and computation, preventing the theoretical reduction in FLOPs from translating into actual speed improvements.
The core solutions are loop order reversal, non-contiguous memory access optimization, staged computation and Softmax optimization, and memory-hardware co-adaptation.
Its greatest value lies in providing a hardware-friendly sparse attention solution for long-context LLM training and inference, which can be directly deployed within the existing GPU ecosystem, achieving an average 1.11x speedup in inference compared to NSA.

### Strengths
1. Problem Significance: Addresses the key bottleneck in deploying sparse attention: the incompatibility between native NSA kernels and mainstream LLMs' small GQA group sizes (typically 1-4 query heads).
2. Core Innovation: Inverts the NSA kernel's loop order (to "outer loop over KV blocks, inner loop over query tokens"), eliminating padding requirements for small GQA groups and significantly reducing redundant computation and memory access. Enhanced by memory management and specialized kernels.
3. Evaluation Results: Comprehensive testing shows substantial improvements: up to 3.5× lower kernel latency, 1.25× faster end-to-end training, and 1.36× faster inference prefill, with consistent performance across configurations.
4. Technical Depth: Theoretically reduces memory access and FLOPs to 21.3% and 56.2% of NSA respectively (GQA=4), complemented by practical engineering solutions for real-world deployment.

### Weaknesses
1. Limited Generalization to Extreme Lengths: Evaluation only up to 64K tokens, lacking analysis of ultra-long contexts (128K/256K). Attention sink effects on FSA's dual-buffer design remain unexamined at extreme scales.
2. Missing SOTA Comparisons: Only compares FSA with vanilla NSA and full attention. Omits comparisons with recent sparse kernels like flashdecoding, limiting perspective on performance trade-offs.
3. Insufficient Accuracy Analysis: Accuracy claims rely solely on Llama3-8B results. Lacks validation across diverse tasks and smaller models, with no study of numerical stability impact from optimizations.
4. Unquantified Deployment Overhead: Intermediate buffer costs are mentioned but not measured for memory-constrained scenarios. No discussion of integration complexity with mainstream frameworks.

### Questions
1. Can FSA adapt to other sparse patterns (e.g., block-sparse, dynamic)? Is the loop inversion strategy compatible beyond NSA's structure?
2.  Any evaluation beyond 64K length? How does attention sink affect the dual-buffer design?
3. Beyond loss curves, can you provide multi-model metrics (e.g., perplexity, QA F1) across diverse tasks?
4. What's the actual memory footprint of intermediate buffers? Any compilation time or distributed inference benchmarks?
5. What about other related work,like flashdecoding?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Flash Sparse Attention to enable efficient NSA computation with a varied number of query heads in each GQA group to improve LLM inference.

### Strengths
1. This paper aims to address an interesting and important problem in long-context LLM applications.
2. The presentation is good with clear writing.
3. The evaluation results are comprehensive and good.

### Weaknesses
1. What precision is used for evaluation? Is it FP8, FP16, or FP32? 
2. I am wondering how would the proposed kernel could scale beyond 64K length, especially for inference.
3. I am wondering if the authors could provide any further insights into optimizing the proposed kernel for different GPU architectures, such as Hopper and Blackwell.
4. Some system-related works on sparse attention are missing [1-3].

[1] InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management, OSDI 2024.

[2] Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference, MLSys 2024.

[3] ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching, ISCA 2024.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Flash Sparse Attention (FSA)—a new kernel implementation for Native Sparse Attention (NSA) that flips the loop order: instead of iterating queries (outer) and KV blocks (inner) as in the vanilla NSA kernel, FSA iterates KV blocks in the outer loop and batches of query tokens in the inner loop. This change targets a practical inefficiency of NSA on modern GPUs when each GQA group has few query heads, which otherwise forces padding and underutilizes tensor cores. FSA adds (i) an indexed, non-contiguous query loader, (ii) a dedicated reduction kernel, and (iii) a separate online-softmax statistics kernel. Experiments on H20/H200 GPUs report up to 3.5× kernel speedups vs. NSA and 1.25× training / 1.36× prefill speedups on Llama3-8B, Qwen3-14B, and Qwen2.5-32B for long contexts.

### Strengths
1. The paper is well motivated. It identifies padding-driven inefficiency in NSA for common small-g GQA regimes and addresses it directly via loop reordering.
2. The paper presents a solid kernel design with detailed techniques such as non-contiguous query batching with early termination; decoupled reduction to avoid atomics; precomputed online softmax stats to maintain numerical correctness.
3. The paper provides comprehensive evaluation with microbenchmarks across GPUs and (BK,T) settings, plus end-to-end training and inference on multiple LLMs and sequence lengths (8k–64k).

### Weaknesses
1. FSA only provides efficiency gains over NSA when each GQA group has few query heads, which limits its impact.

### Questions
typo: L63: "implementation fail" ->  "implementations fail"

### Soundness
3

### Presentation
3

### Contribution
3
