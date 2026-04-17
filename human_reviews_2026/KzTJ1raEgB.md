# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Mixture of Experts (MoE) models have emerged as the de facto architecture for scaling up language models without significantly increasing the computational cost. Recent MoE models demonstrate a clear trend towards high expert granularity (smaller expert intermediate dimension) and higher sparsity (constant number of
activated experts with higher number of total experts), which improves model quality per FLOP. However, fine-grained MoEs suffer from increased activation memory footprint and reduced hardware efficiency due to higher IO costs, while sparser MoEs suffer from wasted computations due to padding in Grouped GEMM kernels. In response, we propose a memory-efficient algorithm to compute the forward and backward passes of MoEs with minimal activation caching for the backward pass. We also design GPU kernels that overlap memory IO with computation benefiting all MoE architectures. Finally, we propose a novel "token rounding" method that minimizes the wasted compute due to padding in Grouped GEMM kernels. As a result, our method SonicMoE reduces activation memory by 45% and achieves a 1.86x compute throughput improvement on Hopper GPUs compared to ScatterMoE's BF16 MoE kernel for a fine-grained 7B MoE. Concretely, SonicMoE on 64 H100s achieves a training throughput of 213 billion tokens per day comparable to ScatterMoE's 225 billion tokens per day on 96 H100s for a 7B MoE model training with FSDP-2 using the lm-engine codebase. On Blackwell GPUs, SonicMoE also achieves a 28.7% and 22.1% relative speedup on the forward and backward pass respectively compared to a highly optimized DeepGEMM baseline on OLMoE-sized 7B MoE models. Under high MoE sparsity settings, our tile-aware token rounding algorithm yields an additional 1.16x speedup on kernel execution time compared to vanilla top-K routing while maintaining similar downstream performance. We open-source all our kernels to enable faster MoE model training. An extended version of this paper can be found on arXiv.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper, SNaX, presents three improvements to current MoE models to allow them to better utilize modern GPU hardware. Specifically, they propose a new implementation that reduces the memory used to store activations during training and allow this activation size to be independent of the expert granularity. Secondly, they improve throughput by overlapping IO with computation. Finally, they propose to constrain the routing strategy such that the number of tokens routed to each expert is a multiple of the tile size of GPUs, reducing the amount of wasted computation resulted from padding. Empirical results show that the propose implementation and algorithm is much more efficient in terms of throughput and memory usage while maintaining performance. It also fosters the development of high-granularity MoE models since it provides better support for such models compared to existing open-source solutions.

### Strengths
- The paper is well-written, the motivation is clear, and the description of the proposed algorithm is easy to understand.
- The experimental results are convincing and the proposed method provides considerable efficiency improvements. The proposed method, which will be open-source as described in Line 107, will be useful for the community at large since it will greatly lower the cost of training large MoE models. Moreover, high-granularity MoE models hold great promise and is the focus of many research works. The method proposed by this paper will be helpful for many researchers.

### Weaknesses
- It would be helpful to see efficiency tests on more hardware other than H100.
- It seems that the focus of this paper lies in training efficiency. It is currently unclear whether parts of the improvements can be used to improve inference efficiency. Nonetheless, a discussion on inference efficiency seems to be important.

### Questions
- Figure 5 and Section 6.2: Floating-point operations per second is usually abbreviated as FLOPS instead of FLOPs.
- I think the citation in Line 264 should be moved to the first reference of Ping-Pong in Line 256. Moreover, "Ping-Pong" and "pingpong" both appears in the paper. The authors, should unify the capitalization for this term.
- Figure 6 is never referred to in the paper.
- Can this method be used for MoE models without a top-K function, such as ReMoE [1]?

[1] ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing

### Soundness
4

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
3

### Summary
This paper addresses the memory bottlenecks and hardware inefficiencies posed by the trend towards increasingly fine-grained and sparse Mixture of Experts (MoE) models. It proposes SNaX, a system and algorithm co-design solution. SNaX tackles these challenges via three core contributions: (1) A memory-efficient MoE algorithm that minimizes activation memory by modifying the backward pass computation graph; (2) An IO-aware GPU kernel design that overlaps memory IO latency with computation ; and (3) A novel 'Token Rounding' routing method designed to reduce wasted compute resulting from tile quantization. Experimental results show that SNaX significantly reduces memory usage (e.g., a 45% reduction for a fine-grained 7B MoE model ) while substantially improving compute throughput (a 1.87x improvement on Hopper GPUs ).

### Strengths
● The paper astutely points out that as MoE models trend towards being more fine-grained and sparse, the training bottleneck is shifting from computation to memory bandwidth.
● Consistent performance improvements over baselines like ScatterMoE and MoMoE across various model scales (1.4B to 120B) strongly demonstrate the practical value of the proposed method.
● 'Token Rounding' is a novel and clever idea that addresses the often-overlooked problem of wasted computation in GEMM operations due to padding, which the paper refers to as 'tile quantization'.

### Weaknesses
● The paper states in Section 6.3 that Token Rounding (TR) is used during training, but it switches back to Token Choice (TC) for evaluation. The authors also admit that "Token rounding is not a token-choice routing method which creates difficulty for autoregressive inference." This is a major concern. The authors do not clearly explain the root cause of this 'difficulty'.
● Furthermore, this training-inference mismatch constitutes a theoretical weakness. Although the results in Table 1 show acceptable PPL and Avg accuracy, it is unclear if this mismatch could negatively impact other aspects of the model, such as generation diversity or the prevalence of 'hallucinations'.
● Moreover, the paper does not discuss the differences in loss convergence between the two methods (TR vs. TC) during the training phase.
● The paper highlights 'Ping-Pong' scheduling and asynchronous TMA as key to its IO-aware kernel design. While their application in SNaX is effective, these scheduling techniques themselves are not new. The authors should more clearly articulate that their core contribution lies in the specific application and adaptation of these techniques to the heavy-epilogue nature of MoE kernels, rather than claiming novelty for the techniques themselves.
● The paper fails to clarify certain technical terms. For instance, what is the "state-of-the-art BF16 MoE kernel" that SNaX is compared against? Additionally, what does the "K dimension" (mentioned in Section 2.1) refer to in the context of GEMM?

### Questions
The paper avoids caching Y_2 by reordering the computation graph. Does this complex backward pass introduce any numerical instabilities, particularly when training in mixed precision (BF16)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a new efficient implementation of MoE operator for reducing the peaking GPU memory of MoE's training process. This work also introduces a grouped-GEMM–aware token rounding method for tackling the wasted tile computation, as the required computation is lower than the minimum value that a hardware unit can compute in one pass. This phenomenon often occurs when training the highly sparse MoEs since each expert is small and some of them may only receive a few tokens in one pass. This work conducts experiments on a broad scale of MoEs (1.4B-120B), and comparisons against ScatterMoE, MoMoE, and MegaBlocks demonstrate its effectiveness.

### Strengths
The writing is clear, and this work targets the efficiency bottlenecks of training highly sparse MoEs, which is an important research direction for the large models community. More and more recent LLMs and VLMs adopt the MoE structure, and as the trending analysis proposed in the appendix shows, recent MoEs tend to increase the sparsity of activated experts.

### Weaknesses
Weakness:
1. Although the motivation of token rounding aims to reduce tile padding waste in grouped GEMM, the proposed method is similar to load balancing and token dispatching strategies. There are many prior works on token-choice and expert-choice. The proposed method's novelty is limited. The related work 2.4 briefly discusses the routing method, but it's better to discuss the hybrid methods for tackling imbalance issues in detail. 
2. May token rounding introduce hardware sensitivity? The strategy depends on grouped GEMM tile sizes and implementation details (which vary across GPUs/libraries/precisions). This could lead to unexpected behavior when open-sourcing the MoE models since the user may fine-tune it on different devices. 
3. The risk of training–inference consistency and behavioral bias. This work trains with token rounding but evaluates with token-choice. Recent work suggests routing consistency matters in RL/align phases; train–infer routing mismatch may increase the chance of mode collapse and bias amplification.

### Questions
see weakness

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
4

### Summary
This paper introduces SNaX, a co-design solution for training fine-grained and sparse MoE models efficiently. The authors identify that increasing MoE granularity and sparsity shifts computation from compute-bound to memory-bound. They propose three contributions: (1) a memory-efficient backward pass maintaining constant activation memory regardless of expert granularity (45% reduction), (2) IO-aware kernels with asynchronous operations achieving 1.87x throughput improvement, and (3) token rounding that aligns token counts to GPU tile sizes, reducing wasted computation by up to 18%.

### Strengths
The paper correctly identifies that modern MoE architectures face a critical shift from compute-bound to memory-bound operations as granularity and sparsity increase, supported by compelling arithmetic intensity analysis. The memory-efficient backward pass cleverly avoids materializing Y₂ through computation reordering, achieving 45% memory reduction while maintaining mathematical equivalence. Token rounding addresses the overlooked tile quantization problem where padding wastes significant compute in sparse MoEs, yielding 18% additional speedup with maintained accuracy. Comprehensive experiments across 1.4B-120B scales show consistent improvements over ScatterMoE (1.87x throughput) and MoMoE, with the planned open-source release providing valuable production-quality kernels to the community.

### Weaknesses
The fundamental incompatibility between token rounding (training) and token-choice (inference) is acknowledged but poorly explained, with authors only stating it "creates difficulty" without technical details or solutions. The training-inference mismatch may impact generation quality beyond perplexity (diversity, hallucinations) but remains uninvestigated. Heavy Hopper-specific dependence (TMA, tile sizes) limits portability with no analysis on A100/TPU performance degradation. While presenting ping-pong scheduling prominently, these are known techniques - the contribution is adaptation to MoE epilogues, not algorithmic novelty. Critical details like the "state-of-the-art BF16 MoE kernel" baseline and GEMM "K dimension" remain undefined, hindering reproducibility.

### Questions
The paper focuses on Hopper GPU optimizations (TMA, async operations). What is the performance degradation on A100 or V100 GPUs? Is SNaX still beneficial without these hardware features, or does it become worse than baselines?

### Soundness
3

### Presentation
3

### Contribution
3
