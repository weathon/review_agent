# Layer-wise Sensitivity-aware Sparsity Allocation for Efficient LLM Inference

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Large Language Model (LLM) inference presents substantial computational challenges when executed on commodity hardware, thereby necessitating the development of efficient acceleration techniques. While existing approaches predominantly focus on uniform compression strategies, they neglect the heterogeneous sensitivity patterns exhibited across different transformer layers. In this paper, we introduce Adaptive Sparsity Allocation Framework (ASAF), a novel approach that integrates rotation-based low-bit quantization with layer-wise adaptive sparsity allocation. The framework comprises two sequential phases with dynamic programming strategy. Phase 1: coarse-grained optimization that determines the optimal number of layer groups and narrows sparsity rate search intervals. Phase 2: fine-grained optimization that determines precise consecutive layer allocation and exact sparsity rates within each group. The joint optimization of layer grouping decisions and sparsity rate assignments creates a combinatorial explosion in the solution space, rendering brute-force approaches computationally prohibitive. To address this challenge, we employ a dynamic programming strategy that efficiently decomposes the exponential search space into manageable subproblems across both phases, achieving practical computational efficiency while guaranteeing global optimality. Extensive experiments conducted on the Llama-2 model family reveal that our proposed framework sustains benchmark accuracy degradation within 1\%, concurrently achieving up to 3.63$\times$ prefill acceleration and 12.63\% memory reduction on NVIDIA RTX 3090 GPUs. This work advances beyond uniform compression strategies by recognizing and exploiting the distinct sensitivity characteristics of different transformer layers, thereby establishing a new paradigm for adaptive LLM compression on commodity hardware.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ASAF, an Adaptive Sparsity Allocation Framework for efficient LLM inference. It addresses the limitation of uniform compression by recognizing that different transformer layers have varying sensitivity to sparsification. ASAF combines quantization with sparsification via a two-phase, dynamic programming-based optimization to allocate sparsity adaptively across layer groups. This approach minimizes computational FLOPs while keeping accuracy degradation under 1%, achieving up to 3.63× prefill acceleration and 12.63% memory reduction on Llama-2 models.

### Strengths
- Proposed a joint optimization framework for quantization and sparsification, identified the problem, formulated it mathematically, and provided a solution.
- Adopted an optimization perspective with a two-phase approach, and proposed a method to tackle the combinatorial explosion challenge.
- Conducted relatively comprehensive experiments.

### Weaknesses
1. The basis for grouping is unclear, and the rationale for assuming consecutive layers can share sparsity is not justified. Grouping merely addresses the combinatorial explosion from a computational standpoint, but the motivation behind this grouping needs further elaboration. 
2. The model was only tested for compression ratio and prefill speed on the Llama-2 series; no results are provided for other models. Llama-2 is already outdated. Additionally, for Llama-3, only accuracy experiments were conducted. However, differences in model architecture and training methods may affect the compression efficacy. Measuring only accuracy cannot fully demonstrate the method's effectiveness. 
3. Prefill speed was only measured at an input length of 2K; what about other lengths such as 512, 4K, or longer?
4. The baseline method was proposed over 1 to 2 years ago. Are there comparisons with more recent methods from the past year?

### Questions
1. What assumption is the sharing of sparsity among consecutive layers based on, and is there any experimental validation for it?
2. To verify the method's generalizability, I suggest adding compression ratio results on models of different sizes from another series.
3. Prefill speed was only measured at an input length of 2K; what about other lengths such as 512, 4K, or longer?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This submission introduces Adaptive Sparsity Allocation Framework (ASAF), an approach for efficient acceleration of LLM inference that combines rotation-based quantisation with layer-wise adaptive sparsity. The selection of the deployed configuration is formulated as a 2-stage dynamic programming optimization approach, in order to make the exploration of the combined search space computationally feasible, that initially determines high-level structural configuration parameters of both approximations (the optimal number of layers groups and sparsity search intervals), followed by fine-grained optimisation of exact consecutive layer allocation and sparsity rates of each group.

### Strengths
- This work studies a timely and interesting problem, by combining approximations that are often studied in isolation in efficiency works for LLM inference.
- The proposed approach demonstrates considerable speed-up to meaningful baselines (in the examined prefill stage), with controlled impact on acucracy.

### Weaknesses
- The main drawback of the proposed approach is the lack of consideration of the whole LLM inference process (prefill + decoding). Although it is acceptable for an approach to focus its optimisation efforts solely in one of the two phases, the impact of the proposed solution to the total inference time (and discussion on the applicability or impact of the proposed method to the other remainder process) is required to fairly evaluate the contribution and effectiveness of the proposed method. 
- Additionally, it is unclear how the proposed hierarchical search formulation would compare to more naive heuristic exploration baselines on the combined optimisation space (constrained to similar search time).

### Questions
Please consider replying on the concerns raised above.

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
This paper introduces ASAF (Adaptive Sparsity Allocation Framework), a method for making LLM inference more efficient by combining rotation-based quantization and layer-wise adaptive sparsity. Unlike prior work that applies uniform compression, ASAF dynamically assigns different sparsity levels to layers based on their sensitivity. The approach uses a two-phase dynamic programming optimization:
1- Coarse-grained phase: Decides how to group layers and narrows sparsity ranges.
2- Fine-grained phase: Determines exact sparsity rates and layer assignments.
Tested on Llama-2 models (7B–70B), ASAF achieves up to 3.6× faster inference and 12.6% lower memory use, with <1% accuracy drop compared to baselines like QuaRot.

### Strengths
- Framing sparsity allocation as a layer-grouped, constrained optimization problem with dynamic programming is elegant.
- The mathematical formulation is clean, and the dynamic programming approach (Algorithms 1 & 2) is well explained. The inclusion of tabulation to precompute FLOP and accuracy costs is a smart engineering choice that enhances reproducibility.

### Weaknesses
- The experiments emphasize prefill acceleration but offer less analysis of end-to-end latency or real-world throughput improvements.
- The proposed method involves precomputation (tabulation tables for FLOPs and accuracy degradation). This could limit practicality for very large models or rapid iteration cycles.
- It is not entirely clear how scalable the DP-based search is as model depth increases beyond 70B-scale architectures.
- The paper doesn’t deeply probe why certain layers are more sensitive or how the learned sparsity patterns correlate with model internals (e.g., attention vs MLP layers).

### Questions
- The paper mentions dynamic programming and tabulation to efficiently explore the search space, but how does computational complexity scale with model depth (e.g., 70B to 180B parameters)?
- The optimization constraint depends on an estimated accuracy degradation function. How is this function obtained in practice: via heuristics, proxy metrics, or direct evaluation?
- Do the learned sparsity rates correlate with identifiable layer characteristics (e.g., attention layers being less prunable than MLP layers)?
- The experiments focus mainly on the Llama-2 family. How well does ASAF generalize to architectures with different scaling patterns (e.g., Mistral, Falcon, or OPT)?
- Reported “prefill acceleration” results are strong, but what is the effect on end-to-end latency or tokens per second under realistic batch sizes and generation lengths?
- How sensitive is ASAF to delta (the allowed accuracy degradation) and the sparsity range?

### Soundness
2

### Presentation
3

### Contribution
2
