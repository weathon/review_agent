# Astraea: A Token-wise Acceleration Framework for Video Diffusion Transformers

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Video diffusion transformers (vDiTs) have made tremendous progress in text-to-video generation, but their high computational demands pose a major challenge for practical deployment. While existing studies propose acceleration methods to reduce workload at various granularities, they often rely on heuristics, limiting their applicability.
We introduce Astraea, a framework that searches for near-optimal configurations for vDiT-based video generation with a performance target. At its core, Astraea proposes a lightweight token selection mechanism and a memory-efficient, GPU-parallel sparse attention strategy, enabling linear reductions in execution time with minimal impact on generation quality. Meanwhile, to determine optimal token reduction for different timesteps, we further design a search framework that leverages a classic evolutionary algorithm to automatically determine the distribution of the token budget effectively. Together, Astraea achieves up to 2.4x inference speedup on a single GPU with great scalability (up to 13.2x speedup on 8 GPUs) while retaining better video quality compared to the state-of-the-art methods (<0.5% loss on the VBench score compared to the baseline vDiT models).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed ASTRAEA, a token-wise cache framework for efficient video generation. ASTRAEA proposes a lightweight token selection mechanism and a memory-efficient, GPU-friendly sparse attention strategy. ASTRAEA also proposed an evolutionary algorithm to automatically determine the distribution of the token budget across timesteps. Experiments show that ASTRAEA achieves better generation quality with lower latency under different video diffusion transformers.

### Strengths
1.The problem studied in the paper is quite valuable, and the acceleration of video DiT generation is crucial.

2.ASTRAEA should be able to be migrated to various DiT architectures with good generalization.

3.The method explanation is relatively clear and easy to understand.

4.The performance and acceleration effect of ASTRAEA appear to be relatively good.

### Weaknesses
1.The acquisition of LSE Score should not be directly provided by existing attention methods such as FlashAttention. How is the efficient and memory friendly acquisition described in the paper achieved? Can specific resource consumption be reported?

2.The sparse attention in Fig.3 appears to bring additional computational complexity, why does the paper claim less computational complexity compared to sparsification for QK? Sparse QK theoretically should be faster, and faster hardware implementation can be achieved using a kernel designed like in SVG. Can you provide a more detailed explanation.

3.The evolutionary algorithm used in the paper appears to be very time-consuming, which puts it at a certain disadvantage compared to other training-free methods. Can you report the performance comparison without using the evolutionary algorithm?

4.The paper lacks the performance analysis of individual components.

5.It seems that the statement in line 463 is incorrect and contradictory to line 464.

6.What are the hyperparameters set in Eq.4 and how should they be set? The paper lacks additional analysis on whether these hyperparameters are sensitive.

### Questions
Please see above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents **ASTRAEA**, a token-level acceleration framework designed to mitigate the high computational costs of video diffusion transformers (vDiTs). The method introduces a lightweight token selection mechanism, a GPU-friendly sparse attention strategy, and an evolutionary search framework to find optimal token budgets across denoising steps.

### Strengths
- **Excellent Scalability**: A key strength of this work is its demonstrated scalability. The framework achieves strong performance scaling across multiple GPUs, showing up to 13.2x speedup on 8 GPUs, which highlights its practical utility for large-scale inference.

- **Good Performance Gains**: The proposed method delivers acceptable and noteworthy performance gains, achieving up to 2.4x inference speedup on a single GPU while maintaining high video quality (e.g., <0.5% VBench loss).

- **Clear Presentation**: The paper is well-written and logically organized. The core concepts, including the token selection mechanism and the sparse attention strategy, are presented clearly, making the work easy to follow.

### Weaknesses
- **Evolutionary Algorithm (EA) Search Cost**: The EA search for finding the optimal token distribution is computationally expensive, with an average search time of 82 GPU hours and some models taking up to 139 hours. While the authors rightly point out this is an offline cost, this is a significant practical hurdle. Further work could be done to optimize this search or analyze potential inefficiencies in the EA process.
- **Lack of Theoretical Justification for EA**: The paper's justification for using an EA is primarily empirical. While the experiments show great results, the paper would be stronger with a more detailed explanation of why EA is well-suited for this specific problem over other search methods. The current justification—that other methods like NAS have "substantial search times"—could be expanded upon with a brief analysis of the search space itself.
- **Missed Structural Optimizations**: The token selection framework appears to treat all tokens uniformly. It does not seem to leverage the inherent structures of video models, such as spatial-temporal sparsity. Furthermore, it does not analyze or exploit the different roles of text tokens versus spatio-temporal video tokens, which represents a missed opportunity for a more granular and potentially more effective optimization.
- **No Hyperparameter Sensitivity Analysis**: The core token selection metric in Equation 4 relies on two key hyperparameters, $w_{\alpha}$ and $w_{\beta}$, to weigh token significance and the non-selection penalty. The paper does not provide a sensitivity analysis for these hyperparameters. This makes it difficult to assess the robustness of the method and understand how critical fine-tuning these parameters is to achieving the reported results. The ablation study in Section 4.3 focuses on high-level design choices rather than these specific equation parameters.

### Questions
See weakness.

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
2

### Summary
This paper presents  a framework for accelerating vDiTs via token selection. The method leverages the multi-timestep nature of diffusion models for token selection, resulting to a GPU-efficient sparse attention strategy, achieving linear inference speedup with minimal quality degradation.

### Strengths
- The idea is novel and conceptually makes sense. It cleverly leverages the multi-timestep nature of diffusion models for token selection.
- The method is supported by solid experiments, and the resulting performance is highly impressive.

### Weaknesses
Since the paper claims to outperform native sparse attention in line 233 to line 241, would it be possible to include a performance comparison with state-of-the-art methods, such as SVG2 [1]?

[1] Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation

### Questions
- Line 203, 204: `We use the LSE score in the previous timestep because LSE scores do not vary across timestep`. What is the basis for this claim?
- Why does Table 1, lines 379–380 say `on two vDiT models` while in fact there are three models?

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
2

### Summary
This paper proposes ASTRAEA, a token-wise acceleration framework for video diffusion transformers (vDiTs). Unlike prior acceleration methods that rely on step reduction or blocking caching, ASTRAEA introduces:

1. A lightweight token selection mechanism that dynamically identifies important tokens at each denoising step with negligible overhead.
2. A GPU-friendly sparse attention strategy that selectively computes queries while retaining full keys/values, ensuring correctness and parallelizability.
3. An evolutionary search framework to automatically allocate token budgets across timesteps, avoiding manual hyperparameter tuning.

### Strengths
1. Operating at the token level, it offers finer granularity than step- or block-level methods, addressing a previously underexplored dimension.
2. Evolutionary algorithm effectively allocates token budgets across timesteps, reducing reliance on heuristics.
3. Clear evidence of multi-GPU efficiency, which is crucial for industrial deployment.

### Weaknesses
1. Evolutionary search may still be computationally expensive, limiting practicality in large-scale or frequently updated deployments.
2. Generality of prompts: Search is conducted on a small set of prompts; broader validation on diverse datasets would strengthen claims of generalization.

### Questions
1. Generalization: If the token budget allocation is searched on one dataset or prompt set, how well does it transfer to unseen prompts or domains?
2. Integration with retraining methods: Could ASTRAEA be combined with distillation-based step reduction to achieve further acceleration?

### Soundness
3

### Presentation
3

### Contribution
2
