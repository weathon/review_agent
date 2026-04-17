# Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable success across diverse artificial intelligence tasks, driven by scaling laws that correlate model size and training data with performance improvements. However, this scaling paradigm incurs substantial memory overhead, creating significant challenges for both training and deployment. While existing research has primarily addressed parameter and optimizer state memory reduction, activation memory—particularly from feed-forward networks (FFNs)—has become the critical bottleneck, especially when FlashAttention is implemented. In this work, we conduct a detailed memory profiling of LLMs and identify FFN activations as the predominant source to activation memory overhead. Motivated by this, we introduce the Mixture-of-Channels, a novel FFN architecture that selectively activates only the top-$K$ most relevant channels per token determined by SwiGLU’s native gating mechanism. MoC substantially reduces activation memory during pre-training and improves inference efficiency by reducing memory access through partial weight loading into GPU SRAM. Extensive experiments validate that MoC delivers significant memory savings and throughput gains while maintaining competitive model performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Large Language Models (LLMs) have demonstrated remarkable success across various domains, driven by increases in both model and data scale. However, this growth also leads to significantly higher memory requirements. This paper focuses specifically on activation memory in feed-forward networks (FFNs). The authors propose Mixture of Channels (MoC), an FFN architecture that selectively activates only the Top-K most relevant channels per token, as determined by the native gating mechanism of SwiGLU. As a preliminary step to support their approach, the authors analyze the memory requirements of an LLM block and characterize the distribution of FFN activations. Additional sections detail the pre-training and inference procedures using MoC, followed by experiments evaluating MoC in terms of both accuracy and efficiency.

### Strengths
* The paper is clear and well structured.
* The empirical results present compelling evidence in support of the proposed method.
* Developing a kernel optimized for MoC on specific hardware is highly impressive. In many cases, the transition from a strong theoretical concept to a practical implementation ends at the stage of custom optimization. The fact that the authors went further to implement this kernel demonstrates a valuable contribution toward making MoC practically applicable.

### Weaknesses
* Theorem 1 isn't very clear - if $b\geq a$, $d_{moc}$ could exceed $f_{ffn}$ which would reduce efficiency. Claiming that MoC is as at least as good as dense FFN does not make a lot of sense. Typically, improving efficiency comes at the cost of reduced expressive power. An interesting direction would be to analyze how closely the model can approximate the original function given a certain efficiency constraint.

### Questions
* The primary motivation for MoC stems from the memory profiling presented in Section 2. However, this analysis was conducted on a relatively simple model, LLaMA 2. Would the observations differ if the same analysis were performed on a more recent model or one that incorporates a mixture-of-experts (MoE) module instead of a dense FFN?
* Inference efficiency - While it is true that for small batch sizes inference performance is primarily limited by I/O bounds, for larger batches it can become compute-bound and dominated by the FLOP count. Could the authors provide additional elaboration or analysis for this scenario?
* Can the authors provide additional downstream task evaluation to table 2 for a more complete image of the models comparison?
* The authors present their approach primarily as a pretraining method; however, an interesting alternative would be to apply MoC as a fine-tuning technique that adapts an already trained model to reduce its memory footprint. Using MoC in this way could also enable evaluation on larger-scale models (≥7B parameters).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed the mixture of channels framework, which preserves the top-k most active channels in FFN during training. Combined with the customized Triton kernel, it can save lots of GPU memory and speed up the end-to-end throughput by 1.13x. Across multiple scales, the MoC can achieve comparable ppl performance while significantly reducing the GPU memory.

### Strengths
1. The idea of inducing sparse computation during pretraining is reasonable, and the memory footprint is significantly reduced.
2. Experiments are conducted on diverse model structures.
3. The paper is well written and easy to follow.

### Weaknesses
1. The end-to-end latency speedup is fair on a single batch setting, which may also be minor under the batching inference.
2. The top-k-based training may induce instability due to the indifferentiable characteristics. I wonder how it compares with those softened masking methods.
3. There is no module-wise ablation study or profiling for the design kernel. I suggest a detailed profiling for the designed kernel.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Mixture-of-Channels (MoC), a new feed-forward layer design that introduces activation sparsity within each Transformer FFN by selecting only the Top-K most activated hidden channels (neurons) for each token. The approach leverages the existing gating signal from SwiGLU to determine which channels are active, thereby reducing the number of intermediate activations stored and updated. The authors report reduced activation memory and improved training and inference throughput, maintaining decent model performance.

### Strengths
1. Clarity:
The paper clearly describes how MoC can be implemented as a more efficient FFN layer using existing SwiGLU gating values.
2. Efficiency benefit:
The approach can reduce activation and gradient memory during training and reduce inference latency on a single token.
3. Compatibility:
The proposed model can be integrated with modern optimized LLM kernels and systems.

### Weaknesses
1. Incremental novelty:
MoC is conceptually very close to CATS (Lee et al., 2024), differing mainly in using topK instead of threshold.
2. Fixed-K limitation:
The Top-K value is globally fixed across tokens, even though different tokens may require varying numbers of active channels. This may underutilize model capacity or cause redundancy for simple tokens.
3. Lack of accuracy validation:
The paper mainly evaluates on memory and throughput metrics. The datasets used are simple, and the models are small and outdated, and accuracy comparisons are missing for modern baselines (e.g., CATS (Lee et al., 2024), MoEfication (Zhang et el., 2021), Learn-to-be-Efficient (Zheng et al., 2024), etc.).

### Questions
How does MoC handle inference when different tokens in the same batch activate different channel subsets? Does this reduce inference efficiency?

### Soundness
2

### Presentation
3

### Contribution
2
