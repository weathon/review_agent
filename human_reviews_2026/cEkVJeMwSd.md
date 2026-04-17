# TAH-QUANT: Effective Activation Quantization in Pipeline Parallelism over Slow Network

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Decentralized training of large language models offers the opportunity to pool computational resources across geographically distributed participants but faces significant network communication bottlenecks, particularly in pipeline-parallel settings. While pipeline parallelism partitions model layers across devices to handle large-scale models, it necessitates frequent communication of intermediate activations, creating challenges when network bandwidth is limited. Existing activation compression methods, such as AQ-SGD, mitigate quantization-induced errors through error compensation but impose prohibitive memory overhead by requiring storage of previous activations. To address these issues, we introduce TAH-Quant (Tile-wise Adaptive Hadamard Quantization), a novel activation quantization framework designed specifically for pipeline parallelism. Our approach integrates fine-grained tile-wise quantization for precise control, entropy-guided token-level adaptive bit allocation for optimal bit usage, and a Hadamard-based transform with pivot element swapping to effectively suppress quantization outliers. We further provide a theoretical analysis, proving that pipeline parallel training equipped with TAH-Quant maintains a convergence rate of $\mathcal{O}(1/\sqrt{T})$, matching that of vanilla stochastic gradient descent. Extensive experiments on diverse LLM tasks demonstrate that \sys achieves aggressive activation quantization (3-4 bits) ratio, which provides up to 4.3$\times$ end-to-end speedup without compromising training convergence, matches state-of-the-art methods, incurs no extra memory overhead, and generalizes well across different training scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a framework to reduce communication costs in decentralized, pipeline-parallel training of large language models. It addresses the bottleneck of transmitting intermediate activations over slow networks through tile-wise quantization (i.e., channel groups). The authors propose entropy-guided adaptive bit allocation for each quantization group, and for groups containing large outliers, they adopt a Hadamard transform with pivot swapping to suppress these outliers and reduce quantization errors. Theoretically, TAH-QUANT is shown to maintain the same convergence rate, O(1/\sqrt{T}), as standard SGD. Empirical studies on GPT-2-XL and Qwen-2.5-3B across multiple datasets demonstrate that it can quantize activations to 3–4 bits, achieving up to 1.33×–4.3× speedups with negligible accuracy loss and no additional memory overhead compared to baselines like AQ-SGD. The method scales efficiently under slow network conditions and generalizes well across both pretraining and instruction-tuning tasks.

### Strengths
-  To the best of my knowledge, TAH-QUANT’s combination of tile-wise quantization, entropy-based adaptive bit allocation, and Hadamard outlier suppression is conceptually novel and well-motivated.
- The paper provides a rigorous convergence analysis proving the same O(1/\sqrt{T}) rate as vanilla SGD, as well as showing the practical efficiency and speedup under the limited bandwidth.
- The contributions of each module (tile size, adaptive allocation, Hadamard transform) are clearly validated, strengthening the empirical credibility of the method

### Weaknesses
- Although the authors show empirical training speedups under limited bandwidth, the computational costs of the bit allocation, outlier detection heuristic, and Hadamard transform are not discussed.
- The paper also lacks an analysis of memory reduction versus compression cost versus batch size versus bandwidth. I recommend that the authors include a micro-benchmark demonstrating the speedup of transmitting a batch of data while sweeping across different batch sizes and bandwidth settings.
- The analysis of the bit allocation mechanism is missing. This component likely affects both the total number of bytes transferred between machines and the overall training speedup. An ablation study on the top-p% threshold is necessary. Furthermore, the hyperparameter p is not specified (or please point me where it is defined or described in the paper).
- Although Algorithm 1 outlines the overall training process, the detailed procedures for bit allocation, outlier detection, and the Hadamard transform are not presented. I recommend that the authors include a pseudo-code algorithm or a schematic figure to illustrate the detailed workflow of the proposed method.
- The experimental models are relative small. Only Qwen2.5-3B (3B parameters) and GPT-2XL (1.5B parameters) are included.

### Questions
- Would the authors provide a discussion on the computational cost of quantization?
- Would the authors include a micro-benchmark evaluating different combinations of batch sizes and bandwidths?
- Could the authors report the memory reduction achieved under different values of top-p%, along with an ablation study analyzing the effect of varying p across different layers and experimental models?
- Would the authors present a detailed description of the proposed quantization and bit-allocation process, perhaps in the form of pseudo-code or a schematic figure?
- If possible, could the authors extend the experiments to larger models to better align with the paper’s discussion on “democratizing large-scale LLM training”?
- How do the authors define “slow network” or “low bandwidth”? Is the 10 Gbps bandwidth used in the paper intended as a formal benchmark from the communication systems domain?

### Soundness
2

### Presentation
2

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
The paper addresses the communication bottleneck of pipeline-parallel training for LLM by compressing activations. It proposes TAH-QUANT (Tile-wise Adaptive Hadamard Quantization) that has: (i) fine-grained tile-wise quantization, (ii) entropy-guided, token-level adaptive bit allocation, and (iii) a Hadamard transform with pivot element swapping. Unlike error-compensation methods (e.g., AQ-SGD), TAH-QUANT claims no extra memory overhead. The authors also provide a convergence analysis showing comparable rates to vanilla SGD, and report activation compression yielding up to 1.33X end-to-end speedups without harming convergence.

### Strengths
- Important novelty: this paper proposed multiple novel techniques in quantizing communication in LLM training, such as tile-wise quantization, entropy-guided bit allocation, Hadamard transform, etc. 
 - Good theory: authors proved theoretically TAH-QUANT has similar convergence rate as vanilla SGD. 
 - Convincing results: TAH-QUANT has up to 1.33X speedup without hurting convergence.

### Weaknesses
- The outlier detection heuristic in Section 3.3 is hand made. Could authors explain how well it generalizes to other training settings? 
 - More experiments may be needed: 
   - The 3090 GPU is not a powerful machine. How will the performance be if we conduct experiments on industrial computation resources, such as A100, H100, H200? 
   - Why do we use 8 pipeline parallel stages on 8 GPUs? In practice, we may not need so many PP stages. If we set the number of PP stages to be smaller than the number of GPUs, for example, on 8 GPUs, if we set PP=2, how will the performance be?

### Questions
- Can we use the proposed TAH-QUANT method in other training techniques, in addition to pipeline parallel? For example, can we extend it to tensor parallel, data parallel, expert parallel?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles the challenge of activation communication bottlenecks in pipeline-parallel LLM training, especially over slow or low-bandwidth networks. The authors introduce TAH-QUANT, a quantization framework that compresses activations more efficiently without hurting convergence. It combines three key ideas: (1) tile-wise group quantization for localized precision control, (2) entropy-guided adaptive bit allocation that assigns either 3 or 4 bits per token based on its information content, and (3) a Hadamard transform with pivot swapping to spread outliers and reduce quantization error.

The paper also includes a theoretical convergence analysis under mild assumptions, showing that TAH-QUANT retains the same 
√T  convergence rate as standard SGD. Experiments on GPT-XLand Qwen2.5 (for both fine-tuning and short pretraining) show that TAH-QUANT matches FP16 and AQ-SGD in accuracy while achieving substantial end-to-end speedups

### Strengths
TAH-QUANT directly tackles a practical and underexplored challenge: activation communication in pipeline-parallel training over limited-bandwidth networks. The method is conceptually simple but well thought out. The combination of tile-wise quantization, adaptive bit allocation, and a Hadamard transform with pivot swapping is elegant and easy to integrate into existing systems.

The theoretical analysis is a strong point. The authors provide √T  convergence guarantees, showing that the method maintains the same rate as standard SGD. This is a meaningful theoretical contribution that helps justify the approach beyond empirical performance.

### Weaknesses
The comparison between TAH-QUANT and AQ-SGD feels incomplete. While TAH-QUANT is meant to serve as a more efficient replacement, the experiments don’t fully explore how the two methods compare on broader benchmarks or zero-shot evaluations. Without this, it’s hard to tell whether TAH-QUANT consistently matches or outperforms AQ-SGD in downstream performance.

The loss curves in Figure 2 also don’t add much clarity. They show that both methods converge, but not how or why their trajectories might differ. If the losses are nearly identical, that deserves more explanation. What exactly enables TAH-QUANT to behave so similarly to AQ-SGD, given the absence of error compensation?

It would also help to include additional quantization settings, like loss curves with the different tile sizes or bit allocation ratios, to show whether TAH-QUANT’s behavior holds across a range of precision levels.

### Questions
1. Can you include zero-shot evaluation results across more benchmarks to make the comparison with AQ-SGD more convincing?

2. If the loss trajectories between TAH-QUANT and AQ-SGD are almost identical, what’s the underlying reason? Does the Hadamard transform or adaptive bit allocation implicitly mimic error feedback?

3. Have you tested multiple quantization configurations (for example, varying INT3/INT4 splits or tile sizes) to show how sensitive convergence is to these parameters?

4. Can you clarify whether the observed speedups mainly come from algorithmic efficiency or implementation differences compared to AQ-SGD?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the communication bottleneck in decentralized large language model training with pipeline parallelism, which is challenge for democratizing LLM training such as enabling participation from universities or startups with slow network connections. The authors propose TAH-QUANT, a tile-wise adaptive Hadamard quantization framework that integrates three core components: (1) fine-grained tile-wise group quantization for localized error control; (2) entropy-guided token-level adaptive bit allocation (3–4 bits) for efficient compression; and (3) Hadamard transform with pivot swapping to suppress quantization outliers.

The paper evaluated across 5 tasks (language modeling, instruction tuning, pre-training) on GPT2-XL (1.5B) and Qwen2.5-3B (3B), TAH-QUANT achieves up to 4.3× end-to-end speedup over FP16 in slow networks

### Strengths
1. TAH-QUANT optimizes pipeline parallelism this by eliminating extra memory overhead while maintaining low-bit quantization (3–4 bits), which is  critical design choice for deployment.

2. Unlike many quantization works that rely solely on empirical validation, the paper provides a solid convergence analysis under standard stochastic optimization assumptions.

### Weaknesses
1. The experiment is restricted to Qwen2.5-3B (3B parameters) for only 6,000 iterations on C4. It would be better to anaylyze model at different scales (smaller and larger) to analysis of how TAH-QUANT performs when model size or training duration changes.

2. The outlier detection heuristic uses a fixed threshold without justification. Different LLMs (e.g., GPT-2 vs. Qwen) or tasks have distinct activation distributions.

3. The paper uses naive quantization for gradients in the backward pass (instead of TAH-QUANT) and claims this is due to more computation load enabling communication-computation overlapping. However, no experiments compare this choice to using TAH-QUANT for gradients.

### Questions
1. Can you provide a breakdown of end-to-end training time to show that the transform’s overhead does not offset communication savings? For example: Does the Hadamard step increase per-batch compute time by <5% , and how does this scale with tile size?


2. The paper only compares TAH-QUANT to AQ-SGD (a 2021 method) as a low-bit activation compression baseline. However, recent works (e.g., 2023–2024) like LightQuant (NeurIPS 2023) or PipeQuant (ICML 2024) also target pipeline parallelism with low memory overhead. Can you provide a qualitative analysis of how TAH-QUANT’s design differs from these methods?

3. Most modern LLM trainings use mixed precision (e.g., FP16 for weights, FP8 for gradients) to reduce memory/compute. The paper trains models in FP16. Could the author address whether TAH-QUANT is compatible with mixed precision?

### Soundness
3

### Presentation
3

### Contribution
3
