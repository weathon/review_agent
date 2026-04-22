# Adaptive Nonlinear Compression for Large Foundation Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4, 4

## Abstract
Despite achieving superior performance, large foundation models (LFMs) have substantial memory requirements, leading to a growing demand for model compression methods.
While low-rank approximation presents a promising hardware-friendly solution, existing linear methods suffer significant information losses due to rank truncation. Nonlinear kernels can enhance expressiveness by operating in higher-dimensional spaces, yet most kernels introduce prohibitive overhead and struggle to support adaptive rank allocation across heterogeneous matrices.
In this paper, we propose a compression method called Nonlinear Low-Rank Approximation with Adaptive Budget Allocation (NLA). 
Instead of relying on linear products, we employ piecewise-linear kernels with a forward-pass optimization operator to approximate weight matrices, enhancing the recovery of high-rank weight matrices from low-rank matrices.
Moreover, considering the heterogeneous representation abilities and dynamic sensitivities of different weight matrices, we adaptively allocate the compression ratio of each weight matrix during the re-training process by cubic sparsity scheduling.
Through evaluations on large language models and vision models across various datasets, NLA demonstrates superior performance while achieving a higher compression ratio compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel compression method for model weights with non-linear low-rank parametrization and adaptive budget allocation. Comparing to other methods, the authors claim improved perplexity / accuracy at equal or higher compression rates, which implies lower memory usage, at the cost of a modest inference-time overhead. The experimental validation includes LLM pretraining and reasoning benchmarks, ImageNet-1K classification for vision transformer, and comparisons against other low-rank compression methods.

### Strengths
The proposed method demonstrates significant performance retention at compression rates over 50%, for example, LLaMA-7B with NLA at 70% compression outperforms SVD-LLM baseline at 60% on Wikitext-2 perplexity. The piecewise linear kernel allows for an efficient implementation thanks to reformulation through a series of matrix multiplications.

### Weaknesses
1. My main concern is an unexplored computational complexity introduced by the method. Naturally, using more expressive non-linear layer structure would allow for a better model performance compared to linear methods, but it does so at a significant computational cost. In Table 15 the inference time is x1.5 higher for NLA compared to SVD, even at a more aggressive compression ratio. The higher inference cost makes the method less likely to be adopted in real deployment. 
Training efficiency is also not discussed; it's unclear how much longer retraining took compared to other baselines.
2. It is not explained how TinyBert was applied for compression of the Swin transformer model. TinyBert is a language model, and if the authors have used the Transformer distillation method proposed in the said paper (Jiao et al., 2020), they should be clearer with their wordings.
3. Different compression rates were used to compare methods in Tables 1, 3, 4, 14, 15, which doesn’t allow for the fair comparison. For example, in Table 3 the compression rates of 40, 50 and 60% were reported for the first 3 methods, then 50 and 60% percent for SVD-LLM and 50 and 70% for NLA.
I understand the authors’ aim to highlight how even with more aggressive compression the proposed method stayed competitive, but all the numbers should be reported nonetheless. A standardized approach to comparisons could be: a) to compare performance at the same compression rates, b) to compare compression rates respective to the same performance level(s), c) to create a plot with (compression rate, performance) axes and compare the trade-off curves for different methods. 
4. The reported experiments lack consistency: the authors report the main results only for the LLaMA-7B model, then in Section 4.2 they use the OPT family models for several ablation studies. 
5. Algorithm 1 lacks analysis with regard to its hyperparameters. While the authors propose the adaptive search algorithm for the optimal values of h, they fix the hyperparameter r = 20. As performance depends on both of them, the dependency on r is not explored.
6. The proposed adaptive budget allocation delivers small absolute gains. It's not the main source of improvement, but it adds computational complexity in training.

### Questions
1. Please report the performance (perplexity / accuracy) vs. inference time for NLA and other methods. Can NLA outperform other methods at the same (or smaller) inference time and memory budget? 
2. How does NLA impact training times?
3. How exactly was TinyBert applied to the Swin Transformer baseline?
4. Is the strategy of fixing r=20 and varying h optimal? Please justify.
5. How does NLA compare with other compression-based methods like SparceGPT and GPTQ?

### Soundness
2

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
This paper proposes NLA (Nonlinear Low-Rank Approximation with Adaptive Budget Allocation), a new model compression framework for large foundation models (LFMs) such as LLaMA-7B and Swin-Transformer. NLA replaces conventional linear low-rank approximations (e.g., SVD-based) with a piecewise-linear kernel that captures nonlinear relationships, allowing higher expressiveness and reduced information loss under high compression ratios. To manage varying layer sensitivities, it introduces an adaptive compression mechanism that allocates compression budgets dynamically via cubic sparsity scheduling based on parameter importance scores. Additionally, a custom forward algorithm enables nonlinear low-rank inference without reconstructing full weight matrices, reducing memory usage. Experiments on language and vision benchmarks show that NLA achieves substantially lower perplexity and comparable or higher accuracy at significantly higher compression ratios than prior methods (e.g., SVD-LLM, FWSVD, ASVD).

### Strengths
## Novelty: 

The paper introduces Nonlinear Low-Rank Approximation (NLA) — a fundamentally new compression paradigm that integrates nonlinear kernel-based approximation with adaptive compression ratio allocation. Unlike traditional SVD-based methods that rely purely on linear decomposition, NLA leverages a piecewise-linear kernel to approximate weight matrices in a higher-dimensional space, effectively recovering information lost during rank truncation.

## Differences from Prior Work:

- Prior low-rank compression methods (e.g., SVD-LLM, ASVD, FWSVD) perform linear approximations, inevitably losing important information at high compression ratios.
- Kernel-based nonlinear approaches have existed but are computationally expensive due to explicit matrix reconstruction. NLA overcomes this via a custom forward operator that performs nonlinear propagation without reconstructing full weight matrices.
- Previous methods use uniform compression across layers, while NLA introduces adaptive budget allocation using importance scoring and cubic sparsity scheduling, enabling per-layer dynamic compression control.

### Weaknesses
### Fairness of Experimental Comparison:

It is unclear whether the proposed NLA and all baseline methods were retrained under the same training protocol. Appendix C provides insufficient training details, particularly for the vision model experiments. For example, there is no description of retraining settings for Swin-Transformer, nor any indication of whether the counterparts in Table 4 were trained under identical configurations. Providing consistent and transparent training settings is essential for a fair comparison.

### Generality and Model Coverage:

The baselines used in this paper are relatively outdated (LLaMA-7B, OPT, Swin Transformer). To demonstrate the generality of NLA, it would be valuable to include experiments on more recent models such as Qwen2.5/Qwen3 or DeepSeek. For vision, evaluating on modern architectures like Vision Transformer (ViT) more widely used on multimodal VLMs would better reflect current practice.
Additionally, scalability across different model sizes should be examined. For instance, if compressing Qwen3-8B by 50% leads to performance degradation such that it underperforms the smaller Qwen3-4B, the practical value of compression becomes questionable. A similar analysis should be provided for vision models (e.g., comparing ViT-B-NLA vs. ViT-S).

### Throughput and Memory Efficiency Claims:
The paper should report throughput and memory consumption alongside perplexity and accuracy, ideally comparing them directly against SVD-based baselines (FWSVD, ASVD, SVD-LLM). While Table 4 presents FLOPs and throughput, the results are inconsistent — NLA achieves higher compression (54.2%) but lower throughput than the original Swin-Transformer. This inconsistency suggests that FLOPs alone cannot justify claims of “comparable throughput.” As shown in Table 4, even with lower FLOPs, NLA exhibits worse throughput than PELA, contradicting the statement at line 462 (“with comparable computation and throughput”). These efficiency claims should be tempered or supported with real GPU inference timing and memory profiling.

### Questions
1. Were the NLA-compressed models retrained under the same training protocols and hyperparameters as the baseline methods (e.g., SVD-LLM, ASVD, FWSVD)?

2. Could you specify the retraining setup (optimizer, learning rate, epochs, batch size, etc.) used for vision models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
NLA compresses neural networks using nonlinear kernel functions instead of linear matrix products, achieving better quality than SVD methods. However, it's much slower at inference than SVD methods, lacks fine-tuning results on key benchmarks, and needs broader baseline comparisons and theoretical justification.

### Strengths
1. Novel application of nonlinear kernels to compression
2. Consistent exp improvements
3. Clever efficient algorithm

### Weaknesses
1. Incomplete Evaluation: Table 3 lacks fine-tuning results for common sense reasoning, making it impossible to fairly assess NLA vs. SVD methods post-fine-tuning. This is important given that fine-tuning is standard practice.
2. Computational Cost: The paper shows better final quality but shower than SVD methods, and likely orders-of-magnitude longer compression time. 
3. Limited Theoretical Foundation: No convergence guarantees, approximation bounds, or principled hyper-parameter selection. Purely empirical. When and why does this work?

### Questions
1. Table 3 shows common sense reasoning results WITHOUT fine-tuning, where all methods perform poorly (0.30-0.37 average). However, Table 2 shows dramatic improvements with fine-tuning for perplexity. Can you provide fine-tuning results for all seven common sense reasoning tasks in Table 3? Specifically, does NLA maintain its 4-point advantage (0.37 vs 0.33) over SVD-LLM after both methods are fine-tuned? This is critical for fair comparison since fine-tuning is standard practice.

2. The paper fixes r=20 and derives h from the compression ratio (Appendix C). Can you justify:  Why r=20 specifically? Did you try r=10, 15, 25, 30?

3. Table 1 shows results at different compression ratios for different methods (ASVD at 40/50/60%, NLA at 50/70%). Can you provide a complete comparison table: Method 40% 50% 60% 70%?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes NLA, a nonlinear low-rank compression method with adaptive budget allocation for large foundation models. It introduces a piecewise-linear kernel operator to enhance expressiveness and avoid reconstructing full matrices, and uses an importance score to allocate compression capacity efficiently. NLA outperforms existing linear methods (like SVD, SVD-LLM, etc) across LLaMA, OPT, and Swin-B at high compression ratios.

### Strengths
- Exploring nonlinear approximation for low-rank decomposition is interesting.

- Experiments demonstrate consistent improvements across other baselines.

- Ablation studies is throughout, including some interesting experiments like End-to-End memory/throughput comparison, capability with quantization, etc.

### Weaknesses
- In Line 188, it's better to explain the reason for selecting the nonlinear kernel format.

- The results in Table 3 looks strange to me. For example, PiQA is a binary classification task, the accuracy around 50% is already random guess, the improvements is not meaningful. Same as other tasks in Line 328. If the experiments is evaluated on the generation format rather than logits comparison, then better select instruction version of the model to ensure the answer follows the prompt format.

- Table 4 shows that NLA reduces throughput, making the method less compelling in practice. In general, memory-efficient techniques can either enable inference that would otherwise result in OOM, or improve throughput in memory-bound scenarios. However, from these two perspectives, it's difficult to directly observe the benefits of NLA. It would be better to demonstrate specific scenarios that NLA can help.

### Questions
- What is the overhead (time, resources) for NLA?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes non-linear low-rank approximation (NLA), a compression method for LLMs and vision transformers that reduces memory footprint. Traditional low-rank compression uses linear SVD truncation, which incurs large information loss at high compression ratios. On the other hand, NLA applies a piecewise-linear kernel to approximate weight matrices in a nonlinear feature space, enabling richer representations from low-rank factors. Furthermore, to avoid reconstructing the full dense matrix from the low-rank factors during inference (which would be impractical), the authors design a custom forward operator that computes outputs directly from low-rank factors, reducing compute and memory overhead.

To handle heterogeneous layer importance, NLA further introduces adaptive budget allocation -- it dynamically assigns different compression ratios to different layers based on sensitivity (gradient x weight) score and a cubic sparsity schedule. Experiments on Llama-7B, OPT, and Swin-Transformer show that NLA outperforms state-of-the-art linear methods (e.g., SVD-LLM, ASVD, FWSVD), achieving lower perplexity and higher accuracy at similar or higher compression ratios. NLA can also be combined with quantization, enabling more aggressive memory savings with minimal accuracy loss.

### Strengths
The paper moves beyond conventional SVD-based linear rank truncation by using piecewise-linear kernels, and demonstrates that nonlinear structure retains more information under high compression. The custom forward operations that enables forward pass without reconstructing the entire matrix is also interesting -- this allows doing memory- efficient inference despite the non-linear mapping. 

The authors also address the varying sensitivities of the different layers -- allowing for non-uniform rank-budget allocation -- which is a key practical issue in compression. Since the proposed methods is compatible with other compression methods like quantization, this suggest potential stacking to improve the compression ratios.

Compared to prior works, it consistently outperforms across LLM and vision benchmarks.

### Weaknesses
I have some concerns, and would appreciate it if they are addressed:

1. My primary concern with the paper is that the presentation can be improved. For example, does nonlinear low-rank attention a more general case of linear SVD based low-rank factorization? Under what conditions does the former boil down to the latter? The Alg. 1 pseudocode should also be properly explained in text.

2. Moreover, the nonlinear formulation in eq. (3) is intuitively motivated and empirically effective, but the paper lacks deeper theoretical justification or analytical guarantees on approximation error or convergence, especially compared to the well-studied linear methods like SVD.

3. The computational overhead of doing the forward pass is discussed, but the experimental section where latency numbers are mentioned in Table 9 requires more elaboration. How re the latency number obtained -- these increased compute cost and latency are not rigorously quantified for large-batch or real-time inference settings. Do they involve customized kernel implementations (CUDA, Triton, etc.) for the forward path?

4. Combination of quantization with low-rank is only vaguely discussed. It would be appreciated if the authors discussed relevant works in this context such as https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html

### Questions
1. What is the function $s(\cdot)$ in eq. (5)?

2. In general, Sec. 3.3 should be explained in more details -- seems like a big jump from prior text.

3. Why have more recent models like Llama 3/4, Wen, etc. not considered? Compressing these newer models are more challenging and results compared to prior works will be more interesting.

### Soundness
2

### Presentation
2

### Contribution
2
