# BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Rotation-based methods have become essential for state-of-the-art LLM quantization by effectively mitigating outliers in weights and activations. Current approaches predominantly focus on optimizing the global rotation matrix to achieve marginal accuracy improvements—a strategy that incurs prohibitive computational costs through full-model backpropagation while offering limited practical utility.
We fundamentally reassess this optimization paradigm and identify two critical error sources that persist even under optimal rotation conditions: (i) channel mean misalignment, which amplifies rounding errors during quantization, and (ii) clipping-induced energy loss, which is exacerbated by the rotation-induced Gaussian-like distributions. Our analysis reveals that directly addressing these issues offers a more effective path to achieving high quantization accuracy.
Based on these insights, we introduce \textbf{BASE-Q}, a lightweight quantization framework that circumvents expensive global rotation learning. \textbf{BASE-Q} employs simple yet powerful transformer-block-wise correction strategy: \textbf{bias correction} to eliminate channel mean variance and \textbf{asymmetric scaling} to compensate for clipping-induced energy loss. This blockwise strategy drastically reduces optimization overhead, enabling efficient quantization of 70B parameter models on a single GPU.
Extensive experiments across diverse LLMs and benchmarks validate the effectiveness of BASE-Q, narrowing the accuracy gap to full-precision models by 50.5\%, 42.9\%, and 29.2\% compared to previous rotation method QuaRot, SpinQuant, and OSTQuant respectively, demonstrating the superiority of our lightweight paradigm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes BASE-Q, a lightweight 4-bit post-training quantization framework for large language models (LLMs) that enhances rotational quantization through two key components: blockwise bias correction to eliminate channel-mean misalignment and asymmetric scaling to compensate for clipping-induced energy loss. The authors argue that instead of pursuing costly global rotation learning (e.g., SpinQuant, OSTQuant), it is more effective to directly correct the residual quantization errors that persist even after optimal rotations. The method is evaluated on a wide range of modern LLMs—including Llama-2/3/3.1/3.2 and Qwen2.5 series with strong empirical results.

### Strengths
1) The shift—from “optimize rotation” to “correct residual errors”—is conceptually fresh and represents one of the few recent works that rethinks the purpose of rotation in quantization, rather than merely tuning it.

2) The paper has a relatively rigorous theoretical analysis on the error decomposition of rotational quantization. 

3) The evaluated LLMs are relatively recent.

4) The corresponding GPU kernels are implemented, and the practical speedups of quantized LLMs using their method are justified.

### Weaknesses
1) The proposed blockwise bias correction introduces an additive term $b_c$ before activation quantization and compensates it post-multiplication (Eq. 14). If I remember correctly, some linear layers in modern LLMs (e.g., attention projections) do not include a learnable bias in their original architecture. It is unclear whether the introduced correction term incurs additional latency or memory overhead during inference in such bias-free layers.

2) In Equation (1), it seems the variable $e_i$  is used without definition.

3) The paper argues that global orthogonal rotations cannot eliminate channel-mean variance. However, it does not discuss whether locally (layerwise) learned invertible transformations—such as those in FlatQuant—could mitigate this issue. If such methods can reduce channel-mean misalignment, how does BASE-Q’s explicit bias correction compare in efficacy or efficiency?

4) Could non-orthogonal but invertible transformations simultaneously reduce both channel-wise variance and inter-channel mean discrepancy?

5) Figure 6 illustrates a fused pipeline combining Triton and CUTLASS kernels. How are memory layouts, synchronization, and data movement coordinated between the Triton-based quantization/dequantization and the CUTLASS-based INT4 GEMM? Are these kernels truly fused—or are their speeds measured separately and summed for total throughput?

6) The paper states that 70B-model quantization takes ~10 hours on a single A800. Could authors break down the time spent on joint optimization of $R_v, b_c, s_a$, and $α$? What fraction of this time is attributable to the proposed bias and asymmetric scaling components versus baseline rotation and scaling?

7) The MSE objective in Equation (7) appears to lack an explicit scale factor in the reconstruction term. 

9) Eq.(12) gives an optimal global scaling $s^2 = \mathbb{E}[\|w\|^2] / \mathbb{E}[\|a\|^2]$ from error propagation (Eq.11). 
Yet BASE-Q uses per-quantizer asymmetric $s_a$ and blockwise symmetric $s_j$, not $s$. 
Clarifying the link—or empirically validating $s_a$'s alignment with this principle—would strengthen the argument.

### Questions
See weaknesses.

### Soundness
3

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
5

### Summary
This paper proposes BASE-Q, a lightweight quantization framework for LLMs. BASE-Q addresses two key error sources in rotational quantization—channel mean misalignment and clipping-induced energy loss—through blockwise bias correction and asymmetric scaling, avoiding the high cost of optimizing global rotation matrices.

### Strengths
1. Compared with the previous approach of optimizing the global rotation matrix, this method can be optimized with fewer GPU resources.
2. The authors conducted extensive experiments on both QWen series and LLaMA series models under the W4A4KV4 quantization configuration. Across these models, BASE-Q consistently achieved performance improvements.

### Weaknesses
1. The author only tested the effectiveness of the proposed method in basic experiments such as Zero-Shot and PPL evaluations. How does this method perform on more complex benchmark datasets like MMLU?
2. While I acknowledge the rationality of the authors' method, it appears that their approach is essentially OminiQuant under rotation conditions. Additionally, the fused-bias technique they employed is not particularly novel—Outlier Suppression++ has also adopted the fused-bias technology. Therefore, the authors' technical contributions are somewhat limited.
3. How long does the authors' method take to quantize a model? The comparison of quantization time remains an important aspect.
4. How is the generalization ability of the authors' method, and does the selection of the calibration dataset affect the performance of the algorithm?
5. Beyond INT4 quantization, does the authors' method exhibit good performance under other quantization modes such as NVFP4?
6. The authors have omitted some important works: Xiang, J. and Zhang, S.Q., 2024. DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation. arXiv preprint arXiv:2412.00648.

### Questions
See weekness.

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
BASE-Q is a lightweight post-training quantization framework for large language models that achieves near-full-precision performance at 4-bit weight, activation and KV-cache (W4A4KV4) quantization. BASE-Q combines fixed Hadamard rotations with two block-wise corrections—channel-wise bias to cancel rounding error caused by misaligned channel means and asymmetric scaling. Extensive results show that BASE-Q narrows the accuracy gap to FP16 by 50.5%, 42.9% and 29.2% vs. QuaRot, SpinQuant and OSTQuant across 12 open-source LLMs (1B–70B) on WikiText-2 perplexity and nine zero-shot tasks, while fused Triton kernels deliver 2.1–2.4× prefill speed-up and 71% memory reduction. While the paper is well motivated by math analysis, the methodology is kind of incremental compared with previous methods, and there are still issues with the presentation (as listed in weakness and detailed questions).

### Strengths
- The authors conduct extensive empirical results to demonstrate the effectiveness of BASE-Q against a set of leading baselines such as SpinQuant and OSTQuant. The ablation study in Table 2 also looks adequate.

- The authors also customize the kernel to further speed up the inference of BASE-Q. Figure 5 looks promising, and the proposed method introduces little overhead.

### Weaknesses
- The proposed method is largely built upon existing frameworks like SpinQuant. Both bias correction and asymmetric scaling are kind of incremental to the baseline. Moreover, the training paradigm (e.g., blockwise training) also follows SpinQuant.

- The writing is kind of hard to follow. Some equations should be explained in more detail (e.g., Equation 7, 8). The necessary derivations are missing. The logic can be a bit messy. For instance, it can be hard to understand the expectation of rounding error (Line 155), yet this is only introduced in more detail later in Equation 10.

- While I appreciate the analysis in Section 3, it seems to overlap quite well with the proposed BASE-Q in Section 4. This makes the reading back-and-forth.

### Questions
- FlatQuant should be considered for comparisons in the main table, since both the model and quantization settings are pretty aligned with its original paper. The results of FlatQuant in Table 9 and Table 10 look weird.

- Section 3.4 is hard to follow. It seems weird why ϵ follows uniform distribution in Equation 9 and Normal distribution in Equation 10. Besides, I doubt the assumption that ϵ follows the uniform distribution, as the rounding error also depends on the distribution of weights, which can be normally distributed.

- It is not clear how the blockwise training is conducted, e..g, what is the definition of a block?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper suggests that while Hadamard rotations (or any globally learned rotation matrix) help with outliers, they fail to fully reduct the variance between channel means to zero which results in poor quantization. To alleviate this, the authors suggest using a learnable bias correction term. Combined with a fixed scaling (offline) and using assymetric quantization, the authors name the obtained method BASE-Q. The results show better performance than several existing methods (QuaRot, SpinQuant, and OSTQuant) after quantization over different architectures including LLaMA 2, 3 and QWEN.

### Strengths
The paper explains the reason behind all of the design choices in the proposed method. The method is effective and shows strong results compared with existing SoTA methods that are used as baselines. The authors include ablations to showcase the importance of each component.

In addition to theoretical efficiency arguments, the authors implemented optimized kernels that allow them to obtain real-world speedup more than 2x when using 4-bit quantization.

### Weaknesses
The writing can be improved quite a lot. Important details are either missing or are scattered in different places. To name a few exampels:

1. The dimensions of newly introduced parameters are not clearly specified. It is hard to understand whether a parameter is scalar, a vector or a matrix.

2. There are two different scaling mentioned in the paper. These are referred by different names in different places. For example, Table 2 calls them unpaired scale and scale. There is no other mentioned of "Unpaired" anywhere else in the text and the latter scale is usually referred to as "Dynamic Assymetric Scale".

As another note, using assymetric quantization can be combined with any of the existing methods which makes it an orthogonal axis. Similarly, all the low-level optimizations to handle the assymetric scale such as fused kernels can also be re-used for existing methods. While it makes sense to use the best recipe to obtain the final results (i.e. the best possible performance of quantized models), it adds an additional confounder when comparing different methods. Still, this issue is mostly resolved by Table 2 which shows the main effect comes from the main contributions of the method.

### Questions
1. Is there an $s^{-1}$ missing from Eq. 14 to be applied on the activations?  Similarly shouldn't $s^{a}$ be inverted before multiplying by the quantized weights?

2. Just to confirm is $R_{qk}$ shared between layers? (and similarly other learned rotation matrices such as $R_{v}$)

### Soundness
3

### Presentation
1

### Contribution
3
