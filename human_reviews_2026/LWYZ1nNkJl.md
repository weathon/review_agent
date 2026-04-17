# Rethinking Residual Errors in Compensation-based LLM Quantization

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 8

## Abstract
Methods based on weight compensation, which iteratively apply quantization and weight compensation to minimize the output error, have recently demonstrated remarkable success in quantizing Large Language Models (LLMs). 
The representative work, GPTQ, introduces several key techniques that make such iterative methods practical for LLMs with billions of parameters.
GPTAQ extends this approach by introducing an asymmetric calibration process that aligns the output of each quantized layer with its full-precision counterpart, incorporating a residual error into the weight compensation framework.
In this work, we revisit the formulation of the residual error.
We identify a sub-optimal calibration objective in existing methods: during the intra-layer calibration process, they align the quantized output with the output from compensated weights, rather than the true output from the original full-precision model. Therefore, we redefine the objective to precisely align the quantized model's output with the original output of the full-precision model at each step. We then reveal that the residual error originates not only from the output difference of the preceding layer but also from the discrepancy between the compensated and original weights within each layer, which we name the 'compensation-aware error'.
By inheriting the neuron decomposition technique from GPTAQ, we can efficiently incorporate this compensation-aware error into the weight update process. Extensive experiments on various LLMs and quantization settings demonstrate that our proposed enhancements integrate seamlessly with both GPTQ and GPTAQ, significantly improving their quantization performance. Our code is publicly available at https://github.com/list0830/ResComp.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an improved residual error formulation for post-training quantization (PTQ) of large language models, building on frameworks like GPTQ and GPTAQ.

The key insight is that prior methods inaccurately model the calibration objective during the layer-wise iterative process. The authors identify that the residual error should not only account for the inter-layer propagated output error but also an intra-layer compensation-aware error arising from the weight update process itself.

### Strengths
This paper demonstrates significant strengths across key dimensions. Its originality lies in a nuanced reformulation of the residual error objective within established PTQ frameworks, identifying and incorporating the previously overlooked compensation-aware error. 

The quality is high, evidenced by rigorous and extensive experiments across multiple model families (Llama 2/3), scales (1B-70B), and quantization settings (weight-only, weight-activation). 

The clarity is commendable; the paper logically builds from the identified limitation to the proposed solution, with clear mathematical derivations and an efficiently described algorithm.

### Weaknesses
In significance, while improvements are consistent, they are often marginal (e.g., <1% accuracy gains in Table 1), raising questions about practical impact. The added calibration memory overhead (Table 5) could be prohibitive for edge devices, yet this trade-off is underexplored. Addressing these points would strengthen the work's relevance and applicability.

### Questions
Question 1 : Sensitivity to Calibration Data: Your experiments use a fixed 128 samples. How sensitive are the performance gains to the number and nature of the calibration samples? If the gains diminish significantly with fewer samples or change drastically with a different calibration dataset, it would impact the practical robustness of the method.

Question2 : Generalizability Beyond Llama Architectures: The empirical validation is comprehensive but exclusively on the Llama family. Have you observed similar improvements on other prominent architectures, such as GPT-style models (e.g., Qwen, Gemma) or encoder-only models? A result on at least one non-Llama model would greatly bolster the claim of general applicability.

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
4

### Summary
The paper shows that compensation-based PTQ methods for LLMs, such as GPTQ and especially GPTAQ, optimize against already compensated activations rather than the original full-precision (FP) layer output, which makes the target drift as we modify the weights inside the same layer. To fix this, the authors re-derive the per-layer objective so that every quantization and compensation step is directly aligned to the FP output, which naturally reveals a missing "compensation-aware" residual term capturing the error from earlier intra-layer updates, and they show that this term can be folded in using the same neuron-wise decomposition and Cholesky tools GPTAQ already uses, adding only modest memory/time overhead and delivering consistent quantization improvements for LLaMA 2/3 across especially in low-bit (3-2 bit) settings where quantization errors accumulate the most.

### Strengths
- The core insight is simple and just proposes that we should match to the original full-precision output instead of a compensated one.
- The derivation is intuitive because focusing on full-precision alignment, the missing compensation-aware term appears by default.
- The method easily fits within GPTAQ style pipelines since it also uses the neuron-wise and Cholesky.
- There are improvements in difficult low-bit and weight-plus-activation scenarios, which is exactly where we care.
- The experiments cover several LLaMA variants and quantization settings.

### Weaknesses
- Reported improvements are generally modest and in some cases the technique even leads to decreases in performance.
- The method introduces additional memory and runtime overhead, which is not always justified by the size of the gains.
- The writing of the paper has inconsistencies, e.g. Table 2 contains a bolding error on L3.1-8B-Inst for C4, where the proposed method is highlighted despite being worse than GPTAQ, which undermines the empirical presentation.
- Table 1 applies bolding only to the authors’ method even when other techniques perform better (e.g. L3-8B), creating presentation bias in favor of the proposed approach.
- The approach does not consistently dominate GPTAQ across models, on some models (again L3-8B) average accuracy even decreases in Table 1.

### Questions
- The evaluation is limited to LLaMA/LLaMA-3 variants. Evidence of the technique across other families (e.g. Qwen, Phi, Gemma), especially those with different activation/normalization patterns?
- It would be useful to extend Tables 5 and 6 to larger model to better understand the impact of the technique.
- How sensitive is the method to calibration set size and distribution?
- Are there quantization/deployment scenarios where aligning strictly to FP outputs is not the right target (e.g. fully quantized activation pipelines), and how would the method adapt there?
- Can the authors correct the bolding inconsistencies and re-run significance checks to ensure that the reported improvements are not due to formatting or selection bias?

I'd be happy to increase my score if the concerns are resolved!

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a method that builds upon GPTQ and GPTAQ for efficient finetuning-free quantization. The main idea is to approximate the unquantized model’s activations using quantized weights during the quantization optimization process, aiming to reduce residual errors and improve quantization accuracy. The authors claim that this approach refines the calibration process and better aligns quantized activations with the full-precision model.

### Strengths
1. The topic of efficient and finetuning-free quantization for large transformer models is timely and relevant.

2. The authors provide a clear motivation to address accumulated quantization error across layers.

3. The structure of the paper and the technical presentation follow a standard quantization analysis format.

### Weaknesses
1. Sections 3.1 to 3.3 are almost entirely copied or rephrased from the GPTAQ paper. The derivations, notation, and even paragraph flow (e.g., the definitions of asymmetric calibration, residual error formulation, and inverse Hessian update) are reproduced with only superficial wording changes. This raises a serious concern of possible plagiarism.

2. The paper merely extends the asymmetric calibration idea already introduced in GPTAQ. The supposed “rethinking of residual error” is essentially a restatement of GPTAQ’s asymmetric calibration mechanism.

### Questions
1. Adding some figures might present the idea of this paper in a better way.
2. The content of this paper is not self-contained. Without reading the GPTAQ paper, it is hard to understand this paper. For example, it is hard to understand why in Eqn. 4, $\mathbf{\tilde{X}}$ is used as the target. The purpose of the background should completely explain the idea of the paper without copying content from previous papers. Concepts in previous papers should be explained in a concise way.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper points out a defective implementation in the previously published GPTAQ method, proposes a corrected algorithm, and demonstrated the effectiveness of the correction.

### Strengths
+ Clear presentation.
+ Sound analysis.
+ Compelling empirical results. 
+ Practical significance.

### Weaknesses
- I have but one issue with the wording when compared against GPTAQ.  If I understand correctly, the formulation of GPTAQ optimization in matrix form is exactly correct and not challenged by this paper; rather, the row-wise iterative algorithm implementation has been defective and is now corrected to truly align with the optimization problem.  So instead of leaving the reader the impression of this being yet another method, it should be clearly shown as a correction to a previously wrongly implemented existing method, which is no less significant.

### Questions
* Minor question on data-efficiency: to achieve the same compensation outcome, does the corrected algorithm require more data than the original GPTQ/GPTAQ?

### Soundness
3

### Presentation
3

### Contribution
4
