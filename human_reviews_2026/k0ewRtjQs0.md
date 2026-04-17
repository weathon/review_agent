# Towards W2A4 LLM Inference: Hybrid SQ-VQ Framework with Adaptive Error Compensation

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Quantization presents a powerful approach for reducing the memory footprint and accelerating the inference of Large Language Models (LLMs). However, it faces a fundamental dilemma: computation-friendly Scalar Quantization (SQ) suffers performance degradation at ultra-low bit-widths, whereas memory-friendly Vector Quantization (VQ) maintains higher accuracy but fails to reduce computational demand. As a result, achieving both computational efficiency and high-fidelity compression in ultra-low-bit regimes (e.g.W2A4) remains a tough challenge. To address this, we propose $\textbf{AEC-SVQ}$, a hybrid framework that synergistically integrates SQ ,VQ for high-performance, ultra-low-bit LLM inference. The framework is built on three  innonvations. To simultaneously address the disparate distributional challenges presented by weight VQ, activation SQ, and codebook integer quantization, we introduce a $\textbf{learned rotation-smooth transformation}$ that adaptively promotes quantization-friendly distributions for weights, activations, and codebooks within the hybrid SQ–VQ scheme. To mitigate the compounding errors caused by the independent quantization of weights and activations, we propose the $\textbf{Cumulative-Error-Aware Vector Quantization (CEAVQ) algorithm}$. CEAVQ adjusts weights to compensate for the cumulative error from upstream quantized layers, thereby proactively aligning with the full-precision output distribution. To ensure robustness against statistical noise from limited calibration data, we introduce a closed-form, data-driven $\textbf{Adaptive Compensation}$. It modulates the compensation strength for cumulative errors, preventing overfitting to calibration set statistics and guaranteeing stable generalization. AEC-SVQ enables a W2A4 pipeline that achieves the memory footprint of a 2-bit model while exploiting the computational efficiency of 4-bit integer arithmetic. On LLaMA-30B, it delivers a 3.6$\times$ speedup and 7.1$\times$ memory saving, establishing a practical frontier for ultra-low-bit LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes AEC-SVQ, a quantization framework for large language models that employs scalar quantization for activations and vector quantization for weights, utilizing INT4 arithmetic units. It introduces an effective transformation for quantization, a novel quantization algorithm that accounts for cumulative errors across quantized layers, and adaptive correction methods to further enhance accuracy.

### Strengths
* The proposal to synergistically use scalar quantization for activations and vector quantization for weights, while also quantizing the codebook to leverage low-precision ALUs, is brilliant and makes this work highly relevant to real-world practitioners.
* AEC-SVQ is extensively compared against several existing solutions and demonstrates promising result.

### Weaknesses
* The overall contributions of this work are somewhat difficult to grasp at a high level.
* The speedup evaluations are incomplete.

### Questions
* First, I would like to ask about two of the three main contributions claimed in the paper: learned transformation and CEAVQ.

Regarding the learned transformation, I am curious how the proposed mechanism differs from existing learned rotation or transformation approaches such as SpinQuant or DuQuant, and why it performs better both quantitatively and qualitatively.

As for CEAVQ, I found it somewhat difficult to understand the core novelty that distinguishes it from prior work, partly due to my lack of mathematical literacy. To me, the objective function in Equation (5) does not appear particularly novel, and it seems that the key contribution lies in Equation (6) and the accompanying discussion. I would greatly appreciate a higher-level explanation of CEAVQ’s conceptual contribution beyond the mathematical formulation.

Minor: Why not just write equation (5) as W^X-WX~? This seems more intuitive to me.

* Second, I am wondering why decode speedup results are not included in the manuscript. I assume that this is because achieving decode speedup would require a specialized kernel supporting vector quantization with quantized codebooks. If that is the case, I understand the challenge. However, including at least a discussion on this limitation or providing partial evidence of end-to-end speedup would make the paper more valuable.

* Third, low-precision floating-point formats (e.g., FP4) are becoming increasingly popular for LLM inference as newer GPUs offer native support. How would the proposed approach extend to support FP4 instead of INT4? What would you expect in terms of performance and accuracy? Would it outperform the INT4 version, or not?

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
4

### Summary
The paper proposes to combine scalar quantization and vector quantization. Essentially the proposal is to use vector quantization on weights, but then quantize the entries inside the codebooks to INT4 representation. Activations are kept in INT4. This by itself is not novel and susceptible to excess noise. The proposed method is to enhance this quantization using learned transformations and other techniques.

### Strengths
-CEAVQ provides formulas for updating weights to account for activation quantization.

### Weaknesses
- The proposed learned transformation in Section 3.1 is given by T=O\Lambda, i.e, a product of a rotation matrix and a smoothing matrix. An expression for the MSE of the layer's gemm output is given in terms of the weight and activation statistics. Then it is claimed that the transformation is theoretically guaranteed to minimize this MSE with a promised proof in the appendix. But the actual construction of T is not given. Therefore, the claim is essentially void.
- The paper assumes INT4 tensor cores. These have been discontinued since the Ampere architecture. In Blackwell, there are NVFP4 tensor cores which may be more fit to explore.

### Questions
- Please provide a construction for the learned transformation T.
- Shouldn't CEAVQ also account for weight quantization noise?
- Please explain why the evaluations of related works are not similar to those in the corresponding papers, e.g., quarot is claimed to have a perplexity in the 5e4 regime - but that paper itself claims good accuracy for extremely low bitwidth quantization.

### Soundness
2

### Presentation
2

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
This paper proposes a method that combines scalar quantization (SQ) and vector quantization (VQ) to quantize large language models (LLMs) to W2A4 precision. The approach employs a learned rotation-smooth transformation and a cumulative error aware vector quantization technique, supported by a layer-wise correction factor. Through this design, AEC-SVQ achieves both high compression rates and computational efficiency.

### Strengths
This LLM quantization paper presents a hybrid approach that combines SQ and VQ with a learned transformation, leading to an intuitive understanding of its benefits, supported by clear mathematical derivations. It also provides a detailed study that examines various aspects and techniques across multiple models. Additionally, it adopts a layer-wise regularization factor to prevent overfitting in the compensation process.

### Weaknesses
Please refer to the questions below.

### Questions
1. Beyond the combination of existing SQ and VQ techniques, what are the main novelties of this paper? The use of a learned rotation matrix has already been explored in prior works such as SpinQuant and OSTQuant, and the CEA column-wise compensation based on WX error appears similar to GPTQ, not to mention the use of codebook quantization. Please clarify which components are borrowed or inspired by prior work and which constitute the novel contributions of this paper.

2. How sensitive is your method to the choice of the calibration dataset?

3. Fine-tuning appears to have a significant impact on the results, which raises some concerns about the robustness of the proposed strategy. Could you elaborate on this point and confirm whether the comparisons with other works are conducted fairly (e.g., fine-tuning applied to your method but not to baselines)?

4. In Section A.5.2, is the analysis referring to OSTQ?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an approach for W2A4 (2-bit weight and 4-bit activation) quantization of large language models. While the motivation is clear and the experiments are conducted on relevant benchmarks, the contribution lacks sufficient novelty and fails to situate itself properly within the current literature on low-bit quantization.

### Strengths
1. The presentation is clear.
2. Extensive experiments are done on Llama and Qwen models.

### Weaknesses
1. The proposed method combines vector quantization for weights and scalar quantization for activations. THis is a setup that has already become standard practice in recent quantization research.
2. Without a clear conceptual or technical innovation, the contribution falls short of the standards for publication.
3. Several recent vector quantization approaches such as AQLM and QuIP# are not included in the comparison. These methods represent the state-of-the-art in efficient LLM quantization with vector quantization and should be considered essential baselines. The absence of such comparisons makes it difficult to assess the actual competitiveness of the proposed method. As it stands, the reported results cannot convincingly demonstrate superiority or even parity with existing solutions.

### Questions
1. It would strengthen the work if the authors provided hardware-level latency or energy efficiency evaluations, as quantization benefits are often hardware-dependent.

### Soundness
2

### Presentation
3

### Contribution
2
