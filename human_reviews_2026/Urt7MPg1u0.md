# Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
1-bit LLM quantization offers significant advantages in reducing storage and computational costs. However, existing methods typically train 1-bit LLMs from scratch, failing to fully leverage pre-trained models. This results in high training costs and notable accuracy degradation. We identify that the large gap between full precision and 1-bit representations makes naive adaptation difficult. In this paper, we introduce a consistent progressive training for both forward and backward, smoothly converting the full-precision weights into the binarized ones. Additionally, we incorporate binary-aware initialization and dual-scaling compensation to reduce the difficulty of progressive training and improve the performance. Experimental results on LLMs of various sizes demonstrate that our method outperforms existing approaches. Our results show that high-performance 1-bit LLMs can be achieved using pre-trained models, eliminating the need for expensive training from scratch.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new 1-bit LLM quantization paradigm to overcome the instability and accuracy loss of existing binary quantization methods. The authors propose a consistent progressive training method to gradually transition from a near-linear quantization function to a sign function, reducing quantization error and preserving pre-trained knowledge: Binary-aware initialization ensures stable optimization. while dual-scaling compensation introduces learnable scaling factors to maintain accuracy. Experiments across various LLM scales show that this approach narrows the performance gap between 1-bit and full-precision models.

### Strengths
The paper is well written.

### Weaknesses
1. PTB perplexity evaluation is confusing. For example, on the 3B model, BitNet b1.58 clearly outperforms the proposed BinaryLLM on C4 and WikiText2 perplexity, but performs significantly worse on PTB perplexity. This inconsistency is also observed in other model evaluations, including the 1.3B experiments. It is recommended to recheck the PTB perplexity evaluation.

2. In Section 4.2, the training data and base LLM are not aligned. In the 130M experiment, BinaryLLM uses SmolLM data, while FBI-LLM is trained on the Amber dataset. The SmolLM data is of notably higher quality. Moreover, compared with OneBit, BinaryLLM adopts a stronger base LLM (e.g., LLaMA-3B). These differences could lead to unfair comparisons.

3. BitNet also proposes a 1-bit LLM [1], but the paper lacks a comparison with this baseline under the same dataset and experimental setting.

I recommend that the authors **conduct comparisons under strictly comparable experimental conditions**, e.g., training on the same dataset and starting from the same BF16 model, rather than simply comparing with results reported in previous papers. This is particularly important considering that all experiments are conducted on relatively small-scale models.

[1] Wang, Hongyu, et al. "Bitnet: Scaling 1-bit transformers for large language models." arXiv preprint arXiv:2310.11453 (2023).

### Questions
See Weaknesses.

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
5

### Summary
This work presents a novel 1-bit training framework for large language models (LLMs). It introduces a progressive training strategy to enable smooth transition from full-precision to binarized weights, complemented by binary-aware initialization and dual-scaling mechanisms to further boost performance. Experimental results demonstrate that the framework requires fewer training tokens while achieving superior performance.

### Strengths
1. For the 130M model, BinaryLLM requires only 20B training tokens. This is significantly fewer than the 1.26T tokens needed for training a model from scratch.
2. The insight that well-trained models are harder to quantize while they still outperform under-trained ones after binarization, is interesting, and this point is thoroughly discussed.
3. The motivation is clear. The method is generalizable, and the performance is satisfactory.

### Weaknesses
1. For the ablation study on progressive training (Table 4), the comparison should use BinaryLLM without binary-aware initialization and dual-scaling. Merely comparing results between BinaryLLM and IR-Net fails to highlight the effectiveness of progressive training.
2. The binary-aware initialization yields only marginal improvements, as shown in Table 9.
3. There are some typos. For instance, line 372 should read "from smallest to largest".

### Questions
1. The authors claim in the paper: "At convergence, inference is performed with the Sign function instead of F(x, t), incurring negligible error." Are there any quantitative results to support this assertion?
2. How did the authors determine the coefficients of the function t(c) = 1.3 × e^(0.22c) − 1.3?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes BinaryLLM, a new framework for training 1-bit large language models directly from pre-trained weights instead of training from scratch. The method introduces consistent progressive training to smoothly transition weights from full-precision to binary form, binary-aware initialization to preserve salient weights across layers, and dual-scaling compensation to balance quantization error and accuracy. Experiments on multiple LLMs show that BinaryLLM achieves great results among 1-bit LLMs, significantly reducing performance gaps with full-precision models while requiring far less training cost.

### Strengths
The paper presents a thoughtful and well-motivated attempt to make 1-bit quantization more practical for large language models. The idea of leveraging pre-trained weights rather than training from scratch is both efficient and timely, and the proposed progressive training and dual-scaling strategies are conceptually sound. The paper is clearly written with comprehensive experiments.

### Weaknesses
Although the authors include additional results on LLaMA2-7B in the appendix, larger-scale validation (e.g., 13B or 30B models) is still missing, leaving some uncertainty about scalability under truly large-model settings. The comparison baselines are reasonably strong, but despite explicitly discussing the instability of 1-bit quantization on newer models such as Qwen3, the authors do not include direct experiments on it. This omission leaves open how well BinaryLLM performs on the latest LLM architectures. Moreover, the discussion on training stability and convergence focuses mainly on the design of the progressive parameter t and its scheduler, without quantitative analysis to substantiate robustness claims. In addition, while binary-aware initialization and dual-scaling compensation are described in detail and empirically shown to help, their theoretical justification and computational complexity are only briefly discussed.

### Questions
1. Could the authors provide results or discussion on how BinaryLLM scales to larger models? Even limited experiments or resource estimates would help clarify whether the proposed training strategy remains stable and effective at larger scales.
2. Since the paper explicitly mentions the instability of 1-bit quantization on newer architectures such as Qwen3, could the authors include or discuss experiments on such models to verify BinaryLLM’s generalization ability to the latest LLM families?
3. The paper explains the progressive parameter t and its scheduling strategy, but lacks quantitative analysis. Could the authors provide sensitivity studies to better support the stability claims?
4. While binary-aware initialization and dual-scaling compensation are shown to be effective empirically, could the authors elaborate more on their theoretical justification to clarify their computational overhead and convergence behavior?

### Soundness
3

### Presentation
3

### Contribution
2
