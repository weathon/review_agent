# A Mechanistic Analysis of Low-Precision Instabilities in Microscaling Formats

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Training large language models is expensive and compute-bound, and it must be repeated as models scale, algorithms improve, and new data is collected. To address this, next-generation hardware accelerators like NVIDIA’s Blackwell increasingly support lower-precision arithmetic formats, including Microscaling (MX) formats. In this work, we investigate the challenges and viability of block-scaled precision formats during model training. Across a broad sweep of weight-activation precision combinations and compute budgets from \( 2 \times 10^{17} \) to \( 4.8 \times 10^{19} \) FLOPs, we generally observe that training in MX formats exhibits sharp, stochastic instabilities in the loss, particularly at larger compute scales. To explain this phenomenon, we conduct controlled experiments and ablations on a smaller proxy model that exhibits instability behavior similar to the language model, sweeping across architectural settings, hyperparameters, and precision formats. These experiments motivate a simple model in which multiplicative gradient bias introduced by the quantization of layer-norm affine parameters and a small fraction of activations can trigger runaway divergence. Through \textit{in situ} intervention experiments on our proxy model, we demonstrate that instabilities can be averted or delayed by modifying precision schemes mid-training.  Guided by these findings, we evaluate stabilization strategies in the LLM setting and show that certain hybrid configurations recover performance competitive with full-precision training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the training dynamics of models trained with microscaling (MX) quantization in the forward and backward passes. MX formats (and other similar block scaled formats such as NVFP4) are become widely used for low precision training, making this a practical area to study. The paper studies this problem by training many models and considering a toy example with a student/teacher model setup. The paper finds that the main reason for loss spikes and divergence training is due to overflow in layernorm parameters during MX quantization, which can be mitigated by modifying the MX quantization algorithm.

### Strengths
- The authors trained a large number of models to show that when applied "incorrectly," MX datatypes can cause training instability. 
- The minimal synthetic model is very interesting, as it is able to mostly model the effect of different norms and activation functions on training instabilities.

### Weaknesses
- The main takeaway of this paper seems to not quantize layernorms, but people generally do not quantize non-decoder layer linear layers already. Can the authors explain why you would want to quantize things beyond linear layers? The point of using hardware supported datatypes for quantization is to accelerate compute bound matrix multiplications, and these operations are almost entirely contained in projection layers and self attention, neither of which appears to be source of training instabilities according to this paper.
- Since the main remedy appears to be avoiding overflow during rounding, I would have liked to see more experiments on different rounding methods that solve this. Increasing the exponent is just one way - https://aisystemcodesign.github.io/papers/FP4.pdf discusses more methods that get around this.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper attempts a systematic study on the stability of MXFP8 format for training LLMs. Through extensive experiments on a proxy model, they find the instabilities to be the result of gradient bias due to quantization. This bias stems mainly from layer-norm weight distribution and sometimes other activations. Bridging to the LLM case, they suggest keeping the layer norm (and activations) in bfloat16, and only quantizing the forward pass.

### Strengths
- The paper is generally well-written
- Systematic approach

### Weaknesses
- While the study is interesting, the findings are not significant: in quantized/quantization-aware training, layer norms are generally skipped as they don't have significant memory or computation overhead (if implemented correctly).
- Some assumptions/conclusions are loose (see Questions below)

### Questions
1. The fact that layer norm should be kept in high precision is not new, e.g., [1] shows even FP16 layer-norm is difficult (even before that for batch norm [2]). Additionally, to my knowledge, practical papers in quantization-aware training such as QuEST [3] already know not to touch the layer-norm layers. Even standard implementations of normalization layers make sure to operate in FP32, such as [4]. So my question is that what is the novelty in this paper's findings?

2. It's not clear to me why such proxy model is "good." In the authors' words: "we caution that stability in this minimal model as a necessary (though perhaps not sufficient) condition for stability in the full LM." Why is this a necessary condition? Note that this means if LLMs fail in a certain setting, then this proxy model will also fail.

3. Can you verify that you have FP32 master weights? Maybe I missed it, but I didn't find it explicitly in the paper.

4. Since the finding is that instability is due to gradient bias, how would the results look like if stochastic rounding is employed? Theoretically, quantization errors due to stochastic rounding should be unbiased. Although I understand it does not affect overflow.

5. Can you elaborate on why there is no layer-norm is the teacher?

6. How would your studies interact with Hadamard transformations [3], which is now common practice in QAT?

In the current state, despite the systematic investigation, I don't believe this paper offers any significant contribution.

[1] https://arxiv.org/pdf/2410.10553
[2] https://arxiv.org/pdf/1710.03740
[3] https://arxiv.org/pdf/2502.05003
[4] https://github.com/huggingface/transformers/blob/4d0b6758b90aa54e4077171e6d42c55e0c01c622/src/transformers/models/llama/modeling_llama.py#L64

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the mechanistic origin of training instabilities observed when applying Microscaling (MX) low-precision formats (FP6/FP8) to LLM pretraining. The authors combine OLMo-based LLM experiments with controlled synthetic proxy models to demonstrate that the observed failures are not attributable to activation choices or hyperparameter settings, but instead arise from systematic quantization effects on LayerNorm affine parameters.

During training, LayerNorm affine parameters become highly clustered, and under MX block-scaled quantization, these values saturate into the same maximum bin due to shared scaling constraints. This saturation introduces persistent gradient bias, which accumulates across updates and ultimately leads to irreversible divergence in training loss. The paper provides empirical evidence for this failure mode, demonstrating both its emergence at LLM scale and its reproducibility within a simplified synthetic setting.

Building on this analysis, the authors propose practical stabilization strategies, including disabling LayerNorm affine quantization, keeping activations in bfloat16, and applying quantization only in the forward pass. Among these approaches, the configuration using MXFP8 (E4M3) weights and BF16 activations achieves validation performance comparable to full BF16 while avoiding instability.

### Strengths
* The paper offers a clear mechanistic analysis of instability in MX-based low-precision training.
* It identifies a coherent failure chain in which LayerNorm affine parameters saturate under block-scale quantization, introducing gradient bias that ultimately triggers training divergence.
* The authors combine OLMo-based LLM experiments with a controlled synthetic proxy model, enabling both realistic evaluation and reproducible causal analysis.
* In-situ intervention experiments further provide empirical evidence that LayerNorm affine quantization is a primary causal factor behind the observed failures.
* The paper proposes practical and easily deployable mitigation strategies—such as disabling LN-affine quantization, using BF16 activations, or restricting quantization to the forward pass.
* These techniques restore stable training while matching the validation performance of full-BF16 systems, underscoring their practical utility.

### Weaknesses
* Approaches such as disabling LN-affine quantization or increasing activation precision offer limited throughput benefits compared to standard BF16 training.
* Most empirical validation is conducted on OLMo-scale models, leaving generalization to frontier-scale systems uncertain.
* Although the paper reports activation overflow phenomena, its relative contribution is not quantitatively evaluated against the LayerNorm-induced mechanism, leaving the importance of this additional pathway insufficiently characterized.
* The paper identifies the core mechanism behind instability, but does not provide a systematic exploration of the conditions under which instability emerges—e.g., across compute scale, dataset size, or architectural variations—making it difficult to determine when MX training is expected to fail.
* Comparisons with other low-precision formats (e.g., per-channel scaling, PQ) are limited.

### Questions
* Can the relative contributions of activation overflow and LN-affine saturation to the observed instability be quantitatively compared?
* Has this failure mechanism been evaluated in other architectures, such as MoE, MQA, or non-Transformer blocks?

### Soundness
2

### Presentation
2

### Contribution
3
