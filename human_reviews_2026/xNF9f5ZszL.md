# MOQE: IMPROVE QUANTIZATION MODEL PERFORMANCE VIA MIXTURE OF QUANTIZATION EXPERTS.

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Quantization method plays a crucial role in improving model efficiency and reducing
deployment costs, enabling the widespread application of deep learning
models on resource-constrained devices. However, the quantization process inevitably
introduces accuracy degradation. In this paper, we propose Mixture of
Quantization Experts( abbr. MoQE), a quantization inference framework based
on the Mixture-of-Experts (MoE) architecture, aiming to jointly improve the performance
of quantization models. MoQE combines multiple quantization variants
of one full-precision model as specialized ”quantization experts” and dynamically
routes input data to the most suitable expert based on its characteristics. MoQE alleviates
the performance degradation commonly seen in single quantization models
through specialization quantization expert models. We design lightweight,
structure-aware router models tailored for both CV and NLP tasks. Experimental
evaluations on ResNet, LLaMA, and Qwen model families across benchmark
datasets including ImageNet, WikiText, C4, and OpenWebText demonstrate that
MoQE achieves performance comparable to SOTA quantization model, without
incurring significant increases in inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes quantizing the model with different methods to obtain multiple different quantized models and learning a router which, given an input, determines which quantized model to use in test time. The authors report the bias effects as their motivation and demonstrate the utility of the proposed method by improving accuracy at a small latency overhead due to model transfer from CPU to GPU.

### Strengths
Effective idea to improve the accuracy of quantized model.
The latency overhead is small.

### Weaknesses
Bias analysis does not address how the bias effects are obtained.
Appendix A.2 and A.3 show the effects, not the causes. 
There will be several factors (e.g., scaling, clipping, ...) to consider to investigate the causes.
For instance, it would be possible to investigate which of scaling (like AWQ) and clipping incurs more bias on which datasets.

### Questions
It would be nice to discuss the following questions, if possible with quantitative results.

What are the real causes of the bias effects?
Does the bias effects get more pronounced on lower precision? If not, why?
Is the proposed method more effective on lower precision? If not, why?
How different is the bias effects between scaling and clipping based quantization methods?
In case of LLM, can best quantized models be different between prefill and decode stages?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MoQE, an MoE-style inference framework that assembles multiple quantized variants of a single FP model as quantization experts,  and trains a lightweight router to dispatch each input to the most suitable expert. MoQE achieves performance comparable to the SOTA quantization model, without incurring significant increases in inference latency. They design domain-specific routers for CV (SEResNet-8 + attention) and NLP (Transformer encoder + attention + MLP). The router is trained by labeling each sample with the lowest-loss expert, with a balancing regularizer to avoid expert collapse. Experiments on Qwen/LLaMA (NLP) with C4/WikiText/OpenWebText and ResNet/MobileNet (CV) report lower PPL/higher top-1 than individual experts; the paper also reports low extra latency (≤~5%) on a single V100S GPU.

### Strengths
1. Clear, modular formulation and training objective. The router is trained with CE + dynamic load-balancing. 

2. Router architectures are task-aware and lightweight. CV uses a three-stage SEResNet-8, then an 8-head self-attention module; NLP uses a Transformer encoder, aligning with modality priors.

3. Evidence for expert complementarity. The paper motivates MoQE via data-dependent expert bias: e.g., “no single quantization method remains optimal on all subsets” (ImageNet sub-datasets; max gap 4.7%).

4. Consistent empirical gains and scaling with #experts. MoQE often outperforms the best single expert across models/datasets (e.g., Table 1; Table 2; Table 4) and improves as the number of experts increases from 2 to 5.

5. Latency and memory analysis. Extra inference time is small, end-to-end overhead remains within 5% even in worst-case frequent switching, and VRAM ≈ single Int8.

### Weaknesses
1. Evaluation scope (datasets & metrics) is narrow for NLP. Results use C4/WikiText/OpenWebText perplexity only; no downstream tasks (e.g., QA, reasoning) are reported. The paper repeatedly frames claims in terms of PPL (Table 1), but no task-level evaluations are provided to validate end-user utility

2. Cost/footprint trade-offs underplayed. While the latency impact is small, the paper concedes extra CPU RAM and disk to keep multiple experts resident. The serving cost implications deserve more emphasis.

3. Comparison fairness around embeddings. For NLP, “the quantization experts’ embedding layer is left in full precision” and “MoQE uses the pre-existing embedding layer of the original full-precision model”. It’s unclear whether baselines also keep embeddings FP; if not, MoQE might enjoy an unfair FP advantage in the input stack.

### Questions
1. Generalization beyond PPL. Most NLP results are perplexity-only on C4/Wiki/WebText . Could you add zero-shot/GLUE/MMLU-style tasks (or CV transfer beyond ImageNet) to validate that MoQE’s gains transfer to tasks rather than only PPL?

2. Router label generation & data leakage. Router labels are assigned by “the expert that yields the lowest loss” per sample (training uses the training set). Is labeling performed on the same data used to train the router? If so, can you provide an evaluation where labels are computed on a held-out set to avoid teacher-student leakage?

### Soundness
3

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
4

### Summary
This paper proposes Mixture of Quantization Experts (MoQE), a novel post-training quantization framework that enhances model accuracy using a Mixture-of-Experts (MoE) architecture. Instead of relying on a single quantized model, MoQE integrates multiple quantization variants of the same full-precision model as “quantization experts” and uses a lightweight router model to dynamically assign each input to the most suitable expert. Extensive experiments on ResNet, MobileNet, LLaMA, and Qwen across datasets such as ImageNet, WikiText, C4, and OpenWebText show that MoQE consistently outperforms individual quantization methods (e.g., GPTQ, SmoothQuant, AWQ) at both Int8 and Int4 bitwidths.

### Strengths
1) Experiments are comprehensive, covering ResNet, MobileNet, LLaMA, and Qwen models across diverse datasets (ImageNet, WikiText, C4, OpenWebText).
2) The paper is generally well written and clearly structured, with extensive figures, tables, and methodological details.
3) The proposed MoQE achieves consistent performance gains over state-of-the-art quantization baseline.

### Weaknesses
1) The paper provides limited theoretical justification for why dynamic routing among quantized experts improves accuracy. A formal analysis of router convergence, expert diversity, or error bounds would strengthen the contribution.
2) While the idea is interesting, it may be perceived as a straightforward combination of two existing ideas, quantization and MoE routing, rather than a fundamentally new theoretical contribution.
3) The paper could be improved by including real-world deployment results, and additional baselines such as adaptive or ensemble quantization methods.
4) The work would benefit from a deeper discussion on when MoQE might fail, e.g., highly homogeneous data distributions or limited quantization diversity, and how the approach could adapt in such scenarios.
5) The performance improvements reported in Tables 1–3 are relatively small compared to the baselines.

### Questions
1) The framework relies heavily on the assumption that different quantization methods introduce complementary biases. Could the authors provide quantitative evidence (e.g., disagreement metrics or diversity indices) showing that these experts are sufficiently distinct to justify a MoE-style combination?
2) The experiments mainly cover models up to Qwen-4B and LLaMA-3B. How would the MoQE system scale for >10B-parameter models, where router overhead and expert management may become non-trivial?
3) The work provides intuitive explanations but lacks a formal analysis of routing optimality or error bounds. Could the authors elaborate on theoretical insights explaining why routing among quantized experts yields lower loss than single-expert quantization?

### Soundness
3

### Presentation
3

### Contribution
2
