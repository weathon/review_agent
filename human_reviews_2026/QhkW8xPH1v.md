# Reliability Scaling Laws for Quantized Large Language Models

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Quantization is a powerful strategy to build capable and resource-efficient large language models (LLMs) by reducing the bitwidth of the parameters. While quantized LLMs achieve state-of-the-art performance on unperturbed inputs using standard predictive metrics, their performance on perturbed inputs, measured using subsidiary reliability metrics, remains underexplored, despite its importance for safe and reliable deployment. To address this gap, we conduct a comprehensive reliability evaluation of quantized LLMs consisting of three key components: (1) Uncertainty: We assess the trustworthiness of LLMs quantized to 2, 3, 4, and 8 bits using six different quantization methods, employing established uncertainty metrics operating at both token and sequence levels. 
(2) Robustness: We design character-level and word-level input perturbations to evaluate the reliability of quantized models under semantically-preserving variations in the inputs that commonly arise in real-world applications.
(3) Reliability scaling trends: We investigate how the reliability scales with the total number of model bits. Interestingly, our study reveals that while the performance scales monotonically with the total number of bits, the reliability scalings show nonlinear trends. Specifically, a reliability peak occurs for 4-bit quantized models, indicating that quantizing moderately sized base models offers the best reliability-efficiency trade-off. Additionally, our empirical findings reveal that quantization can enhance the robustness of LLMs to natural input perturbations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses an interesting problem, and the experiments are empirically robust. The proposed concept of a "Reliability Scaling Law" is insightful, breaking down reliability into three interrelated yet distinct concepts: uncertainty, robustness, and reliability scaling trends. These three focus on different aspects of model performance. However, the current contribution primarily focuses on empirical validation, lacking theoretical depth and systematic comparative analysis.

### Strengths
Reliability is a key limiting factor in the application of large models, but there has been little systematic analysis of how this scales with time. The experimental design compares different models, tasks, sampling parameters, prompts, and other variables. Scaling curves and error decomposition plots clearly illustrate the key findings, and the fitting results in Figures 3–5 are particularly explanatory.

### Weaknesses
**1. Limited Innovation and Insufficient Theoretical Depth:**

The main contribution of this paper is to transfer the scaling law framework of Kaplan et al. (2020) to the "reliability" dimension. However, it does not theoretically model the failure point, and lacks clarity on whether the occurrence of reliability saturation is related to capacity limits or calibration errors.

**2. Inadequately Clear Metric Hierarchy:**

The paper uses "reliability" as a unified metric, but it actually mixes multiple concepts: stochastic consistency, logical soundness, and factual accuracy, resulting in limited interpretability.

Although the authors attempt to decompose reliability using uncertainty and robustness, the numerical definitions and calculation methods of the three are not strictly distinguished. At the same time, in the direction of LLM security and trustworthiness, there are relatively systematic analysis frameworks available for reference ([1,2,3,4]). The author did not include these methods in the evaluation or comparison, which weakened the scientific persuasiveness of the paper.

3. The logical structure is clear, but the language connection is a bit stiff, and there is a need to improve the paragraph transition and argument coherence.

**Ref:**

[1] Hong J, Duan J, Zhang C, et al. Decoding compressed trust: Scrutinizing the trustworthiness of efficient llms under compression. ICML'24

[2] Dong P, Li H, Guo S. Durable quantization conditioned misalignment attack on large language models. ICLR'25

[3] Chen K, Zhang J, Hu J, et al. Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models. ICML'25

[4] Li, Shiyao, et al. "Evaluating quantized large language models.ICML'24

### Questions
Is there a unified scaling exponent for the three dimensions (uncertainty, robustness, reliability)?

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
3

### Summary
The paper investigates the reliability of quantized large language models (LLMs) under semantics-preserving natural perturbations, decomposing reliability into calibration/uncertainty and robustness to perturbations, and evaluating these aspects systematically. The core approach uses the total number of model bits (parameter count × bit-width) as a unified capacity axis to compare scaling trends across model sizes and quantization precisions, fitting a log-quadratic relationship. Across multiple benchmarks and perturbation types, task performance (accuracy and perplexity) increases monotonically with total bits, while reliability metrics are non-monotonic. In most configurations, a 4-bit reliability–efficiency sweet spot emerges. The key conclusion is that, under a fixed memory or storage budget, appropriate low-bit quantization (especially 4-bit) can not only save resources but also improve robustness and calibration to semantics-preserving noise.

### Strengths
1. The study performs experiments across multiple public benchmarks and diverse natural perturbations, uncovering a non-monotonic relationship between performance and reliability metrics and highlighting a clear 4-bit sweet spot.  
2. The results offer practical guidance for deployment, showing that under resource constraints, 4-bit quantization can improve inference efficiency while enhancing model robustness and calibration.  
3. Radar plots over 15 semantically-preserving perturbations show that 4-bit maintains performance, and quantization improves reliability.

### Weaknesses
1. Does not cover newer or more diverse architectures (e.g., Qwen3, Llama 4); evaluation is limited to LLaMA 3 and OPT.  
2. The setup focuses on QA/LM, with generation truncated at 20 tokens, which may bias toward short-answer scenarios; robustness and calibration for long-form generation remain untested.  
3. Although the paper reports the GPUs used and rough runtime, it lacks a systematic comparison of throughput, latency, and memory footprint; “total number of model bits” functions more as a capacity axis than an efficiency axis.  
4. Many of the benchmarks used are relatively dated.

### Questions
1. Have you tried repeating the experiments on the Qwen series (such as Qwen2.5 or Qwen3) or on newer models like Llama 4 to verify the generality of the 4-bit sweet spot?  
2. Are the conclusions stable under different decoding strategies, such as variations in temperature, top-k, or top-p sampling?  
3. Have you compared models with the same total parameter bits on efficiency metrics such as output throughput, latency, and memory footprint, and is using total parameter bits as a unified capacity axis reasonable for efficiency comparisons?

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
4

### Summary
This paper investigates how quantization affects robustness in large language models (LLMs), beyond standard performance metrics like accuracy or perplexity. They conduct a comprehensive reliability evaluation of six state-of-the-art quantization techniques. Their study reveals that while the performance scales monotonically with the total number of bits, the reliability scalings show nonlinear trends.

### Strengths
This paper studies the effect of quantization from an underexplored perspective: reliability under quantization, including robustness to realistic perturbations. 

Experiments are extensive across several model families, sizes, bitwidths, and quantization methods, covering diverse datasets.

This study reveals that while the performance scales monotonically with the total number of bits, the reliability scalings show nonlinear trends. This is a new finding.

### Weaknesses
**Limited backbone model**: The experiments are conducted on LLaMA and OPT series, which are a bit outdated. As an empirical study on LLM quantization, the authors should try more diverse model series to prove that the findings are not restricted to specific models.

**Character-level and word-level perturbations are not efficient for robustness**: The paper is mainly investigating how well a quantized model resists perturbations. However, robustness has a broader meaning: all operations maintaining semantic invariance should be considered.

**Lacking insight**: While this paper proposes some interesting findings, the mechanism behind these findings is unclear, it is too empirical.

### Questions
see weaknesses

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides the first comprehensive analysis of how quantization influences the reliability of LLMs.
The authors examine both uncertainty and robustness dimensions by introducing 15 semantically-preserving input perturbations and evaluating their effects across multiple quantization levels and model families.
Through empirical scaling analyses, the paper uncovers a non-monotonic trend in reliability: while accuracy scales monotonically with total model bits, reliability peaks at 4-bit precision. 
This work offers valuable insights for understanding how quantization impacts model robustness and calibration, suggesting that quantization may even improve resilience to natural perturbations.

### Strengths
**1. Novel reliability-focused analysis of quantization.** This is the first study to systematically analyze reliability scaling laws under quantization, shifting focus from mere accuracy preservation to robustness and uncertainty calibration.

**2. Comprehensive experimental evaluation.** The evaluation is thorough and well-designed, covering numbers of experiments across various base models, quantization methods, bitwidths, datasets, and perturbation types at varying intensities, providing strong empirical evidence for the claims.

**3. Well-motivated perturbation design.** The character-level and word-level perturbations are carefully designed to reflect realistic typed digital communication scenarios rather than adversarial attacks, making the robustness evaluation ecologically valid.

**4. Insightful and counterintuitive finding.** The observed reliability peak at 4-bit precision challenges common assumptions that higher precision monotonically improves reliability, revealing a new dimension of quantization effects.

### Weaknesses
**1. Limited theoretical explanation for 4-bit peak.** While the paper observes and documents the reliability peak at 4-bit quantization across multiple settings, the theoretical explanation remains somewhat superficial, attributing it to a balance between quality degradation and overconfidence without deeper mechanistic analysis of why this particular bitwidth achieves optimal regularization.

**2. Missing generalization to other compression forms.** The study excludes pruning and quantization-aware training, leaving open whether the observed reliability trends generalize to broader compression paradigms.

**3. Insufficient validation.** All experiments focus on QA and language modeling benchmarks; it would strengthen the claims to evaluate reliability scaling in reasoning, dialog tasks.

### Questions
**1. Generalization to other compression methods.** The paper briefly discusses pruning and knowledge distillation in the introduction, but it remains unclear how reliability scaling trends might differ across these alternative compression paradigms. 
Could the authors elaborate on their expectations regarding reliability behaviors under pruning (e.g., structured vs. unstructured) or distillation (e.g., teacher–student reliability transfer)? 
Additionally, how might quantization-aware training (QAT) affect the observed reliability peak-does fine-tuning mitigate or amplify quantization-induced regularization effects? 
It would be valuable if these hypotheses could be empirically verified or at least partially validated through controlled experiments.

**2. Domain generalization to other modalities.** While the study focuses on textual LLMs, do the authors expect similar non-monotonic reliability trends to emerge during compression in other domains, such as the quantization of vision models, zero-shot quantization settings, or PTQ / QAT of VLMs?
Given that quantization in ViTs and multimodal encoders often interacts differently with robustness and calibration, extending this analysis to non-text modalities could provide stronger evidence for the universality of the proposed reliability scaling laws.

**3. Perturbation-specific quantization benefits.** Among the designed perturbations, are there specific categories (e.g., character-level vs. word-level, or distinct perturbations such as emoji or slang) where quantization yields particularly strong or weak robustness gains? 

**Misc.**
Please adjust paragraph formatting to avoid orphaned lines at the end of paragraphs, and ensure figures are better integrated with text to prevent single-line overflows that disrupt the visual flow of the paper.

### Soundness
3

### Presentation
3

### Contribution
3
