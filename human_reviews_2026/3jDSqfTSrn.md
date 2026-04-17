# Is Finer Better? The Limits of Microscaling Formats in Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Microscaling data formats leverage per-block tensor quantization to enable aggressive model compression with limited loss in accuracy. Unlocking their potential for efficient training and inference necessitates hardware-friendly implementations that handle matrix multiplications in a native format and adopt efficient error-mitigation strategies. Herein, we reported the emergence of a surprising behavior associated with microscaling quantization, whereas the output of a quantized model degrades as block size is decreased below a given threshold. This behavior clashes with the expectation that a smaller block size should allow for a better representation of the tensor elements. We investigate this phenomenon both experimentally and theoretically, decoupling the sources of quantization error behind it. Experimentally, we analyze the distributions of several Large Language Models and identify the conditions driving the anomalous behavior. Theoretically, we lay down a framework showing remarkable agreement with experimental data from pretrained model distributions and ideal ones. Overall, we show that the anomaly is driven by the interplay between narrow tensor distributions and the limited dynamic range of the quantized scales. Based on these insights, we propose the use of FP8 unsigned E5M3 as a novel hardware-friendly format for the scales in FP4 microscaling data types. We demonstrate that UE5M3 achieves comparable performance to the conventional FP8 unsigned E4M3 scales while obviating the need of global scaling operations on weights and activations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the counterintuitive behavior in microscaling quantization where smaller block sizes can lead to higher quantization error. The authors provide both experimental evidence across several LLMs and a theoretical framework explaining the phenomenon, attributing it to scale quantization effects. They further propose a hardware-friendly fix using FP8-UE5M3 scale representation, demonstrating improved performance.

### Strengths
The discovery of the “perplexity inversion” phenomenon is novel and well-motivated.

The theoretical modeling is rigorous and matches experimental data convincingly.

The proposed UE5M3 solution is simple, practical, and hardware-friendly.

Writing and figures are clear; experiments cover multiple models and tasks.

### Weaknesses
The experiments mainly focus on inference; it would strengthen the paper to evaluate whether the same anomaly occurs during training.

While hardware feasibility is discussed qualitatively, more quantitative data (e.g., area, latency, or energy cost of adding one exponent bit) would clarify the trade-offs.

The proposed format is only tested on sub-10B models. Given the claim of generality, evaluating larger-scale LLMs (e.g., 30B–70B) would enhance credibility.

Some connections to existing FP8 and mixed-precision deployment standards (e.g., NVIDIA MXFP4, OCP spec) could be more explicitly compared.

### Questions
How does the anomaly behave with integer quantization (INT4) in practical LLM inference, not only synthetic distributions?

Would dynamic or learned scale clipping alleviate the same issue without new hardware?

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
3

### Summary
This paper uncovers a surprise in LLM quantization: making quantization blocks too small ("finer") can paradoxically hurt performance. The authors trace this "perplexity inversion" anomaly to the standard FP8 UE4M3 format used for the per-block scales, which fails to accurately represent tensors with very small values (narrow distributions). They provide a rigorous theoretical model to prove this and propose a simple, hardware-friendly fix: FP8 UE5M3, a new scale format that uses a spare bit to add a 5th exponent bit. This new format solves the anomaly and achieves high accuracy without requiring expensive per-tensor scaling operations.

### Strengths
Identifies the counter-intuitive "finer is worse" quantization anomaly.

Develops a mathematical framework that perfectly explains the why behind the anomaly, which is a significant step beyond just observing it.

### Weaknesses
The theory is heavily based on weight distributions (modeled as Normal), with less focus on how the anomaly impacts different and often asymmetric activation distributions.

The claim of "minimal" hardware cost for UE5M3 is asserted but not analyzed in-depth (e.g., no area or latency estimates).

### Questions
How does this anomaly, and the UE5M3 fix, perform with the different, often-asymmetric distributions of activations?

Did you investigate adding a mantissa bit (i.e., UE4M4) for precision instead of an exponent bit (UE5M3) for range?

Table 1 shows that for several models (e.g., llama-3.1-8b, bamba-9b-v2), your UE5M3 proposal achieves nearly identical performance to the UE4M3-S (with scaling), not a clear gain. Given this, what is the primary motivation for a hardware change when a software mitigation performs comparably?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the unexpected behavior of microscaling FP4 quantization when FP8 scale quantization is used. The authors find that reducing block size does not always reduce error and in fact can worsen accuracy for narrow weight distributions. They develop a theoretical framework that decouples sources of quantization error and show excellent agreement between theory and empirical results across multiple models. Finally, they propose an FP8 UE5M3 scale format that mitigates the anomaly without requiring additional hardware cost, and demonstrate improved model accuracy compared to UE4M3 or per tensor scaling.

### Strengths
I appreciate the solid empirical observation and thorough investigation of a subtle but important anomaly in microscaling quantization. 

The paper formulates a clear theoretical framework that generalizes the understanding of error behavior and matches experiments well. 

The analysis in figures such as Fig 2b and Fig 3c is especially compelling as it isolates the dependence on distribution width and scale quantization. 

The proposed UE5M3 solution is simple, hardware friendly, and demonstrates practical effectiveness.

### Weaknesses
The anomaly is a surprising phenomenon for readers and it may help to offer a concise intuitive explanation earlier in the introduction, rather than waiting until later sections, so that readers understand the high level mechanism before diving into the detailed framework. For example, a short statement that quantization of scales interacts with narrow distributions and reduces effective representable range could improve clarity.

It would also be valuable to expand the discussion to other scale precisions. The paper focuses on FP8 scales versus FP16 and the new UE5M3 format. Discussion on whether the same anomaly is expected for future lower precision formats such as FP4 scales or mixed mantissa exponent configurations would help generalize the insight.

### Questions
Why does FP16 scaling not suffer from this anomaly if the cause is related to the deviation of the maximum weight in a block and scale resolution?

Could you briefly comment on expected behavior for future lower precision scale formats such as FP4 or hybrid exponent mantissa configurations. For example, if microscaling continues to push toward fewer bits for scale, should we expect similar inversion behaviors and would your theoretical framework still apply.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates an unexpected limitation of microscaling quantization—a fine-grained, per-block quantization method increasingly used for efficient training and inference of large language models (LLMs). While smaller block sizes are generally assumed to improve quantization accuracy, the authors discover a quantization anomaly whereby further reducing block size below a threshold increases model perplexity, a phenomenon they term perplexity inversion. They diagnose this anomaly through extensive experimentation across various LLMs and develop a robust theoretical framework that decomposes the Mean Squared Error (MSE) into three distinct contributions, revealing that the quantization of scaling factors and the effect of zero-rounding are the primary drivers of the inversion, especially for narrow tensor distributions. To address this, the paper proposes FP8 unsigned E5M3 scales, demonstrating that this hardware-friendly solution effectively mitigates perplexity inversion by offering an increased dynamic range without the need for global scaling operations.

### Strengths
1. The paper makes a highly original and significant contribution by identifying "perplexity inversion," a counter-intuitive phenomenon where smaller block sizes unexpectedly increase quantization error in microscaling for LLMs. This challenges common assumptions and reveals a critical pitfall for future low-bit quantization efforts.
2. The FP8 UE5M3 solution stands out as a key strength due to its practical and well-reasoned approach to mitigating the identified perplexity inversion. By repurposing an unused bit to extend the exponent range, UE5M3 significantly increases dynamic range, enabling better representation of small-magnitude elements crucial for narrow tensor distributions. This design is not only hardware-friendly, requiring minimal modifications to existing infrastructure, but also achieves comparable or superior performance to more complex per-tensor scaling methods, effectively simplifying the quantization pipeline while preserving or enhancing model accuracy. Its foundation in the paper's theoretical insights ensures it is a targeted and robust solution to the core problem.

### Weaknesses
1. The paper effectively shows perplexity inversion with FP4 elements and FP8 UE4M3 scales. Do the authors observe similar inversion with other low-bit formats (e.g., INT4, INT8, other FP formats) and quantized scales? Clarifying if this mechanism is universally applicable or specific to the studied configuration would define the discovery's scope.
2. The paper emphasizes the hardware-friendly nature of UE5M3, particularly for inference. However, the practical implications of integrating UE5M3 during the training phase, especially for quantization-aware training (QAT), are not fully elaborated. The discussion mainly focuses on FP8 UE4M3 for existing hardware. Clarifying how the extended exponent range of UE5M3 impacts gradient calculations, potential numerical stability issues during training, or if it primarily targets post-training quantization (PTQ) scenarios would be beneficial.
3. While the proposed FP8-UE5M3 format effectively extends the dynamic range of fixed scales, the paper lacks a comparison with adaptive scaling methods such as VS-Quant (Per-vector Scaled Quantization for Accurate Low-Precision Neural Network Inference) and GWQ (Gradient-Aware Weight Quantization for Large Language Models). These approaches also refine scale granularity through per-vector or group-wise learnable scaling. Without an empirical or qualitative comparison, it remains unclear whether UE5M3 offers distinct advantages over these adaptive strategies or if their benefits overlap.

### Questions
1. The paper effectively demonstrates perplexity inversion using FP4 elements and FP8 UE4M3 scales. It would be valuable to understand if this phenomenon is specific to this combination or broadly applicable. Have the authors observed similar inversion when using other common low-bit quantization formats (e.g., INT4, INT8, or different FP variants) with their corresponding scales also quantized? Clarifying whether the identified mechanism (interplay of narrow distributions and limited scale dynamic range) is a universal challenge or uniquely pronounced with the studied configuration would better define the scope and novelty of this important discovery.
2. The benefits of UE5M3 are predominantly highlighted for inference. However, its role in the training phase, particularly within a Quantization-Aware Training (QAT) framework, needs further clarification. Could the authors elaborate on how the extended exponent range of UE5M3 impacts gradient calculations, numerical stability, or convergence during QAT compared to UE4M3? Understanding whether UE5M3 is primarily geared towards Post-Training Quantization (PTQ) or if it seamlessly integrates with QAT, potentially requiring specific modifications or offering distinct advantages, would provide a more complete view of its practical utility.
3. While the UE5M3 format improves scale representation through a fixed hardware-defined design, it would be valuable to compare it with adaptive scaling methods such as VS-Quant (Dai et al., 2021) and GWQ (Yang et al., 2024), which optimize scales per vector or group. These methods share the goal of mitigating quantization errors via finer-grained scaling. An empirical or qualitative comparison discussing complexity, overhead, and accuracy trade-offs would help clarify UE5M3’s unique advantages and positioning within the current quantization landscape.

### Soundness
3

### Presentation
3

### Contribution
3
