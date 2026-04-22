# Block Rotation is All You Need for MXFP4 Quantization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Large language models (LLMs) have achieved remarkable success, but their rapidly growing scale imposes prohibitive costs in memory, computation, and energy. Post-training quantization (PTQ) is a promising solution for efficient deployment, yet achieving accurate W4A4 quantization remains an open challenge. While most existing methods are designed for INT4 formats, the emergence of MXFP4—a new FP4 format with various hardware support (NVIDIA, AMD, Intel)—raises questions about the applicability of current techniques. In this work, we establish a comprehensive benchmark of PTQ methods under the MXFP4 format. Through systematic evaluation, we find that methods like GPTQ consistently deliver strong performance, whereas rotation-based approaches, which are almost used by all state-of-the-art approaches, suffer from severe incompatibility with MXFP4. We further provide the first in-depth analysis of this conflict, tracing its root to a fundamental mismatch between MXFP4’s PoT (power-of-two) block scaling and the redistribution of outlier energy via global rotation. Building on this insight, we propose a simple yet effective block rotation strategy that adapts rotation-based methods to MXFP4, leading to substantial accuracy improvements across diverse LLMs. Our findings not only offer clear guidance for practitioners but also set a foundation for advancing PTQ research under emerging low-precision formats.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the effect of rotations applied in the LLM post-training quantizations, focusing on the MXFP4 data format for both weights and activations.
1. It evaluates the existing quantization methods (RTN, GPTQ, SmoothQuant, QuaRot, OmniQuant, SpinQuant) on MXFP4 in the same set of benchmarks, and compares the accuracy gains between INT4 and MXFP4 when applying rotations for RTN and GPTQ.
2. It points out that the rotation's accuracy gain on MXFP4 is not as good as that on INT4. Through experimental analysis, the observed accuracy differences are attributed to the shape of MXFP4's error curve and the rotation-induced increase in inlier magnitudes.
3. It proposes a blockwise rotation strategy for MXFP4 quantization, which outperforms the full rotation in accuracy and speed.

### Strengths
In terms of originality, this paper provides one of the first analyses of the newly emerged MXFP4 data format and shows comprehensive benchmark results across different methods, models, and tasks.

In terms of quality, the claims and results in the paper look correct and make sense.

In terms of clarity, this paper is well-written and easy to understand. This paper has nice visualizations.

In terms of significance, this paper points out a significant problem of why rotations do not work in MXFP4 quantization as well as in the INT4 quantization, and provides an easy and simple method (block rotation) to mitigate this problem.

### Weaknesses
1. The title "Block Rotation is All You Need for MXFP4 Quantization" is an overclaim. From the accuracies in Table 3, there is a clear gap between the block rotation and the full-precision FP16 baseline. The authors do not prove their block rotation method is optimal over any potential methods for MXFP4 quantization.

2. The claims in Section 4 (Why Rotation Transforms Hurt MXFP4) are purely supported by the analysis on empirical data. It would have been better to also have theoretical guarantees.

3. The proposed method, BRQ, appears in Section 5 (Experiments) without a proper description. I can only know it from the caption of Table 3 that BRQ stands for block rotation transformation, but this is not sufficient. For example, I do not know if it uses RTN, GPTQ, or another quantization method as a backend after applying block rotations. It is also unclear whether the blockwise rotation matrices are shared across blocks or layers, and whether they are general orthogonal matrices or random Hadamard matrices.

4. The performance analysis in Section 5.3 only compares the prefill latency. The decoding latency, which generally dominates the runtime of sequence generation, is missing despite being claimed in Line 466.

5. In Appendix A.2, the calibration dataset used for SpinQuant and BRQ_spin is 800 sequences of length 2048, i.e., 1,638,400 tokens, whereas those of other methods are 262,144 tokens. This creates an unfairness in the benchmarks.

6. The full-precision LLaMA-3 series models use float16 (FP16) while the LLaMA-3 series models and Mistral 7B use bfloat16 (BF16). The authors fail to distinguish the two data types in the paper.

### Questions
1. I do not get the point D (Line 236) in Section 3.2. The statement says that, after rotation, BINT4 outperforms BFP4, and BFP4 outperforms MXFP4. How is it related to the divergent behaviors under FP16 vs PoT scaling? Shouldn’t the last sentence be MXINT4 underperforms compared to its FP4 counterpart (MXFP4)?

2. Section 4.1 classifies the blocks into two types: regular blocks and outlier blocks. However, it is not clear what is considered an outlier. There is a definition in the caption of Figure 6 in Section 4.2. Does this definition also apply to Section 4.1?

3. On Section 4.3, the last bullet point (Line 383), the online rotation should be applied to the input activations of all layers. Why are only the computations of the down-project layers reduced? Why is $R_4$ treated specially here (and in the conclusion on line 483) compared to the $R_1$ to $R_B$ in Equation 1?

4. Potential typos:
- Line 193: 3.35 should be 13.35.
- Line 319: blue area should be yellow area.
- Line 428: 7.68 should be 7.62.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an investigation into the applicability of existing post-training quantization (PTQ) methods for LLMs under the emerging MXFP4 format. The core finding that global rotation-based methods are fundamentally incompatible with MXFP4's block-wise scaling, thus they posit that a simple block-wise rotation (BRQ) is able to mitigate this issue. They provide extensive experiments and analysis. While the topic is timely, the paper suffers from a fundamentally incremental contribution and a lack of technical novelty that is enough to question its value to the quantization community. The core insight is deemed obvious for the target audience, and the solution does not constitute a significant algorithmic advance.

### Strengths
1. The evaluation is comprehensive. It tests multiple models (LLaMA-2/3, Mistral, Qwen), includes both perplexity and downstream task accuracy, and compares against a wide range of strong baselines (GPTQ, OmniQuant, SpinQuant, etc.). The inclusion of a 70B model further strengthens the claims.

2. The explanation of how MXFP4's PoT scaling struggles with large values and how global rotation amplifies small values in regular blocks is clear, intuitive.

### Weaknesses
1. The central problem and its solution are a straightforward, expected outcome for anyone with deep expertise in quantization. Applying a block-level transformation to align with a block-level quantization scheme is a natural and almost trivial engineering adjustment, not a novel research contribution. The MXFP4 format, by design, uses local block scaling (PoT) to contain outliers. Applying a global operation that deliberately spreads out outlier energy directly counteracts the format's core design principle. Therefore, observing a performance collapse is not a discovery; it is a confirmation of a predictable hardware/algorithm mismatch.


2. The proposed BRQ method is a direct and obvious application of existing concepts. It simply restricts the well-known rotation transform to the block granularity defined by the hardware. This does not represent a new algorithm or a conceptual breakthrough.

3. They selected 'datasets and benchmarks'  as the primary area, but there is no new datasets or benchmarks provided.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper identifies a critical mismatch between global rotation‐based quantization methods and modern block-power-of-two (PoT) floating-point formats (e.g., MXFP4) in large language models, noting that traditional rotations can amplify error when applied across hardware blocks. To address this, the authors propose Block-Wise Rotation Quantization (BRQ), applying independent Hadamard rotations within each quantization block so as to contain outlier energy redistribution locally and reduce interference across blocks.
Extensive experiments on models such as LLaMA-3 8B and Mistral 7B show BRQ recovers much of the performance loss seen by standard rotation methods under MXFP4 quantization, e.g., improving perplexity and accuracy by several percentage points while reducing memory and latency. The key takeaway is that quantization methods must align with the underlying hardware format’s scaling mechanics; in particular, block-sized rotations matched to PoT blocks can restore compatibility and effectiveness on next-generation low-bit formats.

### Strengths
- The paper is logically organized, providing a systematic analysis of the incompatibility between MXFP4 and rotation-based quantization methods, making the motivation and contributions easy to follow.

- The authors evaluate across multiple mainstream LLMs (e.g., LLaMA-3 8B, Qwen2.5, Mistral 7B) and compare with various quantization baselines such as GPTQ, QuaRot+, and BINT4, using both perplexity and zero-shot benchmarks.

- The proposed Block-Wise Rotation Quantization (BRQ) specifically addresses the degradation problem of traditional global rotation under MXFP4 format, providing a hardware-aware and theoretically motivated design.

- BRQ significantly improves the performance of rotation-based quantization on MXFP4 (e.g., PPL reduced from 12.78 to 11.95, accuracy increased from 48.83% to 49.87%) while maintaining efficiency and deployment friendliness.

### Weaknesses
-  MXFP4 is a block-wise quantization method, and adopting a block-wise rotation transform seems to be an intuitive idea.

- While the proposed BRQ method is empirically effective, the paper lacks deeper theoretical justification or formal analysis explaining why block-wise rotation achieves better quantization stability.

- In integer quantization, group-wise quantization is often required as well. Why does combining it with a rotation transform not harm accuracy.

- The latest NVIDIA GPUs support NVFP4[1] format and use E4M3 instead of E8M0 scaling factors. Will the method described in this paper fail on more future hardware?

References:

[1] Pretraining Large Language Models with NVFP4

### Questions
Please refer to the  weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses post-training quantization (PTQ) for large language models (LLMs), focusing on the MXFP4 format. It identifies an incompatibility between rotation-based quantization and MXFP4’s power-of-two block scaling, proposing a Block-wise Rotation Quantization (BRQ) strategy to resolve this.

### Strengths
This paper establishes a comprehensive benchmark comparing state-of-the-art PTQ methods (GPTQ, SmoothQuant, QuaRot, SpinQuant) across multiple LLMs under MXFP4, providing strong empirical evidence and highlighting performance gaps in existing methods.

This paper conducts a detailed analysis of the destructive interaction between rotation-based methods and MXFP4’s power-of-two scaling.

Building on the identified issues, this paper proposes a simple yet effective Block-wise Rotation Quantization (BRQ) strategy, which adapts rotation methods to the MXFP4 format and substantially enhances PTQ accuracy across various models and tasks.

### Weaknesses
1. The paper primarily focuses on the theoretical aspects of BRQ and MXFP4 quantization but lacks a detailed evaluation on real-world hardware deployment, such as latency, memory overhead, and computational cost. This leaves a gap in understanding how BRQ performs in practical settings.
2. The experiments predominantly focus on INT4 PTQ algorithms applied to MXFP4. However, the paper does not explore other quantization formats or different model sizes (e.g., INT8, mixed-precision), limiting the generalization of the findings across various use cases.
3. While the paper highlights the issue of rotation method incompatibility with PoT scaling, it does not offer a detailed sensitivity analysis to quantify the impact of parameters such as block size, rotation strategies, or outlier distributions. A deeper exploration of these factors would provide a clearer understanding of the robustness and limitations of the proposed approach.

### Questions
1. While the grouping size and block‑scale strategy for MXFP4 are described (e.g., block size = 32 channels), it’s not clear how sensitive BRQ’s performance is to the choice of block size. Could the authors provide results or ablations showing how performance varies for block sizes of, for example, 16 vs 32 vs 64 channels?

2. The paper reports gains from BRQ in terms of accuracy/perplexity, but it lacks detailed metrics about inference runtime, memory overhead, and extra compute cost introduced by the block‑wise rotations. Could the authors include profiling (on GPU or accelerator) that quantifies the additional operations (e.g., rotation matrix multiplies) and the net latency/throughput impact of BRQ versus baseline quantization?

3. While you benchmarked selected methods on multiple widely adopted LLMs —including LLaMA‑2 7B/13B, LLaMA‑3 8B, LLaMA‑3.2 1B/3B, and Mistral‑7B—the experiments still remain within a rather narrow family of models (primarily the LLaMA/Mistral lineage). Are there particular considerations (e.g., checkpoint availability, architecture uniformity, calibration data, hardware constraints) that motivated this choice?

### Soundness
3

### Presentation
2

### Contribution
3
