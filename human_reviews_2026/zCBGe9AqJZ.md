# Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
The recent hardware-accelerated microscaling 4-bit floating-point formats such as MXFP4 and NVFP4, supported on NVIDIA and AMD GPUs, promise to revolutionize large language model (LLM) inference. Yet, their practical benefits remain unproven. We present the first comprehensive study of  MXFP4 and NVFP4 for post-training quantization, revealing  gaps between their promise and real-world performance. Our analysis shows that state-of-the-art methods struggle with FP4, due to two key issues: 
(1) NVFP4's small group size \emph{provably} neutralizes traditional outlier mitigation techniques; 
(2) MXFP4's power-of-two scale quantization severely degrades accuracy due to high induced error. 
To bridge this gap, we introduce Micro-Rotated-GPTQ (MR-GPTQ), a variant of the classic GPTQ quantization algorithm that tailors the quantization process to FP4's unique properties, by using block-wise Hadamard transforms and format-specific optimizations. We support our proposal with a set of high-performance GPU kernels that enable the MR-GPTQ format with negligible overhead, by rotation fusion into the weights, and fast online computation of the activations. This leads to speedups vs. FP16 of up to 3.6x layer-wise, and 2.2x end-to-end on NVIDIA B200, and of 6x layer-wise and 4x end-to-end on RTX5090. Our extensive empirical evaluation demonstrates that MR-GPTQ matches or outperforms state-of-the-art accuracy, significantly boosting MXFP4, to the point where it nears that of NVFP4. We conclude that, while FP4 is not an automatic upgrade over INT4, format-specialized methods like MR-GPTQ can unlock a new frontier of accuracy-performance trade-offs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides the first comprehensive study of FP4 microscaling floating-point formats—MXFP4 and NVFP4—used in recent NVIDIA and AMD GPUs for LLM quantization. The authors analyze why these formats underperform despite their hardware support: (1) NVFP4’s small group size limits outlier mitigation, and (2) MXFP4’s power-of-two scale quantization causes high quantization error. To address this, they introduce Micro-Rotated GPTQ (MR-GPTQ), a GPTQ variant tailored to FP4, using block-wise Hadamard rotations, MSE-optimized scales, and static activation reordering. They implement this through QuTLASS, a GPU kernel suite supporting fused online rotations with negligible overhead. Experiments on Llama and Qwen models show that MR-GPTQ significantly improves MXFP4 accuracy (closing most of the gap to NVFP4) and achieves up to 3.6×–6× layerwise and 2.2×–4× end-to-end speedups on Blackwell GPUs. The paper concludes that FP4 is not inherently superior to INT4, but format-specific optimization can unlock its full potential.

### Strengths
1. The paper tackles an important and emerging problem—quantization under new FP4 microscaling formats (MXFP4 and NVFP4)—that directly relates to current-generation GPU hardware (NVIDIA Blackwell, AMD CDNA4). This makes the work highly relevant for both research and deployment.
2. The paper implements QuTLASS, optimized GPU kernels that fuse quantization, rotation, and matrix multiplication for FP4 inference. Integrated with the Blackwell tensor core pipeline, these kernels add almost no overhead, effectively bridging algorithm design and hardware deployment.
3. The authors present a clear analytical treatment of quantization error for FP4, explaining the contrasting effects of rotations under Laplace vs. Normal distributions, and showing why existing PTQ methods (e.g., GPTQ, SmoothQuant) underperform on FP4 formats.

### Weaknesses
1. Despite the proposed improvements, FP4 quantization—especially with MXFP4—still suffers from noticeable accuracy degradation. Even with MR-GPTQ, the recovery remains 3–5 % below FP16 for many models, suggesting that FP4 formats are not yet competitive with INT4 or FP8 in accuracy.
2. The methodological novelty is moderate, as MR-GPTQ mainly adapts the existing GPTQ framework to FP4 formats using known ideas such as block-wise rotations and scale optimization. However, the work shows strong engineering and system-level originality—it is the first to deeply study NVFP4 and MXFP4, propose an FP4-specific GPTQ variant, and implement efficient GPU kernels with near-zero overhead. Overall, the contribution is not conceptually new but is practically impactful and timely.

### Questions
The current speed evaluation focuses mainly on end-to-end throughput. Is there any latency measurements for the prefill and decode stages, respectively?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper systematically explores the accuracy–efficiency trade-offs of 4-bit floating-point formats (MXFP4, NVFP4) and introduces QuTLASS, a GPU kernel template that integrates the Hadamard transform at effectively zero additional overhead. The study also examines how INT-family quantization methods interact with FP4 to assess practical benefits for real-world inference.

### Strengths
- The paper is well structured and easy to follow.
- Broad empirical exploration of combining INT quantization methods with FP4 (MXFP4/NVFP4).
- Introduction of QuTLASS, which integrates the Hadamard transform with minimal overhead.

### Weaknesses
- The text states FP8 is “near-lossless,” yet Table 1 lacks an FP8 baseline (e.g., E4M3/E5M2). Including it under identical conditions would anchor FP4’s position more convincingly.
- Although MR-GPTQ runs offline and does not affect inference latency directly, reproducibility details are missing (GPU type/count, hours). Please add a minimal, reproducible configuration.
- (Minor) L238: typo “teh” → “the.”

### Questions
- With FP8 (E4M3/E5M2) included as a baseline under identical tuning, how large and consistent is the gap to FP4 across tasks?
- What is the minimal reproducible setup for MR-GPTQ (GPU count, wall-time, memory)? How does cost scale for small vs. large models?

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
This paper proposes MR-GPTQ (Micro-Rotated GPTQ), a post-training quantization scheme tailored for FP4 formats, aiming to overcome the limited accuracy recovery observed when applying existing PTQ methods to FP4-quantized LLMs.

Although recent support for MXFP4 and NVFP4 on Blackwell-class GPUs enables efficient 4-bit inference, the authors show that conventional INT4-oriented techniques do not transfer well to FP4 due to format-specific constraints—most notably, NVFP4’s small group size and MXFP4’s power-of-two scale quantization (E8M0). Moreover, contrary to observations in prior low-bit quantization work, the authors find that Hadamard-based rotations can degrade MSE in NVFP4, exposing a mismatch between FP4 properties and existing rotation-based normalization strategies.

To address these issues, the authors present MR-GPTQ, which adapts GPTQ’s second-order update mechanism to FP4 via block-wise Hadamard rotation, format-aware scale optimization, static activation reordering, and fused GPU kernels (QuTLASS) optimized for Blackwell.

Experimental results on Llama-3 and Qwen models show that NVFP4 combined with MR-GPTQ recovers approximately 98–99% of FP16 accuracy, while MXFP4—despite its inherently larger quantization error—benefits substantially and approaches NVFP4-level performance. In addition, end-to-end inference achieves 2.2×–4× speedups on B200 and RTX5090 GPUs, demonstrating that FP4 can deliver strong accuracy–performance trade-offs when paired with format-specialized PTQ.

### Strengths
* Provides a systematic and quantitative analysis of the MXFP4/NVFP4 formats, clearly identifying the root causes of accuracy degradation in FP4 PTQ.
* Demonstrates, both empirically and analytically, that rotation-based PTQ techniques are not consistently effective under FP4, motivating the need for format-aware methods.
* Proposes MR-GPTQ, an FP4-specialized variant of GPTQ, which delivers meaningful accuracy improvements by explicitly accounting for FP4 constraints.
* Introduces Blackwell-optimized QuTLASS kernels that minimize the runtime overhead of micro-rotations, yielding practical performance gains (2.2×–4× end-to-end).
* Presents a well-structured pipeline—from analysis to algorithm design, kernel implementation, and evaluation—resulting in a cohesive and well-justified contribution.

### Weaknesses
### Major Weaknesses
* Insufficient motivation for choosing FP4 over INT4 (potential logical gap):
    * The paper adopts FP4 (MXFP4/NVFP4) as the primary target format, yet it does not clearly articulate why FP4 should be preferred over INT4 in PTQ scenarios.
    * A more principled comparison—including distributional properties, quantization behavior, and rotation sensitivity—would help substantiate the motivation for focusing on FP4.
    * If strong INT4 + rotation baselines (e.g., QuaRot/QuIP) can achieve performance comparable to or better than FP4 + MR-GPTQ, the necessity and relative advantage of the proposed approach may become less evident.
* Limited justification and characterization of MR-GPTQ design choices:
    * The paper introduces several components—format-aware scaling, static activation reordering, and block-wise rotation—but does not clearly explain why each is particularly necessary or effective for FP4.
    * A more principled analysis that isolates the contribution of each component (e.g., through ablations or sensitivity) would clarify whether these choices are inherently beneficial, or simply empirical heuristics.
    * It is also unclear under which conditions these components fail, making it difficult to assess the robustness and general applicability of the proposed design.

### Minor Weaknesses
* The source of FP4 rotation degradation is not clearly isolated; the paper does not distinguish format-intrinsic effects from configuration factors (e.g., block size, scaling, calibration), making the necessity of the proposed approach less firmly supported.
* The observed superiority of NVFP4 over MXFP4 is largely expected from format specifications (E4M3 vs. E8M0), so the empirical confirmation offers limited new insight.
* The method primarily applies heuristic extensions to GPTQ without demonstrating a substantial methodological advance, which may make the contribution appear incremental rather than novel.

### Questions
If the authors could clarify or further substantiate the major weaknesses mentioned above, I would be happy to reconsider and potentially raise my evaluation. I would particularly appreciate further clarification on these major concerns.

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
This paper analyzes the quantization error in the NVFP4 and MXFP4 formats and shows that rotation-based quantization could help MXFP4 but hurts the accuracy of NVFP4 when using round-to-nearest quantization. A new quantization method, MR-GPTQ, is proposed to use block-wise Hadamard micro-rotations, static activation reordering, and format-specific optimizations to improve the accuracy of both NVFP4 and MXFP4. The evaluation results show that MR-GPTQ could achieve better accuracy recovery than prior art. The performance results show the overhead introduced by MR-GPTQ is very small, and 2.2x and 4x end-to-end speedup is demonstrated on server-grade (B200) and consumer-grade GPUs (RTX5090).

### Strengths
- A good theoretical analysis and empirical validation of quantization error on the latest quantization schemas (NVFP4 and MXFP4).
- A good kernel implementation that minimizes the overhead of the proposed quantization method and demonstrates strong end-to-end speedup on cutting-edge GPUs.

### Weaknesses
- It would be helpful if the proposed method could be compared against existing quantization methods, FP8 and INT8, which is already used in many production situations.

### Questions
- How does the proposed method compare to FP8 and INT8 in quality and end-to-end performance?
- There is a gap between single-layer speedup and end-to-end speedup. Could you provide some analysis of this? Also, in Figure 5 (right), why is the speedup of FP4 low in a small batch size? I think attention is not a bottleneck here since the sequence length is not very long.

### Soundness
3

### Presentation
3

### Contribution
3
