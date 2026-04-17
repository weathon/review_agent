# MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 2

## Abstract
Training large language models with FP8 formats offers significant efficiency gains. However, the reduced numerical precision of FP8 poses challenges for stable and accurate training. Current frameworks preserve training performance using mixed-granularity quantization, i.e., applying per-group quantization for activations and per-tensor/block quantization for weights. While effective, per-group quantization requires scaling along the inner dimension of matrix multiplication, introducing additional dequantization overhead. Moreover, these frameworks often rely on just-in-time scaling to dynamically adjust scaling factors based on the current data distribution. However, this online quantization is inefficient for FP8 training, as it involves multiple memory reads and writes that negate the performance benefits of FP8. To overcome these limitations, we propose MOSS, a novel FP8 training framework that ensures both efficiency and numerical stability. MOSS introduces two key innovations: (1) a two-level microscaling strategy for quantizing sensitive activations, which balances precision and dequantization cost by combining a high-precision global scale with compact, power-of-two local scales; and (2) automatic scaling for weights in linear layers, which eliminates the need for costly max-reduction operations by predicting and adjusting scaling factors during training. Leveraging these techniques, MOSS enables efficient FP8 training of a 7B parameter model, achieving performance comparable to the BF16 baseline while achieving up to 34\% higher training throughput.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MOSS, a specialized strategy designed to accelerate FP8 training for Large Language Models (LLMs). Traditional training frameworks often face a trade-off between quantization accuracy and computational overhead, dictated by the granularity of quantization. MOSS innovatively proposes methods to better balance these two factors within the FP8 training framework: First, for sensitive activation tensors, it employs a two-step quantization strategy using coarse-grained FP32 quantization scale factors combined with fine-grained MXFP quantization factors, addressing the high computational overhead associated with the dequantization phase in traditional per-group quantization. Second, for weight tensors, it adopts a strategy of real-time prediction of learning rates during training, thereby avoiding the computationally expensive real-time calculation of scaling factors. Experiments demonstrate that their strategy achieves a throughput improvement of approximately 35% over a BF16 baseline method while maintaining the training performance of a 7B model.

### Strengths
1.  **Important Problem, Insightful Perspective:** The trade-off between quantization accuracy and computational overhead is a central focus in the field of quantized training. The paper's focus on FP8 training is one of the most important and active problems in this area, and the proposed solution for balancing these factors is highly instructive.
2.  **Sufficient Novelty:** The two-step quantization framework using coarse-grained FP32 scale factors combined with fine-grained MXFP scale factors effectively addresses the significant computational overhead associated with the dequantization phase within traditional per-group frameworks on CUDA Cores. This approach cleverly leverages high-performance Tensor Cores. The framework for automatic weight quantization is novel; by integrating with the learning rate within the optimizer, it directly bypasses the step of calculating scaling factors in real-time, offering potential for acceleration.
3.  **Relatively Complete Theoretical Analysis:** The theoretical analysis is convincing, whether discussing the motivation for the two-step quantization or the mathematical proof regarding the combination of autoscaling and learning rate. These analyses also incorporate practical considerations. For example, periodically recalculating the true scaling factor during autoscaling maintains mathematical correctness while adhering to the practical requirements of quantized training.

### Weaknesses
The paper has few shortcomings elsewhere, but the ablation experiments regarding the critical aspect of "computational efficiency" are insufficient. For example:

1.  **End-to-End Throughput Impact of Automatic Weight Scaling:** How much throughput improvement does automatic scaling on weights *actually* provide in an end-to-end scenario? The paper only shows improvements for GeMM alone (Table 1). While automatic scaling is certainly better than in-time scaling for GeMM separately, in-time scaling might not cause significant throughput reduction end-to-end. If the accuracy degradation caused by potentially less precise weight quantization under automatic scaling outweighs the throughput benefits end-to-end, the approach would be counterproductive. The authors are encouraged to provide analysis on this aspect.
2.  **Two-Stage Scaling Strategy for Activations:** Although the authors theoretically prove that MOSS can yield higher Signal-to-Noise Ratio (SNR), experimental validation is lacking. Ideally, a training curve like Figure 5 would be helpful, though I fully understand computational resource constraints. Alternatively, extracting activation tensors during the actual training process and providing quantized SNR results for per-group, per-tensor, and MOSS would better support the conclusions in Section 3.1.
3.  **Overall Performance Visualization:** A single, comprehensive experimental result chart showing that MOSS's two-stage activation quantization strategy achieves higher accuracy than per-tensor methods and higher throughput than per-group methods (combining existing ablation study results) would be ideal. This would clearly demonstrate the excellent trade-off achieved and make it easier for readers to grasp.

In summary, while the paper provides detailed theoretical analysis for its two main methods, it lacks corresponding experimental validation. Even a slightly more detailed ablation study with concrete results would significantly strengthen the conclusions. This is somewhat regrettable. A more thorough ablation study would incline me more towards accepting this paper.

### Questions
1.  Have you surveyed other efficient FP8 GeMM implementations besides COAT and DeepSeek? What do you consider to be the advantages of your method compared to highly efficient kernels like DeepGeMM, which rely primarily on system-level code optimizations?
2.  Do you apply any special processing to gradient quantization, similar to the specific handling done for weights or activations?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents MOSS, a framework for efficient and accurate FP8 training of large language models. It addresses the computational overhead and instability issues in existing FP8 methods, which often use mixed-granularity quantization and costly just-in-time scaling. MOSS introduces a two-level microscaling strategy that combines global FP32 scaling with local scaling, and an automatic scaling mechanism based on optimizer properties. Experiments show that MOSS matches BF16 accuracy while improving training throughput compared with other SOTA methods.

### Strengths
1. The paper pinpointly identify the quantization overhead related to on-the-fly quantization.

2. The proposed two-level microscaling method effectively balances numerical precision and hardware efficiency.

3. The proposed optimizer-based automatic scaling mechanism is novel and effective

4. Extensive experiments on pretraining and SFT demonstrates promising performance.

### Weaknesses
1. The evaluation focuses mainly on 7B-parameter models, leaving scalability to larger models (e.g., 14B, 30B+) untested. Considering the huge cost of pretraining such a huge model, demonstrating the effectiveness on SFT should be enough.

2. The authors should provide more explanation about Figure 1 (a). Is there any comparison between the BF16 GEMM baseline, and its a little bit strange about why COAT's FP8 GEMM is 5x slower than the TE baseline or DeepSeek baseline.

3. (minor) There's a missing citation of QServe, where it also observe the quantization overhead introduced on Ampere architecture (as illustrated in Figure 3).

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents MOSS, a new FP8 training framework for large language models that aims to make low-precision training both faster and more stable. The key idea is a two-level microscaling quantization that uses a global FP32 scale together with small local 8-bit power-of-two scales, greatly reducing dequantization overhead without hurting accuracy. In addition, MOSS introduces an automatic scaling method that predicts how scaling factors evolve during training using properties of Adam-style optimizers, avoiding the costly max-reduction steps typical in FP8 systems. Experiments on OLMo-7B and LLaMA-2-7B show that MOSS matches BF16 accuracy while improving training throughput by about 34–47/%, making it a practical and hardware-friendly approach for efficient FP8 large-model training.

### Strengths
1. Clear motivation and practical relevance for FP8 LLM training.
2. Elegant two-level microscaling design balancing accuracy and efficiency.
3. Simple yet effective automatic scaling removing runtime overhead.
4. Strong empirical results: BF16-level accuracy, 34–47% faster.
5. Works on standard GPUs without special hardware support.

### Weaknesses
1. Evaluation is limited to mid-sized models (up to 7B parameters); scalability to larger settings (e.g., 30B–32B models) is not demonstrated.
2. The paper mainly reports throughput improvements, but does not deeply analyze memory, communication, or energy efficiency, which are also key for FP8 training.
3. While results are strong on core GEMM operations, extensions to other components (e.g., LayerNorm, activation functions, or optimizer states) remain unexplored.

### Questions
1. The automatic scaling method assumes that parameter updates remain bounded by the learning rate. Have the authors observed any failure cases when using different optimizers (e.g., Lion or Adafactor) or under larger learning-rate schedules?
2. While the SNR analysis (Theorem 1) shows clear theoretical benefits, the empirical gains in perplexity appear small. Could the authors comment on whether the higher SNR mainly improves training stability (e.g., fewer overflows) rather than end-task accuracy?
3. MOSS focuses on GEMM layers; however, many practical LLM bottlenecks come from nonlinear layers and communication. How difficult would it be to extend microscaling or automatic scaling to these components?

### Soundness
3

### Presentation
3

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
The paper introduces MOSS, an FP8 training framework for large language models that aims to keep Tensor Cores busy while maintaining accuracy. It has two core ideas: a two‑level microscaling scheme for activations (a coarse global scale plus tiny per‑32‑value power‑of‑two scales) that avoids dequantization work in the GEMM inner loop, and an automatic scaling method for weights that predicts scale changes from optimizer settings to remove costly per‑step max‑reductions and extra memory traffic. Implemented with custom Triton kernels on NVIDIA H800 GPUs, MOSS trains 7B‑parameter models. Overall, MOSS offers a practical, hardware‑friendly path to fast and stable FP8 training.

### Strengths
- Kernel‑aware two‑level microscaling keeps the GEMM inner loop on Tensor Cores and shifts dequantization to the epilogue; the mechanism is clearly illustrated.

- Solid empirical parity with BF16 at 7B alongside better throughput.

- The writing and figures are clear and the limitations section is candid about scope

### Weaknesses
- The paper focuses on throughput but does not report memory/communication gains,

- MOSS’s GEMM is slower than DeepGEMM on several shapes (Table 4)

- Longer runs in Appendix B report only MOSS

### Questions
- Fig. 4 fixes the interval at 500; how do accuracy/acceleration change with different intervals or learning‑rate schedules?

- Can you report peak activation memory and inter‑GPU bandwidth for BF16, COAT, and MOSS to substantiate the Intro’s memory/communication claims?

- Table 3 shows 241.8 vs 168.2 samples/s but lists “+47.5%”; by calculation it’s ~44%. Is this a typo or based on a different baseline?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MOSS, a framework for efficient FP8 training of LLMs. Its core contributions are a two-level microscaling strategy for activations to reduce dequantization overhead, and a novel automatic scaling mechanism for weights that exploits the bounded update property of Adam to eliminate runtime max-reduction costs.

### Strengths
1. Automatic weight scaling (Adam’s bounded update) avoids real-time max-reduction, outperforming TE’s delayed scaling, which is novel and efficient.
2. Proofs (SNR, bounded updates) validate designs; experiments with clear metrics compare MOSS to BF16/COAT.
3. Well-structured framework with visualizations and detailed experimental setups for reproducibility.
4. Custom kernels enable MXFP8 on non-native hardware.

### Weaknesses
1. Limited originality of two-level microscaling: The strategy overlaps heavily with the MXFP standard (OCP’s microscaling format), which already defines tensor subblock partitioning and E8M0 local scale factors to optimize FP8’s dynamic range. The addition of a FP32 global scale is also used in NVFP format, limiting originality in this module.
2. Experimental gaps: Figure 5 (OLMo-7B pretraining loss) obscures the BF16 baseline curve for steps > 2000, precluding direct verification of MOSS’s claimed loss alignment with BF16. For steps < 2000, the BF16 curve exhibits unexpectedly worse performance, which is unconvincing. Additionally, the training loss remains far from convergence, casting doubts on MOSS’s long-term stability.

### Questions
1. Could you zoom in Figure 5 to demonstrate alignment with the baseline? 
2. Why were no experiments conducted on newer LLMs like Qwen? Will you validate MOSS on these models, or are there technical barriers?

### Soundness
3

### Presentation
3

### Contribution
2
