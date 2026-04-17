# SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
Recent advances in sequence modeling have highlighted Mamba as a state space architecture offering efficient long-range dependency modeling and providing a viable alternative to Transformers. Building upon this, Mamba-2 introduces the Structured State Space Duality (SSD), which integrates recurrent and attention modes to achieve efficiency and scalability. However, this architectural expansion substantially increases memory and latency overhead, underscoring the need for efficient compression strategies tailored to SSD. In this work, we present SSDi8, the first post-training quantization framework specifically designed for SSD to maintain a persistent INT8 path. SSDi8 introduces a reformulation that decouples element-wise multiplications from matrix multiplications, enabling reuse of quantized activations across modules. Moreover, SSDi8 adaptively quantizes channel-varying activations at cost-effective points, further reducing latency. On the accuracy side, SSDi8 explicitly leverages the intrinsic dimensional decomposition of SSD, exploiting distinct outlier distributions across axes, and incorporates an error correction term based on per-channel error statistics. Comprehensive experiments demonstrate that SSDi8 achieves accuracy comparable to FP16 while delivering up to 1.4X speedup in W4A8 and W8A8 settings. We further validate its robustness in resource-constrained environments by deploying it on the Orin NX device. Code is available at https://github.com/cau-hai-lab/SSDi8.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces SSDi8, a framework designed to accelerate Structured State Space Duality (SSD) computation using INT8 representation. Unlike prior methods (e.g., Quamba2) that quantize only the SSD inputs while performing state updates in FP16, SSDi8 establishes a persistent INT8 execution path by quantizing internal SSD computations, reusing quantized activations, and reformulating element-wise operations to ensure low-precision consistency. The framework further incorporates a sparse-aware reformulation to mitigate quantization errors and a mean-correction mechanism to compensate for layer-wise activation drift. Experiments on Mamba-2 models (1.3B, 2.7B, and 8B) demonstrate that SSDi8 achieves up to 1.47× speedup over FP16 while maintaining accuracy comparable to full precision across six zero-shot benchmarks. Moreover, SSDi8 exhibits robust performance on edge devices (e.g., NVIDIA Orin Nano), highlighting its practical deployability under resource constraints.

### Strengths
SSDi8 provides a approach to quantizing Mamba-2’s SSD modules with 8-bit, addressing quantization sensitivity and memory inefficiency not handled by prior work. Its sparse-aware reformulation and activation reuse are both theoretically grounded and empirically validated, enabling a full INT8 inference pipeline without significant accuracy loss. The paper includes comprehensive experiments across multiple model scales, showing consistent improvements over baselines like Quamba2 in both accuracy and latency. Furthermore, the mean-correction mechanism is elegantly simple yet effective, improving quantized results with minimal overhead. Finally, the inclusion of both theoretical proofs and deployment evaluations (e.g., Orin Nano) enhances the paper’s completeness and real-world relevance.

### Weaknesses
### Major Concerns
- The paper is somewhat difficult to follow, particularly in the presentation of the Sparse-aware Reformulation, which appears to be one of the main contributions. I recommend the authors derive Equation (7) explicitly from the original ChunkState formulation to establish a clearer connection between the reformulated SSD equations and the preceding context.
- The source of the reported accuracy and latency improvements is unclear. To the best of my understanding, Quamba2 [1] already quantizes inputs (x), B, and C to INT8, then dequantizes within the SSM module to perform state computation in FP16. The authors should clarify how SSDi8 maintains state propagation and accumulation in INT8 precision, and why this results in both higher accuracy and lower latency compared to Quamba2. A detailed explanation or ablation isolating the impact of full INT8 state computation would greatly strengthen the technical claims.
- Some related methods, such as HadMamba, are mentioned without citation or discussion. Properly citing these works and clarifying their relationship to SSDi8 would improve the completeness of the related-work section.





[1] Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

### Questions
- Could the authors provide a deeper comparison with Quamba2 [1]? In particular, it would be helpful to include Quamba2 results in Table 5 for a more direct and quantitative comparison. If my understanding is correct, Quamba2 already supports ChunkState quantization Q(X) and quantization of matrices B and C. A side-by-side evaluation would better clarify the incremental benefits of the proposed framework over this closely related baseline.
- Would the authors also provide a more detailed latency breakdown of the entire SSD block? Specifically, analyzing the latency contributions of the input projection, SSD computation, normalization, and output projection layers under different input batch sizes would offer valuable insight into the scalability and bottlenecks of the proposed quantization method. 

I am willing to reconsider and potentially raise my overall score if the above concerns are adequately addressed in the authors’ rebuttal or follow-up discussion.


### Minor Suggestions
- I recommend that the authors simplify the shape conversion and tensor dimension descriptions in the SSD computation steps. The current presentation is dense and difficult to follow, and a clearer explanation (potentially with a schematic or pseudo-code) would improve readability.
- I suggest that the authors include a discussion on large-batch serving scenarios, as quantization behavior and latency characteristics can differ significantly between small-batch (edge inference) and large-batch (cloud serving) settings. Highlighting how SSDi8 performs under these conditions would enhance the paper’s practical relevance and deployment perspective.

### Soundness
3

### Presentation
2

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
This paper presents a post-training quantization framework specifically designed for the Structured State Space Duality (SSD) layers in Mamba-2 models. They enable INT-8 quantization inside SSD blocks through a combination of: a sparse-aware reformulation that decouples element-wise multiplications from matrix multiplications, strategic placement of quantization operations to minimize overhead, and a mean correction mechanism. The authors achieve comparable accuracy to FP16 while delivering up to 1.4x speedup in inference. The work addresses the challenge that existing quantization methods designed for Transformers fail when applied to SSD layers due to their unique computational structure involving recurrent states and channel-varying activations.

### Strengths
- first work to successfully apply INT-8 quantization within Mamba-2's SSD architecture
- Thorough mathematical analysis
- comprehensive experimental validation across multiple model sizes up to 8B
- Deployment validation on resource-constrained devices (Orin Nano)
- clear identification and analysis of why existing methods fail on SSD layers
- well-structured paper with detailed ablation studies

### Weaknesses
- unclear notation and presentation around the reformulation (mixing step-by-step vs chunked vs parallel forward passes)
- limited theoretical justification for why the mean correction works
-some implementation details are vague (e.g., specific quantization schemes for different components)
- the paper could benefit from more intuitive explanations alongside the mathematical formulations
- limited discussion of potential limitations or failure cases
- not immediately clear what the overhead is of computing and storing the quantization scales for the various per-channel schemes

### Questions
1. Do you observe any mismatch between step-by-step vs. chunked vs. parallel forward passes, given that quantization introduces errors in the recurrence?
2. In Section 1, you claim to show results for Hadamard rotation and GPTQ but these are not in Table 1 or other results tables. Could you clarify where these comparisons are?
3. Why is the mean correction applied only to the output projection layer? What happens if applied elsewhere?
4. How sensitive is the method to the calibration dataset size and choice?
5. Could you provide more details on how the sparse-aware reformulation maintains numerical stability?
6. How does the method perform on longer sequences beyond $L=2048$?

### Soundness
4

### Presentation
3

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
This paper presents SSDi8, a post-training quantization method built specifically for Mamba-2’s Structured State Space Duality (SSD) blocks. Instead of applying generic Transformer quantization, SSDi8 restructures SSD computations so they can run almost entirely in INT8. It quantizes the group-wise activations B and C once and reuses them across modules, reformulates ChunkState to keep the matrix multiplications in low precision, and adds a small per-channel mean correction to offset quantization drift. The result is near-FP16 accuracy with up to 1.47× faster inference over FP16 and 1.38× over Quamba2, marking the first fully INT8 data path within the SSD architecture.

### Strengths
A key strength of the paper is its architecture-aware approach to quantization. Rather than applying generic Transformer-style methods, SSDi8 directly models the structure of Mamba-2’s SSD blocks and exploits their specific activation patterns.

The design choices are well motivated by empirical observations, such as head-wise activation variance and per-axis sparsity.

The experiments are comprehensive, covering multiple model scales, bit-width regimes, and hardware platforms, with consistent accuracy close to FP16 and up to 1.47× latency improvement

### Weaknesses
The paper overemphasizes the need for a fully continuous INT8 execution path through SSD without clearly explaining why this is critical for Mamba-2’s efficiency. While the idea of keeping all operations in INT8 sounds appealing, the authors do not show evidence that occasional FP16 operations meaningfully degrade throughput. The claim that the presence of an FP16 activation such as LUTstate “eliminates the efficiency of INT8 GEMM” is not well supported. A mixed-precision path would not inherently destroy INT8 performance; it would simply require limited casting or dequantization overhead. More quantitative profiling is needed to show where and how FP16 operations bottleneck the pipeline.

The novelty claim that SSDi8 is “the first post-training quantization framework specifically designed for SSD” is also overstated. Quamba2 already targets the SSD modules in Mamba-2 and applies structured activation quantization within them. The paper should clarify what makes SSDi8 distinct from existing SSD-aware methods, such as whether the main innovation is the sparse-aware reformulation, activation reuse, or the persistent INT8 recurrence path, rather than relying on the claim of being the first.

### Questions
1. Why is maintaining a fully continuous INT8 execution path so essential for SSD efficiency?

2. Do the authors have profiling data showing where FP16 operations become actual latency bottlenecks in SSD?

3. How much runtime is really lost when LUT_state state remains in FP16—can this be quantified?

4. Could a mixed-precision (INT8 + FP16) path achieve similar performance without full reformulation?

5. The paper claims that FP16 activations “eliminate” INT8 GEMM efficiency—what hardware-level evidence supports that?

6. In what specific ways does SSDi8 differ from Quamba2, which already quantizes SSD blocks in Mamba-2?

7. Is the main novelty the sparse-aware reformulation, the activation reuse strategy, or the persistent INT8 recurrence path?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SSDi8, a post-training quantization (PTQ) framework specifically designed for Mamba-2's Structured State Space Duality (SSD) architecture. The authors first identify key challenges that make naive PTQ methods unsuitable for SSDs, such as heterogeneous activation scales and the mix of element-wise and GEMM operations. To address this, SSDi8 introduces three core contributions: (1) a sparse-aware reformulation that pre-scales activations to maintain a persistent INT8 data path within the SSD block; (2) a "quantize-once, reuse" strategy for channel-varying activations \(B,C\) along the group axis to minimize DRAM traffic; and (3) a lightweight per-channel mean correction technique to mitigate quantization bias.

Empirically, SSDi8 demonstrates near-FP16 accuracy on Mamba-2 models up to 8B parameters. It achieves up to a 1.47x speedup for the SSD block compared to FP16 (1.38x over the Quamba2 baseline) and shows practical latency gains on edge hardware like the Orin Nano. The framework strategically keeps two submodules, ChunkCumsum and ChunkScan2, in FP16 to preserve stability. A theoretical proposition in the appendix further supports the sparse-aware reformulation, showing it can reduce quantization MSE under certain conditions.

### Strengths
- **Principled, SSD-Specific Design:** The work is well-motivated, starting from a clear analysis of why standard PTQ fails for Mamba-2. The proposed sparse-aware reformulation and persistent INT8 state are theoretically grounded (as shown in the Appendix) and effectively target the unique properties of the SSD architecture.
- **Consistent Accuracy Gains:** The method's effectiveness is demonstrated across a range of model scales (1.3B, 2.7B, 8B) with perplexity and accuracy metrics that are nearly on par with the FP16 baseline and competitive with or better than Quamba2. The inclusion of latency measurements on edge hardware (Orin Nano) highlights the practical deployability of the framework.
- **Low-Overhead Accuracy Recovery:** The per-channel mean correction method effectively improves accuracy with a negligible latency overhead (1-2%), offering a practical and efficient solution for mitigating quantization bias.

### Weaknesses
- **Marginal End-to-End Speedup Over Prior Work:** While the SSD block itself shows significant speedups (up to 1.47x), the reported end-to-end latency gains over the most relevant baseline, Quamba2, appear modest. This raises questions about the overall practical impact, as other parts of the model may become bottlenecks. A more detailed breakdown of end-to-end latency would help clarify the real-world benefits.
- **Limited Evaluation on Recent Mamba-2 Models:** The evaluation is confined to the base Mamba-2 models. The paper misses an opportunity to demonstrate the framework's relevance and scalability by applying it to recent, influential foundation models that incorporate Mamba-2, such as Nemotron-4-340B-Base, which use hybrid SSM-Attention architectures and operate on much longer contexts.

### Questions
1.  The evaluation is performed on context lengths up to 4k. I believe this is limited by the base Mamba-2 models. However, a key advantage of quantizing SSDs may become apparent when scaling the sequence length to a longer one. It might be good and interesting to see how the e2e latency gains are there at longer context (e.g., 8k, 16k or longer)

2. Can SSDi8 apply on mabma2-transformer hybrid models [1,2]?  

[1] https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-128k
[2] https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K

### Soundness
3

### Presentation
3

### Contribution
4
