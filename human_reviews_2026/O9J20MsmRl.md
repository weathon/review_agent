# BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Diffusion transformers currently lead the field in high-quality video generation, but their slow iterative denoising process and prohibitive quadratic attention costs for long sequences create significant inference bottlenecks. While both step distillation and sparse attention mechanisms have shown promise as independent acceleration strategies, effectively combining these approaches presents critical challenges---training-free integration yields suboptimal results, while separately training sparse attention after step distillation requires prohibitively expensive high-quality video data. To overcome these limitations, we propose $\textit{BLADE}$, an innovative data-free joint training framework that introduces: (1) an Adaptive Block-Sparse Attention (ASA) mechanism for dynamically generating content-aware sparsity masks to focus computation on salient spatiotemporal features, and (2) a sparsity-aware step distillation paradigm, built upon Trajectory Distribution Matching (TDM), directly incorporates sparsity into the distillation process rather than treating it as a separate compression step and features fast convergence. We validate BLADE on text-to-video models like CogVideoX-5B and Wan2.1-1.3B, and our framework demonstrates remarkable efficiency gains across different scales. On Wan2.1-1.3B, BLADE achieves a 14.10$\times$ end-to-end inference acceleration over a 50-step baseline. Moreover, on models such as CogVideoX-5B with short video sequence lengths, our framework delivers a robust 8.89$\times$ speedup. Crucially, the acceleration is accompanied by a consistent quality improvement. On the VBench-2.0 benchmark, BLADE boosts the score of CogVideoX-5B to 0.569 (from 0.534) and Wan2.1-1.3B to 0.570 (from 0.563), results that are further corroborated by superior ratings in human evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
BLADE unifies sparse attention and step distillation to accelerate video diffusion transformers. It introduces Adaptive Block-Sparse Attention (ASA), which dynamically selects salient regions for computation, and integrates this sparsity directly into a Trajectory Distribution Matching (TDM) distillation process. This joint, data-free approach enables efficient few-step video generation without quality loss. Tested on CogVideoX-5B and Wan2.1-1.3B, BLADE achieves up to 14× faster inference and improved VBench-2.0 scores, showing that sparsity-aware distillation yields both speed and fidelity gains

### Strengths
1. Unifies two dominant acceleration axes—few-step distillation (via TDM) and sparse attention—by making sparsity part of the distillation loop rather than a post-hoc swap. The results matches current practice in few-step distillation (Trajectory Distribution Matching).

2. A diverse set up metric, including VBench, human evaluation, and PSNR/SSIM.

3. Offering both ASA (inference-only) and ASA-G (distillation-aware with global tokens) gives a practical path for immediate speedups and better quality when (light) training is allowed.

### Weaknesses
1. The paper discusses SpargeAttention and VSA but does not include either as baselines. I find ASA (inference-only) to highly resembles SpargeAttention, while VSA represents a trainable sparse attention design closely related to the ASA-G variant.  Moreover, the claim that VSA is limited by video resolution is inaccurate — VSA is not constrained by resolution in their open-sourced implementation. I feel including SpargeAtteniton as baseline is necessary (published at ICML) and including VSA is optional (a more recent work), but the author should discuss the difference.

2.  The distillation component essentially follows Trajectory Distribution Matching (TDM). The overall framework is thus a straightforward combination of two existing techniques. While I find incremental A+B paper to be acceptable, it should be supported by strong results. However,  this paper’s experimental evidence is weak, as discussed in later points.

3. One of the central arguement of this paper is combining sparse attention and distillation in a single training stage is better than doing them sequentially. However, there is not comparions against a. sparse attention tuning and then distillation. b. distillation and then sparse attention tuning.

4. The experimental evaluation is limited to small or medium-sized models and short video sequences at low resolution. No experiments are conducted on Hunyuan Video or Wan 2.1 14B, and no results are presented for long-sequence or high-resolution scenarios — where sparse attention truly matters due to quadratic scaling of attention cost. 

5. The paper claims that ASA improves generation quality, but this is not supported by sufficient evidence. On Wan 1.3B, full attention after distillation achieves a higher VBench score. The only improvement shown is on CogVideoX, an arguably old model released last year, which does not strongly support the generality of the quality improvement claim.

6. In Table 2, the authors evaluate on the H20 GPU but use FlashAttention-2 (FA2) as the dense baseline. On H20, FlashAttention-3 (FA3) should be the standard baseline, as it is roughly 40% faster than FA2. Given the reported effective sparsity rate of 0.798, a theoretical 5× speedup over FA3 is expected, yet only a 3.3× gain over FA2 is reported — indicating significant implementation inefficiencies and optimization headroom that are not analyzed.

7. This claim  “Moreover, on models such as CogVideoX-5B with short video sequence lengths, our framework delivers a robust 8.89× speedup” is misleading in the abstract, the speedup mostly comes from distillation (7.93), putting this sentence make people think sparse attention play a huge part.

8. I believe DMD2 alone can reduce the number of inference steps to 3 or 4 steps on Wan 2.1, which is faster than the proposed sparse attention + TDM solution.

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes BLADE, which unifies Adaptive Block-Sparse Attention (ASA) and Sparsity-Aware Step Distillation, introducing a new sparse attention operator to enhance efficiency. The method achieves 8.9×–14.1× acceleration on CogVideoX-5B and Wan2.1-1.3B with maintained or improved quality. The idea is clear, technically solid, and promising for efficient video generation, though validation on larger models (e.g., Wan2.1-14B) is still needed.

### Strengths
- Effective under both training-free and distillation-based settings.  

- Large speedups with stable quality (VBench and human evaluation confirmed).  

- Robust at high sparsity (~80%), outperforming similar methods.  

- Detailed pseudocode and source code are provided, making the method easy to follow and reproduce.

### Weaknesses
- Lacks large-scale and long-sequence experiments;

- ASA is currently implemented in a custom Triton kernel and Block Sparse Attention library, and a more detailed analysis of the runtime contribution of each component would be helpful.

### Questions
Have you considered including training-free results on larger models such as Wan2.1-14B to strengthen the evaluation?

Could you provide a more detailed breakdown of the runtime for each component of ASA to better explain the performance gap?

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
4

### Summary
The paper introduces BLADE, a framework that integrates Adaptive Block-Sparse Attention (ASA) with step distillation for efficient video generation. It proposes a data-free joint training approach, leveraging ASA to generate dynamic, content-aware sparsity masks and sparsity-aware Trajectory Distribution Matching (TDM) to enhance quality. Experiments on CogVideoX-5B and Wan2.1-1.3B demonstrate significant speedups (up to 14.10×) and quality improvements, validated by VBench-2.0 and human evaluations.

### Strengths
- Integration of adaptive block-sparse attention with step distillation, enabling data-free joint training for efficient video generation.
- ASA mechanism dynamically generates content-aware sparsity masks that enable high sparsity levels, achieving hardware-friendly acceleration without quality loss when combined with distillation training.
- Demonstrates substantial speedups (up to 14.10×) on diverse models like CogVideoX-5B and Wan2.1-1.3B, with consistent quality improvements on VBench.

### Weaknesses
This paper lacks details on experimental settings and comparative results, for example:
- Lack of reporting on specific GPU hours, training batch size, and memory usage for the 100-200 distillation iterations.
- Lack of inference results demonstrating video quality across low-to-high sparsity levels to illustrate the impact.

### Questions
Please refer to the **Weaknesses** above.

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
This paper presents BLADE, a novel framework designed to accelerate video diffusion models. BLADE significantly improves inference speed while maintaining or even enhancing generation quality by combining the dynamic, content-aware Adaptive Block-Sparse Attention (ASA) mechanism with the data-free Trajectory Distribution Matching (TDM) distillation process. Extensive experiments on various video models demonstrate significant improvements in kernel-level efficiency, end-to-end inference speed, and generation quality.

### Strengths
The innovative BLADE framework effectively addresses the computational bottleneck in accelerating inference for video diffusion models by jointly training the sparse attention mechanism (ASA) with trajectory distillation (TDM). This solution not only accelerates the generation process but also maintains high-quality outputs, especially in high sparsity conditions, achieving high-quality video generation with fewer steps, outperforming traditional methods. The paper is clearly motivated, well-written, and the diagrams are easy to understand, ensuring good readability.

### Weaknesses
1. Although ASA's performance is compared with traditional sparse attention methods (e.g., STA, RaA, SVG), the paper does not delve into the impact of different sparsity patterns (e.g., varying threshold settings, block sizes) on generation quality. Ablation experiments with different sparse configurations could provide further insights.
2. BLADE optimizes generation performance by jointly training sparse attention and trajectory distribution matching. The core innovation here is the fusion of sparsity with the distillation process. However, there is a lack of further experimental evidence to demonstrate the advantages of joint training.

### Questions
1. The paper mentions that ASA performs excellently in accelerating the generation process, but increasing sparsity might negatively impact generation quality. Specifically, how can the generation quality be ensured when sparsity is very high, while still maintaining significant computational acceleration?

### Soundness
3

### Presentation
3

### Contribution
3
