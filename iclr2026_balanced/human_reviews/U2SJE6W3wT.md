## Human Reviewer 1

### Summary
This Paper proposes an improved Adversarial Diffusion Compression method. The core idea it to distill knowledge from a large-scale 3D teacher model into a well-designed and lightweight ‘2D + 1D’ student model. This is achieved through a novel adversarial distillation scheme. The proposed approach effectively addresses the critical challenge of preserving both spatial details and temporal consistency, a common problem in the compression of video super-resolution models.

### Strengths
1.	This paper proposing a “2D spatial + 1D temporal” decoupling hypothesis, and introduces a novel lightweight “2D+1D” architecture. This approach drastically cuts down on parameters and computational load, enabling efficient inference.
2.	The paper proposes a novel dual-head adversarial distillation scheme. This scheme effectively balances the richness of spatial details with temporal coherence, which is a critical challenge in the field of video super-resolution.

### Weaknesses
1.	A core assumption of this paper is that a 2D diffusion model is sufficient for synthesizing fine-grained details. However, this assumption is challenged by the experimental results. Currently, the ablation study comparing the 2D and 3D backbones is based only on the DISTS metric. To provide a more balanced and convincing comparison, the authors should consider including additional metrics that measure perceptual quality (e.g., LPIPS) and/or fidelity (e.g., PSNR, SSIM). Furthermore, the qualitative results in Figures 3 and 5 exhibit clear visual artifacts or "hallucinations." These findings suggest that the 2D diffusion model may be insufficient for generating correct details, and therefore, the validity of this core assumption is questionable. The authors should address this limitation, perhaps by discussing the trade-offs of their approach or analyzing why these artifacts occur.
2.	The paper states that the 'detail head' label for real videos is set to 'unlabeled', with the provided justification being to “encouraging the generator to produce more detail-rich frames”. This is a critical design, yet the underlying mechanism or rationale is not sufficiently explained. It is unclear why the more intuitive approach—labeling the details from real videos as 'real'—was not adopted. The authors should clarify whether this decision is supported by empirical findings (e.g., from an ablation study) or if it is grounded in some theoretical justification.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces AdcVSR, an improved adversarial diffusion compression framework tailored for real-world video super-resolution (Real-VSR).
The method builds upon the concept of Adversarial Diffusion Compression (ADC), proposing a “2D + 1D” hybrid architecture that replaces heavy 3D diffusion backbones with a 2D spatial diffusion network (a pruned SD2.1) augmented by lightweight temporal 1D convolutions.
Furthermore, it introduces a dual-head dual-discriminator adversarial distillation scheme, where two discriminators (in pixel and feature domains) independently supervise “detail” and “temporal consistency.”
Experimental results on multiple Real-VSR datasets demonstrate that AdcVSR achieves competitive visual quality while reducing parameters by up to 95% and achieving an 8× inference speedup compared to the teacher model DOVE.

### Strengths
1. The proposed dual-head discriminator effectively disentangles spatial detail enhancement and temporal consistency, addressing a long-standing trade-off in Real-VSR.

2. The results on multiple datasets and metrics (PSNR, LPIPS, MUSIQ, MANIQA, etc.) are convincing and show both efficiency and quality improvements. And the visual quality is also satisfactory.

3. The 2D+1D design combined with adversarial distillation is simple yet efficient, offering clear insights into practical diffusion model compression.

### Weaknesses
The paper lacks formal justification for why the dual-head adversarial loss leads to better convergence or perceptual trade-off control.

### Questions
1. Why does AdcVSR use a video diffusion model as the teacher but an image diffusion model as the backbone? How does a 3D spatio-temporal DiT as the student network compare with the 2D+1D architecture in terms of performance and efficiency? I recommend that the authors include related experiments.

2. Why did the authors choose to use a 2D VAE? I think that employing a 3D VAE could lead to faster inference and better temporal consistency.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes an improved adversarial diffusion compression method for real-world video super-resolution tasks. The authors note that while existing diffusion-based VSR methods can generate detail-rich videos, their inference speed is slow; conversely, one-step approaches are faster but still result in bulky models. To address this, the authors introduce AdcVSR, which distills a DiT teacher model with 3D spatiotemporal attention into a lightweight "2D + 1D" student network (based on a trimmed Stable Diffusion 2.1 backbone combined with 1D temporal convolutions). Additionally, a dual-head dual-discriminator adversarial distillation strategy is introduced to decouple the discrimination of detail richness and temporal consistency in the pixel and feature domains, respectively. This approach significantly improves efficiency while maintaining video quality. Experiments demonstrate that AdcVSR reduces parameters by 95% and accelerates inference by 8 times, while still achieving visual quality comparable to the teacher model.

### Strengths
Originality: The novel "2D + 1D" architecture design, combined with the dual-head discriminator adversarial distillation strategy, effectively decouples the optimization objectives for detail and consistency, demonstrating strong innovation.

Quality: Comprehensive experimental designs, including extensive validation on multiple synthetic and real-world datasets, support the effectiveness of the method through both quantitative and qualitative results. Ablation studies also thoroughly verify the contribution of each module.

Clarity: The paper is well-structured, with detailed method descriptions, and the inclusion of diagrams and pseudocode aids understanding. The writing is generally fluent, and technical details are clearly expressed.
Impact: The proposed method achieves a notable balance between efficiency and quality, offering significant practical value for deployment and providing a feasible pathway for the compression and application of diffusion models in video tasks.

### Weaknesses
Insufficient Comparative Experiments: Although comparisons are made with several SOTA methods, there is a lack of comparison with recent non-diffusion-based efficient VSR approaches, such as those based on CNNs or lightweight Transformers.

Limited Generalization Validation: All experiments are conducted at a fixed resolution (512×512) and frame length (25 frames), without demonstrating performance on longer videos or higher resolutions.
Weak Theoretical Support for Dual-Head Discriminator Design: While experiments prove its effectiveness, there is insufficient theoretical or visual analysis explaining why the "shared backbone + separate heads" design outperforms independent discriminators.

Incomplete Computational Efficiency Comparison: Only parameter counts and inference time are provided, without more detailed efficiency metrics such as FLOPs or memory usage.

### Questions
Is the weight allocation (75%/25%) between the "detail head" and "consistency head" in the dual-head discriminator universally applicable? Would this ratio remain effective across different datasets or tasks?

Were other teacher models (e.g., SeedVR2, DLoRAL) explored? Was DOVE selected solely because its structure is more suitable for this method?

It is recommended to include performance on longer video sequences (e.g., >100 frames) to validate the stability of temporal modeling, and providing failure cases or limitations analysis, such as performance under complex motion or extreme degradation, would be beneficial.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposed an improved adversarial diffusion compression method for real-world Video Super-Resolution. It comprises a 2D SD backbone and several 1D convs to replace the heavy 3D DiT architecture to improves its efficiency. By using a dual-head discriminator, it balances the optimization between detail enhancement and temporal consistency for adversarial distillation scheme. Experiments on synthetic and real-world benchmarks show its potential in compressing Real-VSR model on both model size and inference steps effectively.

### Strengths
* This paper proposes an intuitive explanation to use a ``2D + 1D'' architecture for student model to improve its efficiency.
* Detailed designs on adversarial distillation scheme (e.g., data type, training loss) help improve its clarity.
* Thorough experiments on synthetic and real-world benchmarks demonstrate its effectiveness after distillation.

### Weaknesses
* I remain skeptical about the effectiveness of replacing the 3D DiT architecture with a ``2D+1D'' architecture. As demonstrated in the paper, it uses a radical distillation strategy to achieve a 20-fold reduction in parameter count and single-step inference. It is essential to demonstrate that this strategy does not cause significant performance degradation on real-world benchmarks. Ablation study is only conducted in UDM10 (a small synthetic benchmark). Authors should conduct more analysis on large-scale real-world benchmarks.
* Detailed investigation on model size is missing. This paper provides an appealing results by distilling a 0.6B student model from a 11B teacher model. However, the selection of model size requires further exploration. Authors need to find a boundary between visual quality and model size, and conduct an ablation study about it.
* Motivation of using dual-domain discriminator. As mentioned in Sec 3.3, it replaces the conventional feat-domain discriminator by two discriminators in both pixel and feat domain. However, the motivation of adding one discrimintor in pixel domain remains unclear. Some related questions remain unresolved, such as why two discriminators are not used in the feature domain, and whether using more discriminators would improve performance. 
* Authors should also conduct experiments about temporal quality. By using 1D convs and a relative small model size, the temporal consistency of AdcVSR remians unclear now. Authors should conduct experiments by showing multiple frames of a single video in real-world benchmark.

### Questions
* Using stronger generative backbone. Using SD2.1 as 2D backbone limits the capacity on complex degradations. Authors should consider utilizing a large-scale backbone.
* I strongly recommend authors to provide a video demo or video files corresponding to the results presented in the paper. Given the usage of a ``2D+1D'' architecture, verifying the temporal quality of the final results is essential. Displaying only a single frame from the video within the article is not very convincing. I will consider raising my score after evaluating the video results.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5