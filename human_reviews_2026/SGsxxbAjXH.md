# MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Multi-view generation with camera pose control and prompt-based customization are both essential elements for achieving controllable generative models.  
However, existing multi-view generation models do not support customization with geometric consistency, whereas customization models lack explicit viewpoint control, making them challenging to unify.
Motivated by these gaps, we introduce a novel task, multi-view customization, which aims to jointly achieve multi-view camera pose control and customization.
Due to the scarcity of training data in customization, existing multi-view generation models, which inherently rely on large-scale datasets, struggle to generalize to diverse prompts.
To address this, we propose MVCustom, a novel diffusion-based framework explicitly designed to achieve both multi-view consistency and customization fidelity. 
In the training stage, MVCustom learns the subject's identity and geometry using a feature-field representation, incorporating the text-to-video diffusion backbone enhanced with dense spatio-temporal attention, which leverages temporal coherence for multi-view consistency. In the inference stage, we introduce two novel techniques: depth-aware feature rendering explicitly enforces geometric consistency, and consistent-aware latent completion ensures accurate perspective alignment of the customized subject and surrounding backgrounds. 
Extensive experiments demonstrate that MVCustom achieves the most balanced and consistent competitive performance across multi-view consistency, customization fidelity, demonstrating effective solution of multi-objective generation task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces multi-view customization, a novel task for generating multi-view consistent, customized subjects in new, text-described scenes. The proposed method, MVCustom, uses a video diffusion backbone and two novel inference-stage techniques: DFR for geometric consistency and CLC to realistically fill disoccluded regions.

### Strengths
1. Defines a novel, challenging, and high-impact task at the intersection of customization and 3D-aware generation.
2. Presents a highly innovative inference strategy (DFR+CLC) that cleverly enforces geometric consistency on new, text-driven scenes using 2D priors (depth) and feature rendering.
3. Provides compelling experimental validation (especially in the supplementary videos) demonstrating clear superiority over strong baselines in holistic (subject + background) consistency.

### Weaknesses
1. Inference Cost / Latency: The proposed inference pipeline is considerably complex, involving multiple forward passes (for CLC) , depth estimation, mesh construction, and differentiable rendering (for DFR). This likely results in significant inference latency compared to standard diffusion models. The paper does not discuss or quantify this computational overhead, which is an important factor for assessing the method's practicality.
2. Rliance on External Depth Estimator: The efficacy of DFR is dependent on the accuracy of an off-the-shelf monocular depth estimator. It is well-known that such estimators can be unreliable on transparent, reflective, or texture-less surfaces. While the paper mentions aligning the depth scale (Appendix B.2), it does not deeply explore the impact of depth estimation failures on the final generation quality.
3. Limitation on Object Pose: As noted in Appendix C, the method cannot alter the subject's intrinsic pose (e.g., from a "sitting" to a "standing" teddy bear) because FeatureNeRF learns a canonical pose from the reference images. This is a reasonable limitation, but given its importance, this discussion should be moved from the appendix to the main paper's limitations section.

### Questions
1. What is the quantitative inference latency of the full MVCustom pipeline compared to baselines?
2. How robust is DFR to significant failures from the monocular depth estimator?
3. Can the text prompt influence the subject's intrinsic pose at all, or is it completely fixed by the reference images' canonical pose?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents MVCUSTOM, a  diffusion-based framework explicitly designed
to achieve both multi-view consistency and customization fidelity. Specifically, they propose a novel task, multi-view customization, and  address multi-view consistency and limited data problem in customization.

### Strengths
• The paper is well-written with a logical structure that makes the technical contributions easy to follow.
• The proposed framework integrating dense spatio-temporal attention modules into video diffusion-based backbone is well-justified and addresses limitations in multi-view consistency via temporal coherence.

### Weaknesses
1The author only conduct experiments on a U-Net-based video diffusion model, I wonder if the proposed method can further improve performance on DiT-based models.

Suggestions:
1.	Conduct experiments on DiT-based backbone like Wan2.1?
2.	Include a comprehensive supplementary video showing: Side-by-side comparisons with all baselines (at least 3-5 examples per method)
3.	Add more dense camera pose trajectory to show more consisent.
4.	Create a project page with interactive demos which allow users to adjust the camera parameters.

### Questions
see weakness

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
2

### Summary
This paper introduces a diffusion-based framework named MVCustom, which is designed for multi-view customization with explicit camera pose control. For the training stage, it employs a video-diffusion backbone with spatio-temporal attention. To address the challenge of limited training data during inference, the paper proposes two novel techniques: depth-aware feature rendering to enforce geometric consistency and consistent-aware latent completion to realistically synthesize content in newly revealed regions.

### Strengths
1. The paper defines a novel task of multi-view customization that combines camera control with subject personalization.
2. The proposed inference-time techniques for ensuring multi-view consistency effectively address the challenge of limited training data.
3. The paper is well-structured and easy to follow.

### Weaknesses
1. Could you provide more details on the inference time of MVCustom compared to the baselines? The inference process, particularly the depth-aware feature rendering component, appears to be computationally intensive.
2. On the project page, some results still show obvious hallucinations or inconsistencies when the viewpoint variation is large. Is there further analysis on this specific limitation? A discussion or visualization of typical failure cases would be helpful to understand the method's boundary conditions better.

### Questions
Please refer to the weaknesses.

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
2

### Summary
This manuscript presents a joint multi-view generation with explicit camera pose control and subject customization task, and a corresponding framework to solve this task. More specifically, the authors proposed an MVCustom framework, which based on a diffusion model to generate multi-view images controlled by explicit camera poses with a consistent subject from given reference images. MVCustom is based on AnimateDiff and incorporates an additional feature NeRF and dense spatio-temporal layers to achieve multi-view consistency and subject consistency. In addition, the authors also introduce a depth-aware feature rendering to better achieve geometric consistency, and a consistent-aware latent completion technique to ensure the appearance consistency of both the subject and the surroundings.

### Strengths
The newly proposed task of jointly performing subject customization and multi-view image generation is a direction that has been rarely explored.

### Weaknesses
1. The reviewer is concerned about the practical application value and novelty of the proposed multi-view customization task. This task can be framed as a sparse-view version of controllable (camera pose-conditioned) video customization. The paper’s own choice of a video diffusion backbone (AnimateDiff) and its method of "enhancing temporal coherence into holistic-frames consistency" further reinforces this similarity. The practical distinction and added value over existing controllable video generation or customization frameworks are not sufficiently justified.

2. The framework is built on a U-Net-based video diffusion model (AnimateDiff), which is arguably dated compared to recent transformer-based architectures. While the authors justify this in the appendix (Sec. A.1) as a compatibility choice with FeatureNeRF and frame-level precise camera control, the quality of the generated images is still highly constrained by the base model. Meanwhile, the core mechanism for camera control (pose-conditioned transformer block) is admittedly adopted from a previous work, further limiting its novelty. The other two main contributions, depth-aware feature rendering and consistent-aware latent completion, are presented as inference-stage techniques but lack rigorous quantitative validation.

3.	As for the quantitative results shown in Tab. 2, the quantitative results in Table 2 do not show clear and comprehensive superiority, but a mixed trade-off rather than a decisive win, making the paper's claim of being the "only one that achieves consistently strong performance" an overstatement. Meanwhile, while the figures (e.g., Fig. 4) show improvements over baselines, inconsistencies are still apparent. This is particularly noticeable in the supplementary videos, where flickering and geometric instability suggest that holistic multi-view consistency is not fully solved, especially for the synthesized backgrounds that are not geometrically constrained by the FeatureNeRF model (explicit 3d modeling). To more robustly evaluate the claimed multi-view and geometric consistency, the authors should attempt to reconstruct an explicit 3D model (e.g., a mesh or Neural Radiance Field) from the generated multi-view outputs. This would provide a much stronger and more objective measure of 3D consistency than 2D metrics.

4. The ablation study in Section 4.3 is insufficient as it is purely qualitative. The paper introduces several key techniques (Depth-aware Feature Rendering, Consistent-aware Latent Completion) but fails to provide any quantitative analysis to demonstrate their individual impact on the final metrics (e.g., in Table 2). Figure 5 offers only a single visual comparison, which is not enough to fully validate the effectiveness of each proposed module.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
