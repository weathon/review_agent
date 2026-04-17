# Time-to-Move: Training-Free Motion-Controlled Video Generation via Dual-Clock Denoising

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Diffusion-based video generation can create realistic videos, yet existing image- and text-based conditioning fails to offer precise motion control. Prior methods for motion-conditioned synthesis typically require model-specific fine-tuning, which is computationally expensive and restrictive. We introduce Time-to-Move (TTM), a training-free, plug-and-play framework for motion- and appearance-controlled video generation with image-to-video (I2V) diffusion models. Our key insight is to use crude reference animations obtained through user-friendly manipulations such as cut-and-drag or depth-based reprojection. Motivated by SDEdit’s use of coarse layout cues for image editing, we treat the crude animations as coarse motion cues and adapt the mechanism to the video domain. We preserve appearance with image conditioning and introduce dual-clock denoising, a region-dependent strategy that enforces strong alignment in motion-specified regions while allowing flexibility elsewhere, balancing fidelity to user intent with natural dynamics. This lightweight modification of the sampling process incurs no additional training or runtime cost and is compatible with any backbone. Extensive experiments on object and camera motion benchmarks show that TTM matches or exceeds existing training-based baselines in realism and motion control. Beyond this, TTM introduces a unique capability: precise appearance control through pixel-level conditioning, exceeding the limits of text-only prompting. Visit our project page (https://time-to-move.github.io) for video examples and code.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TTM, a training-free framework for motion control in image-to-video (I2V) models. The core of the method is a "dual-clock denoising" strategy: it uses less noise ($t_{strong}$) in user-specified motion regions to strongly constrain motion, while using more noise ($t_{weak}$) in other regions (like backgrounds) to allow for natural dynamics. Experiments show this training-free method achieves results comparable to, or even better than, state-of-the-art (SOTA) methods that require training.

### Strengths
1.  **Practicality and Novelty:** The paper addresses a key pain point in video generation (training-free motion control). "Dual-clock denoising" is a concise and effective innovation that elegantly solves the SDEdit trade-off between motion fidelity and background artifacts. The framework can be used as a plug-and-play module for different I2V models.
2.  **Experimental Validation:** The experiments are thorough, covering object motion, camera motion, and joint appearance control. Comparisons against SOTA methods like DragAnything and GWTF are convincing, demonstrating the method's effectiveness both quantitatively (Tables 1, 2) and qualitatively (Figs 4, 6). The ablation study in Appendix A also strongly supports the necessity of the dual-clock design.

### Weaknesses
1.  **Relation to RePaint:** The paper should more clearly articulate its distinction from RePaint. The blending operation during the $t_{strong} \le t < t_{weak}$ phase appears very similar to RePaint's. The core innovation seems to be the **temporal control** of this blending (i.e., stopping at $t_{strong}$ for joint refinement) rather than the blending mechanism itself, which needs clarification.
2.  **Hyperparameter Sensitivity:** The key hyperparameters $(t_{weak}, t_{strong})$ require manual tuning for different models (e.g., (36, 25) for SVD vs. (46, 41) for CogVideoX), which somewhat reduces its "plug-and-play" convenience. An analysis of sensitivity to these parameters is recommended.
3.  **Dependence on Input Quality:** The method's effectiveness depends on the quality of the "crude animation" $V^w$ and the mask $M$. Particularly for the camera motion task, it relies on external depth estimation models and complex 3D reprojection (as detailed in Appendix B.2), which adds a "hidden cost" to its application.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper Introduces a training-free method for controllable video generation. It firstly generates a crude edited video according to user's input. Then the edited video serves as the guiding signal for a pretrained video diffusion model and inferences in a SDEdit-style. To achieve better performance, authers modify SDEdit and propose a Dual-clock inference pipeline which apply different denoising timestep for mask and unmasked region.

### Strengths
1. The method adopts a user-friendly setup that allows users to drag the object they want to move. Studying this problem plays an important role in advancing controllable generation toward practical applications. 
2. The proposed training-free method is straight-forward and efficient. The method also shows board application for various pretrained video diffusion models, extending the function of video foundation model.
3. The author’s writing and figures in the paper are clear and easy to understand.

### Weaknesses
1. The warping method. The quality of this method highly depends on the quality of the directly warped video. However, the author only briefly mentioned that the warped video is generated through forward warping, without providing detailed steps or references for the process.
2. The dual-clock denoising strategies. The dual-clock is a slight adaptation of SDEdit and lacks insight or improvement of SDEdit. And I think it's highly depend on the choice of $t_{weak}$ and $t_{strong}$.
3. Experiments. The authors do not provide ablation study. I think the author should ablate different warping method and the dual-clock denoising strategy to see which part contributes more to the final result.

### Questions
1. The warping method. Can you describe more about the forward warping method? Also, in the provided Object Control demos (second column, fifth row), the warped video not only warps the white ball but also warps the two other balls it collides with. I’m curious how this was achieved. Additionally, I wonder how the method performs on actions that are difficult to warp, such as complex interactions or occlusions between objects. If the warping quality is poor in such cases, would it significantly affect the subsequent generation?
2. The dual-clock denoising strategies. Could the authors provide some empirical guidelines or example values for selecting  $t_{weak}$ and $t_{strong}$? Is there a generally applicable choice for these parameters, or do they need to be tuned separately for each video?
3. Experiments. This paper lacks ablation study and the authors should provide more ablation studys on different warping method and the dual-clock denoising strategy.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Time-to-Move (TTM), a training-free framework for controllable video generation using image-to-video diffusion models. TTM introduces the concept of using crude reference animations (e.g., cut-and-drag edits or depth-based warping) as direct motion guidance signals. TTM uses dual-clock denoising, which assigns different noise schedules to masked (motion-controlled) and unmasked regions, allowing strict enforcement of motion guidance where specified and natural evolution elsewhere. Experiments across multiple diffusion backbones show that TTM offers strong performance compared to state-of-the-art training-based and training-free baselines.

### Strengths
- The training-free and plug-and-play natures of TTM are valuable. No fine-tuning or architectural changes are required, making the method broadly applicable across various existing models and lightweight.
- Region-dependent noising is effective, balancing fidelity to user-specified motion with natural background dynamics.
- Generalization results across backbones look good.

### Weaknesses
- Objects introduced later rather than in the first frame cannot be anchored, limiting use cases involving scene entry, new dynamic object, or sudden occlusion.
- The dual-clock denoising scheme requires tuning specific settings per backbone, which could hinder plug-and-play deployment in practice.
- Strict full-object masks must be provided. Control signals relying on sparse cues are more user-friendly in some workflows.

### Questions
- What is the trade-off in inference time and computational overhead? What is the average inference time increase (or decrease) when using TTM on a base model compared to its vanilla inference process?
- How robust is the method to segmentation errors? For example, when there is motion blur or non-rigid motion, it is difficult to provide an accurate mask.

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
The paper proposes Time-to-Move (TTM), a training-free, plug-and-play motion control method for image-to-video diffusion models. Core ideas: (i) generate a crude reference animation (cut-and-drag or depth reprojection) and adapt SDEdit to inject its motion, and (ii) a region-dependent “dual-clock” denoising that uses a lower noise level inside a motion mask and a higher one elsewhere to balance adherence vs. realism. The method claims strong object/camera motion control across SVD, CogVideoX, and Wan backbones with no extra training, with quantitative gains on MC-Bench and DL3DV.

### Strengths
1. Training-free & backbone-agnostic; integrates with multiple I2V backbones.

2. Clear articulation of the dual-clock idea with ablations.

3. Competitive metrics and clean qualitative examples for object and camera motion control.

### Weaknesses
1. Missing related-work discussion: structural-noise initalization like in video-MSG [1]

2. Sensitivity to the applied template (reference animation + mask + schedule). TTM critically depends on: (a) how the warped video is produced (cut-drag vs. depth reprojection, inpainting strategy), (b) the mask quality, and (c) the chosen (t_weak, t_strong) schedule. The paper acknowledges per-model tuning and that identity is anchored only by the first frame, but doesn’t deeply analyze robustness to template variance 

3. Even though figures highlight cleaner results vs. baselines, the pipeline can inherit issues from the template (e.g., depth reprojection tearing/holes) and from hard masking. The paper partly recognizes this but artifact analysis is light.

[1] Training-free Guidance in Text-to-Video Generation via Multimodal Planning and Structured Noise Initialization, 2025

### Questions
please see weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
