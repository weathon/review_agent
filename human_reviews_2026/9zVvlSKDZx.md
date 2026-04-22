# InstanceAnimator: Multi-Instance Sketch Video Colorization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We propose InstanceAnimator, a novel Diffusion Transformer (DiT)-based framework for multi-instance sketch video colorization.
Existing animation colorization methods rely heavily on a single initial reference frame, resulting in fragmented workflows and limited customizability. To eliminate these constraints, we introduce a Canvas Guidance Condition that allows users to freely place reference elements on a blank canvas, enabling flexible user control. To address the misalignment and quality degradation issues of DiT-based approaches, we design an Instance Matching Mechanism that integrates the instances with the sketch and noise channels, ensuring visual consistency across different sequences while maintaining controllability. Additionally, to mitigate the degradation of fine-grained details, we propose an Adaptive Decoupled Control Module that injects semantic features from characters, backgrounds, and text conditions into the diffusion model, significantly enhancing detail fidelity.  Extensive experimental results demonstrate that InstanceAnimator effectively enables better user control in multi-instance colorization, producing high-fidelity results with strong temporal consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces InstanceAnimator, a diffusion transformer (DiT)-based framework for multi-instance sketch video colorization. Unlike existing methods that rely heavily on a single reference frame, InstanceAnimator incorporates a Canvas Guidance Condition to allow flexible placement of multiple reference elements, an Instance Matching Mechanism to enforce consistency between sketches and references, and an Adaptive Decoupled Control Module to preserve fine-grained semantic details for both characters and backgrounds. The paper presents thorough experimental results—including quantitative metrics, qualitative figures, and ablations—demonstrating improved controllability, identity preservation, temporal consistency, and usability compared to several strong baselines.

### Strengths
1. Unlike previous image-to-animation architectures, this paper provides a reference-to-animation solution, which simplifies the animation production pipeline.
2. Allows users to freely position and edit reference elements, enabling more realistic animation workflows
3. The framework is extensively evaluated across four strong baselines with appropriate metrics (FVD, SSIM, LPIPS, CLIP, Temporal) as summarized in Table 1, and ablation studies (Table 2) convincingly establish the impact of each module.

### Weaknesses
1. The key problem of reference-to-video is actually the data pre-process. For example, in MovieGen, the authors find that training solely on the above paired data makes the model easily learn a copy-paste shortcut solution, i.e., the generated video always follows the expression or the head pose from the reference face. Thus, they proposed to collect reference data from outside of the video clip. However, in this paper, the authors use SAM to crop the reference instances from the first frames. I suspect that the shortcut problem exists in the proposed method and the authors ignore it.
2. Another problem of reference-to-image(video) is the instance matching. The position of the instances would interchange with each other. Although the authors proposed an Instance Matching Mechanism, it is a learning-based algorithm. It cannot convince me that the model can make sure that the characters would not exchange.
3. The third problem is the background extraction. Directly matting from the first frame will create silhouette, which need image inpainting to fix. However, the imperfect inpainting will still leak sketch information during training, which would also cause a shortcut problem. This problem is not mentioned in this paper.

### Questions
In the anime production process, multi-view character design sheets are often used as character references. Why wasn't this form of reference sheet used?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an instance-aware DiT-based video generation framework. It eliminates the need for complete frame conditions, allowing users to input multiple instances and generate visually consistent videos. The authors propose a Canvas Guidance Condition and an Instance Matching Mechanism to integrate multiple instance features and an Adaptive Decoupled Control to inject semantic features.

### Strengths
- The primary contribution is shifting the animation conditioning from complete frames to the instance level. This approach is more user-friendly and has the potential to simplify the practical animation workflow.
- The paper is easy to follow.
- The experimental results effectively demonstrate that InstanceAnimator can produce high-fidelity video results.

### Weaknesses
- The description of "instance matching" in Section 3.3 is ambiguous. Equation 2 introduces instance-specific latent features ($Z_{\text{inst}}^i$), but the paper fails to explain how these features are utilized or injected into the network. The methodology is difficult to understand without Figure 4, indicating a need for significant improvement in the clarity of the writing.
- Insufficient Instance Correspondence Mechanism: A more critical issue is the lack of a clear explanation for how instances are matched. The paper implies that simply concatenating multiple instance features along the temporal axis is sufficient. This seems inadequate for establishing a robust correspondence between a specific reference instance and its corresponding sketch.
    1. How does the model resolve instance correspondence in multi-person scenes? The case in Figure 6 (top-right) is trivial, as the poses in the reference and the first-frame sketch are identical.
    2. If the reference characters are in different poses, how would the model correctly map the appearance (e.g., costume) from the correct reference to the correct sketch? This is a non-trivial problem, especially in animation, where sketches of different characters can be highly similar.
- This paper is not the first to perform instance-conditioned sketch colorization. For example, MangaNinjia[1] can also achieve sketch colorization from instance references. A plausible alternative workflow would be to use MangaNinjia to generate the first frame and then employ a standard Image-to-Video model to generate the animation. The paper would be significantly strengthened by including a discussion and quantitative comparison against such a two-stage baseline.

[1] Liu, Zhiheng, Ka Leong Cheng, Xi Chen, Jie Xiao, Hao Ouyang, Kai Zhu, Yu Liu, Yujun Shen, Qifeng Chen, and Ping Luo. "Manganinja: Line art colorization with precise reference following." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5666-5677. 2025.

### Questions
- In Figure 4, the "extra instance tokens" are concatenated with noisy latents and sketch tokens along the temporal axis. However, the "C3 reference" is not. What is the design rationale for this asymmetrical handling of conditioning inputs?

- Is CLIP the optimal choice for encoding background and instance features in the "Adaptive Decoupled Control Module"? When a user inputs a specific background image, the expectation is often for high-fidelity detail preservation (e.g., texture, specific objects), not just semantic alignment. This potential limitation seems apparent in the qualitative results. In Figure 6, the male character's hat color (bottom-left) and the child's shirt color (bottom-right) both deviate from their respective reference images. These inconsistencies in appearance preservation do not meet user expectations for this task.

### Soundness
2

### Presentation
2

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
This paper presents InstanceAnimator, a DiT-based framework designed for multi-instance sketch video colorization.
The authors claim that the proposed method tackles limitations in existing methods, including dependency on a single reference frame, misalignment between references and sketches, and loss of fine details.

Key technique contributions include the Canvas Guidance Condition for user-controlled placement of reference elements, the Instance Matching Mechanism to ensure correspondence via latent feature fusion, and the Adaptive Decoupled Control Module for injecting decoupled semantic features from instances, backgrounds, and text.

The approach is evaluated quantitatively against baselines using metrics like FVD, SSIM, LPIPS, temporal consistency, and CLIP score, with ablations and a user study demonstrating improved controllability and fidelity.

### Strengths
The paper demonstrates a multi-instance colorization method in sketch videos, improving single-frame reference colorization to enable flexible, instance-based control that aligns with anime production workflows. The technical design is reasonable, including the canvas guidance and decoupled control module that effectively address identified gaps in DiT-based models. The motivation, method descriptions, and figures are easy to follow. This work has values to reduce the extensive human labor in anime/cartoon production workflows.

### Weaknesses
1. While the claims are generally supported by numerical results, the sketch fidelity remains unsatisfactory in some samples. For example, in the teaser figure, we can clearly infer that the model output does not follow the sketches (2nd row: Chihiro's face does not follow the sketch in the 3rd frame; 4th row: the girl's mouth does not follow the sketch in 3-5th frames). Since sketch fidelity is essential in video colorization and anime production, this defect is unsatisfactory in a video sketch colorization method. In the supplementary video, this phenomenon is even clearer. Does this defect always occurs? Therefore, I suggest the authors to include comprehensive analysis/comparison on the sketch fidelity of the proposed method.

2. The acknowledged limitations in handling complex multi-character motions with backgrounds suggest room for robustness improvements, perhaps through expanded training data. 

3. Although the method claims to support arbitrary numbers of instances, it lacks explicit experiments demonstrating performance with more than three instances and provides no analysis of how computational complexity scales with the number of instances.

### Questions
1. Following the weakness 1, I would like to ask the authors to conduct an analysis of the sketch fidelity of the proposed method and explain the potential reason that causes this poor fidelity.

2. The paper mentions potential flaws in multi-character background control due to data scarcity. What specific improvements are planned for future work?

3. What is the maximum number of instances tested in experiments, and were there any evaluations specifically on more than 3 instances?

4. What is the model's computational complexity with respect to the number of instances?

Although the method and the paper presentation is good, I tend to give the score of 4 given the unsatisfactory sketch fidelity. 
I am willing to raise my scores if the author rebuttal provides reasonable explanation and addresses my concerns.

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
4

### Summary
This paper presents InstanceAnimator, a diffusion-transformer-based framework for sketch video colorization with multiple reference instances. The key idea is to remove the dependency on a single reference frame by introducing (1) a Canvas Guidance Condition for flexible multi-instance placement, (2) an Instance Matching Mechanism to align sketches and references, and (3) an Adaptive Decoupled Control Module to inject detailed semantic information from text, background, and instances. Experiments on animation datasets show some quantitative and qualitative improvements over existing reference-based colorization methods.

### Strengths
- The problem setting—multi-instance sketch video colorization—is interesting and relevant for creative AI and animation generation.

- The overall pipeline is clearly presented, with detailed ablations showing the contribution of each module.

- The qualitative results demonstrate visually appealing outputs with good color consistency and controllability.

### Weaknesses
- The proposed Instance Matching mechanism seems to only establish random associations between sketches and instances, without any explicit spatial alignment. While the Canvas Guidance provides a weak positional prior, it does not guarantee that instances placed on the canvas will appear at the intended locations in the generated sequence. What if users need to swap two characters? This limits the controllability for professional animation use cases.

- The quantitative improvement over strong baselines (e.g., ToonComposer) is rather small. The contribution seems more engineering-oriented than scientific.

- The method is not compared with LayerAnimate (ICCV 2025), which addresses a highly related problem of layer-level animation control using similar diffusion-based techniques. Without this comparison, it’s unclear whether the proposed system actually advances the state of the art in multi-instance or layered animation scenarios.

- The paper claims to support diverse and flexible instance control, but no examples show the same sketch colorized by very different or incompatible characters. Most examples use references already well-aligned with the original sketch poses, which weakens the claim of true multi-instance generalization.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
