# MIMIC: Mask-Injected Manipulation Video Generation with Interaction Control

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Embodied intelligence faces a fundamental bottleneck from limited large-scale interaction data. Video generation offers a scalable alternative, but manipulation videos remain particularly challenging, as they require capturing subtle, contact-rich dynamics. Despite recent advances, video diffusion models still struggle to balance semantic understanding with fine-grained visual details, restricting their effectiveness in manipulation scenarios. Our key insight is that reference videos provide rich semantic and motion cues that can effectively drive manipulation video generation. Building on this, we propose MIMIC, a two-stage image-to-video diffusion framework. (1) We first introduce an Interaction-Motion-Aware (IMA) module to fuse visual features from the reference video, producing coherent semantic masks that correspond to the target image. (2) then utilize these masks as semantic control signals to guide the video generation process. Moreover, considering the ambiguity of the motion attribution,  we introduce a Pair Prompt Control mechanism to disentangle object and camera motion by adding the reference video as an additional input. Extensive experiments demonstrate that MIMIC significantly outperforms existing methods, effectively preserves manipulation intent and motion details, even when handling diverse and deformable objects. Our findings underscore the effectiveness of reference-driven semantics for controllable and realistic manipulation video generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a two-stage, image-to-video diffusion framework for generating manipulation videos with fine control over object interactions. First, an Interaction-Motion-Aware (IMA) module extracts semantic masks from a reference video to guide future frames. Then, these masks are injected into a diffusion model to steer the video generation process. A Pair Prompt Control mechanism helps disentangle object motion from camera motion by conditioning on both the target image and the reference video. 
The method is shown to preserve manipulation intent and detail, even when dealing with deformable or diverse objects, outperforming existing baselines in both visual fidelity and control.

### Strengths
1. The paper addresses an interesting and important problem with a novel solution.
2. The paper proposes several new ideas that could be leveraged for future research in different direction, including the IMA attention, Pair Prompt Control module and the idea of using ICL for learning motion from motion masks of similar tasks.
3. Experiments are sound and the visual results show the superiority of MIMIC over previous work.

### Weaknesses
1. The method depends on relevant reference videos. This might limit its applicability in scenarios where such references are not available or are very different from the target.
2. It is unclear how the reference videos are chosen.
3. The method is limited to generating 16-frame videos.

### Questions
1. How did the authors choose the reference videos? 
2. Did the authors try to generate longer videos by concatenating generations? (e.g. use the last frame of a previously generated video as the initial frame of a new generation)
3. Did the authors try multi-object or multi-hand interaction scenarios? e.g. manipulate different objects with 2 hands or a hand and a gripper in the same video?
4. Are there failure cases you exhibited, such as specific objects/motions/lighting conditions etc.?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MIMIC, a two-stage diffusion framework to generate manipulation videos by leveraging a reference video for semantic and motion cues. The method first generates a sequence of interaction masks (Stage I) and then renders the final video (Stage II), aiming to provide a scalable data source for embodied AI. The authors demonstrate strong quantitative and qualitative performance over existing methods.

### Strengths
- The paper addresses the important and challenging problem of data scarcity for training embodied intelligence systems.
- The proposed two-stage framework is logical. Decoupling the generation process into (1) motion/interaction understanding (mask generation) and (2) high-fidelity rendering (video generation) is a sensible approach to decompose a complex problem.
- The qualitative comparisons provided in the Supp. video are compelling.
- The ablation studies are informative and effectively validate the contributions of the proposed components, i.e., the IMA Attention and the Pair Prompt Control mechanisms .

### Weaknesses
- Relation to Video Editing: The related work section focuses on "Video Motion Customization" and "Interaction Video Generation" but does not explicitly discuss the field of generative video editing. The task in this paper shares conceptual overlap with reference-based video editing. It would strengthen the paper to include this discussion, clarify the key distinctions, and situate the work relative to recent highly relevant papers in the HOI image/video editing domain, such as [1][2].
- The paper does not clearly explain how the (reference, target) video pairs are constructed for training. Is a video simply used as its own reference? Are pairs formed in some other way (based on action label)? This is a critical methodological detail that is currently missing and needs to be explicitly stated.
- The paper's description of Stage I is confusing. It's trained to "identify" the manipulated object , but Grounding-SAM2 is already used for annotation. Why train a generative model for a recognition task? Why Grounding-SAM2 isn't used at test time to provide the initial object mask, allowing the diffusion model to focus on its core task: generating the new motion pattern (i.e., the mask trajectory)?
- It is unclear what the scope of the tested actions and objects is. For instance, do the experiments demonstrate generalization to unseen object categories, or new instances of seen categories? The experimental setup and data splits should be described more explicitly to clarify what level of generalization is being evaluated.
- About the use of language: Appendix A.2 says that the method relies on structured language templates (e.g., "move [something] down") and an NLP pipeline to "extract structured action-object pairs". I feel this should be explicitly stated in the main paper, as it clarifies that the model is not conditioned on free-form, natural language prompts, right?
- About the masks: the text repeatedly describes them as "binary masks". However, numerous figures visualize multi-class masks, with hands/grippers and manipulated objects colored differently. The authors need to clarify whether the masks are binary (e.g., interaction vs. background) or multi-class (e.g., hand vs. object vs. background). Also Ln204 “soft binary mask”, what does “soft” mean?

[1] Affordance Diffusion: Synthesizing Hand-Object Interactions

[2] HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness

### Questions
My overall feeling is that the paper presents a strong method, but the presentation is unclear in several critical areas, as detailed in the comments. I hope the authors can clarify these points in their response. Happy to increase my score once these are clarified.

### Soundness
3

### Presentation
2

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
This paper proposes MIMIC, a novel two-stage image-to-video diffusion framework designed to generate realistic and controllable manipulation videos. The method first utilizes an Interaction-Motion-Aware (IMA) module to generate a sequence of semantic interaction masks guided by a reference video. Subsequently, it introduces a Pair Prompt Control mechanism that conditions the final video generation on both the predicted masks and the original reference video, effectively disentangling object motion from camera motion and enhancing controllability.

### Strengths
1. The authors' decomposition of the task into two stages, "mask generation" and "video rendering," is a reasonable design. Furthermore, the authors demonstrate the necessity of this decomposition through ablation studies (such as "Two-Stage vs. One-Stage" and Figure 6) and clearly show the contributions of each key module (IMA and PPC).

2. The authors identified the problem with using only masks for control. The Pair Prompt Control (PPC) part is a clever solution. Figure 6 (background drift vs. static background) shows how this part is effective at separating object motion from camera motion.

### Weaknesses
1. The IMA module just copies the motion from the reference video and does not understand the physics of the target scene. The predicted masks in Figure 6 support this, appearing to be a direct motion transfer that ignores the specific geometry and constraints of the target object.

2. The examples in the paper are all very similar, showing only robot-to-robot or hand-to-hand tasks. This makes it unclear if the method can work for more general cases where the reference and target are different. For instance, it is not shown if a human hand video can be used to control a robot arm.

3. The method's first stage requires reference manipulation masks as input, but the paper does not explain how these are obtained. Class-agnostic models like SAM are difficult to automatically and accurately prompt to segment only the manipulated object in complex videos. Meanwhile, class-specific semantic segmentation models (e.g., Mask R-CNN) would restrict the method to only predefined object categories, causing it to fail on any new objects.

### Questions
1. The experiments only show in-domain cases. Have the authors tested cross-domain generalization? For example, can a video of a human hand be used to guide a robot arm?

2. Figure 6 shows that lacking the IMA module leads to failure. With the IMA module, how well does it work if the reference motion is complex or the object is occluded?

3. The paper mentions that the input of Stage I includes text, but it appears to be unused in the architecture (Figure 2) and the IMA module description. Could the authors clarify how exactly the text is used in Stage I?

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
The paper proposes MIMIC, a two-stage image-to-video diffusion framework MIMIC designed for manipulation video generation. The propsoed training framework consists of two-stages: 1) Condition on a reference video, reference video interation mask, and target image, generte the target video interation mask. 2) condition on reference video, reference video interation mask, target image and masked target image, generate the target video. The paper use DynamiCrafter as the base model. The model was trained and evaluated on human-hand and robotic-gripper datasets built from SSv2, BridgeV2, and Fractal datasets. It was compared against major baselines such as DynamiCrafter, CogVideoX, MotionClone, FlexiAct, MotionDirector, etc.

### Strengths
- The paper is well-written and easy to follow, presenting its ideas and methodology clearly.

- The paper combines quantitative metrics, MLLM-based semantic evaluation, and human preference studies, providing a comprehensive assessment of model performance.

- The paper conducts extensive ablation studies, including analyses of the Two-Stage vs. One-Stage framework and the effectiveness of each key component, further validating the proposed approach.

### Weaknesses
- **Design choice of motion representation**: I was curious about the motivation for adopting masks as the abstraction for motion information rather than alternative representations such as tracking points or optical flow [1]. While masks provide clear spatial localization, they might be too rigid in preserving the object shape from the reference video, potentially limiting flexibility in motion adaptation.

Reference: [1] VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models (ICML).

- **Impact of base model**: Would employing a more powerful base model (e.g., a larger or more recent video diffusion backbone) further improve the overall generation quality and temporal coherence? Since the paper mentions limitations due to the capacity of the current base model, it would be interesting to discuss how the proposed approach might scale with stronger foundational architectures.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
