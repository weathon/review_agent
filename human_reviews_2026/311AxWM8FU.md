# Target-Aware Video Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We present a target-aware video diffusion model that generates videos from an input image, in which an actor interacts with a specified target while performing a desired action. The target is defined by a segmentation mask, and the action is described through a text prompt. Our key motivation is to incorporate target awareness into video generation, enabling actors to perform directed actions on designated objects. This enables video diffusion models to act as motion planners, producing plausible predictions of human-object interactions by leveraging the priors of large-scale video generative models. We build our target-aware model by extending a baseline model to incorporate the target mask as an additional input. To enforce target awareness, we introduce a special token that encodes the target's spatial information within the text prompt. We then fine-tune the model with our curated dataset using an additional cross-attention loss that aligns the cross-attention maps associated with this token with the input target mask. To further improve performance, we selectively apply this loss to the most semantically relevant attention regions and transformer blocks. Experimental results show that our target-aware model outperforms existing solutions in generating videos where actors interact accurately with the specified targets. We further demonstrate its efficacy in two downstream applications: zero-shot 3D HOI motion synthesis with physical plausibility and long-term video content creation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a method to make image-to-video (I2V) diffusion models "target-aware", enabling them to generate videos where an actor plausibly interacts with a specific object present in the initial frame. The core problem identified is that existing I2V models, even when prompted, often fail to interact with the intended object and may hallucinate new ones. The proposed solution extends a base I2V model to accept a segmentation mask of the target object as an additional condition. The key contribution is a novel cross-attention loss applied during fine-tuning. This loss explicitly forces the model to align the cross-attention map of a special token with the spatial region of the input target mask. This alignment is selectively applied only to the video-to-text (V2T) cross-attention regions and to specific transformer blocks identified as most semantically relevant.
To train this model, the authors curate a new dataset from BEHAVE and Ego-Exo4D, consisting of 1290 video clips showing pre-interaction and interaction phases. Experiments demonstrate that this method significantly outperforms baselines in interaction accuracy (measured by a new "Contact Score" metric and extensive user studies) without degrading overall video quality. The paper also showcases two downstream applications: zero-shot 3D Human-Object Interaction (HOI) motion synthesis and long-term video content creation.

### Strengths
- The paper addresses a clear, practical, and important limitation of current video generation models. The "target-unawareness" of I2V models is a well-known failure case, and solving it is a key step toward using these models as effective "world models" or motion planners, as the authors suggest.
- The core idea of using a special token and enforcing its cross-attention map to match a target mask is shown to be effective. It directly injects spatial grounding into the text-conditioning mechanism of the transformer. The simplicity of the method (implemented as a LoRA module and an extra input channel) makes it practical and easy to adopt.
- Reasonable metrics are proposed and used, and decent improvements can be obtained.

### Weaknesses
- The curated dataset is quite small (1290 clips). While effective for this task, it may limit the diversity of learnable interactions. How well does the trained model generalize?
- The "Contact Score" is a good first-pass metric for this new task, but its definition is somewhat lenient. It is defined as a success if the contact detector finds an overlap with the target mask in at least one frame. This metric might not differentiate between a brief, accidental touch and a semantically correct, sustained interaction (e.g., a full "pick up and hold" action). This low bar could inflate the reported scores, though the qualitative results and user studies do suggest the generated interactions are indeed plausible.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of target-unaware video generation, where existing models fail to ensure an actor interacts with a specific target object designated by the user. The authors propose a method that feeds a target object's mask as an additional condition and introduces a novel cross-attention loss. During training, this loss forces the attention map of a special text token ([TGT]) to align with the input target mask. This allows the model to connect a text command to a specific spatial location in the scene, thereby generating target-aware interaction videos.

### Strengths
1. The paper is well-written, and the illustrative view clearly presents the entire pipeline.
2. The paper defines a clear and practical problem (target-aware video generation), It has excellent application value in multiple fields.
3. The ablation studies and quantitative analysis are thorough. The results clearly demonstrate the effectiveness of each module in the method and provides a reasonable explanation for the selection of hyperparameters.

### Weaknesses
1. Cross-attention loss cannot be considered as one of the innovations. There have already been numerous works in image generation and video generation that design losses at the attention level for fine-tuning models or optimizing generation results.
2. The amount of data used for training is insufficient. With only 1K+ video clips, the highly adaptable DiT architecture is prone to overfitting. Additionally, the data primarily consists of single-person indoor scenes, so the effectiveness of this method in outdoor or more complex scenarios requires further validation.
3. The examples demonstrating control over both the source actor and the target object are somewhat limited. It would be helpful to include more instances of interactions between objects (not just robotic arms) to showcase the generalization capability on the source actor.

### Questions
In the examples shown in the paper, the target object occupies a much smaller area relative to the actor. I am curious about how the model would perform when the target object takes up a significant portion of the frame (e.g., a car very close to the camera as the target object).

### Soundness
4

### Presentation
4

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
This paper introduces a target-aware video diffusion model that enables a subject in an input image to interact with a user-specified target object using only a segmentation mask and a text prompt. The method extends a state-of-the-art I2V diffusion transformer (CogVideoX) by incorporating a target mask and introducing a special [TGT] token whose cross-attention maps are constrained to align with the mask via a cross-attention supervision loss. Selective application of this loss to specific cross-attention regions (V2T) and transformer blocks leads to stronger target conditioning. The paper also curates a dataset for this task and demonstrates applications such as zero-shot 3D HOI motion synthesis and long-term video creation. Experiments show clear improvements over baselines in target-interaction accuracy without degrading video quality.

### Strengths
+ Target-aware video generation for HOI and robotics planning is an important and timely direction. The paper identifies a real limitation in current I2V models (target ambiguity) and formulates a useful new capability.
+ Experiments systematically measure both target alignment and video quality, with diverse baselines and ablations. The Contact Score and user studies convincingly support the claims. Qualitative results are compelling, including multi-target, same-category objects, and generalization to animals and robotics scenes.

### Weaknesses
- The curated training dataset (1,290 clips) is relatively small and largely human-object centric. While results are strong, scaling considerations are not fully explored. It is unclear how well the method would generalize to more complex scenes or multiple simultaneous targets beyond simple masks.
- The paper hints at automatic mask extraction, but practical workflows for users remain unclear. Additional discussion on user burden or future plan for text-only grounding integration would be beneficial.
- Captioning uncertainty may cause noisy supervision; more detail on how noisy captions affect training would help strengthen the case.

### Questions
- How sensitive is the approach to inaccurate target prompts during training (e.g., captioning incorrectly naming the target)? Could self-refinement or contrastive prompts help?

- Could the proposed cross-attention loss be extended to temporal consistency (e.g., enforcing [TGT] attention along tracks across frames)?

- How does performance change if the target mask is inferred from a text-driven grounding model at test time instead of being provided manually?

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
This paper presents a target-aware video diffusion model that generates videos from an input image, mainly focusing on the human-object interaction generation. The key idea is to involve the target masks as new conditions with a special token. The experiments verify the effectivenss.

### Strengths
1. The idea is straightforward and reasonable.
2. The visualization is sufficient.

### Weaknesses
1. Integrating the mask condition is reasonable with expected improvement for interaction generation. Therefore, the technological innovation of this work is not significant. Using masks to control the video generation process is not novel. The author should reelaborate the insights of this paper.

2. The quantitative comparison is rather limited, making it less convincing. The author only compared one baseline. The influence of different backbones should be further analyzed to strengthen the claims of the paper.

3. I am confusing of the exact role of [tgt] token. In fact, [tgt] can simply encode the position information of the input mask as textual tokens. What are the benefits of this compared to not having the [tgt] token and only inputting the mask as the condition?

4. Providing the analysis of failure cases is beneficial.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
