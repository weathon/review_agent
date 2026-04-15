# MotionDirector: Motion Customization of Text-to-Video Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 8, 5

## Abstract
Large-scale pre-trained diffusion models have exhibited remarkable capabilities in diverse video generations. Given a set of video clips of the same motion concept, the task of Motion Customization is to adapt existing text-to-video diffusion models to generate videos with this motion. For example, generating a video with a car moving in a prescribed manner under specific camera movements to make a movie, or a video illustrating how a bear would lift weights to inspire creators. Adaptation methods have been developed for customizing appearance like subject or style, yet unexplored for motion. It is straightforward to extend mainstream adaption methods for motion customization, including full model tuning, parameter-efficient tuning of additional layers, and Low-Rank Adaptions (LoRAs). However, the motion concept learned by these methods is often coupled with the limited appearances in the training videos, making it difficult to generalize the customized motion to other appearances. To overcome this challenge, we propose MotionDirector, with a dual-path LoRAs architecture to decouple the learning of appearance and motion. Further, we design a novel appearance-debiased temporal loss to mitigate the influence of appearance on the temporal training objective. Experimental results show the proposed method can generate videos of diverse appearances for the customized motions. Our method also supports various downstream applications, such as the mixing of different videos with their appearance and motion respectively, and animating a single image with customized motions. Our code and model weights will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The task of Motion Customization involves adapting these models to produce videos showcasing specific motions using reference video clips. However, conventional adaptation methods often entangle motion concepts with appearances, limiting customization. To address this, "MotionDirector" is introduced, employing a dual-path Low-Rank Adaptions (LoRAs) architecture and an appearance-debiased temporal loss, which effectively decouples appearance and motion, enabling more versatile video generation.

### Strengths
- The paper identifies the challenge in generalizing customized motions across diverse appearances. The integration of motion in the video appears great, and this effect can be attributed to the decoupling treatment of the temporal module.
- It proposes a dual-path architecture designed to separate the learning of appearance and motion.
- The visual results show that the proposed method outperforms multiple baseline methods on motion control and video object replacement.

### Weaknesses
- The explanation of the decentralized temporal loss is not very clear. It might be beneficial to verify the effect of this loss through more ablation experiments, especially in the context of video visualization.
- Training LoRA does not appear to be computationally intensive, but it's advisable to specify the training cost in the article.
- There are concerns about the generalizability of this method for video motion extraction. It's worth considering the possibility of developing a unified video motion extraction module to address this issue.
- There are no supplementary videos, which makes the paper less convincing.

### Questions
See above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the concept of Motion Customization in text-to-video diffusion models and proposes a method called MotionDirector, which applies dual-path inserted LoRAs to decouple the learning of content and motion. It also incorporates an appearance-debiased temporal loss to refine the learning process further. The paper validates the approach through experiments on two benchmarks, demonstrating its superiority in terms of motion fidelity and appearance diversity.

### Strengths
- The paper is well-written and easy to follow.
- The proposed appearance-debiased temporal loss sounds reasonable.
- I appreciate Fig 4, which shows the denoising paths of different conditions.
- The demo quality is good.

### Weaknesses
1. **Limited Novelty**: 
   - The concept of decoupling the learning of content and motion is not new and has been explored in works like "Align your Latents" [1] by NVIDIA. The dual-path architecture seems to be a reiteration of this idea.
   - The methodology largely builds upon existing techniques like Low-Rank Adaptions (LoRAs).

2. **Lack of Justification for Appearance-Debiased Temporal Loss**: 
   - The paper introduces an appearance-debiased temporal loss but does not provide a thorough explanation or justification for its effectiveness.
   - The introduction of a hyperparameter $\beta$ is not accompanied by a sensitivity analysis, leaving its impact on the model's performance unclear.

3. The video length is too small (number of frames), it seems only experiments on video length equal to 16 are conducted. Considering that it requires 8 minutes to fit 16 frames, it becomes very time-consuming and even inapplicable for longer videos.

4. Technical contribution is weak. 

[1] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presented MotionDirector, a diffusion-based pipeline for text-to-video video editing. The paper emphasized the design of motion-appearance decoupling dual-path architecture and a special appearance-de-biased temporal loss. The experiments are conducted on UCF Sports and LOVEU-TGVE-2023.

### Strengths
1. The way that authors decouple the motion and appearance when using LoRA is novel and smart.  
2. The qualitative experiment results are convincing. 
3. The paper generally reads well.

### Weaknesses
1. I personally do not see a necessity that especially formulates the task of motion customization. It is a subset of video editing tasks. Meanwhile, the motion pattern is not generated from scratch nor adjustable. 
2. There is no discussion of failure cases, which can provide important insights for the video editing field.

### Questions
1. I really love the motivation of appearance-debiased temporal loss. Especially, the illustration in Figure 4 is intriguing and meaningful. However, I expect the authors to provide more discussion and analysis for this part. Including but not limited to answering the following questions:

	* Is there a more theoretical and/or experimental proof for the hypothesis: motion primarily impacts the underlying dependencies between these point sets, whereas the distances between different sets of points are more influenced by appearance?

	* Is there a better way to evaluate the effectiveness of AD loss? In the paper, there are only two sample videos showcasing the impact of adding AD loss to the training.

2. Do the authors test how many frames can be consistently generated using the proposed method?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes MotionDirector, a dual-path LoRA architecture to decouple the learning of appearance and motion within videos for transferring the motion. It also proposes a new loss function for debiasing appearance bias in temporal information. Experiments show that the proposed method can generate diverse videos with desired motion concepts.

### Strengths
1. The paper is well-written and easy to follow 
2. The idea of dual-path model combining LoRA is intersting, 
3. Experimental results show the effectiveness of the proposed method to achieve the transfer of target actions.

### Weaknesses
1. In this paper, the LoRA technique is used to decouple the learning of appearance and dynamics in reference videos. Does this method require separate training for each specific set of videos for a particular motion to generate videos? How does the video quality fare beyond the distribution?

2. The authors have designed two branches for learning the appearances and dynamics of videos. In the temporal branch, the authors have included spatial LoRA and share parameters with the spatial branch. Does such inclusion of spatial LoRA in the temporal path interferes with learning temporal information?

3. In Figure 4, video 1 & video 2, and video 2 & video 3 are relatively close, but they do not seem to have much in common (e.g., the three videos do not share the same motion or appearance). On the contrary, video 3 and video 4 are farther apart. Therefore, I am skeptical about the claim that the distance between clusters is determined by the appearance of the videos. It's important to base such claims on statistical results from a larger sample rather than a few videos, as the latter can lead to a higher degree of randomness. I suggest gathering more video data to support the argument.

4. In Figure 4, Part D, the authors mention that it represents the visualization of appearance-debiased latent codes, but it's not clear how it relates to Part C. How does Part D reflect appearance debiasing?

5. In the section "Temporal LoRAs Training" on page 6, why does inserting spatial LoRA into the temporal path allow the temporal LoRA to ignore the appearance information in the training dataset? Why wont it affect spatial LoRA during the training of the temporal path?

6. How was Equation 6 derived? Why was this form of loss function with sampling used to eliminate appearance bias in the temporal information? Epsilon_anchor is used as an anchor, so why is there another epsilon_i later on? What is the purpose of having two anchors?

### Questions
see weaknessess

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
