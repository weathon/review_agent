# Dreamix: Video Diffusion Models are General Video Editors

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 3, 5

## Abstract
Text-driven image and video diffusion models have recently achieved unprecedented generation realism. While diffusion models have been successfully applied for image editing, none can edit motion in video. We present the first diffusion-based method that is able to perform text-based motion and appearance editing of general, real-world videos. Our approach uses a video diffusion model to combine, at inference time, the low-resolution spatio-temporal information from the original video with new, high resolution information that it synthesized to align with the guiding text prompt. As maintaining high-fidelity to the original video requires retaining some of its high-resolution information, we add a preliminary stage of finetuning the model on the original video, significantly boosting fidelity. We propose to improve motion editability by using a mixed objective that jointly finetunes with full temporal attention and with temporal attention masking. We extend our method for animating images, bringing them to life by adding motion to existing or new objects, and camera movements. Extensive experiments showcase our method's remarkable ability to edit motion in videos.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a text-driven motion edition method based on video diffusion models. The authors leverage a cascade of VDMs and use a mixed-finetuning objective (full temporal attention and masked temporal attention) for high-fidelity retention. The authors also extend their method to animation generation. Extensive experimental results show impressive creativity of the proposed method, especially in the supplementary materials.

### Strengths
Motion edition is a challenging and important topic. The authors propose a systematic solution for text-driven motion edition based on video diffusion models, which has a positive impact on the entire community. The authors present a good number of experiments validating the effectiveness of their approach and demonstrate excellent performance.

### Weaknesses
1. Limited fidelity to original videos. Ensuring the fidelity of the original video is a thorny issue. Also, in the presentation:

**Spatially** The background has become smooth and appears to be peeling (ref. Figure 7). The authors need to explain what efforts the authors have made to address this issue.

**Temporally** Lack of integration of action semantics and temporal modeling, which leads to the incongruity of target motions. *e.g.,* the transition from eating to dancing in Figure 1.

2. The proposed method is computationally expensive due to the introduction of VDMs.

### Questions
1. In page 5, the authors say:*Technically, the model is trained on a sequence of frames u by replacing the temporal attention layers by trivial fixed masks ensuring the model only pays attention within each frame, and also by masking the residual temporal convolution blocks.* What do you mean about temporal attention layers? Why use an unordered set of M frames? In my opinion, the former is a technical operation at the model level, while the latter is a processing at the data level. Is there any necessary connection between the two technologies?

2. Can the authors explain why extending this method to animation generation works? After all, an image cannot provide temporal information about actions, and how to ensure the continuity of video frames is a key issue.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
introduce Dreamix, a video editing framework that combines low-resolution spatio-temporal information from the original video with high-resolution synthesized information aligned with a text prompt. The framework maintains fidelity to the original video by using a preliminary stage of finetuning the model on the original video. The authors propose a mixed finetuning approach that improves motion editability by training the model on both the original video and its individual frames. Additionally, Dreamix can be used to animate still images and perform subject-driven video generation. The authors provide extensive experiments to showcase the capabilities of Dreamix.

### Strengths
1. the proposed framework can be applied top multi-tasks like Video Editing, Image-driven Videos and Subject-driven Video Generation.

2. the Mixed Video-Image Finetuning is sound and with reasonable performance 

3. The paper presents extensive experiments that demonstrate the ability of Dreamix.

### Weaknesses
1.  although the Mixed Video-Image Finetuning strategy is sound, the overall technical novelty is kind limited.  it is an application of VDMs with sophisticated strategy during finetuing. Also, it would be interesting to see how much would the base VDM would effect the finetuing results.

2. although the paper claims high fidelity and quality for video editing, the resolutions are still low and with blurred details.  It looks like the input video/image provides layout information and the details are generated by VDMs. 

3. for Image-driven Videos, the method can not maintain the consistent between the input and first frame, and the quality is worse than more recent work like animatefiff.

4. for Subject-driven Video Generation, the appearance feature and details can not be maintained, like the   toy in figure 2 and bear in figure 6.

### Questions
Please check weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a video editing method with a video diffusion model. The proposed method can edit both motion and appearance, and it can acquire information from images or animate a single image. Dreamix achieves good results on the collected videos.

### Strengths
1. The proposed method can achieve various video editing and generation applications.
2. Extensive ablation studies are conducted to demonstrate the effectiveness of each component.

### Weaknesses
1. The comparison with previous works is not comprehensive. For text-based video editing, FateZero [ref-1] and TokenFlow [ref-2] are more advanced methods designed for video editing. 
For subject-driven video generation, Animatediff [ref-3] can learn a Lora model from several images and generate corresponding videos. For animating a single image, VideoComposer [ref-4] can generate video conditioned on the single image and text, which does not require additional transformation. These methods should be discussed and compared in the experiments and related work.

[ref-1] Fatezero: Fusing attentions for zero-shot text-based video editing.

[ref-2] TokenFlow: Consistent Diffusion Features for Consistent Video Editing.

[ref-3] Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.

[ref-4] VideoComposer: Compositional Video Synthesis with Motion Controllability.

2. The performance of motion editing is not satisfactory. This paper claims motion editing as one of the key contributions. However, as shown in Figure 7, the motion and appearance are from the video generation model (Uncond.), while Dreamix seems only to help to learn the background. From my perspective, motion editing should focus on keeping the same identity while changing the motion, instead of keeping the background. 

3. The proposed method heavily relies on the choice of video generation model. For both single-image animation and subject-driven video generation, only frame-level finetuning is adopted ($\alpha$ in Eq.3 is 0). This requires the video generation model to handle single images, which may not be satisfied easily by other video generation models. It is better to verify the method on an open-source video generation model (e.g., [modelscope](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis))

4. During video editing, the degraded source video with noise is used as the initialization. In comparison, most video editing methods adopt DDIM inversion as initialization. It is better to compare the two methods.

5. **Minor**. I hope the authors can check the `\citep` and `\citet` in the LaTex template and improve the writing.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a novel extension to the video diffusion model by introducing video editing capabilities. The fine-tuning method of the combination of frame-level and video-level is used to ensure the motion and appearance quality. The experimental results demonstrate the model's effectiveness in video editing, image-to-video conversion and object-driven video generation.

### Strengths
1.	The model can accomplish multiple video editing tasks and show great visualized results.
2.	The fine-tuning strategy enhances the editing capabilities of VDM.
3.	The paper is well-structured, capable of clearly elucidating its core ideas.

### Weaknesses
1. This paper conducted editing experiments on a single VDM base model, making it difficult to ascertain whether the proposed method is applicable to other VDMs e.g., modelscope or if the observed results are solely due to the characteristics of the base model. This is somewhat inconsistent with the title of the paper.   

2. The comparisons with Tune-A-Video are not fair. Tune-A-Video is finetuned on image diffusion model but dreamix is finetuned on video diffusion model.


3. When compared with other VDMs, only the motion editing capabilities (Figure 7) were assessed, and there is no comparison of appearance editing capabilities. This doesn't adequately showcase the impact of mixed-attention fine-tuning.

4.  Lacks specific information. Imagen Video operates on a series of diffusion models. It is unclear which components undergo fine-tuning during the process. Is there a necessity to fine-tune the super-resolution diffusion models as well? or only key frames diffusion model is enough.

### Questions
Please see the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
