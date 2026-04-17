# Generating Human Motion Videos using a Cascaded Text-to-Video Framework

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Human video generation is becoming an increasingly important task with broad applications in graphics, entertainment, and embodied AI. 
Despite the rapid progress of video diffusion models (VDMs), their use for general-purpose human video generation remains underexplored, with most works constrained to image-to-video setups or narrow domains like dance videos. 
In this work, we propose CAMEO, a Cascaded framework for general human Motion vidEO generation. It seamlessly bridges Text-to-Motion (T2M) models and conditional VDMs, mitigating suboptimal factors that may arise in this process across both training and inference through carefully designed components. 
Specifically, we analyze and prepare both textual prompts and visual conditions to effectively train the VDM, ensuring robust alignment between motion descriptions, conditioning signals, and the generated videos. 
Furthermore, we introduce a camera-aware conditioning module that connects the two stages, automatically selecting viewpoints aligned with the input text to enhance coherence and reduce manual intervention. 
We demonstrate the effectiveness of our approach on both the MovieGen benchmark and a newly introduced benchmark tailored to the T2M–VDM combination, while highlighting its versatility across diverse use cases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the same problem as HMTV, which connects a text-to-motion module with a motion-to-video generator. The authors first disentangle the text prompt into a motion prompt and a semantic prompt. The motion prompt is then converted into a motion sequence using a text-to-motion model, which is subsequently rendered as guidance videos. A tailored conditioning strategy is developed for the motion-conditioned video diffusion model (VDM). Camera pose selection is handled by an early denoising stage of a text-to-video model. The results demonstrate both quantitative and qualitative improvements over the vanilla text-to-video baseline and the prior method, HMTV.

### Strengths
1. The observation that large body movements are generated earlier, while finer details appear later, is interesting. Based on this observation, the authors design a conditioning strategy that improves performance.

2. The approach of using a text-to-video model to generate a reference video, and then estimating the approximate camera pose from the early frames of the generated human shapes, is also interesting.

3. The paper demonstrates that the generated motion videos can be edited using the SDEdit approach, which represents a meaningful and practical application of the proposed method.

### Weaknesses
1. The paper proposes to recaption videos and disentangle the text prompt into two complementary parts: motion caption and semantic caption. However, there is no ablation study that evaluates each part separately. What would the results be if the videos were recaptioned into a single caption instead? In Table 2, the paper compares results with and without text refinement, but the improvement is unclear, and in fact, the motion metrics drop after refinement.
2. There is no ablation study or sufficient explanation regarding the hyperparameter choices and the intuition behind the diffusion timestep sampling. In particular, the rationale for using a truncated normal distribution, the mean reduction, and the increase of standard deviation should be further clarified.
3. The argument in Lines 200–210 and Figure 3a needs revision. As I understand it, both the vanilla training and the proposed method use the same motion but different text prompts as inputs. If the visual conditions are the same, why does the vanilla training fail to capture the fine-grained motion? Does this imply that the generated motion is not aligned with the motion condition?
4. The camera pose selection is based on an existing text-to-video model, rather than being predicted within the proposed pipeline. This raises doubts about the accuracy of the extracted poses. It would be more elegant and meaningful if the camera poses were predicted directly by the model.
5. The quantitative results in Table 1 are not particularly strong, especially on the MovieGen benchmark.
6. The paper should discuss related work that studies a related problem:
Move-in-2D: 2D-Conditioned Human Motion Generation, CVPR 2025.
7. There are no supplementary videos provided, making it difficult to assess the visual quality of the model.

### Questions
1. In the motion editing experiments in Figure 6, all results appear to use the same text prompt. Are there any results showing edits using different text prompts?
2. Why is the first-stage text-to-motion model not retrained on the proposed dataset? Would retraining it improve performance?
3. In Figure 4, should HumanVid actually be CamAnimate? Do both HumanVid and Ours use the same motion sequences as input, while VDM is trained on different data and uses a different backbone?

### Soundness
3

### Presentation
3

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
This paper aims at human video generation from text prompts. The proposed method, named CAMEO, is a cascaded T2V generation pipeline, which consists of two parts: T2M and VDM. The T2M component utilizes an off-the-shelf model (STMC) to generate low-dimensional SMPL motion sequences, which are then rendered (conditioned on the camera-view-selection) and fed into a 2D-motion-conditioned video diffusion model to produce the final video. CAMEO is compared with recent methods (HTMV and CamAnimate) and outperforms them in most quantitative evaluation metrics. The paper also presents ablation studies regarding the choice of text refinement strategy and the importance of the view selection module.

### Strengths
1. **Clear pipeline**. It connects a text-to-motion module and a motion-conditioned video diffusion model for more robust human motion video generation, which is technically sound.
2. **Good caption design**. The caption re-captioning with motion/semantics split reduces conflicts during VDM training.
3. The Camera view selection idea is simple yet effective, leveraging early denoising results in the text-to-video diffusion model to extract view changes.

### Weaknesses
1. The approach is quite similar to HMTV and does not demonstrate a significant conceptual or methodological improvement.
2. The procedure depends on early denoising frames and SMPL estimation; robustness to text domains (e.g., stylized, occlusions) is unclear.
3. The proposed text refinement contributes only marginal improvements, as shown in Table 2; ablation results suggest it may not be a major factor.
4. The base models used in comparison differ, making the claimed superiority less meaningful, as the quality of the base T2V model has a substantial influence on the resulting human motion videos.
5. No user evaluation or human preference test is provided to validate the claimed perceptual improvements.
6. Table 1 performance markings are incorrect -- the “best” and “second-best” indicators do not match the actual numerical values, which weakens result's credibility.

### Questions
How robust is the camera module? What happens for stylized or animation-like prompts?

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
Authors propose a framework to decompose Text to Motion and Video Diffusion Models to generate videos conditioned well on Motion and view points.

Authors bring in techniques (e.g. refinement by LLMs using rendering tools) to make the pipeline continuous end to end with minimal (or no) need for human interruptions from prompts to final video.

A new dataset is also proposed.

### Strengths
Comprehensive and Complete pipeline. Authors study the video human centric video generation as an integrated system and do not leave the bottlenecks (like view point conditioning) out of the solution.

The provided qualitative results look promising.

The new dataset brings some more novelty to this work.

Quantitative results are competitive on MovieGen with state of the art, and is mostly the best on the proposed dataset.

### Weaknesses
Although I appreciate the completeness of the approach, but there is not much novelty in each element being used and to some extend the method seems like an ad-hoc utilization of some off-the-shelf tools. The VDM controlNet is the only part that is trained by a new condition.

### Questions
1- It seems to me that Tab 2, ablation study, is not supporting the contribution of different steps. Any clarification on this?

2- Is there an analysis on visibility of different body parts? I feel in most of the qualitative results, the lower body is not visible. Can this be controlled by m_{1:k}?

3- Is there any measurement on the diversity of the view points in the dataset, and the generated videos?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
the paper proposes a text-to-motion augmented cascaded text-to-video generation framework for human motion video generation

### Strengths
- the paper proposes a feasible solution for reliable human motion generation in video generation models
- the proposed framework clearly improves the correctness of generated human body structures and fidelity of generated poses and motions
- extensive experiments demonstrate the effectiveness of the proposed modules and the effectiveness of semi-explicit 3D controls

### Weaknesses
- while the paper demonstrates improvements in terms of single-person motion for the video generation task, the novelty of the proposed framework seems to be limited: the proposed framework feels like a combination of existing components, which are added up together to solve a specific and narrowed-down problem. while the paper acknowledged the limitations of generalizability, it still unclear how robust the proposed framework is when handling more complicated cases for single-person scene video generation, e.g., what happens when camera distance significantly changes, or can the proposed framework handle the cases where the person is occluded or partially out of the frame or temporally missing in some frames
- while the paper adopted VBench for quantitative comparisons, the reliability of the VBench metrics is still not well justified. based on the reviewer's experience, some scores might favor specific aspects of videos while ignoring the actual visual quality. it is highly recommended to conduct a user study to validate the effectiveness of the proposed method considering the human evaluation is still the most reliable metric for video generation tasks
- the quality of demonstrated applications of motion editing and camera view editing seem to be not good in terms of subject consistency

### Questions
please refer to weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2
