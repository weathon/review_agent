# Generative Photographic Control for Scene-Consistent Video Cinematic Editing

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Cinematic storytelling is profoundly shaped by the artful manipulation of photographic elements such as depth of field and exposure. These effects are crucial in conveying mood and creating aesthetic appeal.  However, controlling these effects in generative video models remains highly challenging, as most existing methods are restricted to camera motion control. In this paper, we propose CineCtrl, the first video cinematic editing framework that provides fine control over professional camera parameters (e.g., bokeh, shutter speed). 
We introduce a decoupled cross-attention mechanism to disentangle camera motion from photographic inputs, allowing fine-grained, independent control without compromising scene consistency. To overcome the shortage of training data, we develop a comprehensive data generation strategy that leverages simulated photographic effects with a dedicated real-world collection pipeline, enabling the construction of a large-scale dataset for robust model training. Extensive experiments demonstrate that our model generates high-fidelity videos with precisely controlled, user-specified photographic camera effects.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CineCtrl, a video-to-video editing framework that enables fine-grained control over professional photographic parameters (bokeh, exposure, zoom, color temperature) alongside camera trajectories. The authors propose a decoupled cross-attention mechanism to separate camera motion from photographic effects and construct a large-scale training dataset combining synthetic and real-world videos with simulated photographic effects.

### Strengths
1. First work to address explicit photographic parameter control in video editing
2. Both synthetic and realistic data construction pipelines are valuable.
3. User study shows clear preference over baselines (Table 4)

### Weaknesses
1. Parameters are normalized relative adjustments, requiring users to understand what "K=0.7" means for their specific video. The author should contain these contents in the paper. Besides, the reader does not have a sense of how "good" the metric number represent for, e.g. how much the value of 0.7 is better than 0.5 in the Bokeh metric?
2. This paper claims that it can also control the camera trajectory, but it lacks relative metric? Will the camera trajectory control accuracy worse than the ReCamMaster after finetune?
3. Since all the metric of Table.2 is from VBench, the author should provide corresponding metric values of the base model, e.g. the base video diffusion model and the ReCamMaster.
4. In the qualitative comparisons in Figure 4, there should exist the ground truth results for reference.
5. As stated in Line236, the motivation of Camera-decoupled cross attention is to alleviate the undesired entanglement between the trajectory and photographic controls, thus, in Table 3, only provide the Bokeh CorrCoef metric is not enough.
6. The synthetic data generation process should be more detailed.

### Questions
1. Are all the model in the table 1 and 2 finetuned from the same base model? If yes, which model?
2. What is the actual effect strength? The qualitative results show subtle changes - can the model produce dramatic photographic effects?
3. What does the composition of the synthetic data look like? For example, how many samples have a complex mixture of effects and how many samples have isolated control? If a sample has a complex mixture of effects, how does it generated? In the row2 of figure 4, it seems that the stitching will cause the degraded visual quality. If the complex sample is generated using the similar technique, there is an issue that the visual quality of synthetic dataset is low.
4. What does the "Note that the stitching baseline is excluded since it directly uses these simulations." in line416 means?

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
5

### Summary
This paper presents CineCtrl, a unified framework for video photography effect editing. It supports editing of bokeh, focal length, exposure, color temperature, and simple camera motion. The method introduces a disentangled attention mechanism to separate the control of camera intrinsics and extrinsics. Both the paper and the supplementary demos show strong visual results.

### Strengths
1. This is the first unified framework for video photography effect editing, extending the idea of Generative Photography (Yuan et al. CVPR 2025) to the video domain. It enables joint control over multiple camera parameters — bokeh, focal length, exposure, and color temperature.

2. The proposed approach has strong potential in video editing, generative AI for photography, and visual effects applications.

### Weaknesses
1. The disentanglement analysis is not deep enough. In Eq. (5), features from camera intrinsics and extrinsics are directly added — how can this design theoretically achieve disentanglement?

2. There is no clear design for disentangling multiple intrinsics (e.g., focal length vs. exposure). How can these parameters be independently controlled without interference? The paper and supplement lack examples showing the same source video with multiple intrinsics changed simultaneously.

3. The paper does not explain why paired data is needed for training. While this idea was validated in Generative Photography (CVPR 2025), here the motivation is unclear — given that video frames already share scene consistency, the need for paired data should be supported by an ablation.

4. The paper lacks discussion or results on fine-grained control (e.g., bokeh K = 0.3 vs. 0.31), which would demonstrate the precision limit of CineCtrl.

5. The simulation methods used to synthesize the four photographic effects lack necessary citation to Generative Photography (Yuan et al. CVPR 2025).

### Questions
Overall, I have a positive impression of this work. The main issues to address are the intrinsic disentanglement design and the fine-grained control demonstration. If the authors can clarify or improve these points, I’d be happy to raise my score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for scene-consistent video cinematic editing that enables fine-grained photographic effect control (e.g., exposure, depth of field, color temperature, zoom) on real videos. The method builds upon a pre-trained text-to-video backbone and introduces a camera-decoupled cross-attention module to separate photographic control from camera trajectory control. To train and evaluate the model, the authors construct a hybrid dataset combining simulated and real-world videos with controlled photographic variations. Experiments demonstrate improvements in photographic effect accuracy, video quality, and scene consistency.

### Strengths
1. The paper explores an interesting and relatively unexplored direction—adding fine-grained photographic control to video editing.
2. It includes a reasonable data collection pipeline and a simple module design that yields some improvements over baselines.
3. The overall writing and presentation are clear.

### Weaknesses
1. Overstated novelty and incomplete related work discussion.
The paper emphasizes its novelty but omits several closely related works, especially in the image domain (e.g., arXiv:2412.02168
), which already demonstrate strong results on similar photographic controls. The paper briefly dismisses these methods as “text-conditioned,” but this difference seems superficial, since both textual and numerical conditions are ultimately embedded as vectors. While I acknowledge the novelty on the video side, I would like to see a clearer explanation of what makes camera-level fine control more challenging in videos compared to images.

2. Limited methodological novelty.
The proposed decoupled cross-attention essentially extends existing camera trajectory conditioning by adding more camera-related dimensions. The design change seems incremental rather than fundamentally new. It would be helpful to clarify what unique insights or training challenges arise from including these additional camera parameters, and how they qualitatively differ from position parameters.

3. Insufficient experimental validation.
a) Baselines. The comparison set is limited. Many related image-based methods could, in principle, be adapted to videos frame by frame, and should be included as baselines rather than only self-implemented ones.
b) Ablations. The paper heavily argues for the effectiveness of the decoupled cross-attention, but lacks comparisons with other conditioning strategies (e.g., text-based injection, additive modulation as in ReCamMaster). The effect on model complexity and parameter count should also be reported.
c) Rationale vs. implementation. While the task itself is meaningful, the implementation seems less justified—since the training data are generated using conventional algorithms, it remains unclear what benefits the learned model provides compared to these traditional methods in terms of efficiency, latency, or effect quality.

### Questions
1. Could the authors clarify what makes camera control in video generation fundamentally harder than in image generation, beyond temporal consistency?
2. How does the proposed decoupled cross-attention differ conceptually and functionally from existing trajectory- or motion-conditioned attention mechanisms?
3. Have the authors compared their method against recent image-domain controllable generation works (e.g., [arXiv:2412.02168]) to better contextualize their novelty?
4. Could additional ablations be provided to show the impact of the proposed conditioning design versus simpler baselines?
5. What advantages does the learned control approach provide over traditional camera control methods used in creating datasets in terms of efficiency, latency, or effect quality?

### Soundness
2

### Presentation
2

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
The paper proposes CineCtrl, a video-to-video (V2V) cinematic editing framework that enables fine-grained, scene-consistent control over professional photographic parameters—including bokeh, exposure, color temperature,  focal length, and also camera trajectory.

The core contributions are as follows:

1. Beyond camera extrinsics, the paper introduces a more comprehensive set of photographic control parameters for video translation, covering professional attributes such as bokeh, exposure, color temperature, and focal length.

2. It develops a data preparation pipeline that constructs both synthetic and real paired datasets of “the same scene under different photographic settings,” enabling supervised learning of controllable cinematic effects.

### Strengths
### Originality
1. The paper explores an under-addressed aspect of controllable video translation—photographic controls such as bokeh, exposure, color temperature, and focal length—rather than only camera trajectories. This significantly broadens the scope of controllable video translation.
2. The data side is also non-trivial: physically inspired simulation for 4 kinds of photographic effects + a real-data curation pipeline so the model doesn’t overfit to synthetic depth or simple zooms.

### Technical Quality
1. The method is built on top of a strong, modern video diffusion backbone (DiT-based) and reuses a proven camera-control encoder (from ReCamMaster) for the trajectory branch, which makes the engineering story credible.
2. Evaluation considers both: (i) effect accuracy (CorrCoef for bokeh/zoom/exposure/color) and (ii) scene/video consistency (LPIPS, CLIP-V, VBench metrics), which is appropriate.

### Clarity
1. The paper has a clear structure and is easy to read.
2. Figure 2 very clearly shows where the two control streams enter the DiT block.

### Significance
1. This work is valuable, as it enables a single model to modify real videos with precise photographic adjustments—for example, producing the same scene with a tighter focal length, warmer tone, shallower depth of field, or a shifted camera trajectory.

### Weaknesses
1. **Limited Novelty and Missing Comparison.**
   The proposed *Camera-Decoupled Cross-Attention* is conceptually similar to existing decoupled cross-attention mechanisms such as IP-Adapter (Sec. 3.2.2), but the paper does not clearly explain the differences or cite related works, reducing the perceived originality.

2. **Data Pipeline Reliability.**
   The data synthesis pipeline depends heavily on depth estimation (“Video Depth Anything”) and bokeh simulation, both of which are error-prone and can fail on thin structures or dynamic scenes, potentially limiting data quality and model robustness.

3. **Lack of Quantitative Disentanglement Analysis.**
   Although the decoupled attention empirically outperforms naïve fusion, the paper does not provide quantitative evidence of control independence (e.g., verifying that changing exposure does not affect motion or depth-of-field), leaving disentanglement only qualitatively demonstrated.

### Questions
1. **Originality Concern.**
   My main concern lies in the originality of this work. Although extending controllable video generation to include richer photographic parameters is meaningful, the core method — *Camera-Decoupled Cross-Attention* — appears highly similar to prior approaches. The authors should clearly articulate how their design differs from existing methods and what novel insights it provides.

2. **Control Resolution.**
   The paper states that all controllable parameters are normalized to ([0,1]) or ([-1,1]). How fine is the actual control in practice? For instance, for the bokeh parameter, can users reliably perceive a difference between 0.3 and 0.35, or is the control effectively quantized into only a few visually distinct levels?

3. **Design of Dual Encoders.**
   The model employs two learnable encoders — one for camera extrinsics and another for other photographic parameters. What is the motivation for this two-encoder design? Why not assign a separate encoder to each controllable parameter for potentially finer disentanglement and flexibility?

### Soundness
3

### Presentation
3

### Contribution
2
