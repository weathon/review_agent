# EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Controlling camera motion in video diffusion models is highly sought after for content creation, yet remains a significant challenge.
Recent approaches often create anchor videos
(i.e., rendered videos that approximate desired camera motions)
to guide diffusion models as a structured prior,
by rendering from estimated point clouds following camera trajectories. 
However, errors in point cloud and camera trajectory estimation often lead to inaccurate anchor videos during training. Furthermore, these inherent errors lead to higher training cost and inefficiency, since the model is forced to compensate for rendering misalignments.
To address these limitations, we introduce EPiC, an efficient and precise camera control learning framework
that constructs well-aligned training anchor videos
without the need for camera pose or point cloud estimation.
Concretely, we create highly precise anchor videos by masking source videos based on first-frame visibility.
This approach ensures strong alignment, eliminates the need for camera/point cloud estimation, and thus can be readily applied to any in-the-wild video
to generate image-to-video (I2V) training pairs.
Furthermore, we introduce Anchor-ControlNet, a lightweight conditioning module that integrates anchor video guidance in visible regions to pretrained video diffusion models, with less than 1\% of backbone model parameters. By combining the proposed anchor video data and ControlNet module, EPiC achieves efficient training with substantially fewer parameters, training steps, and less data, without requiring modifications to the diffusion model backbone.
Although being trained on masking-based anchor videos, our method generalizes robustly to anchor videos made with point clouds at test time, enabling precise 3D-informed camera control.
EPiC achieves state-of-the-art performance on RealEstate10K and MiraData for I2V camera control task, demonstrating precise and robust camera control ability both quantitatively and qualitatively.
Notably, EPiC also exhibits strong zero-shot generalization to video-to-video (V2V) scenarios. This is compelling as it is trained exclusively on I2V data, where anchor videos are derived with only source videos' first frame as visibility referencing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes EPiC, a method for camera-controlled video diffusion. Previous methods rely on either Plucker coordinates or depth-based warping techniques using point clouds to condition the video model with camera control. In contrast, this work uses optical flow to create anchor videos to guide the generation process. The paper shows this prevents artifacts of point cloud-based techniques and leads to precise camera control.

### Strengths
- Novel conditioning mechanism: Using visibility-based masking with optical flow instead of point-cloud-based conditioning is novel.

### Weaknesses
- Short camera trajectories: The demo videos are not very convincing, since the amount of camera motion is very limited and the videos are very short.
- Imprecise camera control: For “Arc Right” in the supplementary website, the bottom result of the third column (city at the coast): the camera continues to move even though it should stop after the first half of rotation. So the camera is not staying fixed at a final pose. The camera seems to continue moving for many cases, especially when the input video has camera motion. So I wonder if there is an issue that the input camera motion of the video is not considered.
- Missing comparisons: FloVD [1] is only briefly discussed, i.e., it does not use anchor videos but uses optical flow. It would have been great to compare with FloVD. Moreover, comparisons with GEN3C [2] as pointcloud-based method are missing. Furthermore, there are no comparisons with ReCamMaster [3] for the task of video-to-video camera retargeting,
- Anchor-ControlNet is nothing novel: Using a lightweight ControlNet to save parameters is pretty standard. For example, AC3D as one of the baselines also does this, if I understand the approach correctly.

[1] Jin et al., FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis, CVPR 2025 \
[2] Ren et al., GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, CVPR 2025 \
[3] Bai et al., ReCamMaster: Camera-Controlled Generative Rendering from A Single Video, ICCV 2025

### Questions
While the method is sound, the results are not that convincing. The camera motion is limited and it seems to not be that precise. Moreover, there are some comparisons with recent approaches missing.

I would like authors to address following questions:
- Why is the amount of camera motion that limited? Because of the base video model?
- Why does the camera continue moving for many video-to-video cases even though it says to be fixed after the first movement?
- Why were recent methods like GEN3C or ReCamMaster not used for comparisons?

I think that the paper makes sense but the results and missing comparisons lead to a negative rating. I am happy to consider a rebuttal and open to adjusting my score, though it might be unlikely to accept the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents EPiC (Efficient and Precise Camera Control), a framework for efficient learning of camera motion control in video diffusion models (VDMs). Instead of relying on point-cloud rendering and camera trajectory estimation—which often introduce pixel-level misalignment—the authors propose a visibility-based masking strategy to construct well-aligned anchor videos directly from source videos. They further introduce a lightweight Anchor-ControlNet, accounting for less than 1% of the backbone parameters, which conditions video generation on anchor videos through visibility-aware output masking. EPiC enables efficient training (only 5K videos and 500 iterations) and achieves state-of-the-art camera control accuracy on RealEstate10K and MiraData, while also demonstrating strong zero-shot generalization from image-to-video (I2V) to video-to-video (V2V) tasks. Extensive quantitative, qualitative, and ablation studies validate the proposed approach’s efficiency, robustness, and alignment advantages over existing baselines such as CameraCtrl, AC3D, and ViewCrafter.

### Strengths
1. Clear and interpretable design.
2. The idea of ​​converting the problem into anchor-video construction is very interesting.

### Weaknesses
1. Converting camera-guided video generation into the task of supplementing an anchor video is an interesting idea. However, when applied to I2V, although the authors preserve foreground dynamics by masking foreground regions during guidance, the background is effectively forced to remain static. This imposes a limitation when treating video generation as a world model. Moreover, the approach depends to some extent on the performance of foreground extraction models (e.g., *GroundingDINO* used in the paper), which requires users to provide additional inputs.

2. In the Introduction, the authors state that the Anchor-ControlNet is “injected into the first 25% of backbone layers”, while the original ControlNet is injected into 50% of layers. What specific design or empirical analysis motivated selecting 25%? Some works (e.g., *InstantStyle*) experimentally show that particular layers predominantly control generation style, and therefore only control those layers. Did the authors observe similar layer-wise effects? Please provide more design details and/or empirical evidence supporting the 25% choice.

3. Minor :

   a. Line 76: change “We propose” → “we propose” (lowercase “we”).  
   b. Make figure references consistent throughout the manuscript; do not mix full form and abbreviation. For example, Line 367 uses “Figure 1” while Line 375 uses “Fig. 4”. Please standardize notation across the paper.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EPiC, an efficient framework for camera control learning in video diffusion models. It constructs well-aligned anchor videos through first-frame visibility masking. The authors further design a lightweight Anchor-ControlNet architecture with visibility-aware output masking. EPiC achieves state-of-the-art camera control accuracy while requiring only a fraction of the training cost compared to prior methods.

### Strengths
1.	The visibility-based masking for anchor video generation is conceptually simple yet effective.
2.	The method achieves SOTA quality and camera control scores on both I2V and V2V test sets, and the ablation studies verify the effectiveness of the proposed methods.

### Weaknesses
1. The principal contribution lies in constructing more precisely aligned anchor videos to improve training efficiency. Is the proposed module plug-and-play, and could it likewise enhance other diffusion-based video models?
2. I recommend including a failure-case analysis to more thoroughly illustrate the method’s limitations.
3. During inference, how robust is EPiC to substantial errors in point-cloud-rendered anchors? Since it is trained with high-quality mask-based anchors, this discrepancy could introduce a train–test mismatch.

### Questions
Do the authors plan to release the mask-based anchor-video training dataset? Its availability would greatly benefit the research community.

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
4

### Summary
This paper introduces EPiC, a novel framework for efficient camera control in video diffusion models. It replaces error-prone 3D point clouds with a new method for creating "anchor videos" by directly masking source videos based on optical flow. This produces perfectly aligned training data from any video, enabling remarkable efficiency. Combined with a lightweight Anchor-ControlNet, EPiC achieves state-of-the-art results with a fraction of the data and compute of previous methods, and generalizes well to video-to-video tasks.

### Strengths
1. The core strength is the visibility-masking method for creating aligned anchor videos. This elegantly bypasses the need for 3D reconstruction, enabling more efficient training in data and compute than prior work.
2. EPiC achieves SOTA results on standard I2V camera control benchmarks (RealEstate10K, MiraData), demonstrating that the efficiency gains do not compromise final quality.
3. The model shows excellent zero-shot generalization to V2V tasks and demonstrates strong capability in handling dynamic scenes, a key advantage over methods trained primarily on static data.

### Weaknesses
1. The framework operates on a relative scale determined by an external depth estimator at inference time. This may prevents users from specifying precise, real-world camera movements (e.g., "move 2 meters").
2. While qualitative results are promising, the paper lacks quantitative validation on a benchmark with dynamic objects and ground-truth camera motion (e.g., RealCam-Vid).  This would allow for a direct and fair comparison of dynamic handling capabilities with methods like RealCam-I2V.

### Questions
1. The interactive interface for drawing trajectories is a strong feature. Is the motion scale purely relative to the scene's estimated depth, or is there a mechanism to map it to an absolute, metric scale for consistent control?
2. How robust is the training data generation to challenging scenarios where optical flow often fails, such as very fast camera motions (e.g., drone footage) or large-scale rotations?

### Soundness
3

### Presentation
3

### Contribution
3
