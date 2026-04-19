# DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5

## Abstract
Despite remarkable research advances in diffusion-based video editing, existing methods are limited to short-length videos due to the contradiction between long-range consistency and frame-wise editing. Recent approaches attempt to tackle this challenge by introducing video-2D representations to degrade video editing to image editing. However, they encounter significant difficulties in handling large-scale motion- and view-change videos especially for human-centric videos. This motivates us to introduce the dynamic Neural Radiance Fields (NeRF) as the human-centric video representation to ease the video editing problem to a 3D space editing task. As such, editing can be performed in the 3D spaces and propagated to the entire video via the deformation field. To provide finer and direct controllable editing, we propose the image-based 3D space editing pipeline with a set of effective designs. These include multi-view multi-pose Score Distillation Sampling (SDS) from both 2D personalized diffusion priors and 3D diffusion priors, reconstruction losses on the reference image, text-guided local parts super-resolution, and style transfer for 3D background space. Extensive experiments demonstrate that our method, dubbed as DynVideo-E, significantly outperforms SOTA approaches on two challenging datasets by a large margin of 50%~95% in terms of human preference. Compelling video comparisons are provided in the anonymous project page https://dynvideo-e.github.io. Our code and data will be released to the community.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work present a novel framework of DynVideo-E that for the first time introduces the dynamic NeRF as the video representation for large-scale motion- and view-change human-centric video editing. With a set of customized designs and training strategies, it outperforms SOTA approaches by a large margin on human preference.

### Strengths
Long term video editing consistency has been improved by a large margin;
Qualitative video results shows great performance gain.

### Weaknesses
The whole customized process is a bit lengthy with a lot of customization, which potentially makes the reproductivity difficult.

### Questions
Could you explain a bit more about the part of Text-guided Local Parts Super-Resolution?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a dynamic NeRF-based approach to handle video editing in 3D space. The proposed method uses a deformation field to propagate the edits to the entire video. The authors introduce several design improvements that enhance the editing performance. The experimental results demonstrate the effectiveness of the proposed approach, as shown by both qualitative and quantitative evaluations.

### Strengths
1. The paper is well-written, easy to follow, and features illustrative figures that effectively convey the concepts. 

2. The proposed method is well-motivated and adeptly addresses existing limitations by employing 4D representations for video editing. This approach integrates motion and view changes, while the deformation field guarantees consistency throughout the edited video.

3. The experimental evaluation is comprehensive, offering both qualitative and quantitative evidence that demonstrates the superiority of the proposed method over baseline approaches.

### Weaknesses
1. One limitation of the proposed method is the requirement for calibrated camera poses, which may not be readily obtainable for all videos. This constraint could potentially restrict the applicability of the approach in certain scenarios or require additional preprocessing steps to estimate camera poses.

2. Similar to other recent NeRF-based generation and editing methods, the proposed approach can be time-consuming. This factor may hinder its adoption in real-time applications or situations where rapid editing is necessary.

3. As a NeRF-based approach, the edited videos should ideally support free-viewpoint rendering. However, the qualitative results presented in the paper only show editing results with the aligned timestamp and viewpoint as the input video. This aspect raises questions about the method's ability to generate consistent and accurate results across different viewpoints, which is a key advantage of NeRF-based approaches.

### Questions
1. COLMAP can sometimes fail with moving objects in the scene for camera calibration. It would be helpful if the authors could provide more details on how to run COLMAP in such cases and if there are any specific parameters or settings that can be adjusted to improve its performance.

2. It would be beneficial if the authors could provide an analysis of the time required for a single editing operation using the proposed approach. This information would help to understand the practical feasibility of the proposed method for real-world applications.

3. It would be interesting to see an edited result with an arbitrary camera trajectory or a static human subject while the camera is moving (e.g., bullet time effect) instead of only showing results with viewpoints aligned to the input video. This would highlight the versatility of the proposed method and its potential for a wide range of applications.

4. As another dynamic NeRF-based approach for video editing, the paper does not mention the work by Zhang et al., "Editable free-viewpoint video using a layered neural representation" (SIGGRAPH 2021). It would be helpful if the authors could provide a brief comparison between their proposed method and the approach presented in Zhang et al.'s work in the related works section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a method for consistent editing of large-scale motion- and view-change human-centric videos. Specifically, the proposed method exploits dynamic NeRF for the video representation, and integrates several techniques including the Score Distillation Sampling (SDS) from both 2D personalized diffusion priors and 3D diffusion priors, reconstruction losses on the reference image, text-guided local parts superesolution, and style transfer for 3D background space. The experiments validate the effectiveness of the proposed method.

### Strengths
+ The paper is clear and easy to follow.
+ Multiple existing techniques are combined into the whole pipeline.

### Weaknesses
- About the novelty: The paper is a system paper that combines several existing works, including HOSNeRF, Zero-1-to-3, and Magic123, without too much novel insight. For example, the basic video representation follows the existing work HOSNeRF, and the only difference is the removal of object state designs for the specific task in the paper. Both 3D and 2D priors follow the existing works, Zero-1-to-3, and Magic123. From these points, the novelty of the paper mainly lies in the integration of such existing works.
- About the application scenario: The proposed method relies on the dynamic human NeRF reconstruction, making it limited to human-centric video and less interesting. 
- About the experiment: A quantitative comparison is also expected for the ablation. Moreover, only a rather small dataset is utilized for the test, and more videos in the wild are expected.

### Questions
Please refer to the weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
