# 3DiffTection: 3D Object Detection with Geometry-Aware Diffusion Features

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3

## Abstract
We present 3DiffTection, a cutting-edge method for 3D detection from posed images, grounded in features from a 3D-aware diffusion model. Annotating large-scale image data for 3D object detection is both resource-intensive and time-consuming. Recently, large image diffusion models have gained traction as potent feature extractors for 2D perception tasks. However, since these features, originally trained on paired text and image data, are not directly adaptable to 3D tasks and often misalign with target data, our approach bridges these gaps through two specialized tuning strategies: geometric and semantic. 
For geometric tuning, we refine a diffusion model on a view synthesis task, introducing a novel epipolar warp operator. This task meets two pivotal criteria: the necessity for 3D awareness and reliance solely on posed image data, which are readily available (e.g., from videos). For semantic refinement, we further train the model on target data using box supervision. Both tuning phases employ a ControlNet to preserve the integrity of the original feature capabilities. In the final step, we harness these capabilities to conduct test-time prediction ensemble across multiple virtual viewpoints.
Through this methodology, we derive 3D-aware features tailored for 3D detection and excel in identifying cross-view point correspondences. Consequently, the resulting model emerges as a powerful 3D detector, substantially surpassing previous benchmarks, e.g., Cube-RCNN, a precedent in single-view 3D detection by 9.43\% in AP3D-Near on the Omni3D-ARkitscene dataset. Furthermore, 3DiffTection showcases robust data efficiency and remarkable generalization to cross-domain data.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduce 3DiffTectio, a 3D detection model using posed images, based on a generative diffusion model. It overcomes the limitations of current diffusion models in 3D tasks and leverage two controlnet to refine the diffusion feature to be 3D-aware, ultimately excelling at identifying cross-view correspondences. The proposed method outperforms predecessors like Cube-RCNN by 9.43% on a specific dataset and showcases impressive data efficiency and cross-domain generalization.

### Strengths
1. This paper is quite novel, revealing that the features of generative models are also suitable for downstream perception tasks.
2. The figures and datasets chosen in the paper effectively elucidate its motivation and the viability of the proposed method. 
3. The performance is quite good.

### Weaknesses
1. I am quite doubt whether the geometric ControlNet truly introduces 3D awareness. Although they trained the ControlNet on posed images using novel view synthesis, the inclusion of a warping operation in the ControlNet suggests that the diffusion model is simply performing an image completion on the warped features.
2. The method is trained on video data, which means it posses the piror knowledge on general 3D scene. In contrast, the baseline method has not been trained on posed images, making this comparison somewhat unfair.
3. For perception tasks, the size of the model and its runtime need to be considered. The combination of ControlNet + diffusion might make the model inefficient.

### Questions
See weakness.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces "3DiffTection," an advanced methodology for 3D detection from posed images, leveraging features from a 3D-aware diffusion model. The approach adeptly addresses the challenges associated with annotating large-scale image data for 3D object detection. By integrating geometric and semantic tuning strategies, the authors have augmented the capabilities of existing diffusion models, ensuring their applicability to 3D tasks. The method notably surpasses previous benchmarks, demonstrating high label efficiency and robust adaptability to cross-domain data.

### Strengths
1. The methodology effectively circumvents the challenges of annotating large-scale image data for 3D object detection.
2. Through the integration of geometric and semantic tuning strategies, the authors have enhanced the capabilities of diffusion models

### Weaknesses
1.The performance on a broader range of datasets is missing, and it should also be compared with more recent research.

2.Semantic ControlNet lacks a more comprehensive analysis.

### Questions
1.Could you provide further explanations regarding how Semantic ControlNet and Novel View Synthesis assist in enhancing models, along with corresponding analyses?

2.Could you present comparative performance results of the more models across the more datasets?

1. In the stage2, is the input noised source image or pure gaussian noise?
2. is there any other generative model can get the same improvement by embedding geometry and semantic control?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript addresses the task of 3D object detection from posed images by leveraging the 2D feature space of pre-trained large diffusion models and exploiting ControlNet to integrate 3D geometric awareness and auxiliary semantic infomation. Extensive experiments on Omni3D datasets demonstrate the effectiveness of the method.

### Strengths
* The manuscripts first proposes to improve 3D awareness by aggregating features with ControlNet from auxiliary views 
* The method proposed in the manuscript achieve significant margins over comparable baselines.

### Weaknesses
* The novelty seems limited. Though with the insight of integrating 3D awareness and closing the domain gap with auxiliary semantic information, the actual practice is adopting existing work ControlNet (Zhang et al., 2023)[^1]. The proposed method is more like an application of ControlNet on a specific task (in this case, the task of 3D object detection from posed images).
* The sampling strategy on the epipolar line needs clarification. If the line of sight is blocked by objects, it is unreasonable to include features sampled behind the blocking objects. It is recommended to provide more details on how to avoid aggregate sampling features from blocked views.
* Minor problems in presentation. *Diffusiondet: Diffusion model for object detection* appears twice in the *Reference* section

[^1]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.

### Questions
See *Weaknesses* section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
