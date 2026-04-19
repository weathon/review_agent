# 2D-Supervised Monocular 3D Object Detection by Global-to-Local Reconstruction

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6

## Abstract
With the rise of big models, the need for data has become increasingly crucial. However, costly manual annotations may hinder further advancements. In monocular 3D object detection, existing works have investigated weakly supervised algorithms with the help of additional LiDAR sensors to generate 3D pseudo labels, which cannot be applied to ordinary videos. In this paper, we propose a novel paradigm called BA$^2$-Det that utilizes global-to-local 3D reconstruction to supervise the monocular 3D object detector in a purely 2D manner. Specifically, we use scene-level global reconstruction with global bundle adjustment (BA) to recover 3D structures from monocular videos. Then we develop the DoubleClustering algorithm to obtain object clusters. By learning from the generated complete 3D pseudo boxes in global BA, GBA-Learner can predict 3D pseudo boxes for other occluded objects. Finally, we train an LBA-Learner with object-centric local BA to generalize the 3D pseudo labels to moving objects. Experiments conducted on the large-scale Waymo Open Dataset show that the performance of BA$^2$-Det is on par with the fully-supervised BA-Det trained with 10% videos, and even surpasses some pioneering fully-supervised methods. Besides, as a pretraining method, BA$^2$-Det can achieve 20% relative improvement on KITTI dataset. We also show the great potential of BA$^2$-Det for detecting open-set 3D objects in complex scenes. Anonymous project page: https://ba2det.site.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents the BA2-Det method, a novel paradigm for 2D supervised monocular 3D object detection and tracking. The method utilizes global-to-local reconstruction techniques to generate and refine 3D pseudo labels for objects. It achieves comparable performance to fully supervised methods, even with only 2D annotations. BA2-Det demonstrates robustness in different depth ranges and shows potential for detecting open-set 3D objects.

### Strengths
1. BA2-Det eliminates the need for expensive 3D ground truth or LiDAR data by utilizing 2D annotations. It introduces a novel pipeline that effectively generates and refines 3D pseudo labels from monocular images, achieving accurate and robust object detection.

2. BA2-Det leverages the power of global and local reconstruction techniques to recover 3D structures from monocular videos. This enables the generation of comprehensive 3D pseudo labels for objects, even in complex and occluded scenes.

3. Experimental results on the Waymo Open Dataset demonstrate that BA2-Det performs on par with fully supervised methods trained with only 10% of the training videos. It also surpasses some pioneering fully supervised methods, indicating its strong performance and potential for various 3D object detection tasks.

### Weaknesses
1. Assumptions and Limitations of Monocular 3D Object Detection: The paper assumes that the BA2-Det method is applicable to ordinary videos, but it does not explicitly discuss the limitations of monocular 3D object detection using only 2D annotations. Addressing the inherent limitations and challenges associated with monocular 3D object detection, such as scale ambiguity, occlusion handling, and depth estimation accuracy, would provide a more comprehensive understanding of the method's scope and potential limitations.

2. Simplistic Approach for 3D Object Label Generation: The method for generating 3D pseudo labels relies heavily on global and local reconstructions. While the paper mentions the use of global bundle adjustment (BA) and object-centric local bundle adjustment (LBA), it lacks a comprehensive discussion on the limitations and potential shortcomings of this approach. Additionally, the paper does not provide detailed explanations of the reasoning or justifications behind using specific methods, algorithms, or models within the BA2-Det pipeline.

3. Lack of Comparison with More Recent Methods: The paper only compares BA2-Det with several baselines and a fully supervised BA-Det method. However, it would be beneficial to include a comparison with more recent state-of-the-art methods for a comprehensive evaluation of BA2-Det's performance.

4. Insufficient Analysis of False Positives: The paper briefly mentions the presence of false positives at longer distances but does not provide a detailed analysis of why these false positives occur and how they can be addressed. More insights into the causes and potential solutions would enhance the understanding of the model's limitations.

### Questions
The motivation behind this paper is good, but the authors need stronger evidence, including formula derivations and data explanations, to support the effectiveness of their method. Additionally, the presentation of the paper could benefit from further refinement.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a method for learning 3D bounding box detection from 2D bounding box annotations in videos, based on BA-Det. The approach includes three main steps: obtaining point clouds through Structure from Motion (SFM) and clustering, fitting cuboids and refining them by the learning-based refiner to create pseudo-GT of cuboids, and finally, iteratively training the detector and updating pseudo-GT. The proposed method is evaluated on Waymo Open Dataset (WOD) and KITTI, and it performs comparably on KITTI against the fully-supervised baselines. It also shows applicability to indoor scenes with open-set settings combined with segment-anything’s 2D detection.

### Strengths
- An extensive ablation study discusses valuable information that significantly affects learning, such as the variation in self-training strategies (Table 5) and the threshold values for the depth of pseudo-GT (Table 6).
- The proposed components of the pipeline add up to the final performance (Table 3)
- Applicability is demonstrated not only for road scenes but also for indoor scenes.

### Weaknesses
**Major**
- The paper does not discuss relevant works that learn 3D cuboids from 2D supervision, such as MonoDR [1] and nerf-from-image [2].
- The proposed components of the pipeline are perceived to be a combination of straightforward approaches based on existing techniques without significant technical innovation. For instance, numerous works exist for tracking-based temporal feature integration, clustering employs existing algorithms, and pseudo-GT for training and iterative updates of pseudo-ground truth has also been explored in previous work [3].
- Quantitative evaluation on WOD, which is more challenging than KITTI, shows that the BA-Det and MonoFlex with 10% GT outperform the proposed method significantly by a large margin. I wonder whether this gap becomes closer if 10% GT is used for the proposed method.
- Utilizing a pre-trained extra tracking module in the pipeline and comparing it with the baselines that lack such a module could be seen as an unfair comparison.

**Minor**
- The best results in the table are not highlighted in bold font, making it difficult to discern the superiority of the proposed method.
- The caption of Table 2 and the table header fail to specify the metric used. Ensuring the table is self-contained would aid the reader.


[1] Beker et al. Monocular Differentiable Rendering for Self-Supervised 3D Object Detection. ECCV 2020.

[2] Pavllo et al. Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion. CVPR 2023.

[3] Zakharov et al. Autolabeling 3D objects with differentiable rendering of SDF shape priors.

### Questions
- What is the performance of the proposed method using 10% of GT, compared to BA-Det with the same setting?
- Does this method work comparably with the recent self-supervised pose estimation (+shape reconstruction) method nerf-from-image, when combined with a 2D detector?
- What is the duration of the entire training process?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Amid the rising demand for extensive data in the era of large-scale models, manual annotations pose cost and resource challenges, potentially impeding further progress. Within the domain of monocular 3D object detection, existing research efforts have explored weakly supervised methodologies integrating LiDAR sensors to generate 3D pseudo labels, typically unsuitable for conventional video data. This study introduces an innovative approach, BA2-Det, employing global-to-local 3D reconstruction to supervise the monocular 3D object detector in a purely 2D context.

The proposed paradigm leverages scene-level global reconstruction coupled with global bundle adjustment (BA) to recover 3D structures from monocular videos. To delineate object clusters, the DoubleClustering algorithm is developed. Learning from the complete 3D pseudo boxes generated through global BA, the GBA-Learner predicts 3D pseudo boxes for other occluded objects. Furthermore, training an LBA-Learner with object-centric local BA enables the generalization of 3D pseudo labels to moving objects.

Experimentation on the extensive Waymo Open Dataset reveals that BA2-Det performance aligns with the fully-supervised BA-Det model trained with 10% of videos and even surpasses some pioneering fully-supervised techniques. Additionally, as a pretraining method, BA2-Det demonstrates a remarkable 20% relative improvement on the KITTI dataset. The study also highlights the substantial potential of BA2-Det in detecting open-set 3D objects within complex scenes.

### Strengths
(1) The authors present a novel paradigm referred to as BA2-Det, focusing on supervised monocular 3D object detection within a 2D framework. This approach involves generating 3D pseudo labels and facilitating the learning process for the monocular 3D object detector, emphasizing a perspective based on global-to-local 3D reconstruction.
(2) In an effort to overcome the challenge of training a 3D object detector without 3D labels, this paper has developed three key modules: DoubleClustering, GBA-Learner, and LBA-Learner.
(3) In their experiments using various datasets, BA2-Det demonstrates high-quality pseudo labels for training 3D detectors. This approach shows comparable performance to fully supervised methods and a 20% relative improvement on the KITTI dataset when used for pretraining. Additionally, it shows promise for detecting open-set 3D objects in complex scenes.

### Weaknesses
(1) In the context of Figure 2, is it possible to train Global BA and Local BA together in an end-to-end manner?
(2) In Table 1, it would be beneficial to compare BA-Det trained with 10% data against BA-Det and MonoFlex, which were trained using similar data proportions. When trained with 100% of the available data, the proposed method shows a performance gap compared to BA-Det. Notably, there exists a 12%, 27%, and 30% gap on 3D AP5, 3D APH5, and 3D AP50, respectively.
(3) In Table3, are there any ablation studies with both 3D and 2D supervision?

### Questions
I'm positive about this paper. However I have some concerns in Weaknesses. I hope the authors could response to those questions in the rebuttal. Then I'll make the final decision.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
