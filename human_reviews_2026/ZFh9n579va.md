# AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Single-view RGB model-based object pose estimation methods achieve strong generalization performance but are fundamentally limited by depth ambiguity, clutter, and occlusions. Multi-view pose estimation methods have the potential to solve these issues, but existing works rely on precise single-view pose estimates or lack generalization to unseen objects. To address these challenges, we introduce AlignPose, a 6D object pose estimation method that aggregates information from multiple extrinsically calibrated views and generalizes to unseen objects. The contributions of this work are threefold. First, leveraging powerful, frozen features from a foundation model, AlignPose iteratively minimizes the discrepancy between rendered and observed images across multiple viewpoints, enforcing geometric consistency without object-specific training. Second, robust handling of noisy inputs is achieved by aggregating pose candidates from an arbitrary single-view pose estimator via 3D non-maximum suppression. Third, extensive experiments on three BOP benchmarks (YCB-V, T-LESS, ITODD-MV) show AlignPose sets a new state of the art, especially on challenging industrial datasets where multiple views are readily available in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AlignPose, a new method for generalizable multi-view 6D pose estimation. 
Given pose candidates from an off-the-shelf single-view estimator, AlignPose aggregates them via 3D Non-Maximum Suppression (NMS) in a common coordinate system. 
Then the poses are refined through multi-view feature-metric alignment. 
The refinement minimizes a robust multi-view feature-metric cost function between DINOv2 features from rendered 3D model projections and observed images across calibrated views using Levenberg-Marquardt optimization. 
AlignPose achieves state-of-the-art performance on three BOP datasets, outperforming previous multi-view methods by leveraging frozen foundation model features for zero-shot generalization.

### Strengths
++ The performance gains on YCB-V, T-LESS, and ITODD-MV demonstrate the effectiveness of AlignPose's multi-view refinement strategy over existing methods.

++  This paper introduces a straightforward adaptation of FoundPose's feature-metric refinement to a multi-view setting, integrated with 3D NMS, and demonstrates its effectiveness with promising results.

### Weaknesses
-- The work lacks an ablation study to justify the choice of the robust cost function, including a comparison with alternatives and an analysis of its hyperparameters.

-- The experimental results lack data on time or speed. Analyzing the runtime is essential for understanding the practicality of this method.

-- The 3D NMS method used in this paper lacks a comparison with the translation-based 3D NMS in FreeZev2 [a].

-- The methodological innovation is somewhat limited. The multi-view feature-metric refinement approach presented here is essentially a straightforward extension of the single-view feature-metric refinement method in FoundPose and, therefore, lacks significant innovation.

[a] Accurate and efficient zero-shot 6D pose estimation with frozen foundation models. Caraffa et al., 2025.

### Questions
-- What is the rationale for selecting the current three BOP datasets (YCB-V, T-LESS, ITODD)? Two additional industrial datasets from the BOP-Industrial benchmark, IPD and XYZ-IBD, also provide multi-view evaluation data and appear suitable for assessing the method proposed in this paper. It would be valuable to include an evaluation on these datasets.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents AlignPose, a generalizable multi-view 6D object pose estimation method that does not require object-specific training. Instead of relying on single-view predictions or category-specific models, AlignPose aggregates pose hypotheses from arbitrary single-view estimators and refines them by minimizing a multi-view feature-metric alignment loss using deep learning features. The method leverages 3D non-maximum suppression to consolidate pose candidates across views and performs optimization to enforce geometric consistency across all images. Experiments on YCB-V, T-LESS, and ITODD datasets show that AlignPose achieves state-of-the-art performance in unseen object pose estimation.

### Strengths
This paper introduces AlignPose, a refinement method for unseen object pose estimation. It optimizes a consistent object pose in the world frame, jointly utilizing initial pose estimates from multiple views.

AlignPose introduces a multi-view feature-metric alignment loss with non-maximum suppression, which optimizes the object pose by aligning rendered object features with real images.

The refined pose is obtained by using a Levenberg-Marquardt optimization algorithm, ensuring the robustness of the refinement.

The method achieves state-of-the-art results on multiple datasets, including YCB-V, T-LESS, and ITODD. It outperforms previous refinement approaches in unseen object pose estimation.

### Weaknesses
The problem formulation is not clear enough. To my current understanding, the authors decompose the object pose $T_{CO}$ into two transformations, $T_{CW}$ and $T_{WO}$. This is a bit confusing since we often assume that the world frame and object frame are aligned in object pose estimation. Otherwise, it is unclear how to define the world frame beyond the object frame. A more detailed explanation would be important to improve clarity and help readers better understand the problem setup.

The comparisons appear to be unfair. The authors assume the ground-truth camera pose is available, meaning that $T_{CW}$ is known. This would significantly simplify the object pose estimation and make the comparisons with other methods unfair. Moreover, in real applications, the camera poses are often unknown or noisy. For instance, to get these poses, one needs to run some algorithms such as colmap and VGGT. The results are not always accurate.

The method relies on object meshes, which makes it inapplicable in some scenarios where the object meshes are unavailable. A discussion regarding this limitation is missing, but important. 

The presented multi-view alignment method is a bit straightforward and quite similar to bundle adjustment. Given the object mesh, many alternatives could be used. For example, using the initial object pose to render an RGB image and aligning the rendered image with the query image to refine the pose.

### Questions
How to use the initial object pose in practice?  In Eq.2, which transformation stands for this pose? I guess $T_{WO}$ in this equation is derived from the initial object pose. Is it correct?

In Eq.3, a confidence score is computed, but how to use this score in the experiments is missing. How to use this score to facilitate the pose refinement? Is it important?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method to estimate the poses of objects in the scene given calibrated images from multiple view points. The method involves first getting initial coarse estimates of object poses from each input image by running an off-the-shelf object pose estimator, and then refine those estimates using a bundle adjustment technique that searches for 6DOF object pose estimates that ensures consistency between features registered to 3D from multiple viewpoints. The method achieves state-of-the-art performance on this refinement task.

### Strengths
1. The authors perform rigorous evaluation on multiple datasets, and find that their method consistently outperforms competing approaches. 
2. The presentation of the method is easy to understand with the equations that have been written. The approach seems like a reasonable thing to try. 
3. Strong results are shown on both seen and unseen object categories, showing that the method can generalize well given initial coarse estimates of object poses are in the ballpark of the right answer.

### Weaknesses
1. This idea of using DINO feature space metrics for bundle adjustment has already been explored in other contexts like structure from motion (see [1, 2] below). In fact, it seems like the equations in that paper are more or less equivalent to what is proposed here. I don’t think that paper is cited. 
2. The contribution seems a bit narrow here. I think this idea has been known for a while now, and is an integral part of standard bundle adjustment pipelines.  It’s just something that one would do by default for object pose refinement if they are aware of the general pose estimation literature. So I don’t think it’s adding a lot of value to write an entire paper showing that it can work well in this setting. 

[1] DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model
[2] Pixel-Perfect Structure-from-Motion with Featuremetric Refinement

### Questions
1. My main question is that what’s really new in this paper apart from applying a well known feature space bundle adjustment technique to the multi-view 6DOF object pose refinement problem? 
2. Did the authors find that the algorithm had to be changed in a crucial way for it to work?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper deals with object pose estimation given multi-view RGB inputs. The method is based on frozen features from vision foundation models (such as DINOv2). The method performs optimization based on feature-metric refinement. Then it aggregate multiple views by 3D Non-Maximum Suppression (NMS). The method is evaluated on BOP benchmarks (YCB-V, T-LESS, ITODD-MV) and shows-  improvements over single-view methods.

### Strengths
- The method shows great generalization to unseen objects since it does not need training on unseen objects.
- The performance is much stronger than baselines such as CozyPose on BOP benchmarks.
- The paper presentation is good.

### Weaknesses
- It seems that the paper is only leveraging the features from existing vision foundation models (e.g. DINOv2) to do LM optimization of feature loss. The paper is not training any new models. This is OK if the method works, but it would seem that the contribution of this paper is limited.
- It would be interesting to see how the performance would be with different vision foundation models.
- There are some other related works that also uses LM optimization on visual features to do object pose estimation (e.g. https://arxiv.org/abs/2104.00633) and are not compared against or mentioned in this paper. Not sure how much novelty this paper contains if considering other related works.

### Questions
- I wonder how sensitive the method is due to the error of camera calibration (intrinsic and extrinsic)?
- Is there any failure cases? It would be great if some failure cases are shown and analyzed.

### Soundness
4

### Presentation
4

### Contribution
2
