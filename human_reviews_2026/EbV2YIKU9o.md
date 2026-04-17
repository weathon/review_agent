# 3DRot: 3D Rotation Augmentation for RGB-Based 3D Tasks

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
RGB-based 3D tasks, e.g., 3D detection, depth estimation, 3D keypoint estimation, still suffer from scarce, expensive annotations and a thin augmentation toolbox, since many image transforms, including rotations and warps, disrupt geometric consistency.
In this paper, we introduce 3DRot, a plug-and-play augmentation that rotates and mirrors images about the camera's optical center while synchronously updating RGB images, camera intrinsics, object poses, and 3D annotations to preserve projective geometry, achieving geometry-consistent rotations and reflections without relying on any scene depth.
We first validate 3DRot on a classical RGB-based 3D task, monocular 3D detection. On SUN RGB-D, inserting 3DRot into a frozen DINO-X + Cube R-CNN pipeline raises $IoU_{3D}$ from 43.21 to 44.51, cuts rotation error (ROT) from 22.91$^\circ$ to 20.93$^\circ$, and boosts $mAP_{0.5}$ from 35.70 to 38.11; smaller but consistent gains appear on a cross-domain IN10 split. \rev{Beyond monocular detection, adding 3DRot on top of the standard BTS augmentation schedule further improves NYU Depth v2 from 0.1783 to 0.1685 in abs-rel (and 0.7472 to 0.7548 in $\delta<1.25$), and reduces cross-dataset error on SUN RGB-D. On KITTI, applying the same camera-centric rotations in MVX-Net (LiDAR+RGB) raises moderate 3D AP from about 63.85 to 65.16 while remaining compatible with standard 3D augmentations. Because it operates purely through camera-space transforms, 3DRot drops into diverse RGB-based 3D tasks and multi-modal pipelines without architectural changes or depth supervision.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes 3DRot, which introduces a 3D rotation augmentation for RGB-based 3D tasks by rotating/mirroring scenes around the camera's optical center while preserving projective geometry through synchronized updates of RGB images, camera intrinsics, and 3D annotations.  This method provides a data-efficient solution that significantly improved performance on monocular 3D detection, achieving gains comparable to multi-dataset training using only a single dataset.

### Strengths
++ The augmentation does not require depth information.

++ The 3DRot augmentation has been validated to be effective on the RGB-based 3D detection task.

### Weaknesses
-- While the authors acknowledge that standard flipping (reflection) augmentation violates chirality, they claim to use a chirality-preserving method. However, the methodology is only briefly mentioned in the ablation study. A detailed explanation of how chirality is preserved should be provided in the Method section to ensure clarity and readability.

-- The depths and NOCS maps in Figure 2 do not appear to be utilized in the experiments. It seems necessary to incorporate them into relevant tasks for coherence; otherwise, they may seem superfluous.

-- The claim of cross-domain generalization is not sufficiently substantiated. The experimental validation is limited to a single task and two datasets, raising concerns that the reported gains may be specific to that particular experimental setup. To robustly support this claim, the authors should demonstrate the effectiveness of their method across a wider range of tasks and datasets. Furthermore, the potential need to re-tune parameters of the augmentation strategy for different domains remains an open question.

-- Overly simple derivations and basic formulas can be omitted from the main text to free up space for experimental details.

-- A limitation of the study is the absence of a comparison with other augmentation techniques like 3D copy-paste, both in isolation and combined. This omission makes it difficult to discern the specific advantages of the proposed method.


-- A discussion of the limitations and potential failure cases of the proposed augmentation method should be included.

Minor: 

-- The dimensionality is usually denoted using \mathbb{R}, but this paper uses several different (and inconsistent) notations.

### Questions
-- The method relies on camera intrinsic parameters. How would its performance be affected by approximated or inaccurate camera intrinsics?

### Soundness
2

### Presentation
1

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
The paper proposes a geometric image augmentation for RGB-based 3D vision tasks (e.g., 3D detection, 3D keypoint estimation). The augmentation rotates the camera around its optical center in a way that preserves projective geometry and therefore the 2D–3D correspondences between images and 3D labels. The authors give a mathematical derivation of the augmentation and present experiments showing improved performance on indoor 3D detection benchmarks when used with a Cube R-CNN detector and a DINO-X backbone.

### Strengths
-	Rigorous derivation: The mathematical formulation that guarantees preservation of 2D–3D correspondences is clean and convincing.
-	Empirical ablation: The paper includes ablations about flipping/rotation configurations and shows that the proposed rotation improves accuracy compared to no-rotation.
-	Conceptual simplicity: The idea is intuitive and could be widely useful if shown to be robust across models and datasets.

### Weaknesses
Limited Evaluation:

- The authors did not compare their method with alternative augmentation strategies. Only the relative gain against no augmentation is reported. Other augmentations could include non-geometric augmentation (jittering) or geometric ones as discussed in related work.
- The biggest gain in performance in Table 3a) is achieved by leveraging DINO-X as the backbone of Cube R-CNN which is not the contribution of this paper.
- The augmentation strategy is only evaluated on one model (DINO-X + Cube R-CNN).

Limited Contribution:

-	Based on the conducted evaluation, the contribution of this work seems to be limited to the community.

### Questions
-	Is there a reason that 3DRot is only evaluated on indoor scenes? Does the technique also apply for outdoor scenes (e.g. KITTI dataset) for example?
-	What are the boundaries for the rotation angles? As their absolute value increases, more and more pixels are padded, hindering model’s training.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a data augmentation method that work on the rotation of the viewing direction of the camera. The method rotates and mirrors images about the camera's optical center. The method does not need depth information. The author introduces the math foundations for the transformations of pixel locations based on which the remap is defined. The data augmentation method is evaluated by the performance gain of the 3D detection task.

### Strengths
- The idea of augmenting the data using camera view direction change is a nice and interesting idea.
- The authors presented detailed math derivation to backup the method.
- The method does not need depth information.
- The performance gain is notable on 3D detection task on various datasets.

### Weaknesses
- I would suggest the authors to take a closer look at related works. It would be quite surprising if the same or similar idea has never been used by others. It is likely that people already used this augmentation method for quite a long time yet did not publish it because it seems obvious and trivial. I would suggest the authors put more effort in investigating the prior works.
- Though depth information is not required, this method still needs camera intrinsics. Such information may not be available for random image data. So this may limited the scope of this method being useful.
- I am not sure if 3D detection is a good task for evaluating the data augmentation method, since the performance gain seems to be modest. Would other tasks be better where the performance gain is more significant?

### Questions
- Roll rotation seems to be OK which is what vanilla random rotation augmentation already did. But yaw and pitch may have some problems. The principal point is supposed to be at the center of any image. But after changing the view direction, the principal point will not be at the center. How did the authors deal with such situation? Maybe this has been mentioned in the paper but I may have missed it.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a 3D rotation augmentation for monocular 3D object detection. The main proposed design is to apply 3D rotations and reflections on the camera's optical center, and update the camera intrinsics, 3D annotations, and RGB images accordingly. Experimental results are conducted on the 3D object detection task to validate the effectiveness of the proposed method.

### Strengths
- The proposed 3D rotation augmentation is reasonable and supported by theoretical derivations and proof.
- Experimental results on the 3D detection task show the effectiveness of the proposed augmentation.
- The paper is well written and easy to follow.

### Weaknesses
- The experimental evaluation of the proposed augmentation is not thorough:
  - The paper claims the augmentation for RGB-based 3D tasks, but only the 3D object detection task is evaluated. Evaluation on more tasks, e.g., monocular depth estimation, would be beneficial. 
  - The 3D object detection is only conducted on 10 categories and only on indoor SUN RGB-D. Evaluation on more diverse categories and larger datasets that include both indoor and outdoor scenes will improve the validation of the generalizability of the proposed method. 
  - Only one algorithm framework is utilized for the benchmark. Can this augmentation benefit more 3D detection frameworks?
- Considering that more and more large-scale and diverse datasets are proposed, e.g., ScanNet, Matterport3D, and ARKitScenes, it's not clear whether the proposed augmentation still provides a large performance gain in these larger-scale real-world datasets.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
