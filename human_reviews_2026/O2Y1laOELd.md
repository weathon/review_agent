# MVMP-HMR: Multiview Multi-Person Human Mesh Recovery Under Large Scenes with Occlusions

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Human mesh recovery (HMR) refers to recovering the human 3D meshes from images. Most existing HMR tasks focus on multi-person from a single image or a single person from multiple views. And the evaluation benchmarks used in these methods usually contain quite small numbers of humans or under small scenes, which is unreliable for real applications with severe occlusions. Thus, we present Multiview Multi-Person HMR (MVMP-HMR), a multiview model for multi-person whole-body human mesh recovery from multi-view images under occluded scenes. Specifically, MVMP-HMR first fuses multiple views to obtain a 3D feature volume for all persons, and then the pelvis joint from a 3D pose estimation net is utilized to acquire the human query of each person from the 3D feature volume. Finally, the human queries are cross-attentioned with the 3D feature volume and integrated to decode each person's 3D meshes. Besides, two novel losses are put forward to further enhance the model performance: the orientation loss and the 3D joint density loss, dealing with the orientation and pose ambiguities in the mesh predictions under the occluded scenes. Furthermore, a large synthetic MVMP-HMR dataset is proposed, which consists of 15 multiview scenes with up to 50 camera views and 30 persons. Experiments demonstrate that the existing state-of-the-art (SOTA) HMR methods cannot perform well on the proposed large MVMP-HMR benchmark, and the proposed MVMP-HMR model's advantages over existing SOTAs under large scenes with severe occlusions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an approach for multi-person whole-body human mesh recovery model from multiview images and introduce a multiview HMR benchmark with multiple persons in large occluded scenes. The key idea is to first fuses multiple views to obtain
a 3D feature volume for all persons, and then the pelvis joint from a 3D pose estimation net is utilized to acquire the human query of each person from the 3D feature volume. Finally, the human queries are cross-attentioned with the 3D feature volume and integrated to decode each person’s 3D meshes. A orientation loss and a 3D joint density loss is developed for training.

### Strengths
This work introduces multiview multi-person HMR task under large scenes with severe occlusions, and accordingly develop a method for MVMP-HMR. 

A large MVMP-HMR dataset is introduced for training and evaluation. 

Experiments demonstrate the superiority of the proposed method. 

The paper is well-written and easy to read.

### Weaknesses
The paper is submitted to the track of 'datasets and benchmarks', while the authors highlight the method throughout the paper without introducing and analyzing the dataset in detail (including factors such as scene complexity, diversity, occlusion ratio, number of views, count of person, etc).

In my opinion, the proposed method actually has very limtied novelty. Most technical components are off-the-shelf. The orientation loss is devised because the collected dataset has the correpsonding supervision, with very limited contribution. 

The experiments are not convincing. First of all, most compared methods are tailored for single-view HMR, rather than multi-view HMR. Besides, some recent multi-view HMR methods are not compared, e.g., HeatFormer. 

The dataset is contructed with the HMR method of Baradel et al. 2024. Therefore, a direct comparison with the method is necessary. 

I doubt how the proposed method generalizes to unseen data, as the collected data contains only about 60K frames covering 15 scenes from GTA-V. 

The paper lacks visual ablation studies for validation. 

Discussion on the computational complexity and efficiency should be included.

### Questions
How does the number of views affect the performance of the proposed method?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles an interesting problem of regressing multi-person 3D pose, shape, and position from multi-view images, with a particular focus on handling occlusions in large-scale scenes. The contributions can be summarized in two parts. First, the authors construct a synthetic multi-view multi-person dataset to facilitate the training and evaluation of their approach. Unlike BEDLAM, which provides ground-truth 3D meshes, their dataset is generated from GTA-V, and the SMPL-X annotations are derived by projecting the results of Multi-HMR from camera coordinates to the world coordinate system. Second, the paper proposes a model that fuses multi-view features to localize each person’s 3D position in the scene. These estimated positions are then used to sample the corresponding feature vectors for SMPL-X parameter regression. Experiments are conducted on the proposed synthetic dataset, Human3.6M, and CMU Panoptic. Notably, instead of using the ground-truth annotations of Human3.6M and Panoptic for evaluation, the authors follow the same strategy as for their synthetic dataset by employing Multi-HMR’s predictions as pseudo ground truth to compute the number in Table.

### Strengths
1.The problem addressed in this paper is interesting and relatively underexplored in the current literature.
2.The idea of tackling this challenging task through the joint design of a dedicated dataset and a corresponding algorithm is reasonable.

### Weaknesses
1. Quantitative evaluation lacks rigor, leading to concerns about result reliability.
In experiments on Human3.6M and CMU Panoptic, the authors do not use the original ground-truth annotations provided by the datasets. Instead, they adopt pseudo ground truth generated by Multi-HMR to compute metrics. Since the accuracy of these pseudo labels is unverified, the reported quantitative results have limited credibility. For the same reason, the usefulness of the proposed synthetic dataset, which is built upon such pseudo annotations, is also questionable.

2. Qualitative results are not fully convincing.
For large-scale scene settings, it would be more appropriate to compare against methods (like CrowdRec) trained on datasets like GigaCrowd. Moreover, according to the visualization in Figure 6, it remains unclear whether the compared methods were correctly implemented and executed, raising doubts about the fairness of the comparisons.

3. The claimed contributions are somewhat overstated.
There already exist several multi-view human mesh recovery methods for single-person scenarios (e.g., HeatFormer) and multi-person 3D pose estimation approaches (e.g., VoxelTrack). Therefore, the statement “No existing research has focused on this issue in the HMR area” is inaccurate and should be moderated.

4. Model design lacks sufficient justification and comparative analysis.
The proposed architecture separates spatial localization from 3D human parameter regression, which is conceptually similar to BEV's designs. However, the paper does not provide adequate qualitative or quantitative comparisons to demonstrate the advantages of this design choice.

### Questions
Could the authors elaborate on how the Multi-HMR predictions are processed to serve as ground-truth annotations in the synthetic dataset? Specifically, is any multi-view fusion or optimization applied to refine these estimates, or are they simply selected based on the best-matching 2D projection from a single view?

What is the rationale for using Multi-HMR-generated pseudo ground truth instead of the official ground-truth annotations provided by datasets such as Human3.6M or CMU Panoptic? It would be helpful to clarify the motivation behind this choice, as the substitution may significantly affect the reliability of the reported metrics.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces MVMP-HMR, a novel framework for multiview multi-person human mesh recovery in large-scale, heavily occluded scenes. The key contributions are three folds: 1. A multiview fusion model that projects single-view features into a 3D volume, samples human queries using pelvis joints, and decodes SMPL-X parameters via a Human Transformer Block (HTB). 2. novel losses: an orientation loss (based on joint-derived body axes) and a 3D joint density loss (using Gaussian-smoothed heatmaps), which mitigate orientation and pose ambiguities under occlusion. 3. A synthetic dataset, MVMP-HMR, generated via GTA-V, featuring up to 30 people, 50 camera views, and diverse outdoor scenes with realistic occlusions. Experiments show MVMP-HMR outperforms single-view HMR and 3D pose estimation baselines on the proposed benchmark and existing datasets (Human3.6M, Panoptic). Ablation studies validate the contributions of the new losses, feature fusion strategies, and architecture choices.

### Strengths
1. The dataset is useful to the human pose esitmation community. Before I have never seen a dataset with so many people identities in a scene.

### Weaknesses
Major questions: 

1.	The inference time should be mentioned. The whole pipeline seems to be heavy, and the time consumed may be proportional to the person numbers in the scene. 

2.	It seems that the paper does not mention how the MPJPE computed. Was it computed on the joint locations of SMPL-X, or the 3D keypoints (e.g. 98 3d body keypoints of the proposed dataset, the keypoint annotations of Human3.6M)? 

3.	The experiment has flaws. (1) As a multi-view pipeline, only one multi-view method VoxelPose is compared. There are too many papers regarding the multi-view human pose estimation, such as “4D association[1]”, “mvpose[2]”, “Avatarpose[3]”, “GeoAvatar[4]”, etc. However, the paper only compares to single-view human mesh recovery methods with simple extension to multi-view fusion. Such comparison can not demonstrate the superiority of the proposed method. 

4.	As the paper predict SMPL-X instead of SMPL, a detailed comparison on hands and faces is necessary. Otherwise, there is no need to use SMPL-X. 

[1] 4d association graph for realtime multi-person motion capture using multiple video cameras,” in CVPR, 2020.

[2] “Fast and robust multi-person 3d pose estimation and tracking from multiple views,” TPAMI, 2021.

[3] AvatarPose: Avatar-guided 3d pose estimation of close human interaction from sparse multi-view videos. ECCV 2024. 

[4] GeoAvatar: Geometrically-Consistent Multi-Person Avatar Reconstruction from Sparse Multi-View Videos, CVPR 2025. 

5. As the paper title highlight two key words: "large scene" and "occlusions", the paper does not show any experiments regarding "large scene" or "occlusions". If compared with single view methods, the multi-view information naturally handles occlusions to some degree. But  nothing new is designed to further deal with it.

6. Experiments on the number of camera views are necessary to evaluate the effectiveness and generalization ability of the paper. The proposed dataset contains 50views, therefore supporting such experiments. 

Minor: 

1.	What’s the unit of metrics in Table.2? 

2.	At Line. 181, “eg.” -> “e.g.” 

3.	At L.178, “joints outputted from a 3D pose estimation branch…” As I may understand, the “joints” only contain pelvis joint? The “joints” expression may guide readers to feel that more than one joints were predicted for each body.

### Questions
This paper has critical experimental flaws. I do not think that the authors could fully address them within a short revision period.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
- The paper addresses the problem of multi-view, multi-person mesh recovery, with a special focus on occlusion and crowding.
- MVMP-HMR is a multi-view mesh regression model with the following stages: (1) multi-view feature extraction; (2) projection to construct a 3D feature volume; (3) in parallel, a 2D pose + 3D root estimation network predicts each person’s 3D root; (4) query the 3D feature volume at the predicted roots; (5) a transformer decoder regresses per-person mesh parameters from the queried features.
- The key idea is multi-view aggregation with per-person mesh decoding.
- As related datasets do not exist at scale, MVMP-HMR contributes a large-scale synthetic dataset with 10–30 subjects and 50 camera views, covering a large outdoor scene (~30 m × 30 m).
- Evaluations are conducted on multi-view datasets: the MVMP-HMR test set, Human3.6M, and Panoptic Studio. Key baselines include TokenHMR, Multi-HMR, and VoxelSMPLX.
- The quantitative evaluations show consistent improvements over baselines across datasets.

### Strengths
- The paper is well organized, clear, and narratively sound; technical details are presented with intuitive explanations.
- The problem of simultaneous multi-view, multi-person mesh recovery is challenging; the work adopts a popular architecture to tackle it and reports promising results.
- The paper also contributes a synthetic dataset that could benefit the community (if done right); current datasets lack the proposed scale in scene extent, camera views, and number of subjects.

### Weaknesses
- Limited Technical Novelty: MVMP-HMR’s core design—multi-view aggregation with root-centric decoding—closely follows prior multi-view 3D keypoint estimation frameworks like VoxelPose (ECCV 2020), TesseTrack (CVPR 2021) etc. The primary change is predicting mesh parameters instead of 3D keypoints; thus, the technical contribution is limited and incremental.

- Generalization to various multi-view setups: Sec. 3.2 mentions that the construction of the 3D feature volume consists of using provided camera intrinsics and extrinsics, making the system sensitive to the training camera distribution. As far as I can tell, the method is evaluated only (not trained) on Human3.6M and CMU Panoptic Studio. While this shows some generalization over baselines, the absolute metrics are notably lower than 3D keypoint localization methods. It would be helpful to demonstrate the generalization ability of the model on the in-the-wild multi-view camera configurations from Ego-Exo4D (CVPR 2024) or EgoHumans (ICCV 2023).

- Ground-truth Quality: Despite procedural synthesis, the mesh annotations are derived via heuristics and off-the-shelf methods, raising concerns about quality of annotations at scale. To truly take advantage of the synthetic setup, it would be beneficial to directly pose and cloth and rig characters and exporting mesh parameters, which is commonly done in popular synthetic mesh datasets like AGORA and BEDLAM. Please elaborate on why pseudo ground truth was chosen and what quality checks are in place.

- Missing Qualitative Results: The paper lacks qualitative results on real images. Figure 5 shows results only synthetic images (no supplementary). Please consider adding qualitative comparisons on real datasets (Human3.6M, CMU Panoptic Studio) to highlight improvements over baselines (e.g., VoxelSMPLX).

### Questions
My questions are primarily centered on the weaknesses mentioned above:

1. How would you distinguish MVMP-HMR's architecture compared to existing multi-view 3D pose methods?
2. Please provide evidence of generalization to out-of-distribution camera configurations.
3. Why pseudo ground truth was chosen and what quality checks are in place?
4. Please consider demonstrating the performance qualitatively on real images.

### Soundness
3

### Presentation
2

### Contribution
2
