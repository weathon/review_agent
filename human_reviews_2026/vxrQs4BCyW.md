# LiDAR-Anchored Collaborative Distillation for Robust 2D Representations

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
As deep learning continues to advance, self-supervised learning has made considerable strides. It allows 2D image encoders to extract useful features for various downstream tasks, including those related to vision-based systems. Nevertheless, pre-trained 2D image encoders fall short in conducting the task under noisy and adverse weather conditions beyond clear daytime scenes, which require for robust visual perception. To address these issues, we propose a novel self-supervised approach, Collaborative Distillation, which leverages 3D LiDAR as self-supervision to improve robustness to noisy and adverse weather conditions in 2D image encoders while retaining their original capabilities. Our method outperforms competing methods in various downstream tasks across diverse conditions and exhibits strong generalization ability. In addition, our method also improves 3D awareness stemming from LiDAR’s characteristics. This advancement highlights our method’s practicality and adaptability in real-world scenarios. The code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Collaborative Distillation to improve the adaption ability of 2D image encoder with 3D awareness by leveraging 3D LiDAR as self-supervision. It improves the robustness of 2D encoders to noisy and adverse weather conditions. Experiments show that this approach outperforms competing methods in various downstream tasks, highlighting the capability in real-world scenarios.

### Strengths
1.	This paper is well-written and easy-to-follow. Figures are clear for the visualization of design and effectiveness.
2.	It is reasonable to utilize LiDAR distillation to improve the robustness of the domain adaptation ability of the 2D vision encoder.
3.	Experiments on 2D semantic segmentation and monocular depth estimation are extensive to show the effectiveness of collaborative distillation.

### Weaknesses
1.	As adverse conditions are utilized for self-supervised training, the setting in your paper seems to belong to the field of (unsupervised) domain adaptation, which has been investigated previously, like [1][2][3]. Or, the distinction of these settings should be illustrated clearly.
2.	In experiments, only the CD approach is utilized for comparison based on the visual foundation models: ViT pretrained on DINOv2. Not aiming for domain adaptation, a lot of classic self-supervised representation learning approaches have been proposed. These approaches should also be compared to verify your specific ability for the robustness of domain adaptation.
3.	Experiments are conducted based on DINOv2. More baselines and pre-trained foundation models can be applied to verify the generalization ability of your design.
4.	The correspondence between 2D and 3D and Distillation from LiDAR to image has been investigated in previous perception works in autonomous driving, which is not new for the representation learning approach. 

[1] Feng, Zeyu, Chang Xu, and Dacheng Tao. "Self-supervised representation learning from multi-domain data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
[2] Achituve, Idan, Haggai Maron, and Gal Chechik. "Self-supervised learning for domain adaptation on point clouds." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.
[3] Xu, Jiaolong, Liang Xiao, and Antonio M. López. "Self-supervised domain adaptation for computer vision tasks." IEEE Access 7 (2019): 156694-156706.

### Questions
See Above

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
4

### Summary
This manuscript introduces LiDAR-Anchored Collaborative Distillation (CD), a two-stage self-supervised framework designed to improve the robustness of 2D image encoders under adverse weather and noisy conditions. The key idea is to leverage LiDAR representations as stable self-supervision anchors for 2D encoders.

- Stage 1 (Pre-alignment): Aligns LiDAR features with 2D clear-weather features to create a reliable 3D anchor space.
- Stage 2 (3D-anchored supervision): Uses these aligned 3D features to pull degraded 2D features (e.g., night, rain, noise) toward the clean 2D feature space.

The manuscript claims consistent improvements over DINOv2 baselines across multiple downstream tasks (semantic segmentation, depth estimation, and video panoptic segmentation), and generalizes well to both indoor and outdoor out-of-domain datasets.

### Strengths
(+) The manuscript addresses an important and realistic gap: the lack of robustness of vision foundation models (e.g., DINOv2) under adverse conditions. The motivation for using LiDAR as a more stable and weather-invariant modality for self-supervision is clearly justified and timely.

(+) Extensive experiments on nuScenes, nuImages, and some other benchmarks (KITTI, NYUd, ADE20k, Cityscapes) confirm that CD consistently improves robustness and generalization. The use of both few-shot and full-label setups adds credibility to the findings.

(+) The manuscript provides several clear figures showing the feature distribution shift toward stable regions. Ablation studies (Stage 1 effect, corruption robustness, etc.) are thorough and support the methodological choices.

### Weaknesses
(-) While the framework is well-motivated and cleanly executed, it primarily combines known techniques — cross-modal distillation, bidirectional matching, and stop-gradient supervision — into a new use case. The innovation lies more in formulation and application (LiDAR as a teacher for robustness) than in new architecture or loss design.

(-) Lack of discussions with several closely related works. There is a line of 2D-3D cross-modal knowledge transfer work, which, in this manuscript, was omitted for discussions and/or comparisons. Notable works include xMUDA (CVPR'20), xMoSSDA (TPAMI'22), CLIP2Scene (CVPR'23), SuperFlow (ECCV'24), ScaLR (CVPR'24), and LiMoE (CVPR'25). The authors are suggested to discuss these closely related works to ensure the claims made are grounded.

(-) The approach assumes accurate calibration between LiDAR and cameras, which may not always hold in real-world deployment. Discussion of calibration noise, sparse LiDAR data, or generalization to other modalities (e.g., radar) would strengthen applicability claims.

(-) The manuscript shows consistent performance gains but does not discuss computational overhead, convergence stability, or potential degradation when LiDAR data is unavailable during pretraining.

### Questions
In addition to the weaknesses mentioned in the above section, the authors are suggested to clarify on the following minor questions:

(-) Carify whether the “bi-directional distillation” uses shared or separate projection heads and if the stop-gradient operation is symmetric or alternated during training.

(-) Include a small analysis on training stability or feature collapse prevention under the two-stage setup.

(-) Discuss failure cases more concretely (e.g., scenarios where LiDAR sparsity or missing points weaken supervision).

(-) In Fig. 1 and 2, consider explicitly labeling “Stage 1” and “Stage 2” arrows for readability.

(-) In Section 4.3, briefly explain why improvements are stronger on indoor OOD datasets despite pretraining on outdoor-only data.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a self-supervised learning pipeline, called Collaborative Distillation, which leverages the ToF sensor's low sensitivity to the adverse weather conditions for the 2D RGB image representation.
The strategy is two-staged. Align the 3D ToF sensor's pointcloud encoder to the pre-trained 2D RGB representation in the clear image, and distill the 3D ToF sensor's pointcloud encoder to the 2D RGB representation of the scenes with adverse conditions. The self-supervised training strategy is utilized for the various semantic tasks (depth estimation, segmentation).

### Strengths
The idea is simple and straightforward to understand.
The figure is easy to understand, and the paper is easy to read.,
The work leverages complementary sensor properties, and this is considered to be a viable way to solve the nuisance variables.

### Weaknesses
The selection of the Dinov2 is questionable.
The evaluation dataset is limited to the nuScenes dataset.
The t-SNE plot (Fig. 3) is not convincing.
The Waymo dataset is considered another set to test out, which has adverse weather conditions as well.

### Questions
1. Usage of the DINOv2.
Why DINOv2?: DINOv2 is supposed to learn the invariant feature under the photometric/geometric perturbations, which may occur in adverse weather conditions as well. Therefore, the linear probing of the DINOv2 feature would not be adequate (given that DepthAnything also trains the DINOv2 features). Another point is that the training data of DINOv2. DINOv2 is trained on object-centric images, not the outdoor driving scenes. Therefore, it is obvious that DINOv2 cannot perform better than DINOv2+ and Ours. From the reviewer's perspective, this undermines the proposed method's significance. Could the authors clarify?

2. What does the t-SNE plot (Fig. 3) imply?
The left two plots (Dino, Dinov2) divide the adverse weather data, and the right plot shows the self-supervised model dividing the adverse weather data into two chunks, and the right does not seem to be better, given that the t-SNE plot is a dimensionality reduction method to plot the high-dimensional data space into just a two-dimensional space. What is the purpose of the Fig. 3?

3. Monocular Depth Estimation model trained with the proposed method?
Connected to question 1, the reviewer was wondering if the proposed self-supervised method can generalize to the MDE models?

4. The sensitivity study on the LiDAR quality
The proposed method leverages the complementary characteristics of the camera and LiDAR, where the camera is dense but sensitive to the domain shift [1], and the LIDAR is robust to the nuisance variables but sparse. This implies that stage 2 will be affected by the LiDAR's sparsity and noise level.
Based on the question, the reviewer hopes to see the sensitivity study on the sparsity/noise of the LiDAR data.

[1] Park et al., "Test-Time Adaptation for Depth Completion," CVPR 2024

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Collaborative Distillation (CD), a self-supervised method to improve the robustness of pretrained 2D image encoders to adverse weather (e.g., rain, low-light, night, corruptions) thanks to guidance from Lidar. The authors argue that 2D models fail because they are pretrained mostly on clear-day images. CD leverages robust 3D Lidar information in a two-stage process. In stage 1, a 3D encoder is aligned to a frozen 2D encoder's clear-day features, creating a "reliable anchor". In stage 2, the 2D encoder is finetuned on all available data (including clear and adverse) by distillation from the frozen pretrained 3D encoder (anchor). This process "pulls" the 2D features from adverse conditions toward the stable, clear-day feature distribution.

The CD-enhanced 2D encoder is evaluated on multiple downstream tasks, semantic segmentation and depth estimation, demonstrating improved robustness in-domain (on nuScenes) and strong out-of-domain generalization (on KITTI, NYUd, Cityscapes, ADE20K).

### Strengths
**Significance**
- This work deals with an important and highly practical problem for real-world applications like autonomous driving: the reliability of perception models under challenging environmental conditions.
- This is definitely an area that is worth studying further


**Clarity**
- I find the paper well written with multiple visualizations, qualitative visualizations, and detailed implementation information.
- The paper is in general easy to follow and the reasoning of the authors is clear.


**Quality**
- The CD approach is evaluated under several settings, in-distribution and out-of-distribution, on multiple tasks and datasets and distribution shifts
- There are numerous other experiments in the appendix with different pretrained 2D encoders and data splits
- CD shows consistent performance boosts across multiple downstream settings, different tasks and datasets

### Weaknesses
**Not completely valid assumptions**
- The authors argue that 3D Lidar representations are highly robust to adverse weather conditions compared to 2D image representations. While this is true for night time conditions, Lidar is quite brittle under rain, fog and snow [a], [b], [c].
- Lidar does shine however in precise 3D information and geometry which might be in fact here one of the main contributing factors to performance boosts.
- The authors assume that the LVD-142M dataset upon which DINOv2 was pretrained, is predominantly composed of clear daytime images. The dataset is private and we cannot really know it composition. It could be clear images, but also night, indoor, outdoor, etc., but we can only guess.

**Limited scope of experiments**
- The authors argue the practical side of this approach and its adaptability to real-world conditions. However, from an autonomous driving standpoint (the type of data that CD needs, paired image and lidar data, is currently found mostly in this type of application), the experimental setup is not convincing.
- The bulk of the in-domain experiments are conducted on nuscenes 2D semantic segmentation. Here the annotated point clouds are projected in the image space and used as ground truth labels. However the point cloud is actually sparse on nuscenes (32-beams lidar) and many of the pixels are not labelled for both training and testing, making it quite an imperfect setting. In fact this protocol is actually established by the authors and not really used in the community.
- Relevant experiments on automotive datasets like nuscenes would be BEV 3D object detection or BEV semantic segmentation, as done in several of the related works in this area, such as the ones mentioned in related works section "Improving 3D awareness of 2D representation"

- To go further OOD the authors leverage the synthetic images from Cityscape-DVPS converted to rainy and night with Stable Diffusion. However there are specific Cityscapes related datasets (for semantic segmentation at least) with adverse weather conditions (night, rain, fog, snow) ACDC [f], multiple corruptions Cityscapes-C [g], with aggregated distributions shifts BRAVO challenge [h]. They could be used directly as they are just additional test sets for Cityscapes-trained models.

**Limited scope of baselines**
- In-domain experiments use just DINOv2 as baseline. However the authors mention in related work different methods that improve 3D awareness of 2D representations: UniPAD, BEVDistill, DistillBEV etc., but also OccFeat [i]  (not cited, but related) that distills DINOv2 features projected on point clouds
 
- The depth estimation experiments are lacking widely-used baselines such as DepthAnythingV2 (it produces relative depth but could be rescaled or fine-tuned)[d] or DepthPro (producing absolute depths)[e] 

- The comparisons with other methods is mostly done on OOD settings but even there it's not quite apple-to-apples as the considered methods (FiT3D, Condense) do not use Lidar supervision

**Contributions**
- The stage 1 of CD is presented as a contribution of the authors, however the setup is identical to the one from ScaLR (Puy et al., 2024), excepting the fact that CD trains only on daytime data
- The idea of distilling a Lidar encoder in image encoders has been explored thoroughly in the literature (many of the methods are in fact cited by the authors in the related work) and the approach itself is not really novel. What is novel is the use of self-supervised Lidar encoders and the splitting of the data across stages (day images in stage 1, full data in stage 2).
- For the latter I'm not fully convinced about the claim as it might be specific to nuscenes. It would be informative to test the distillation on other datasets, as done in ScaLR (Puy et al.), where the authors run image to lidar distillation on KITTI, Pandar 64, Pandar GT, etc.
- Besides I'm wondering what if in stage 2, CD uses directly the ScaLR checkpoint distilled trained on several datasets and lidars? The authors show better generalization in this way.


**Missing related works**
- OccFeat [i] is pre-trained by distilling DINO features projected on point clouds, towards improving multi-camera BEV perceptions
- Image-lidar co-teaching has been proposed also for unsupervised domain adaptation with results on both 3D and 2D semantic segmentation [j]


**Minor Misc.**
- Typos: 
    + L051: faces
    + in C3, nuscenes is mentioned for the OOD datasets
    + in table 3, the meaning of dagger is not explained

- In general pretraining DINOv2 on automative data as it is, is not expected to provide good results as the content and image sizes are very different. See [k] for a discussion

- The t-sne plot does not look convincing as the two distributions are still quite separated

**References:**

[a] Bijelic et al., A Benchmark for Lidar Sensors in Fog: Is Detection Breaking Down?, arXiv 2019

[b] Bijleic et al., Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather, CVPR 2020

[c] Linnhoff et al., Simulating Road Spray Effects in Automotive Lidar Sensor Models, IEEE Sensors 2022

[d] Yang et al., Depth Anything V2, NeurIPS 2024

[e] Bochkovskii et al., Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, ICLR 2025

[f] Sakaridis et al., ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding, ICCV 2021

[g] Franchi et al., Robust Semantic Segmentation with Superpixel-Mix, BMVC 2021

[h] Vu et al., The BRAVO Semantic Segmentation Challenge Results in UNCV2024, ECCV workshops 2024

[i] Sirko-Galuchenko et al., OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks, CVPR workshops 2024

[j] Jaritz et al., xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation, CVPR 2020

[k] Chen et al., MultiSiam: Self-supervised Multi-instance Siamese Representation Learning for Autonomous Driving, ICCV 2021

### Questions
This paper takes an interesting direction of study: how to improve 2D representations by taking useful cues and signals from 3D representations of Lidar points clouds.

I find the endeavor of the authors nice, with a good story. However I do have several concerns regarding the novelty of the method, the chosen experimental settings and the limited to no comparison with relevant baselines in more standard tasks such as BEV perception, 3D object detection.

My current rating is leaning towards reject at this time, but I'm looking forward for the rebuttal.

Here are a few questions and suggestions that could be potentially addressed in the rebuttal or in future versions of this work (please note that suggested experiments are not necessarily expected to be conducted for the rebuttal):

1. Study the impact of stage 1 data split, but using stronger 3D encoder for stage 2, e.g., ScaLR trained on multiple datasets.

2. Inclusion of semantic segmentation evaluation on actual rain, fog, snow images. The model trained on Cityscapes can be directly tested on ACDC for that.

3. Comparison with DepthAnythingv2 on depth estimation tasks. What it DepthAnythingv2 was used in CD in place of DINOv2?

4. What if the 2D encoder and the 3D encoder are trained together without the alternate freezing, like in UniPad?

5. Extension of automotive experiments to BEV semantic segmentation and 3D object detection and comparison with relevant methods

### Soundness
2

### Presentation
3

### Contribution
2
