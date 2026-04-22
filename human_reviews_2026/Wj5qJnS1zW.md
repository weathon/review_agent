# CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Cloth Dynamics Splatting (CloDS), an unsupervised, visual-only framework for Cloth Dynamics Grounding, which aims to learn cloth dynamics from multi-view videos under unknown conditions without direct physical supervision. The core strength lies in its ability to bridge the gap between 2D visual observations and 3D physical representations for highly deformable materials, a significant challenge in the field.

### Strengths
- The paper clearly defines a new and challenging problem, Cloth Dynamics Grounding, which focuses on unsupervised learning of cloth dynamics solely from visual data.

- The introduction of Spatial Mapping Gaussian Splatting, a mesh-based Gaussian splatting module, provides a differentiable mapping between 2D pixel space and 3D geometry. The proposed dual-position opacity modulation in SMGS is a clever solution to address severe self-occlusions and large non-linear deformations inherent to cloth dynamics. For the mesh-based gaussian splatting, there are relative work that should be cited.

[1]SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

[2]real-time large-scale deformation of gaussian splatting

[3]VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality

[4]Recent Advances in 3D Gaussian Splatting

- CloDS achieves performance close to fully mesh-supervised methods on the CDG task, demonstrating effective unsupervised dynamics learning.

### Weaknesses
- The method assumes an initial mesh state ($M_1$) is available to build the initial Gaussian component representation. Although robustness to initial mesh errors is analyzed, I still suggest that some visual results should be prepared and presented to incorporate the results reported in FigureS.2.

- Performance degrades under complex lighting conditions due to temporal inconsistency caused by shadows and illumination, suggesting the current approach is sensitive to visual changes beyond pure geometry and dynamics.

- How is the initial mesh $M_1$ "extracted from the initial frame $I_1^{1:N}$ via 2D Gaussian Splatting"? Is this step fully unsupervised, or does it rely on any pre-trained model, shape priors, or a fixed template mesh? A brief explanation of how $M_1$ is obtained would clarify the unsupervised visual-only premise.

- The comparison to video prediction models is strong, but since a key challenge is 3D-aware modeling, a comparison to other geometry-aware, unsupervised methods (e.g., scene flow-based approaches or other particle-based visual grounding methods like NeuroFluid, adapted for cloth) would further solidify the value of the DVC and SMGS approach for this task.

- DeepFashion3D is also an impressive cloth dataset, some evaluations on the dataset are more encouraged.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper learns cloth dynamics from video. The approach first reconstructs a mesh from images via Gaussian splatting, then applies a mesh-based neural simulator to model dynamics in the mesh domain. Experiments show performance comparable to or better than selected baselines.

### Strengths
* The method learns cloth dynamics directly from video.
* The method achieves performance comparable to approaches trained on ground-truth mesh data.

### Weaknesses
1. Clarity and consistency. The writing is unnecessarily complex. If I understand correctly, a simple and clear description would be: learn dynamics directly from videos by first performing video-to-geometry grounding, then training a dynamics model on the grounded meshes. Also, there appears to be a typo/inconsistency: Equations (7) and (9) for geometry should take the same input parameters; please verify and correct.
2. Related work coverage (missing citations). Given the focus on data-driven, mesh-based cloth simulation, the Related Work should include additional existing work (e.g., [1,2,3]). In particular, [3] also learns cloth dynamics from multi-view videos and addresses more complex scenarios (richer appearance, human body interactions). Please discuss [3] in more detail and, if feasible, compare against the pipeline in [3].
3. Dataset scope and realism. The current dataset appears limited (a single cloth and ~120 videos of trajectories). Compared to [3], this setting may be simplistic. Please either expand the number/diversity of training and test videos or report results on established real-world datasets (e.g., those used in [3]) to demonstrate robustness and generality.
4. Experimental protocol and reporting. Since results are reported over 20 trajectories, include mean $\pm$ std across trajectories to reflect variability. Additionally, please report inference time (e.g., FPS/latency on a specified GPU) to quantify simulator efficiency.


[1]. Santesteban, et al. Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On. CVPR 2021
[2]. Shao, et al. Towards Multi-Layered 3D Garments Animation. ICCV 2023.
[3]. Rong, et al. Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photorealistic Appearance from Multi-View Video. 3DV 2025.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors in this paper presents a method for cloth dynamics grounding. Cloth dynamics are learned from visual observations under unknown conditions without physical supervision.

### Strengths
- The use of  Spatial Mapping Gaussian Splatting to establish a mapping between the 2D pixel space and 3D space is interesting. SMGS handles large deformations and severe self-occlusion by using both relative and absolute positions of the Gaussian components. This design ensures an accurate mapping between the 2D and 3D spaces during rendering.

### Weaknesses
- The visual results are shown under wind force, it would have been interesting to see cloth dynamics under various type snd source of forces e.g. objects colliding with cloth. How to model them inside the current framework.
- A detailed analysis on cloth-cloth collision, cloth-object collision is missing.
- Do add following relevant references under neural garment simulator GarSim: Particle Based Neural Garment Simulator WACV 2023 and GenSim: Unsupervised Generic Garment Simulator CVPR 2023

### Questions
refer to the weakness section

### Soundness
3

### Presentation
3

### Contribution
4
