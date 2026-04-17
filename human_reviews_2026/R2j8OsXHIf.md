# SG2Loc: Sequential Visual Localization on 3D Scene Graphs

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Visual localization in complex environments remains a critical challenge for robotics and AR applications. Sequential localization, where pose estimates are refined over time, is important for autonomous agents. However, traditional methods often require storing extensive image databases or point clouds, leading to significant storage overhead. This paper introduces a novel, lightweight approach to sequential visual localization using 3D scene graphs. Our method represents the environment with a compact scene graph, where nodes represent objects (with coarse meshes) and edges encode spatial relationships. For each image in the localization phase, we extract per-patch semantic features, predicting object identities. Localization is performed within a particle filter framework. Each particle, representing a camera pose, projects the coarse object meshes from the scene graph into the image, assigning object identities to patches based on visibility. The similarity of the per-patch features, in the input image, and object features from the scene graph determines the weight of a particle. Subsequent images are incorporated sequentially, refining the pose estimate. By leveraging a compact scene graph and efficient semantic matching, our method significantly reduces storage while maintaining performance on real-world datasets. The code will be made public.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a method for sequential visual localization based on compact 3D scene graphs. The approach models environments as graphs, where nodes represent coarse object meshes and edges encode spatial relations. During localization, semantic features extracted from image patches are compared with projected object identities within a particle filter framework to estimate camera poses. The method is evaluated on real-world datasets and reports comparable localization accuracy while reducing storage requirements.

### Strengths
The paper is clearly structured and generally easy to follow. The idea of leveraging a compact scene graph for sequential localization is interesting, and the paper demonstrates careful engineering and integration of existing components.

### Weaknesses
1. Motivation and scope.

The motivation of the paper is not clearly aligned with the proposed approach. The authors argue that existing scene coordinate regression (SCR) and absolute pose regression (APR) methods struggle in complex, large-scale environments (line 91, page 2), yet the experiments are limited to small, static indoor datasets such as ScanNet and 3RScan, which do not convincingly demonstrate the claimed advantages. In contrast, prior works—including SCR and APR methods—have been evaluated on more challenging benchmarks such as Cambridge Landmarks and Aachen-Day-Night, which involve illumination changes, dynamic scenes, and weather variations. In addition, according to Table 3, the proposed approach does not show a clear advantage over SCR methods in terms of storage. It is also worth noting that many existing visual localization approaches operate without explicit gravity alignment or auxiliary sensors, so the benefits of the proposed method are not fully clear.

2. Missing Related Work and Limited Novelty.

The technical novelty of this paper over SceneGraphLoc appears limited, as the proposed framework mainly extends it to handle image sequences. Furthermore, the paper omits discussion and comparison with several relevant recent works. The statement (line 92, page 2) that the method avoids storing image databases or point clouds is not a significant advantage, since many modern methods—such as SCR, APR approaches—already share this property. Moreover, these existing methods typically support full 6-DoF pose estimation, whereas the proposed method appears to handle only 4-DoF localization.

In addition, comparisons with more recent and representative baselines are missing. For instance, GLACE [1] and R-SCoRe [2] have demonstrated strong results on large-scale benchmarks (e.g., Cambridge, Aachen, Hyundai Department Store), and differentiable representation-based GS-CPR [3] achieves efficient pose refinement without image databases and point clouds. Including such baselines would make the experimental evaluation more convincing and better clarify the contribution of this work.

[1] GLACE: Global Local Accelerated Coordinate Encoding, CVPR 2024

[2] R-SCoRe: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization, CVPR 2025

[3] GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting, ICLR 2025

### Questions
If the authors believe that existing SCR or APR methods perform poorly on large and complex scenes, I strongly encourage including at least two larger and more challenging benchmarks—such as Cambridge Landmarks, Hyundai Department Store, or Aachen-Day-Night—and comparing against GS-CPR, GLACE, and R-SCoRe.

If the key contribution is improved storage efficiency or the avoidance of image/point cloud databases, please provide more thorough comparisons with recent differentiable-representation-based methods and discuss these aspects more explicitly in the related work and experiments.

Why does this paper not provide the recommended Reproducibility statement and the Use of Large Language Models (LLMs) statement?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a lightweight visual localization method for sequential image inputs, utilizing compact 3D scene graphs as the scene representation.
In a 3D scene graph, each node corresponds to an object instance, with each instance characterized by a set of multi-modal attributes including RGB frames, a point cloud, textual annotation and a coarse 3D mesh.

The authors formulate the camera localization as a particle filtering problem, where particle weights are determined by combining semantic similarity, the depth and color scores between the query image and the map projection. 
The proposed approach allows for pose searching and optimization over multiple iterations, with further refinement achieved through *RANSAC-PnP*. 

Experimental results on two public datasets demonstrate that the method achieves competitive localization performance while significantly reducing storage requirements.

### Strengths
- The concept of leveraging 3D semantic scene graphs for camera relocalization is innovative and well-justified. By compactly integrating high-level semantic relationships with geometric representations, they provide a sufficiently informative foundation for accurate localization.
- The task is formulated as an iterative particle filtering problem, where particle weights are computed through multi-modal comparisons between the query and the rendered representations (semantic, depth, and color maps).

### Weaknesses
1. The core system is conceptually simple: a standard particle filter in $SE(3)$ with a similarity-based observation model. Particle filtering for robot localization has been extensively studied, including with visual or learned models.  Although simplicity is not a drawback in itself, I'm questioning the suitability of particle filtering for this specific task: 
	1. The framework relies on a considerable number of hyper-parameters, especially the initial particle sampling strategy, which uses four predefined heights, appears quite contrived to me.
	2. The multi-round particle optimization and ray tracing operations are computationally expensive, potentially limiting the method's adoption in real-time or resource-constrained applications
2. The evaluation may be not entirely fair and convincing. While the method is designed for sequential input, it is only compared against baselines targeting per-frame localization. Besides, the direct use of ground-truth poses provided by the datasets as ego-motion raises another major concern on its practical performance.
3. It is suggested to provide more details on how these baselines are adapted and implemented, as the statistics reported are questionable - particularly in `Tab.1`, for *HLoc* and *MeshLoc* that rely on image retrieval, the average position errors seem unexpectedly large.
4. The claimed low-memory benefit and performance gain are not sufficiently compelling. Established scene coordinate regression methods, like *ACE*, *GLACE* and *R-SCoRe*, demonstrate similar or even better savings while achieving  top performance within minutes—and crucially, *without relying on additional priors such as depth, meshes, or annotations*.

### Questions
1. I remain concerned about the unexpectedly large average pose errors reported for *HLoc* and *MeshLoc*. The authors should verify their implementation. It is also recommended to constrain the final pose estimates within the scene's bounding box as a straightforward sanity check.
2. The selection criteria for a fix-length of sequential inputs remains unclear to me. For instance, it is not specified whether frame downsampling is applied to form the query sequence, or how frames in a sequence are chosen. This pre-processing step should be clearly detailed.
3. The results in `Tab.8` indicate that semantic signals are the most contributive factor, which is not fully straightforward to me, since intuitively depth cues are more important.

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
3

### Summary
This paper introduces SG2Loc, a method for sequential visual localization that relies on 3D scene graphs instead of traditional dense 3D maps or large image databases.

The key idea is to represent an environment as a compact graph of object nodes and spatial relationships, where each node includes a coarse mesh and a semantic embedding.

Localization is formulated as a particle filtering problem that refines the camera pose over time.
The method achieves comparable accuracy to classical approaches such as HLoc and MeshLoc while requiring 10× less storage.

### Strengths
1. Overall, the paper is well-written and clearly structured.

2. The work extends the use of scene graphs in visual localization to the sequential setting, which is a meaningful and natural progression for this line of research.

3. While the use of a particle filter is not new, the paper integrates it effectively with semantic and geometric cues from scene graphs, resulting in an effective approach.

4. The method is compared against well-known baselines and achieves comparable performance while requiring less storage (with the exception of ACE).

5. The authors include thorough ablation studies that help clarify the impact of key design choices.

### Weaknesses
1. The contribution is primarily an integration of existing components, scene graphs, particle filtering, and standard similarity measures, rather than a fundamentally novel idea.

2. The method relies on pre-built, labeled 3D scene graphs with coarse object meshes, but the paper does not discuss how such graphs are generated.

3. The system estimates only 4 degrees of freedom (assuming known gravity direction), making it unsuitable for general 6-DoF localization and resulting in an unfair comparison with fully 6-DoF baselines such as HLoc and ACE.

4. The approach is considerably slower than retrieval-based methods.

5. Experiments are conducted only on indoor datasets, leaving their performance in outdoor scenes unexplored.

6. While the paper emphasizes low storage requirements, it does not discuss or compare to other memory-efficient visual localization methods such as SceneSqueezer [A] and [B].

-[A] Scenesqueezer: Learning to compress scene for camera relocalization. CVPR 2022. 

-[B] Differentiable product quantization for memory efficient camera relocalization. ECCV 2024.

### Questions
1. Regarding W3, can you please show an experiment where you estimate the gravity direction using GeoCalib (Veicht et al., 2024)?

2. Please clarify how you construct the scene graph for each scene.

3. Please discuss/compare the memory-efficient methods mentioned in W6.

4. Is it possible to show a small experiment on an outdoor scene?

minor: In line 199, there is a small mistake: a 14×14 grid results in 196 patches, not 144 as mentioned.

### Soundness
3

### Presentation
3

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
The paper introduces SG2Loc, a sequential visual localization framework that utilizes 3D scene graphs as the underlying map representation. Unlike prior image- or point-cloud-based approaches, SG2Loc employs a particle filter operating on a 4-DoF state space, where each particle’s observation likelihood is computed by comparing ray-casted object predictions against semantic patch embeddings derived from a SceneGraphLoc-style encoder. The method further integrates SSIM and depth consistency terms, adopts coarse-to-fine search and KLD-based adaptive resampling, and refines the final pose using RoMa + PnP. The main claim is that SG2Loc achieves competitive localization accuracy with a dramatically smaller storage footprint, demonstrating the viability of semantic scene-graph maps for sequential localization tasks.

### Strengths
S1. The paper makes a clear and logical extension of SceneGraphLoc from single-frame reasoning to sequential probabilistic localization. The integration of semantic, photometric, and geometric cues in a unified particle-filter framework is well motivated and technically sound.

S2. The overall system is well-structured, including motion prediction, adaptive resampling, and pose refinement. The coarse-to-fine search is a sensible addition that improves robustness in practice.

S3. Experimental results indicate that the method achieves accuracy comparable to strong baselines while using significantly less map storage, supporting its motivation for efficient localization.

### Weaknesses
W1: The system mainly combines existing components, such as scene-graph-based embeddings,  particle filtering, SSIM/depth fusion, and RoMa+PnP refinement, without introducing a fundamentally new algorithmic contribution. The conceptual leap beyond SceneGraphLoc is relatively small.

W2: The semantic likelihood assigns a high score only when the predicted object ID from ray-casting matches the predicted object ID from the image patch. This hard matching assumption ignores soft uncertainty in detection and segmentation, which could make the system brittle to misclassification, occlusion, or open-set objects. A more principled probabilistic treatment would have been preferable.

W3: The paper combines semantic, photometric, and depth likelihoods, but does not specify how these are normalized or weighted. Without clear scale calibration or hyperparameter justification, the combined likelihood may behave unpredictably across scenes.

W4. The transition model seems to rely on ego-motion estimation between frames, but details on how this motion is obtained are unclear.

### Questions
Q1. How robust is SG2Loc to semantic segmentation errors or to dynamic scenes where object layouts change?

Q2. How is ego-motion estimated between frames for the particle filter’s prediction step?

Q3. How are the weights between semantic, photometric, and depth likelihoods tuned? Are they fixed, or learned?

Q4. How sensitive is the performance to sequence length and particle count?

### Soundness
2

### Presentation
2

### Contribution
2
