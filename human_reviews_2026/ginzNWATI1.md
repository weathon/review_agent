# From Tokens to Nodes: Semantic-Guided Motion Control for Dynamic 3D Gaussian Splatting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Dynamic 3D reconstruction from monocular videos remains difficult due to the ambiguity  inferring 3D motion from limited views and computational demands of modeling temporally varying scenes. While recent sparse control methods alleviate computation by reducing millions of Gaussians to thousands of control points, they suffer from a critical limitation: they allocate points purely by geometry, leading to static redundancy and dynamic insufficiency. We propose a motion-adaptive framework that aligns control density with motion complexity. Leveraging semantic and motion priors from vision foundation models, we establish patch-token-node correspondences and apply motion-adaptive compression to concentrate control points in dynamic regions while suppressing redundancy in static backgrounds. Our approach achieves flexible representational density adaptation through iterative voxelization and motion tendency scoring, directly addressing the fundamental mismatch between control point allocation and motion complexity. To capture temporal evolution, we introduce spline-based trajectory parameterization initialized by 2D tracklets, replacing MLP-based deformation fields to achieve smoother motion representation and more stable optimization. Extensive experiments demonstrate significant improvements in reconstruction quality and efficiency over existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a semantic-guided motion-adaptive framework for dynamic 3D Gaussian Splatting (3DGS) reconstruction from monocular videos. The key idea is to use vision foundation models to extract semantic and motion priors that guide node allocation (“From Tokens to Nodes”) and introduce spline-based trajectory parameterization for node motion. The authors claim that this design aligns control-point density with motion complexity and yields smoother, more stable optimization. Experiments on Hyper-NeRF and N3DV datasets are presented, showing improved PSNR/SSIM compared to SC-GS, Grid4D, and other baselines.

However, after close inspection, the technical novelty appears limited. Several core components replicate or closely follow prior works such as SC-GS, Superpoint-GS, H3D-DGS, and MoSca. The improvements demonstrated experimentally are modest and insufficiently supported by analysis or ablation. The paper is well written, but the contributions are overstated relative to their actual methodological advancement.

### Strengths
1. **Clarity and organization** – The paper is clearly structured, with explicit method description, and comprehensive visuals (Figure 1–3). The pipeline is easy to follow.
2. **Integration of semantic priors** – The idea of leveraging pretrained vision foundation models (VFM) for adaptive node placement is conceptually appealing and fits the current trend of combining large-scale priors with geometric reconstruction.
3. **Empirical validation** – The paper reports quantitative and qualitative results on two widely used dynamic scene datasets (Hyper-NeRF, N3DV), demonstrating that the proposed pipeline functions in practice.

### Weaknesses
1. **Lack of genuine novelty** –  
   - The *node-based deformation representation* (Sec. 4.2) is almost identical to prior works such as SC-GS [1], Superpoint-GS [2], and H3D-DGS [3]. The node weighting mechanism and dual quaternion interpolation are directly inherited from SC-GS and MoSca[4] with minimal modification.  
   - The *image-to-space projection* in Sec. 4.3 follows the same design as H3D-DGS, where control points are back-projected from image patches using estimated depth. The only difference is the additional compression heuristic, which is conceptually similar to spatial clustering used in K-Means-based approaches (e.g., H3D-DGS achieving similar sparsity with 10% of control points).  

2. **Experimental insufficiency** –  
   - Key baselines such as 4DGS [5] are missing in several figures (e.g., Figure 2), and qualitative results appear cherry-picked. In Figure 3, for example, the Cook Spinach sequence shows that 4DGS visually outperforms the proposed method in both facial and object regions.  
   - The claimed superiority is not consistently demonstrated: PSNR/SSIM gains over strong baselines are marginal (≤1 dB) and may not be statistically significant.  
   - No comparison against simpler node initialization (e.g., K-Means clustering GS according to spatial position) or trajectory models (linear vs. spline) is provided to justify additional complexity.  
   - The *spline-based trajectory* (Sec. 4.4) merely replaces existing linear or MLP deformation with cubic interpolation. Given that both N3DV [6] and Hyper-NeRF [7] are low-motion datasets, linear interpolation would likely suffice. No temporal interpolation or high-motion sequence evaluation is provided.  

```
[1] Huang, Yi-Hua, et al. "Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

[2] Wan, Diwen, Ruijie Lu, and Gang Zeng. "Superpoint gaussian splatting for real-time high-fidelity dynamic scene reconstruction." arXiv preprint arXiv:2406.03697 (2024).

[3] He, Bing, et al. "S4d: Streaming 4d real-world reconstruction with gaussians and 3d control points." arXiv preprint arXiv:2408.13036 (2024).

[4] Lei, Jiahui, et al. "Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[5] Wu, Guanjun, et al. "4d gaussian splatting for real-time dynamic scene rendering." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

[6] Li, Tianye, et al. "Neural 3d video synthesis from multi-view video." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[7] Park, Keunhong, et al. "Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields." arXiv preprint arXiv:2106.13228 (2021).
```

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of dynamic 3D reconstruction from monocular videos using 3D Gaussian Splatting. It identifies a key limitation in current sparse control methods: control points are often allocated based purely on geometry (e.g., uniform sampling), leading to redundancy in static regions and insufficient density in dynamic regions. To overcome this, the paper proposes a motion-adaptive framework. The core contributions are: (1) Motion-Adaptive Node Initialization (MANI), which leverages semantic and motion priors from Vision Foundation Models (VFMs) via patch-token correspondences to compress nodes adaptively, concentrating control density in dynamic areas (2) Spline-Parameterized Node Trajectories (MS), which replaces MLP-based deformation fields with cubic splines initialized by 2D tracklets, aiming for smoother motion and more stable optimization. Experiments on the Hyper-NeRF and N3DV datasets demonstrate improvements in reconstruction quality and efficiency over state-of-the-art methods.

### Strengths
1.The paper accurately identifies and targets a fundamental limitation in existing sparse control methods for dynamic 3DGS – the mismatch between geometrically uniform control point allocation and non-uniform motion complexity. The concepts of static redundancy and dynamic insufficiency are well-articulated.
2.The proposed MANI method is highly novel and intuitive. Using VFM tokens to guide an adaptive compression process that preserves nodes in dynamic regions while aggressively merging them in static ones is a strong contribution. The visualization in Figure 4 effectively demonstrates this capability.
3.The method achieves state-of-the-art results on the Hyper-NeRF and N3DV datasets under the challenging monocular setting, outperforming numerous recent baselines quantitatively and qualitatively. The ablation studies effectively validate the contributions.

### Weaknesses
1.The proposed framework introduces a significant number of hyperparameters. These include parameters for node binding (K neighbors, RBF radius), the MANI process (patch size, initial voxel size, number of iterations, compression rates, similarity/dynamic score weights), spline parameterization (number and selection of keyframes K). Tuning such a large set of parameters can be challenging, raising concerns about the method's applicability and robustness across different types of scenes or datasets. The paper lacks an analysis of how sensitive the final performance is to the choice of these hyperparameters
2.The method heavily relies on pre-trained VFMs for crucial inputs like semantic tokens, depth maps, segmentation masks, and 2D tracklets. Errors or inaccuracies from these upstream models (e.g., poor depth estimation in textureless regions, inaccurate tracking during occlusions, noisy segmentation) could propagate and significantly degrade the quality of node initialization (MANI) and motion supervision.

### Questions
1.A key potential benefit of sparse control methods is enabling motion editing. Since MANI initializes nodes based on semantic tokens , do these nodes correspond to meaningful semantic parts? Have the authors explored manipulating these nodes for motion editing or control? Are there any demonstrations showing this capability?
2.Compared to prior work like SC-GS , which primarily relied on geometric node placement, does the semantic nature of your nodes unlock more advanced or more intuitive editing possibilities? For example, could one edit the motion of specific semantic parts (like "arm" or "leg") by manipulating a group of associated nodes?
3.Given the large number of hyperparameters identified in Weakness #1, how were they tuned for the experiments? Is there a recommended strategy for setting these parameters for new scenes? How robust is the performance to variations in key parameters like the MANI compression rates or the number of spline keyframes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. This paper focuses on dynamic 3D reconstruction from monocular videos and proposes a sparse, node-based deformation framework.
2. This paper proposes Motion-Adaptive Node Initialization and spline-parameterized node trajectories with cubic Hermite.
3. Experiments on Hyper-NeRF and N3DV show improved quality and efficiency over previous methods.
4. The idea seems to be novel and the objective experimental results are convincing.

### Strengths
1. Motion-aware sparsity allocates more control nodes in motion-heavy regions via semantic/motion priors from vision foundation models, which makes sense and demonstrates a new direction for optimizing 3D/4D reconstruction.
2. Spline-parameterized node trajectories replaces MLP deformation fields for smoother, more stable motion with cubic Hermite initialized from 2D tracklets.
3. They provide enough ablations that isolate contributions of their framework.
4. The objective experimental results are convincing enough to demonstrate the superiority of their method.

### Weaknesses
1. In my view, videos are necessary for papers in 4D reconstruction to show the effectiveness and superiority. If there is no videos, there is no enough subjective comparisons. Objective metrics and image/frame comparisons are not enough.
2. This idea of 2D motion-guided deformation is not novel, and many papers have proved the effectiveness. There should be more visualizations and comparisons with such kinds of methods.
3. The method depends heavily on VFM priors. What will happen when there is severe distortions or shifts in the VFM priors? As far as we know, VFM's priors are not that robust and stable so far.

### Questions
1. I am curious about the actual effectiveness of trajectory initialization. It fits splines only for translations while starting rotations from the identity. Does it show slow convergence or demand stronger regularization in cases with pronounced rotational or articulated motion?
2. Despite incorporating multiple constraints (RGB, depth, masks,...), are there some scenarios may require stronger geometric priors or multi-view supplementation due to scale and occlusion ambiguities under monocular setups?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a motion-adaptive dynamic 3D reconstruction framework that allocates control points based on motion complexity rather than static geometry. By leveraging semantic and motion priors from vision foundation models and introducing spline-based trajectory parameterization, the proposed approach achieves smoother motion, higher reconstruction quality, and improved efficiency over prior approaches. Extensive experiments demonstrate the effectiveness.

### Strengths
1. Leveraging the  vision foundation models for 4D Gaussian Reconstruction seems interesting, though it is not novel.
2. The experiments demonstrate the effectiveness based on two important datasets.
3. Motion-Adaptive Node Initialization seems interesting.

### Weaknesses
1. The comparison is slightly old. I feel confused that the paper did not compare directly with [A] and [B].
2. The whole pipeline is quite complex, and this paper did not provide any significant progress.
3. These ideas all make sense, but they’re quite ordinary.

[A] Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds
[B] Himor: Monocular deformable gaussian reconstruction with hierarchical motion representation.

### Questions
n/a

### Soundness
3

### Presentation
2

### Contribution
2
