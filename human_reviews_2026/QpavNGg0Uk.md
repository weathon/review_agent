# LOGAussian: Efficient Local Gathering for Online Feed-forward 3DGS

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Feed-forward 3D Gaussian Splatting (3DGS) has attracted increasing attention for its broad applicability and real-time inference capabilities. Despite recent progress through advanced backbones, prevailing pipelines remain offline: concatenating per-view, pixel-aligned splats introduces redundancy and, under streaming input, accumulates errors over time. To address this, we propose \textbf{LOGAussian} (LOcal GAthering Gaussians), a lightweight post-hoc module that maintains an incrementally updated and render-ready global 3D Gaussian representation from sequential posed images without per-scene optimization. The approach integrates global scene context and models local correlations among predicted Gaussians, explicitly accommodating sparse-view inputs with depth noise and geometric imprecision.
With about 1\% additional parameters, our approach yields a compact, consistent set of splats while maintaining or improving rendering quality as the stream progresses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LOGAussian, a post-hoc lightweight module designed to enable online processing for feed-forward 3D Gaussian Splatting (3DGS). The authors claim that the module performs local and global fusion of Gaussian splats over streaming frames, using a hybrid 2D projection + 3D serialization strategy with attention-based pruning and refinement. The method is evaluated on RealEstate10K and DL3DV datasets using DepthSplat as the backbone, reporting improved PSNR/SSIM and reduced Gaussian redundancy.

### Strengths
1. The paper provides extensive references and a reasonably clear experimental protocol.

2. The idea of integrating 2D projections and 3D serialization is intuitive and might inspire minor future extensions in online fusion.

### Weaknesses
1. The main pipeline is a hybrid of DepthSplat’s cost volume and standard patch-based transformer refinement. The proposed local fusion is essentially a masked attention plus pruning head, not conceptually new.

2. Overclaimed efficiency. Despite “lightweight” claims, the model still relies on multi-head attention and repeated fusion steps. No inference-time analysis or FLOP comparison is provided.

3. The method is described as “online,” yet the global fusion is done every M frames, making it quasi-offline.

### Questions
1. Can the authors provide exact runtime measurements (FPS) compared to DepthSplat and MVSplat for fair evaluation of “efficiency”?

2. How is the “1% additional parameters” computed—does it include the attention weights and dual MLP heads?

3. Did the authors attempt to train with unposed or partially posed sequences to justify the claimed “online” robustness?

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
This paper aims at fusing redundant 3DGS in existing feed-forward 3DGS methods using a learnable network. Unlike previous feed-forward methods, which produces abundant 3DGS in a pixel-aligned way. To do this, it a lightweight module which incrementally updates and refine global 3DGS from sequential images. The network updates 3DGS of an incoming frame alongside the global 3DGS via a local self-attention mechanism. The network is trained solely on the RE10K dataset and evaluated on the RE10K and DL3DV10K test set. Experiments show that the baseline (DepthSplat) performance is largely improved in both the sparse view and dense view settings.

### Strengths
(1) The related work and method parts are well-written and easy to understand.

(2) The incremental updating strategy is effective to remove redundant 3DGS, and the performance is greatly improved -- compared to DepthSplat.

(3) The proposed method can be used as a plug-in to many other feed-forward 3DGS methods, which is an important contribution to the community.

### Weaknesses
- There are some formatting issues:
(1) A "." is omitted at Line309: "offsets for all Gaussian attributes The refined feature ..."

(2) The qualitative comparisons of "Visualization of the challenging floaters issue" is included into Table 2 which makes the formatting very strange. It is recommended to arrange it into an individual figure.

(3) At line 468, Table 4 is not referenced correctly.

- Though the progressive updating strategy is effective in removing 3DGS, its performance in handling long sequences is unclear: when the number of 3DGS grows, it will needs longer time and more GPU memories to fuse/update 3DGS, which could limit its application in longer videos.

- The experiment part lacks some important baselines which are also related to this work, e.g., ZPressor [1], Long-LRM [2]

[1] zpressor: bottleneck-aware compression for scalable feed-forward 3dgs. NeurIPS 2025
[2] Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats. ICCV 2025

### Questions
I am curious about the performance of the method in full sequence videos/images. I would appreciate and raise my score if the authors can provide additional experiments on this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces LOGAussian (LOcal GAthering Gaussians), a lightweight post-hoc module that enables online and incremental processing for feed-forward 3D Gaussian Splatting (3DGS). LOGAussian leverages 3D serialization and 2D projection to model local correlations among Gaussians, followed by an importance score–based pruning and refinement module to update or remove redundant global splats. The proposed module adds only about 1% additional parameters to DepthSplat, while significantly improving both rendering quality and computational efficiency on the RealEstate10K and DL3DV datasets.

### Strengths
(1) Well-motivated problem: The paper clearly identifies a key limitation of existing feed-forward 3DGS methods, which operate offline and cannot handle sequential or streaming inputs.

(2) Simple and effective design: The proposed approach introduces a mask head for filtering and a residual prediction head for updating Gaussian attributes, offering a straightforward yet efficient way to maintain compact and consistent scene representations.

(3) Strong experimental results: The method achieves consistent quantitative improvements over DepthSplat across multiple datasets and varying input-view configurations.

### Weaknesses
The reviewer finds the authors’ approach interesting and potentially impactful. However, the presentation lacks important implementation details, making it difficult for readers to fully grasp the proposed method.

(1) The paper introduces a threshold τ to generate the Gaussian mask, which determines how many Gaussians are retained. This hyperparameter is crucial for understanding pruning behavior, yet the authors do not provide its default value or any ablation study analyzing its effect.

(2) The authors mention that the global model is not updated on every frame but instead uses a sliding window of the most recent M frames. Since M directly affects inference speed and rendering quality, an ablation study and the default setting should be reported.

(3) The paper does not report inference time, which is essential for evaluating the claimed efficiency. Additionally, reporting peak GPU memory usage would help quantify the computational overhead more precisely.

(4) The proposed method appears incompatible with unposed-image settings such as NoPoseNoProblem [1], where the first frame serves as the canonical space. LOGAussian depends on the base model to process each new incoming view, which may limit its applicability in unposed or pose-free scenarios.

[1]  No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (ICLR2025)

### Questions
(1) Can LOGAussian handle sequences with thousands of views?

(2) Can LOGAussian be integrated with MVSplat, and if so, how does it perform in that setting?

### Soundness
3

### Presentation
1

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
LOGAussian introduces a lightweight module that makes feed-forward 3D Gaussian Splatting (3DGS) suitable for online, real-time scene reconstruction. Traditional 3DGS methods process fixed image sets offline, leading to redundancy and drift when handling streaming inputs. LOGAussian addresses this by incrementally fusing new frames into a consistent global 3D Gaussian model using a local correlation-aware attention mechanism. It combines 2D projections and 3D spatial serialization to detect redundant or inconsistent splats, pruning and refining them efficiently with minimal computational overhead. With about 1% additional parameters, LOGAussian yields a compact, consistent set of splats while maintaining or improving rendering quality as the stream progresses

### Strengths
The method achieves a significant reduction in the number of Gaussians used to represent the scene even using the local correspondence matching between Gaussians and the global fusion further decreases the Gaussian count. 
Cross dataset generalization is also shown with 12 image sequences and 100 frame evaluations.

### Weaknesses
- In section 4.3 a sliding window of M frames is described to perform the global fusion periodically. But the details about M are not obvious.

- Compute used at inference and measures like inference time and GPU memory usage could add further value on top of the number of Gaussians reported.

- The serialization design choices like patch size remain unablated.

- In lines 414-417 perhaps it would be useful to quantify between depthsplat-o and depthsplat-or and the former actually slightly improves over depthsplat as reported in table.2.

- Please follow the submission format, the citation format seems incorrect.

### Questions
- Could the authors quantify floaters based on their importance based contribution to the final pixel color and then compare how many of them are removed via the proposed method.

- In equation 5 does the residual update include all Gaussian parameters and can it be quantified in any way perhaps where the most updated features can be identified and their impact on quality isolated.

### Soundness
3

### Presentation
3

### Contribution
3
