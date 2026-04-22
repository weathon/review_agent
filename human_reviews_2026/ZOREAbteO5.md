# PTNET: A PROPOSAL-CENTRIC TRANSFORMER NET- WORK FOR 3D OBJECT DETECTION

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
3D object detection from LiDAR point cloud data is important for autonomous driving systems. Recent two-stage 3D object detectors struggle to achieve satisfactory performance due to limitations in proposal quality, stemming from the degradation of geometric detail information in the generated proposal features caused by high sparsity and uneven distribution of point clouds, as well as a lack of effective exploitation of surrounding contextual cues in the independent proposal refinement stage. To this end, we propose a Proposal-centric
Transformer Network (PTN), which includes a Hierarchical Attentive Feature Alignment (HAFA) module and a Collaborative Proposal Refinement Module (CPRM). More concretely, to obtain multi-granularity proposal representations, HAFA employs a dual-stream architecture that extracts both coarse-grained voxel features and fine-grained point features to enhance proposal features, then harmo-
nizes them through a feature alignment network in a unified space. The CPRM first generates object queries for all objects and then establishes contextual-aware interactions to extract complementary information from semantically similar and spatially relevant proposals. PTN achieves promising performance on large-scale Waymo and KITTI benchmark, demonstrating the superiority of PTN.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the proposal-quality bottleneck in two-stage LiDAR 3D detectors by introducing PTN, a proposal-centric transformer composed of two key parts: Hierarchical Attentive Feature Alignment (HAFA) and a Collaborative Proposal Refinement Module (CPRM). HAFA is a dual-stream design that fuses coarse, multi-scale voxel features with fine, foreground point features and then aligns them in a unified space to strengthen geometric detail in proposals. CPRM creates hybrid object queries—top-K proposal queries plus learnable random queries—and performs 3D parameter-guided deformable attention across proposals to share contextual cues, especially helpful under occlusion and sparsity. Experiments on Waymo and KITTI report consistent gains; e.g., Waymo test mAP/mAPH (L2) of 72.7/70.6 with strong category results, and competitive KITTI performance, alongside ablations justifying each component.

### Strengths
1. Clear problem framing: identifies two concrete issues—loss of geometric detail in proposals and lack of cross-proposal context—and directly designs modules to address each.
2. Well-motivated dual-stream features: combining grid-sampled voxel tokens (coarse) with raw foreground point cues (fine) is intuitive and technically grounded, with an explicit alignment step.  
3. Ablations support claims: both HAFA and CPRM contribute; component-wise tables and studies on query counts/NMS thresholds/decoder depth are informative.

### Weaknesses
1. Added complexity & runtime: HAFA + CPRM introduce nontrivial overhead and multiple hyperparameters (e.g., grid sizes, query counts, NMS/thresholds); the paper discusses trade-offs qualitatively but detailed compute/memory costs and scaling behavior are limited.
2. RPN dependence: although CPRM adds random queries, the pipeline still leans on RPN quality; failure modes when RPN proposals are poor (domain shift, long-range sparse objects) are not deeply analyzed.

### Questions
1. Deformable attention specifics: How are reference points and offsets parameterized from 3D boxes? What’s the exact number of sampled keys per query and the computational/memory cost per decoder layer?  
2. Query budgeting: Beyond global K, did you try class-aware (per-class) query budgets or adaptive K per scene without discretized intervals? How sensitive is performance to K and the random-query count M across datasets?

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
This paper proposes a Proposal-centric Transformer Network (PTN) for 3D object detection from LiDAR point clouds. PTN addresses two main challenges in two-stage 3D detection: degradation of proposal geometric details due to point sparsity and ineffective use of contextual cues during refinement. PTN introduces two core modules: Hierarchical Attentive Feature Alignment (HAFA), which extracts and aligns coarse voxel and fine point features within each proposal, and Collaborative Proposal Refinement Module (CPRM), which integrates contextual interaction among spatially and semantically related proposals via hybrid queries and deformable attention. Extensive experiments on the Waymo and KITTI benchmarks demonstrate performance improvements over representative prior methods.

### Strengths
1. The Collaborative Proposal Refinement Module is the first proposal-centric transformer to refine the bounding boxes generated from RPN. The idea is interesting, and the performance gain is significant. With the random queries, it can recall weak proposals caused by distance or occlusion.
2. The final performance on the Waymo Open dataset (single-frame setting) is impressive, especially on small objects.
3. Ablation studies isolate the effect of each module, providing transparency on their contributions to accuracy and speed. Table 9 offers inference speed comparisons per module, supporting claims of balanced efficiency and performance.
4. The paper contextualizes itself against a range of competing methods, including both DETR-style and non-Transformer 3D detectors, and references and discusses most major prior works.

### Weaknesses
1. One of the effects of CPRM is that it is an end-to-end module. However, the authors still apply NMS before the module, which weaken the end-to-end feature. Moreover, I would like to see a comparison between CPRM and classic RCNN module (e.g. Voxel-RCNN) with NMS.
2. Section 3.3.3 introduces a “3D parameter-guided deformable attention,” but omits critical specifics: how offsets are computed per proposal, how attention ranges are constrained spatially, and what prior (if any) is imposed by the box parameters. A clear symbolic or algorithmic formulation is needed for reproducibility and transparency.
3. The multi-frame performance is not superior to pervious work CenterFormer. Does the proposal collect features from the "tail" caused by object movements?
4. Current state-of-the-art works, such as FSD and FSDv2, are not cited and compared.

### Questions
Is the position embedding based on xyz only? How can such position embedding represents "occlusion" between proposals?

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
3

### Summary
This paper introduces PTNET, a novel Proposal-centric Transformer Network designed to enhance two-stage 3D object detection from LiDAR point clouds. PTNET targets two key limitations of current two-stage methods: 1) the degradation of geometric details in proposal features due to point cloud sparsity and pooling operations; and 2) the failure to leverage contextual clues from neighboring proposals during the refinement stage, which traditionally treats each proposal independently. Experimental results demonstrate that PTNET achieves state-of-the-art (SOTA) performance on the Waymo and KITTI large-scale benchmarks.

### Strengths
- The paper is well-motivated, introducing DETR into the second stage of 3D detectors to aggregate information from the full ROIs
- The authors conduct extensive experiments on two major autonomous driving benchmarks (Waymo and KITTI), demonstrating state-of-the-art or highly competitive performance across multiple categories, especially for pedestrians and cyclists. The multi-frame input results further validate the model's robustness.

### Weaknesses
- While the combination of HAFA and CPRM is well executed, the individual ideas (multi-granularity alignment and proposal interaction through attention) resemble previous works such as PV-RCNN++[1] and ConQueR[2]. The contribution may thus be seen as an engineering refinement rather than a conceptual breakthrough.
- The fine-grained branch (FPFR) in HAFA is meant to recover geometric detail from "raw foreground point clouds." As described in Section 3.2.2, this module "first selects foreground points $\text{P}'$ whose locations are inside the proposal $b$." This introduces a limitation: if an RPN proposal $b$ is poor (e.g., a tiny bounding box due to heavy occlusion), FPFR can only access points within that incomplete proposal. It can therefore only "sharpen" the features within the existing, poor boundary, but seems inherently unable to "complete" the true geometric shape by accessing points that belong to the object but lie outside the RPN's initial prediction.

[1] Shi, Shaoshuai, et al. "PV-RCNN++: Point-voxel feature set abstraction with local vector representation for 3D object detection." International Journal of Computer Vision 131.2 (2023): 531-551.
[2] Zhu, Benjin, et al. "Conquer: Query contrast voxel-detr for 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
- How robust is the adaptive query number estimation (Eq. 5–6) to datasets with different object count distributions, such as nuScenes or Argoverse?
- The CPRM employs deformable cross-attention among proposals. What is its computational complexity compared with standard cross-attention, and how does it scale when the number of proposals increases?

### Soundness
3

### Presentation
2

### Contribution
2
