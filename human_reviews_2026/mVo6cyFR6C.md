# WorldTree: Towards 4D Dynamic Worlds from Monocular Video using Tree-Chains

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Dynamic reconstruction has achieved remarkable progress, but there remain challenges in monocular input for more practical applications. The prevailing works attempt to construct efficient motion representations, but lack a unified spatiotemporal decomposition framework, suffering from either holistic temporal optimization or coupled hierarchical spatial composition. To this end, we propose WorldTree, a unified framework comprising Temporal Partition Tree (TPT) that enables coarse-to-fine optimization based on the inheritance-based partition tree structure for hierarchical temporal decomposition, and Spatial Ancestral Chains (SAC) that recursively query ancestral hierarchical structure to provide complementary spatial dynamics while specializing motion representations across ancestral nodes. Experimental results on different datasets indicate that our proposed method achieves 8.26% improvement of LPIPS on NVIDIA-LS and 9.09% improvement of mLPIPS on DyCheck compared to the second-best method. Code: https://github.com/iCVTEAM/WorldTree.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author(s) propose a unified hierarchical spatiotemporal decomposition framework for the scene to recover a 4D dynamic world from monocular video. It consists of two key components. First, a Temporal Partition Tree (TPT), which recursively splits the video’s timeline into a tree of sub-intervals (coarse-to-fine) for sequential optimization at multiple time scales, and second, a Spatial Ancestral Chains (SAC), which links each node of the time-tree with a chain of inherited scene representations from its ancestors. In essence, each time segment (node) has its own local dynamic model, while also leveraging ancestral information (broader motion context and coarse geometry from higher levels) as a complementary spatial dynamic prior. This inheritance-based design allows model specialization at different temporal scales without losing global context. The overall optimization iteratively refines the scene: starting from a root (entire video) model, then subdividing in time and re-optimizing each segment with inherited parameters in a tree structure. Notably, the tree nodes are optimized in parallel at each level, leveraging the fact that disjoint time segments can be processed independently. The SAC mechanism ensures that finer segments still incorporate multi-level spatial information by querying the chain of ancestor nodes for each segment. Qualitative results show improved and temporally coherent reconstructions of moving subjects than baselines.

### Strengths
1. The paper presents a hierarchical decomposition of a dynamic reconstruction problem in both time and space. Prior monocular methods generally either optimized a single global motion field over the entire video or imposed a fixed spatial hierarchy. In contrast, the paper proposes an inheritance-based time segmentation that recognizes that different temporal intervals can exhibit distinct motion patterns. Next, Spatial Ancestral Chains ensure that each temporal segment's model includes multi-scale spatial context rather than starting from scratch. This hierarchical motion specialization may help decouple global and local motions.

2.Using BFS expansion, all segments at a given depth can be optimized in parallel. Next, the use of 3D Gaussian splatting as the underlying scene representation looks like a wise choice, i.e., it provides explicit control over primitives and fast rendering.

3. The use of the SAC mechanism, i.e., aggregating the Gaussians of ancestor nodes when rendering a child's frame, effectively layers coarse and fine geometry, preventing the child from needing to re-learn large static structures or global motion from scratch.

### Weaknesses
1.A notable limitation is the dependence on external 2D vision models for initialization. Specifically, monocular depth estimation, optical flow, and feature tracking are used to lift 2D priors into 3D scene representations.
The paper does not show how errors in these inputs can affect the final result. Presumably, inaccuracies in initial depth or camera pose estimation could propagate through the hierarchical optimization.
Suggestion to improve: It would strengthen the work to either (a) demonstrate some robustness analysis (e.g., running the method with intentionally perturbed or lower-quality priors to see if it can recover) or (b) discuss approaches to reduce reliance on these models.

2. The TPT currently uses a very simple scheme for splitting the video (fixed). Each interval is bisected at the midpoint in time (a binary partition), and this continues to a fixed tree depth. This coarse-grained, uniform partition may not be optimal for every video. Real dynamic scenes can have uneven motion. E.g., a subject may remain still for a while then move rapidly, or different time segments have different complexity. 

The paper could be improved by addressing this design choice. Perhaps providing reasoning for why uniform binary splits were chosen may be simplicity? easier parallelism?, etc.

3. WorldTree's pipeline seems to implicitly assume that the input video has a mixture of static and dynamic content, and that at least a portion of the scene can be treated as static for initialization. The authors perform a "Static Warm-Up" and Bundle Adjustment (BA) at the root stage, probably to establish camera poses and a base model of the scene before modeling non-rigid motion. This likely works well if, for example, the background is static and only an object or person is moving. The authors should clarify this assumption. Are there failure cases when the scene violates these assumptions?

### Questions
Q1. In Sec. 3.3, the paper states that the common ancestors of different nodes are independent of each other but have the same optimized initialization. Kindly clarify what this means in practice? For example, when a parent node (covering a certain interval) is split into two children, do you duplicate the parent's Gaussian primitives and motion parameters into each child's ancestral chain as an initialization, and then allow each branch to optimize those copies independently?

Q2. The current implementation always splits intervals evenly at the midpoint in time. Did the author(s) experiment with other criteria for the partition point? For instance, could one split an interval at a frame where the motion or reconstruction error is highest (indicating a change in dynamics)? If not, how sensitive is the method to the exact placement of the split?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present WorldTree, a framework for 4D dynamic scene reconstruction from monocular video. It introduces a hierarchical spatiotemporal decomposition via two key components: (1) Temporal Partition Tree (TPT) – a coarse-to-fine hierarchical temporal optimization structure. (2) Spatial Ancestral Chains (SAC) – hierarchical spatial composition from ancestor nodes to decouple motion representations. These innovations allow improved performance on monocular dynamic reconstruction benchmarks (e.g., NVIDIA-LS and DyCheck) without requiring multi-view input or manual point prompts. The method shows impressive gains in mLPIPS and mPSNR and is backed by a complete ablation study.

### Strengths
- The paper is well-written and the results are comparable to the state-of-the-art methods if not better.
- The approach reduces the reliance on expensive external priors such as COLMAP points or manual masks, pushing towards a more practical problem setup.
- The method achieves state-of-the-art performance on the NVIDIA-LS and DyCheck benchmarks, with comparable results to methods using stronger priors.

### Weaknesses
- While the TPT design that uses coarse-to-fine temporal partitioning is scalable, the binary split heuristic may limit the adaptiveness in scenes with irregular motion patterns.
- It seems that the transition between subtree boundaries is not explicitly handled, which might lead to edge artifacts in the final reconstruction. It would be nice to see more details on how the method handles the transition.
- Generalization to real-world videos such as those grabbed from the Internet (or simply DAVIS dataset) is not tested.
- It is not analyzed how the external priors such as RAFT or Metric3D-v2 would affect the method performance. This might cause a problem for videos with ambiguous geometry or fast motions.

### Questions
- How does the method handle occlusion? Is there a single accumulated canonical representation built for the entire scene?
- Your Temporal Partition Tree (TPT) uses a fixed binary split strategy. Did you consider or experiment with adaptive or learned temporal partitioning based on motion complexity or energy? If so, how did they compare? How sensitive is the reconstruction quality to the tree depth?
- SAC inherits motion features from ancestor nodes. How does the method ensure that outdated or inaccurate ancestral representations do not propagate errors into child nodes?
- Are there common failure cases you observed (e.g., fast motion, occlusion, camera jitter)? It would be helpful to know where the method performs poorly.

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
This paper introduces WorldTree, a comprehensive framework designed to improve the efficiency and quality of 4D dynamic scene reconstruction from monocular video. The central contribution lies in its dual decomposition strategy: the Temporal Partition Tree (TPT), which facilitates coarse-to-fine temporal optimization via inheritance, and the Spatial Ancestral Chains (SAC), which handles hierarchical spatial decomposition. This structural approach directly addresses known bottlenecks in dynamic NeRF/Gaussian Splatting literature, specifically the computational overhead of holistic temporal optimization. The methodology is clever and provides a fresh perspective on spatiotemporal modeling.

### Strengths
1. The division of the 4D space into TPT (Temporal) and SAC (Spatial) is the most substantial contribution. The TPT’s inheritance-based optimization scheme is a highly promising avenue for reducing the redundancy and computational load associated with optimizing motion across long video sequences, potentially leading to better temporal coherence.

2. The framework explicitly aims to overcome the coupling inherent in many hierarchical methods. If the TPT successfully decouples temporal optimization, it offers a crucial step towards scaling dynamic reconstruction to very long, complex videos.

3. Committing to monocular input significantly broadens the practical applicability of the work, moving the field closer to real-world use cases where calibrated multi-view rigs are unavailable.

### Weaknesses
1. Dependency on External Segmentation (SAM): The reliance on external segmentation tools like SAM (mentioned in the Appendix and implied by comparisons to HiMoR/SplineGS) is a significant point of concern. If the performance gains are largely attributed to clean, pre-processed dynamic masks, the "end-to-end" nature and robustness of WorldTree in unconstrained settings are compromised. A clearer analysis is needed to quantify the degradation when using noisy or no segmentation masks.

2. Scalability and VRAM Limits: The paper notes a constraint on the maximum number of motion nodes (32) due to VRAM capacity on the NVIDIA-LS dataset. While this is a practical limitation of the current implementation (building on MoSca), it raises critical questions about the theoretical scalability of the Tree-Chains spatial representation itself. Is this limit due to the underlying MoSca architecture, or is the complexity of the Tree-Chains structure the bottleneck? This should be analyzed and discussed in the main text.

### Questions
See weaknesses above.

### Soundness
3

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
4

### Summary
This paper proposes a approach for monocular dynamic reconstruction. The core contributions consists of two components:
(1). Temporal Partition Tree (TPT): a hierarchical binary partitioning of the video time sequence. It can be used to refine the deformation modeling in temporal segments.
(2). Spatial Ancestral Chains (SAC): a mechanism that propagates Gaussian representations from ancestor nodes to descendant nodes through ancestral chains, which can alleviate information loss during Gaussian point inheritance.

Additionally, this paper contributes an enhanced dataset based on NVIDIA-LS.

### Strengths
For the experimental results, the comparisons as well as the ablation studies support the paper’s claims. For the presentation, the writing is generally clear, for example, the paper provides a comparison diagram with previous works, and the ablation study results are presented comprehensively.

### Weaknesses
For the experiments, the paper mentions an improvement in computational efficiency, but it lacks corresponding experimental data on computational time cost.
From the perspective of novelty, the main contributions of this paper are TPT and SAC. TPT is used for video segmentation, while SAC supplements the current information by reusing information from higher-level ancestors. However, this does not constitute a significant theoretical breakthrough. On the other hand, the proposed mechanisms rely on strong assumptions, such as reasonable temporal segmentation and proper root node initialization. Therefore, these characteristics could also become new limitations of this paper (compared with methods that directly impose various monocular video priors onto 3D Gaussians[1]).

[1] D. Wu, F. Liu, Y.-H. Hung, Y. Qian, X. Zhan, and Y. Duan, “4d-fly: Fast 4d reconstruction from a single monocular video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 16 663–16 673.

### Questions
(1). Could you provide more experimental validation on how the hierarchical depth of TPT and the ancestor chain length of SAC affect performance?
(2). If the video is very long, would the binary partitioning in TPT result in an overly deep tree? What impact would this have on performance?
(3). Does the ancestor chain in SAC introduce redundant information? Has this potential issue been considered?

### Soundness
3

### Presentation
3

### Contribution
2
