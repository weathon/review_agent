# PointLAM: Local Attentive Mamba for Efficient Point-based 3D Object Detection

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
3D object detection from LiDAR faces a fundamental trade-off between computational efficiency and the preservation of fine-grained geometries. The dominant voxel-based paradigm achieves efficiency by quantizing massive point clouds, but at the cost of inevitable information loss. Conversely, point-based methods excel at capturing precise geometries by directly processing raw points, yet have been constrained by the prohibitive complexity of their core operators for downsampling and spatial feature modeling.
In this work, we tackle this dillema by introducing PointLAM, a novel framework for point-based 3D object detection that excels both in performance and efficiency. We systematically address the long-standing bottlenecks in point-based models through two synergistic designs. First, we propose a Dynamic Point Sampler (DPS) that intelligently curates an information-rich and structurally representative subset of raw points. It leverages a novel Deviation Network (DevNet) to capture each point's local distinctiveness, followed by a Doubly Sorted Sampling (DSS) strategy that retains the most informative points to reduce the workload of the 3D backbone. Second, our 3D backbone synergizes Bi-Directional Mamba (BDM) layers for global context modeling, and novel, lightweight Local Multiplicative Aggregation (LMA) layers for efficiently capturing intricate local geometries without computationally expensive neighborhood queries.
Extensive experiments show that PointLAM sets a new benchmark for efficient point-based 3D object detection. On both nuScenes and Waymo datasets, PointLAM not only significantly surpasses prior point-based models but also achieves comparable performance against strong voxel-based competitors like LION and DSVT. Crucially, these competitive results are achieved with a fraction of the model parameters and latency, demonstrating a superior balance between accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents PointLAM for point-based 3D object detection. First,  the paper proposes Dynamic Point Sampler (DPS) that curates an information-rich and structurally representative subset of raw points. Second,  synergizes Bi-Directional Mamba (BDM) layers for global context modeling, and  Local Multiplicative Aggregation (LMA) layers for efficiently capturing intricate local geometries.

### Strengths
1. The paper presents a couple of modules including Dynamic Point Sampler (DPS), Doubly Sorted Sampling (DSS), Local Multiplicative Aggregation (LMA) layer Bi-Directional Mamba (BDM) layers to capture intricate geometric patterns, improve efficiency, and enhance the overall performance. 
2.  PointLAM achieves competitive performance on large-scale 3D object detection benchmarks including nuScenes and Waymo. As a point-based method, it is even more efficient than the strong and highly efficient voxel-based competitors LION and DSVT, with only a fraction of their complexity in terms of parameters, operations, and latency costs.

### Weaknesses
1. The key contribution seems to be efficient implementation details, which are not very fundamental. These contributions seem somehow empirical and the paper lacks deep analysis and key concept. 
2. The overall performance is just comparable with or fall behind existing works. Only efficiency is improved. 
3. The authors do not provide qualitative comparisons. I think such comparsions will help readers to better understand the key contirbutions.

### Questions
1. Why do you evaluate on Waymo and nuScenes validation set, not test set?

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
This paper presents PointLAM, a point-based 3D detector that operates directly on raw point clouds without voxelization. The method introduces two modules—DPS for sampling distinctive point features and DSS for preserving informative points through the backbone—and integrates a Mamba-based component to further boost performance. Experiments and ablations demonstrate the effectiveness of these designs.

### Strengths
- `Strong empirical results:` The method delivers competitive performance on nuScenes and Waymo, with gains over prior point- and voxel-based baselines.

- `Effective point-based design:` Operating directly on raw point clouds avoids quantization artifacts. The DPS module and DSS module contribute improvements, and the Mamba-based component further boosts performance.

- `Thorough evaluation:` Experiments and ablations attribute gains to each component, making the source of improvements easy to understand and compare across settings.

- `Clear presentation:` The paper is well written and easy to follow, with a modular architecture and clear descriptions that facilitate reproducibility.

### Weaknesses
`Motivation and claimed advantage over voxelization:`
The paper argues for operating on raw points, but in practice both point-based and voxel-based pipelines downsample/aggregate and inevitably discard information. Voxel methods ingest raw points, perform quantization (pillars/voxels), and use VFE/PFE to aggregate; point-based methods also subsample aggressively to keep computation tractable. Without a formal analysis, it’s unclear that PointLAM fundamentally mitigates information loss relative to strong voxel baselines.

`Limited technical novelty; clarity of differences to prior work:`
- Deviation network vs VFE/PFE (e.g., SECOND, PointPillars [1]): The proposed feature aggregation appears conceptually similar to voxel encoders that pool local neighborhoods. Please clarify the concrete differences. An ablation replacing the proposed module with a standard VFE/PFE would help quantify novelty and necessity.

- Doubly Sorted Sampling (DSS) vs F-FPS (3DSSD [2]): DSS seems close to F-FPS or hybrid feature–distance sampling. What is the exact distinctiveness criterion, sorting strategy, and complexity (e.g., O(N log N) vs O(Nk))?

- Bi-directional Mamba vs VisionMamba [3] and DSVT [4]: The Bi-directional mamba part looks related. Please articulate the architectural and algorithmic differences.

[1]. Pointpillars: Fast encoders for object detection from point clouds

[2]. 3dssd: Point-based 3d single stage object detector.

[3]. Vision mamba: Efficient visual representation learning with bidirectional state space model

[4]. Dsvt: Dynamic sparse voxel transformer with rotated sets


`Baseline coverage for neighbor search:` The voxel-grid–based neighbor query is interesting, but it would be more compelling with direct comparisons to alternative neighborhood constructions, especially Point Transformer v3 (PTv3) [5]. Please report accuracy and efficiency compared to the neighboring search method listed in [5].

[5]. Point transformer v3: Simpler faster stronger

### Questions
N/A

### Soundness
3

### Presentation
2

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
This paper introduces PointLAM, a novel point-based 3D object detection framework that addresses the efficiency-precision trade-off in LiDAR-based detection. The method combines two key innovations: (1) Dynamic Point Sampler (DPS) with Deviation Network (DevNet) and Doubly Sorted Sampling (DSS) for intelligent point downsampling, and (2) PointLAM blocks that synergize Bi-Directional Mamba (BDM) for global context and Local Multiplicative Aggregation (LMA) for local geometry modeling. The approach achieves competitive performance with voxel-based methods while maintaining superior efficiency.

### Strengths
1) PointLAM achieves competitive performance on both nuScenes (72.2 NDS) and Waymo (73.6 L2 mAPH) datasets, matching or exceeding strong voxel-based competitors like LION and DSVT while using significantly fewer parameters and achieving faster inference.
2) The paper provides thorough comparisons across multiple metrics, datasets, and efficiency measures (parameters, FLOPs, latency), demonstrating the method's practical advantages.

### Weaknesses
1) While the combination is novel, individual components are relatively incremental. The Deviation Network is essentially a simple feature difference operation, and BDM uses standard axis-based serialization without significant innovation over existing Mamba adaptations for 3D data.
2) The paper lacks comprehensive ablation studies on key components. What happens with different deviation formulations? How sensitive is performance to the choice of k in DSS? The impact of different serialization strategies in BDM is mentioned but not thoroughly analyzed.

### Questions
1) How does the method perform when the point cloud density varies significantly across the scene?
2) What is the computational overhead of the temporary voxelization in LMA compared to direct neighborhood queries?
3) How sensitive is the method to the hyperparameter choices (k in DSS, kernel size in LMA)?

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
3

### Summary
This paper introduces PointLAM, a novel point-based 3D object detection framework that effectively balances computational efficiency with geometric precision in 3D object detection. Its main contributions include:
1.	A Dynamic Point Sampler (DPS) that intelligently selects informative points via a Deviation Network (DevNet) and Doubly Sorted Sampling (DSS), reducing computational load while preserving structural details.
2.	A Local Multiplicative Aggregation (LMA) module that efficiently captures fine-grained local geometries without expensive neighborhood queries, combined with Bi-Directional Mamba (BDM) layers for global context modeling with linear complexity.
The experiment in this paper is thorough, benchmarking PointLAM on nuScenes and Waymo datasets against state-of-the-art voxel-based (e.g., LION, DSVT) and point-based methods. It demonstrates superior or competitive performance in accuracy (NDS/mAP) while significantly reducing parameters, FLOPs, and latency.

### Strengths
The paper presents PointLAM, an innovative point-based 3D object detection framework that achieves an impressive balance between computational efficiency and geometric fidelity. Its originality lies in revisiting the point-based paradigm—long overshadowed by voxel-based methods—and successfully overcoming its inefficiency through two synergistic designs: the Dynamic Point Sampler (DPS) and the Local Multiplicative Aggregation (LMA) layer. The DPS, with its Deviation Network and Doubly Sorted Sampling strategy, introduces a novel feature-based approach for point selection, effectively addressing the classic bottleneck of expensive or lossy sampling. The LMA module further contributes by providing an elegant, lightweight means to model local geometries without explicit neighborhood queries, complemented by Bi-Directional Mamba layers for efficient global context modeling.
The technical quality of the paper is moderate, with clear algorithmic formulations, thorough ablations, and strong empirical validation across nuScenes and Waymo benchmarks.
In terms of significance, PointLAM redefines the feasibility of efficient point-based detection and could inspire a new research direction focusing on lightweight, direct point processing architectures.

### Weaknesses
This paper presents a strong contribution, but several weaknesses could be addressed to further solidify its impact.
1.	While the efficiency gains are impressive, the analysis of the trade-offs introduced by the novel LMA module remains somewhat superficial. The LMA's reliance on a transient voxel grid is a clever trick to avoid k-NN, but it inherently reintroduces quantization, which the paper initially criticizes in voxel-based methods. The paper would be strengthened by a deeper investigation into this apparent contradiction. For example, an analysis of how the performance on very small or thin objects (which are most susceptible to quantization artifacts) compares to a baseline with explicit k-NN would be highly informative. 
2.	The presentation of this paper needs to be improved. For example, the algorithmic submodules shown in Figure 3 cannot be easily matched to the overall framework based on the textual description, and the caption of the figure could be improved for better clarity.

### Questions
1.	Novelty over Prior Mamba-Based Methods:
PointLAM claims to be the first efficient point-based Mamba detector, but prior works like PointMamba and Mamba3D also address point-level modeling. Could the authors elaborate more concretely on what differentiates PointLAM from these in terms of both architecture and theoretical motivation?
2.	Visualization and Interpretability:
Could the authors provide visualizations of the sampled points or learned local feature maps to better interpret what DPS and LMA actually focus on?

### Soundness
2

### Presentation
3

### Contribution
3
