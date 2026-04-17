# RegionUDF: Region-Aware Unsigned Distance Fields for Surface Reconstruction from Point Clouds

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Distance fields offer a powerful representation for continuous geometry, yet current learning-based neural unsigned distance fields (UDFs) remain limited in their ability to capture data patterns and generalize to real-world open surfaces. Point-Based methods mitigate grid quantization errors but current work often oversmooth local details, as query features are obtained solely through interpolation of point-wise features which are aggregated over large receptive fields. To address this, we propose a $ \textit{discriminative region representation} $ that fuses narrow neighborhood features with broader contextual point-wise features, and a $ \textit{primitive-based region representation} $ that decomposes the query region into triplet-defined primitives, enabling the detailed encoding of local surface geometry and the clear distinction of multi‑layer structures. Building on these designs, we propose $ \textit{RegionUDF} $, a region-aware UDF framework that achieves state-of-the-art open-surface reconstruction on both object- and room-level scenes, with additional validation on watertight shapes. Extensive experiments on synthetic and real-world datasets demonstrate superior accuracy and robust cross-domain generalization. Our source code will be available at $ \textit{[no-name-for-blind-review]} $.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes a UDF learning method from point cloud. It extracts two hierarchical features, including narrow region feature and point-wise feature, and then blend them as the local feature descriptor. The network is trained and tested on several datasets for evaluation.

### Strengths
The work introduces narrow, fine-grained region features to capture local details.

It decomposes the local neighborhoods into a set of triplets for primitive-based features.

### Weaknesses
The feature extraction of Equation (1) is not well-motivated. It'd better to explain the designing paradigm of the neural network clearly.

It seems that the K of neighboring points is important for the feature extraction. But the setting is unclear. 

The trainings on different datasets are individually and the testing are also individually. Every neural network can only be used for the same category data. The generalization ability is limited. 

The baseline methods are a bit out of date.

### Questions
$\varphi$ is defined after Equation 3, but it first appears in Equation 1.

In the first equations of Equation 1 and 2, the second $p\in N_q$, but what is the first $p$?

What is the impact of different K? K is a sensitive parameter, but it is not discussed.

How to extract surfaces from UDF?

### Soundness
3

### Presentation
2

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
This paper proposes a region-aware unsigned distance function representation for surface reconstruction from point clouds. The key idea is to incorporate region-level and primitive-based features into the query function, aiming to enhance local geometric representation.

### Strengths
1. The proposed region-primitive feature construction is simple and can be easily integrated into standard UDF reconstruction pipelines.

2. The overall method is presented in a clear and structured manner, with well-defined modules and straightforward implementation steps, which makes the approach easy to follow.

### Weaknesses
1. The main contribution focuses on modifying feature construction within a standard UDF framework, without introducing any new geometric priors or reconstruction paradigms. This represents a relatively incremental improvement compared to recent advances such as segment-based or kernel-based reconstruction methods.

2. The paper lacks comparison with strong baselines like NKSR, which is widely recognized as a high-precision, non-learning benchmark for surface reconstruction.

3. The performance gains mainly over older UDF methods (e.g., GeoUDF, published two years ago) are modest and largely expected given the richer local feature aggregation. Without demonstrating clear advantages over stronger baselines, the contribution falls short of the level typically expected for a top-tier venue.

4. Although the motivation is to enhance local region representation, the experiments do not convincingly demonstrate qualitative superiority in challenging areas such as thin structures and boundary regions.

### Questions
1. The Gaussian noise level used in your experiments (σ = 0.005–0.01) appears substantially low (e.g., σ = 0.02–0.05 or with outliers). This raises concerns about the strength of the claimed robustness. Could you justify this choice of noise level, and discuss how your method would perform under more realistic or higher noise conditions? Do you expect the relative gains over existing UDF baselines to persist in such settings?



2. Your primitive-based feature uses three neighboring points to form a local patch representation. Conceptually, this seems to introduce a slightly richer local prior compared to the two-point segments used in SALS, but it is unclear why three points were specifically chosen. If the goal is to encode more complex local geometry, why not use four or more points to capture higher-order structures? Is there a principled reason for using a triplet instead of other configurations, or is this choice mainly empirical? How does the representational capacity of a triplet compare to a segment or larger patches?



3. Since the method builds on local feature enrichment rather than introducing a structural constraint, it remains unclear whether the observed improvements are robust to more challenging conditions. Could you explain why this formulation should maintain its advantages under high noise levels, sparse sampling, or real-world data, and how it provides more than incremental feature engineering?



4. A key motivation of the method is enhancing local geometric representation. However, the current experimental evidence does not clearly isolate performance in thin structures or boundary regions, which are typical failure cases for UDF methods. Could you provide targeted analysis or additional results to substantiate the claimed improvements in these challenging areas?

### Soundness
3

### Presentation
4

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
This manuscript proposes RegionUDF. a region-aware unsigned distance field (UDF) framework that improves the accuracy and robustness of open-surface reconstruction by fusing fine-grained local region features with global contextual point features. RegionUDF demonstrates good performance across synthetic and real datasets, object-level and scene-level reconstruction, and cross-domain generalization.

### Strengths
1. A good-written manuscript.
2. Introducing a novel and effective discriminative region representation that explicitly fuses fine-grained local neighborhood features with broad contextual point-wise features. Enhances the ability to express local geometric structures, especially for complex non-manifold structures. 
3. Mitigating the over-smoothing problem inherent in point-wise features, as convincingly demonstrated by its superior qualitative results.

### Weaknesses
1. No comparison of optimizing speed and memory usage is provided.
2. The manuscript lacks a discussion of failure cases.
3. UODF (CVPR’24) should be included in comparison methods.

### Questions
1. UODF (Unsigned Orthogonal Distance Fields: An Accurate Neural Implicit Representation for Diverse 3D Shapes, CVPR’24), which performs well on watertight shapes, should be included in comparison methods.
2. Optimizing speed, memory usage, and model parameters should to be reported and compared with main baselines.
3. In Tab.8 in Suppl., why gradient loss seems less critical for RegionUDF? Only depends on the introduction of region features?
4. Will the proposed method fail on multi-layered or complex topological structures? I suggest authors report the failure cases for discussion.

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
3

### Summary
RegionUDF proposes a region-aware UDF for surface reconstruction that fuses point-wise contextual features with local region features extracted from the K-NN neighborhood of each query. The method also introduces a primitive-based decomposition (angle-sorted triplets on a projected sphere) that builds primitive features which are aggregated into the final query feature.

### Strengths
One major technical contribution is the combination of wide-context point-wise features with narrow region features that addresses a weakness of point-based UDF methods (oversmoothing vs. noisy local detail). The primitive decomposition is a plausible way to capture multi-patch local structure.

The authors evaluate on watertight ShapeNet, ShapeNet Cars (open), ABC / non-manifold ABC, ScanNet and Matterport3D (room-scale). They report improvements across object and scene scales (tables / figures).

### Weaknesses
The authors still rely on MeshUDF / GeoUDF meshing heuristics and remove vertices beyond a threshold. They note this as a limitation and suggest a dedicated UDF->manifold extractor as future work. This is important because some reconstruction improvements may come from meshing heuristics rather than the learned field itself. Please be explicit about how much improvement remains when using the same meshing pipeline for all methods.

Are the improvements stable across different meshing heuristics (MeshUDF vs GeoUDF) when they are applied identically to all methods? Please report both sets of numbers or at least clarify.

In the ablation study, the gains could partially stem from a stronger backbone (PointTransformer V2) or network capacity. The ablation compares to POCO-style baselines but I’d like a controlled experiment: same backbone capacity, with and without the region/primitive module. It would be better to show parameter counts and runtime to rule out trivial capacity explanations.

L470: “Abaltion” → “Ablation” appears in the appendix headings

### Questions
The primitive construction uses spherical projection and angle-sorting. The paper acknowledges ambiguity in multi-layered structures (Sec A.2), but does not quantify failure cases. How often primitives span different surfaces, and whether that harms the result?

You report training times on 2×RTX3090 (56h / 24h). What is the inference time to reconstruct a single room (including meshing) at resolution 128 and 256? Peak GPU memory?

### Soundness
3

### Presentation
3

### Contribution
1
