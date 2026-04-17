# CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity. Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances. However, this long-standing paradigm suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual "popping" artifacts during transitions, degrading the user experience. We argue that the explicit, primitive-based nature of the emerging 3D Gaussian Splatting (3DGS) technique enables a more ideal paradigm: Continuous LoD (CLoD). A CLoD approach facilitates smooth and seamless quality scaling within a single unified model, thereby circumventing the core problems of DLOD. To this end, we introduce CLoD-GS, a framework that integrates a continuous LoD mechanism directly into a 3DGS representation. Our method introduces a learnable distance-dependent decay parameter for each Gaussian primitive that dynamically adjusts its opacity based on viewpoint proximity. This allows for the progressive and smooth filtering of less significant primitives, effectively creating a continuous spectrum of detail within one model. To train this model to be robust across all distances, we introduce a virtual distance scaling mechanism with point count regularization. Our approach not only eliminates the storage overhead and visual artifacts of discrete methods but also reduces the primitive count and memory footprint of the final model. Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from a single model, delivering high-fidelity results across a wide range of performance targets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents CLOD-GS, a continuous level-of-detail framework for 3D Gaussian Splatting that enables smooth, pop-free quality scaling within a single model. Each Gaussian learns a distance-dependent opacity decay, allowing fine-to-coarse detail control without discrete model swaps. A coarse-to-fine training strategy with virtual distance scaling and sparsity regularization ensures robustness across viewing scales. CLOD-GS reduces Gaussian count and memory while maintaining or improving rendering quality on benchmarks like BungeeNeRF and Tanks & Temples.

### Strengths
1. The LoD topic is meaningful.
2. The paper is well written and clearly organized.

### Weaknesses
**Major**

1. The coarse-to-fine training strategy is similar to that used in Octree-GS.
2. No demo has been submitted; therefore, the performance of the proposed method cannot be effectively demonstrated.
3. This is an LoD paper, which should be related to large-scale reconstruction for the topic to be meaningful. However, the paper only conducts experiments on small-scale scenes, and the reported metrics are similar to those of the baselines. I believe these results could easily be surpassed by tuning the hyperparameters. Also, I thinks the authors should compare with hierarchical-3dgs and more lod baselines.

**Minor**

1. The paper lacks references to key related work on Level of Detail (LoD) rendering. The authors should consider citing the following works:
    - *Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian*
    - *A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets*
    - *Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes*
    - *Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes*

### Questions
please see the weakness above.

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
4

### Summary
This paper proposes CLoD-GS, a continuous level-of-detail (CLoD) mechanism embedded directly into the 3D Gaussian Splatting (3DGS) representation. Instead of maintaining multiple discrete LoD models (which incurs storage overhead and produces visual "popping" during switching), this work introduces a learnable distance-dependent decay factor for each Gaussian, allowing opacities to vary smoothly with viewpoint distance.

### Strengths
1. Continuous control without model duplication, avoids DLoD storage and popping artifacts.

2. Minimal parameter overhead (~1.6% per Gaussian), practical for deployment.

3. Unified training strategy that naturally encourages Gaussian sparsity at distant views.

### Weaknesses
1. The method assumes that viewing distance is the dominant factor in determining perceptual relevance. However, perceptual saliency often depends on texture frequency, geometric edges, semantic relevance, and shading sensitivity, not merely spatial distance.

2. The paper does not provide sensitivity analyses showing whether these parameters generalize across scene types.

3. Since the decay factor is learned independently for each Gaussian, the simplification behavior lacks global structural organization.

### Questions
1. Could semantic or frequency-based saliency be introduced alongside distance? Would this improve LoD behavior in highly textured scenes?

2. How stable is the learned decay factor across camera trajectory distributions? Does it overfit camera placement biases?

3. Could this be extended to dynamic scenes where Gaussian parameters evolve over time?

4. The opacity threshold scaling seems hand-tuned. Have you explored adaptive thresholds learned per scene?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CLoD-GS, which augments each 3D Gaussian with a learnable distance-decay parameter to modulate opacity continuously as viewpoint distance changes. This creates a single representation that supports smooth quality–performance trade-offs at test time. A complementary training strategies and regularizers are incorporated for effective learning.

### Strengths
1. The proposed distance-adaptive opacity gives a smooth transition within a single representation, avoiding multi-copy asset storage and popping artifacts of DLoD.
2. The virtual distance scale offers a continuous speed-quality curve, which is practical for real-time budegets.
3. This paper is well-written and easy to follow.

### Weaknesses
1. Prior efforts have designed sophisticated LoD for 3DGS [A], which are very effective even in large-scale scenes; a direct, side-by-side comparison with [A] is missing and would strengthen the empirical case and clarify trade-offs.
2. The effectiveness of distance-driven opacity in regions with high-frequency textures or potential aliasing—central issues in LoD—is not convincingly demonstrated.
3. Because opacity modulation retains the finest Gaussians and culls softly, rasterization workload may remain high. A report of FPS and visible-splat counts may be included to demonstrate the efficiency of the proposed method.
4. Statements such as "treating preliminary 3DGS as the key innovation" and "consistently achieves state-of-the-art rendering quality" appear stronger than the presented evidence supports.

[A] A hierarchical 3d gaussian representation for real-time rendering of very large datasets. [SIGGRAPH 24]

### Questions
1. Does the distance-driven opacity exhibit hole artifacts under aggressive thresholds or grazing views? Some visual or quantitative results could demonstrate the generalizability of the proposed method.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a simple yet effective way to make 3D GS models smoothly adjust their rendering quality without switching between discrete versions. Traditional discrete lod methods require multiple stored models, leading to high memory use and visible popping when changing between levels. The paper overcomes this by introducing a learnable distance-based opacity control that allows each Gaussian to fade smoothly as the camera moves farther away. Combined with a coarse-to-fine training process that encourages the model to use fewer points at larger viewing distances, this approach produces a single unified model capable of continuous, artifact-free LoD scaling. Experiments on several public datasets show that CLoD-GS achieves comparable or better visual quality than existing methods, while using about 30–40% fewer Gaussians and memory. The method adds minimal computational cost, integrates easily into existing pipelines, and provides high-quality, scalable rendering results.

### Strengths
- The paper introduces a simple, learnable distance-based opacity mechanism that enables continuous LoD in 3D Gaussian Splatting without adding complex structures or large overhead.
- Experiments show consistent improvement in visual quality and memory efficiency across multiple datasets and methods.
- The paper is well written, provides detailed implementation settings, ablation studies, and public datasets, making the work easy to understand and verify.

### Weaknesses
- The paper lacks deeper analysis of why the proposed opacity function and training design work optimally or how parameters affect convergence and quality.
- Experiments focus only on static scenes. How about generalization to dynamic or large-scale environments remains unexplored?
- Only briefly mentions similar approaches without in-depth experimental or conceptual comparison.

### Questions
Same as weakness

### Soundness
3

### Presentation
3

### Contribution
3
