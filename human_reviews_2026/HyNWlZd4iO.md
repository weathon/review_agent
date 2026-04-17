# GOOD: Geometry-guided Out-of-Distribution Modeling for Open-set Test-time Adaptation in Point Cloud Semantic Segmentation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Open-set Test-time Adaptation (OSTTA) has been introduced to address the challenges of both online model optimization and open-set recognition. Despite the demonstrated success of OSTTA methodologies in 2D image recognition, their application to 3D point cloud semantic segmentation is still hindered by the complexities of point cloud data, particularly the imbalance between known (in-distribution, ID) and unknown (out-of-distribution, OOD) data, where known samples dominate and unknown instances are often sparse or even absent. In this paper, we propose a simple yet effective strategy, termed Geometry-guided Out-of-Distribution Modeling (GOOD), specifically designed to address OSTTA for 3D point cloud semantic segmentation. Technically, we first leverage geometric priors to cluster the point cloud into superpoints, thereby mitigating the numerical disparity between individual points and providing a more structured data representation. Then, we introduce a novel confidence metric to effectively distinguish between known and unknown superpoints. Additionally, prototype-based representations are integrated to enhance the discrimination between ID and OOD regions, facilitating robust segmentation. We validate the efficacy of GOOD across four benchmark datasets. Remarkably, on the Synth4D to SemanticKITTI task, GOOD outperforms HGL by 1.93%, 8.99%, and 7.91% in mIoU, AUROC, and FPR95, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GOOD, a framework for open-set test-time adaptation (OSTTA) in 3D point cloud semantic segmentation. GOOD addresses the imbalance between in-distribution (ID) and out-of-distribution (OOD) points by introducing two key branches: the Superpoint Representation Branch (SRB) and the Temporal Pseudo-label Branch (TPB). Experiments demonstrate that GOOD achieves superior results compared to existing methods.

### Strengths
1. The paper is clearly written, and the figures are effective in illustrating the technical flow.
2. The design of SRB and TPB is technically sound: SRB clusters point clouds into superpoints and defines confidence metrics to better distinguish ID and OOD regions, while TPB enforces temporal consistency across sequential frames to produce more stable pseudo-labels.
3. The experimental evaluation is extensive, providing strong evidence of the method’s effectiveness.

### Weaknesses
1. For 3D open-set segmentation, there are models such as OpenScene that leverage open-vocabulary queries to achieve open-set segmentation. A comparison with such models would be valuable. In particular, how would be the performance difference, and what are the differences and benefits of adopting an open-set test-time adaptation approach compared to the open-vocabulary segmentation paradigm?
2. Missing related work. Several recent works on 3D segmentation are relevant and should be cited:
   - OpenScene: 3D Scene Understanding with Open Vocabularies (CVPR 2023)
   - Rethinking few-shot 3d point cloud semantic segmentation (CVPR 2024)
   - Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation (ICLR 2025)
   - Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model (CVPR 2025)

### Questions
Please refer to the weakness part and address the concerns there.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to more effectively distinguish ID and OOD samples through indicators such as superpoint confidence and purity, as well as ID superpoint prototypes. Furthermore, it designs a temporal pseudo-label method to generate more accurate pseudo-labels. Experimental results show that it achieves optimal performance under the open-set TTA setting, which demonstrates the effectiveness of its method.

### Strengths
1.This paper achieves a better distinction between ID and OOD samples through superpoint clustering and the utilization of superpoints, which is well demonstrated by visualization. Therefore, it exhibits strong innovation and rationality.

2. 
The paper's method is simple and efficient, and can be applied to many point cloud segmentation methods.



3.The paper's experiments and visualizations are comprehensive, and the results all demonstrate the effectiveness of its method.

### Weaknesses
1.The paper reports the running time but fails to report the computational overhead such as GFLOPs.

2.The definition of symbols in the formulas is rather confusing. For example, the definition of the subscript of y$_{j,c}$  in Eq. 1 is not clearly explained.

### Questions
1.Is it possible to add more experiments with different backbones to demonstrate the generality of the method?

2.In Table 4, the increase in mIOU metric when adding more components is not significant. Could the reasons for this be analyzed?

3.The impact of noisy samples be considered when constructing prototypes? This is quite common in TTA methods for 2D images.

4.In Table 4, does " superpoint representation (SR)" refer to the use of superpoint purity and confidence in GMM? It should be explained in more detail.

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
3

### Summary
This paper addresses the challenging problem of open-set test-time adaptation (OSTTA) in 3D point cloud semantic segmentation, where models must adapt to unseen target domains containing both known and unknown categories. The authors identify a key limitation in existing OSTTA methods—most were designed for 2D image data and perform poorly on 3D point clouds due to geometric complexity and severe imbalance between in-distribution (ID) and out-of-distribution (OOD) samples. To address this, the paper proposes GOOD (Geometry-guided Out-of-Distribution Modeling), a novel framework that leverages geometric priors to cluster point clouds into superpoints, thereby transforming point-level imbalance into a structured representation. GOOD introduces (1) a Superpoint Representation Branch (SRB) with a Gaussian Mixture Model and prototype learning to distinguish ID and OOD superpoints, and (2) a Temporal Pseudo-label Branch (TPB) that enforces temporal consistency for reliable pseudo-label generation. The method is evaluated on four benchmarks, showing certain improvements over existing TTA and OSTTA baselines.

### Strengths
The paper presents a clear and well-motivated method that effectively extends OSTTA to 3D point clouds, with experimental validation and meaningful performance improvements.

### Weaknesses
1. The paper assumes access to reliable ego poses for temporal alignment, which might not hold in real-world scenarios—this assumption and its limitations are not discussed.

2. Prototypes are updated via EMA using pseudo-labeled superpoints. If the initial pseudo-labels are noisy, prototypes may drift toward incorrect centroids and subsequently reinforce wrong ID/OOD assignments. The paper does not analyze prototype stability over time or propose safeguards (re-weighting, outlier removal, confidence gating).

3. TPB depends on aggregating multiple frames via ego-pose alignment for K-NN temporal neighborhoods. In practice, pose noise (e.g., localization drift) or missing pose data will break temporal correspondence. The paper does not quantify sensitivity to alignment error nor propose how the method behaves when only single-frame inputs are available.

4. Several design choices (EMA coefficient α, cosine similarity threshold τ, use of Dice loss for ID pseudo-label learning) are empirically reasonable but not theoretically motivated. This weakens understanding of when and why the method will succeed or fail.

5. The paper mentions partitioning the point cloud into sub-regions and applying RANSAC to each, but does not specify sub-region division criteria or RANSAC hyperparameters. Only the original RANSAC literature is cited, and no task-specific settings for 3D point clouds are provided. This ambiguity makes the step impossible to replicate, which undermines the method’s practicality.

6. Appendix A.3.5 details the performance of different superpoint confidence metrics via Table 11 and concludes that "Purity+Entropy balances closed-set and open-set performance", but the main text (Section 3.2.2) only defines the confidence formula "C_k^sup=C_k^ent∙C_k^pur" and does not cite the appendix’s comparative results to support why "product" is better than other combinations. This makes the main text’s formula choice lack empirical support, and readers need to flip to the appendix to understand the rationale, disrupting the logical flow of the argument.

7. Eq.3 uses "z_i^t" to represent superpoint embedding, but the preceding text clearly defines "z_k^t" as the mean embedding of superpoint k—this confusion between "i" (point-level) and "k" (superpoint-level) makes readers unsure whether similarity is calculated per point or per superpoint.

### Questions
1. The SRB uses a GMM on purity and entropy. Have you experimented with other methods to avoid the Gaussian assumption? If so, why was GMM chosen?

2. The paper notes that prototypes are updated via EMA using pseudo-labeled superpoints, but does not quantify how prototype drift (caused by noisy initial pseudo-labels) affects long-term ID/OOD discrimination performance. Have you conducted experiments to measure prototype stability over consecutive test frames? Additionally, do you have strategies to mitigate drift, such as re-weighting pseudo-labels by superpoint confidence or removing outlier pseudo-labels before prototype update?

3. The TPB relies on multi-frame alignment via ego poses to build temporal neighborhoods, but real-world pose noise may corrupt this alignment. Have you quantified GOOD’s sensitivity to pose error (e.g., testing with artificial pose noise added to frames)? Specifically, how does pose noise impact the accuracy of temporal pseudo-labels and final segmentation performance?

4. Fig. 3 does not seem to be cited or mentioned in the main text. Perhaps a brief explanation can be given for it.

### Soundness
3

### Presentation
3

### Contribution
3
