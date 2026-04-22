# Splat Feature Solver

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Feature lifting has emerged as a crucial component in 3D scene understanding, enabling the attachment of rich image feature descriptors (e.g., DINO, CLIP) onto splat-based 3D representations. The core challenge lies in optimally assigning rich general attributes to 3D primitives while addressing the inconsistency issues from multi-view images. We present a unified, kernel- and feature-agnostic formulation of the feature lifting problem as a sparse linear inverse problem, which can be solved efficiently in closed form. Our approach admits a provable upper bound on the global optimal error under convex losses for delivering high quality lifted features. To address inconsistencies and noise in multi-view observations, we introduce two complementary regularization strategies to stabilize the solution and enhance semantic fidelity. Tikhonov Guidance enforces numerical stability through soft diagonal dominance, while Post-Lifting Aggregation filters noisy inputs via feature clustering. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on open-vocabulary 3D segmentation benchmarks, outperforming training-based, grouping-based, and heuristic-forward baselines while producing the lifted features in minutes. Demo Video, \textbf{Code} and \textbf{demo website} are all inside the supplementary.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
1 The paper formulates feature lifting in splat-based 3D representations as a sparse linear inverse problem AX=B , and proposes a closed-form row-sum preconditioner solver with a provable (1+β) -approximation error bound under convex losses.

2 It introduces two regularization strategies—Tikhonov Guidance (to enhance diagonal dominance and numerical stability) and Post-Lifting Aggregation (to filter noisy SAM masks via clustering)—and evaluates the method on open-vocabulary 3D semantic segmentation using mIoU on LeRF-OVS and 3D-OVS benchmarks.

3 Comprehensive ablation studies validate each component, and experiments across multiple splat kernels (3DGS, 2DGS, DBS) and feature backbones (CLIP, DINOv2, ResNet, etc.) demonstrate state-of-the-art performance with minutes-level runtime, confirming both effectiveness and generality.

### Strengths
The paper presents a strong and cohesive contribution by formulating feature lifting in splat-based 3D representations as a sparse linear inverse problem with an original and theoretically grounded perspective that unifies and improves upon prior heuristic, training-based, and grouping-based methods. The proposed closed-form solver with a provable (1+β) -approximation error bound enhances both originality and technical quality, while the two lightweight yet effective regularization strategies (Tikhonov Guidance and Post-Lifting Aggregation) address real-world noise and inconsistency issues without sacrificing efficiency. The work is clearly presented with a logical flow from problem definition to theoretical analysis and extensive experiments across kernels, features, and benchmarks. Its significance lies in enabling fast, general, and high-fidelity semantic enrichment of 3D scenes—advancing open-vocabulary 3D understanding with practical impact and theoretical insight.

### Weaknesses
1 The paper lacks a clear and detailed pipeline diagram—Figure 1 is overly abstract and fails to illustrate concretely how high-dimensional features are assigned to Gaussian splats, making the core lifting mechanism hard to grasp.

2 Despite claiming SOTA performance on LeRF-OVS, the paper provides minimal qualitative comparisons (only Figures 2 and 8, each against a single baseline), severely limiting confidence in the method’s robustness across diverse scenes.

3 Table 1(b) reports cosine similarity across feature types but doesn’t link these metrics to downstream task gains, raising questions about its necessity.

4 Additionally, the paper suffers from formatting issues. Such as overly large table captions, excessively long figure titles, and inconsistent font sizes in visuals—detracting from readability and professionalism.

### Questions
Similar to weakness.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Splat Feature Solver (SFS), a self-supervised framework designed to learn 3D scene representations using 3D Gaussian Splatting (3DGS) as the core rendering primitive.
The key idea is to reconstruct multi-view images from learnable Gaussian feature fields, optimizing both photometric and perceptual losses without camera supervision. The authors claim that the model learns geometry-aware features from raw multi-view imagery and achieves competitive results on downstream 3D tasks such as novel-view synthesis and depth prediction.

### Strengths
1， The paper targets a highly relevant goal，efficient and scalable self-supervised 3D representation learning using Gaussian splatting, an area of growing academic and industrial interest.

2，Compared to NeRF-style volumetric sampling, the splatting-based pipeline is computationally lighter and supports faster convergence. The engineering design is practical and well-motivated.

3， The pipeline, loss functions, and training strategy are described with good clarity. Figures are intuitive and well-illustrated.

4，Experiments across multiple datasets show consistent, if modest, improvements over previous self-supervised 3D baselines. Ablation results are included to demonstrate the influence of loss terms and feature solvers.

### Weaknesses
1, The method essentially reuses the existing Gaussian Splatting pipeline as a self-supervised pretext task, with minor modifications to the loss formulation. The “feature solver” concept adds no clearly new principle beyond standard photometric reconstruction with latent feature regularization. The contribution is incremental and primarily engineering-driven.

2, The paper does not provide any analysis explaining why the proposed self-supervised optimization leads to meaningful 3D representations. There is no exploration of feature alignment, depth consistency, or the information content of Gaussian features. The claim of “self-supervised 3D understanding” is thus empirically unsubstantiated.

3, Although the method is described as “self-supervised,” it implicitly assumes access to approximate camera poses or adjacency constraints during training. The paper does not clarify how SFS handles unposed or unordered images. True pose-free capability is not demonstrated.

4, Reported performance gains over NeRF-based SSL or other Gaussian-based SSL approaches (e.g., GS3, UniSplat) are small (often <1% absolute improvement). Key comparisons to Pose-Free Gaussian Fields, PixelSplat, or DUSt3R are missing, leaving the evaluation incomplete.

### Questions
1, How does SFS perform on unposed or pose-free datasets compared to pose-supervised ones?

2, Could the authors provide any quantitative measure of learned 3D consistency (e.g., reprojection error or latent geometry alignment)?

3, Are the improvements statistically significant across multiple runs?

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
3

### Summary
This paper addresses the feature lifting problem, which aims to transform vision foundation model features (e.g., DINO/CLIP) into Gaussian-splatting-based representations. The authors innovatively formulate this problem as a sparse linear inverse problem and propose a closed-form solver. In addition, they introduce two modules to suppress the noise inherent in foundation model features. Experimental results demonstrate the proposed method’s effectiveness and generalization across different Gaussian splatting kernels. Overall, the paper presents solid contributions both theoretically and practically.

### Strengths
* The paper formulates feature lifting as a sparse linear inverse problem and derives a closed-form solution, which is elegant and theoretically sound.
* The mathematical derivations and reasoning are solid and well-motivated.

### Weaknesses
* The space allocation in the manuscript is unbalanced — too few visualizations are included in the main text, while most figures are deferred to the supplementary material. Moreover, some figure captions are vague.
* The paper lacks discussion and comparison with feed-forward models related to VGGT, such as Anysplat, which can also lift DINOv2 features to Gaussian-splatting representations. Considering their feed-forward nature, such models are likely to offer faster runtime performance.
* In Table 1 (b) and (c) lack highlighted numerical values, making it difficult to visually discern the performance trends.
* The statement “Third, many existing methods are specialized for particular feature types or geometric kernels, which may limit generalization across broader settings” requires further clarification. Specifically, which feature types are those methods specialized for? It would also strengthen the paper to include visual evidence showing that the proposed approach generalizes better across diverse cases.

### Questions
* In Figure 2, why are the segmentation boundaries so noisy? Moreover, for the two similar and adjacent eggs, why does the method only able to segment one of them?
* According to Section 3.2, the solver is closed-form and enables one-shot estimation without iterative SGD. Theoretically, this should result in very fast inference, yet the runtime is still reported as 1–3 minutes. Could the authors clarify what factors contribute to this computational cost?

### Soundness
3

### Presentation
2

### Contribution
3
