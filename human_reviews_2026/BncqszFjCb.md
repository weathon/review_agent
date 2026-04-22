# Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advancements in computer vision have successfully extended Open-vocabulary segmentation (OVS) to the 3D domain by leveraging 3D Gaussian Splatting (3D-GS). 
Despite this progress, efficiently rendering the high-dimensional features required for open-vocabulary queries poses a significant challenge. 
Existing methods employ codebooks or feature compression, causing information loss, thereby degrading segmentation quality.
To address this limitation, we introduce Quantile Rendering (Q-Render), a novel rendering strategy for 3D Gaussians that efficiently handles high-dimensional features while maintaining high fidelity. 
Unlike conventional volume rendering, which densely samples all 3D Gaussians intersecting each ray, Q-Render sparsely samples only those with dominant influence along the ray. 
By integrating Q-Render into a generalizable 3D neural network, we also propose Gaussian Splatting Network (GS-Net), which predicts Gaussian features in a generalizable manner. 
Extensive experiments on ScanNet and LeRF demonstrate that our framework outperforms state-of-the-art methods, while enabling real-time rendering with an approximate ${\sim}43.7\times$ speedup on 512-D feature maps.
Code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Quantile Rendering (Q-Render), an efficient feature rendering algorithm for 3D Gaussian Splatting. Q-Render sparsely samples quantile Gaussians that dominate transmittance along each ray, cutting complexity from O(NC) to O(N+KC). Integrated into a Gaussian Splatting Network (GS-Net), it enables scalable training with high-dimensional CLIP features for open-vocabulary 3D segmentation. Experiments on ScanNet and LeRF-OVS show state-of-the-art accuracy and up to 43.7× faster rendering.

### Strengths
1. Efficient Rendering Design: Q-Render introduces a principled quantile-based sampling strategy that substantially reduces computation for high-dimensional feature rendering without sacrificing accuracy.
2. Solid Integration & Generality: The method integrates seamlessly into 3D Gaussian Splatting pipelines and generalizes well across neural backbones, bridging 2D foundation models and 3D representations effectively.
3. The paper is well written and clearly organized.

### Weaknesses
1. In figure 6, I think the performance improvement of Q-Render is minor and I think the motivation of the authors apply the quantile rendering is not very strong.
2. The feed-forward pipeline for 3d scene understanding has already been applied in early methods, like SIU3R (SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment) and SegMASt3R (SegMASt3R: Geometry Grounded Segment Matching).
3. No demo has been submitted; therefore, the performance of the proposed method cannot be effectively demonstrated.

### Questions
1. What are the number of input views?
2. What is the training time / GPU memory?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Quantile Rendering (Q-Render), a novel and efficient rendering algorithm for high-dimensional features in 3D Gaussian Splatting (3D-GS). Traditional 3D-GS-based rendering densely accumulates all Gaussians intersecting each ray, which becomes computationally prohibitive when rendering high-dimensional embeddings such as 512-D CLIP features. Q-Render addresses this by sparsely sampling “quantile Gaussians”, those that contribute most significantly to the transmittance change along the ray.

### Strengths
1. The authors evaluate on two large-scale benchmarks with extensive ablations, qualitative visualization, and speed analyses.

2. The approach achieves >40× rendering speedup for 512-D features while improving mIoU—impressive for real-world scalability.

3. Bridges 2D foundation models (CLIP, SAM) with 3D Gaussian representations—a timely and valuable direction for the ICLR community.

### Weaknesses
1. The quantile sampling justification is intuitive but lacks a quantitative analysis of approximation error relative to volume rendering.

2. Although ablations are provided, an adaptive or learned K would make the method more robust and generalizable.

3. The paper primarily focuses on indoor datasets (ScanNet, LeRF-OVS). Outdoor or multi-view generalization tests would strengthen the claim of scalability.

4. Only MinkUNet and PTv3 are explored. An analysis of architecture-agnostic behavior would enhance generality.

### Questions
1. Could the authors provide a theoretical bound or empirical analysis quantifying the approximation error between Q-Render and full volume rendering?

2. Is it possible to make K adaptive based on the transmittance variance along each ray?

3. How sensitive is the approach to the noise in Gaussian opacity or density estimation?

4. Can Q-Render be applied to RGB image rendering or only to feature maps?

5. What is the memory footprint of Q-Render compared to top-K or compressed-feature methods?

6. Have the authors tested Q-Render for dynamic scenes or time-varying Gaussians?

### Soundness
2

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
4

### Summary
This paper proposed quantile rendering to accelerate the rasterization of 3D Gaussians. Specifically, the proposed method selects a subset of critical 3D Gaussians that have a significant influence on the final rendering results and skips the rest. The experiments on 3D open vocabulary segmentation show that the proposed method can speed up the rendering while achieving state-of-the-art performance.

### Strengths
The idea is interesting and well-motivated.
The performances shown in the experiments are good.

### Weaknesses
1. Paper presentation: In the abstract, the GS-Net and open-vocabulary 3D semantic segmentation are mentioned, without introducing their relationship with the Q-Render, making it hard to follow. The topic of the paper is unclear. If the proposed method is designed for general high-dimensional feature rendering, why is it only evaluated on the 3D open-vocabulary semantic segmentation task? If the method is specifically designed for 3D open-vocabulary semantic segmentation, then there is a lack of introduction to the specific problem.
2. Unreliable experiments: The final method used for 3D open-vocabulary semantic segmentation consists of two parts: the per-Gaussian feature extracted by GS-Net and the Q-Rendering procedure. It is hard to justify the separate contribution of the two parts to the performance gain in Tab. 3. To my understanding, as Q-Render only requires Gaussians with high-dimensional features as input, it is not hard to disentangle the two parts: (1) Replace the rendering algorithm of the baselines using Q-Render and compare their performance. (2) Replace the feature extraction model of the baselines using the GS-Net and compare their performance.
3. Minor typos: in L.199, the reference line 6 of Algorithm 1 should be enclosed in parentheses.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Q-Render, an efficient rendering algorithm for high-dimensional feature rendering in 3D Gaussian Splatting. Q-Render sparsely samples dominant Gaussians through transmittance change analysis. The authors integrate Q-Render with a 3D neural network (GS-Net) and evaluate it on two open-vocabulary 3D semantic segmentation benchmarks: ScanNet and LeRF-OVS, demonstrating superior performance.

### Strengths
1. The paper proposes a quantile sampling strategy that identifies critical Gaussians through transmittance analysis, which is theoretically motivated and practically effective.

2.  The method demonstrates superior performance and efficiency on both ScanNet and LeRF-OVS open-vocabulary 3D semantic segmentation benchmarks, achieving ~43.7× speed gains when rendering 512-D feature maps.

### Weaknesses
1. The paper lacks theoretical analysis of Q-Render's approximation error to volume rendering or convergence guarantees. The mathematical justification for the normalization operation (line 20 in Algorithm 1) is insufficient.

2. While K significantly impacts performance, the paper lacks an adaptive strategy for selecting K. Why is uniform partitioning (k+1)/(K+1) chosen? Have adaptive thresholds been considered?

3. The paper does not sufficiently analyze when Q-Render might fail or how it performs on scenes with non-uniform transmittance distributions.

4. In Table 2, the authors "reproduce" baseline results but use different training and evaluation setups, which may not provide a fair comparison.

5.  The necessity of de-voxelization is not independently validated. The paper lacks failure case analysis showing scenarios where Q-Render underperforms.

### Questions
1. Can you provide an error bound for Q-Render's approximation to volume rendering? What are the theoretical guarantees for the normalization step?

2.  Is the optimal value of K related to scene complexity and feature dimensionality? Can you design an adaptive strategy for K selection?

3. Performance anomaly: Why does Q-Render with K=40 achieve higher mIoU (50.85) than volume rendering (49.02) in Table 6? Does this suggest that sparse sampling has a regularization effect?

4.  How can the information loss from voxelization (Figure 7) be quantified? Are there better voxelization strategies to mitigate this loss?

5. Threshold design: Is uniform partitioning of transmittance thresholds optimal? Have you considered scene-adaptive thresholds based on local transmittance distributions?

6.  How is de-voxelization specifically implemented? Is K fixed during both training and inference? How do you handle regions with very small transmittance changes?

### Soundness
3

### Presentation
3

### Contribution
3
