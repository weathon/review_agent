# Expo-GS: Exposure-Aware Signed Distance Function in Gaussian Splatting for High Dynamic Range

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
High dynamic range novel view synthesis (HDR-NVS) remains challenged by geometric artifacts and radiometric distortions under multi-exposure conditions, primarily due to existing methods ignoring exposure and over-relying on color cues. Inspired by the integrated processing of color and structure of the human visual system (HVS), we propose Expo-GS, a novel framework that decomposes HDR-NVS into three interpretable components, namely,  Irradiance Field Training, Geometry Field Training, and Interactive Joint Training. Central to Expo-GS is the exposure-aware signed distance function (Expo-SDF), which dynamically reweights geometric supervision via localized exposure reliability estimation, suppressing noisy gradients from unstable regions while enhancing structure learning in well-exposed areas. Building on this, we design an interactive optimization strategy that synchronizes Gaussian primitive growth and pruning with evolving Expo-SDF cues, enabling exposure-aware density control and eliminating hallucinated structures near exposure transitions. Experiments show that Expo-GS significantly outperforms prior methods on both synthetic and real-world datasets. It achieves a peak PSNR of 39.06 dB under HDR settings and up to 41.38 dB in the LDR-OE configuration, excelling in preserving high-frequency textures and maintaining structural consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Expo-GS, an exposure-aware framework for HDR novel view synthesis (HDR-NVS). It introduces an Exposure-aware Signed Distance Function (Expo-SDF) integrated into 3D Gaussian Splatting (3D-GS), and a three-stage pipeline consisting of: 1. Irradiance field training for color regression; 2. Geometry field training via exposure-modulated SDF; and 3. Interactive joint optimization of geometry and radiance. The paper claims improved geometric fidelity and radiometric consistency under multi-exposure HDR conditions.

### Strengths
+ The work addresses HDR-NVS, a challenging setting not well covered in Gaussian Splatting literature.
+ Introducing exposure reliability weighting into geometric supervision is conceptually interesting.
+ Multiple ablation studies and qualitative comparisons demonstrate the framework’s impact on geometry and color reconstruction.

### Weaknesses
- The experiments are limited to the HDR-NeRF dataset, with training and testing conducted under similar exposure distributions. It remains unclear how well Expo-GS generalizes to unseen exposure levels, camera response curves, or dynamic lighting variations. It is beneficial to test the model on cross-dataset or real-world outdoor scenes where exposure and tone mapping differ significantly.
- The training strategy (8k + 12k + 10k iterations) seems complex. However, the paper provides little analysis of its sensitivity or convergence behavior. How crucial is the geometry field stage for the final results?
- The derivation of Expo-SDF assumes that a single Gaussian dominates locally, which may not hold in regions of dense overlap or transparency. How does the pseudo-SDF behave when multiple Gaussians contribute comparable densities?

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper Expo-GS introduces a new framework for High Dynamic Range Novel View Synthesis (HDR-NVS) by incorporating an exposure-aware geometric module into the 3D Gaussian Splatting (3D-GS) pipeline. The approach decomposes the task into color, geometry, and exposure components, inspired by the mechanisms of human visual perception. This design elegantly tackles a major limitation in previous methods. The proposed Exposure-Aware Signed Distance Function (Expo-SDF) makes a notable technical contribution by improving the robustness of geometry learning under complex lighting conditions.

### Strengths
1. The paper presents a comprehensive experimental evaluation, demonstrating state-of-the-art performance on standard HDR-NVS benchmarks and providing thorough ablation studies to validate the contribution of its components.

2. The proposed Expo-SDF module is a well-executed integration of an exposure-weighting mechanism into an SDF framework; however, this combination feels more like a straightforward and incremental assembly of existing concepts rather than a fundamental architectural innovation.

3. The paper is well-written and the figure is clear.

### Weaknesses
1. Overall, this paper presents an effective approach that combines HDR rendering with an exposure-aware SDF. However, the modular and stage-wise nature of its technical design makes the work read more like a well-engineered integration of existing techniques than a fundamentally innovative contribution.

In particular, the core component, Expo-SDF, is essentially an intuitive exposure-weighted extension of the traditional SDF, with limited conceptual novelty. While the paper successfully demonstrates that combining HDR and SDF is effective, it does not provide sufficient justification that such a combination is necessary or deeply integrated at the algorithmic level. Consequently, the contribution of this work lies more in its solid engineering implementation and practical system design than in proposing a genuinely new research direction or model architecture with substantial theoretical insight.


2. The comparison methods are too limited. Basically, only HDR-NeRF and HDR-GS among the compared methods are in a similar direction. The lack of sufficient comparative experiments makes the results less convincing. Perhaps some related methods, such as those in (1)–(4), could also be included for comparison. The authors’ expertise in this area is also questionable.

(1). High Dynamic Range Novel View Synthesis with Single Exposure (ICML 2025)

(2). HDR-HexPlane: Fast High Dynamic Range Radiance Fields for Dynamic Scenes (3DV 2024)

(3). LTM-NeRF: Embedding 3D Local Tone Mapping in HDR Neural Radiance Field (TPAMI 2024), this work is an extension of HDR-NeRF.

(4). HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields (ECCV 2022)


3. The paper does not include a dedicated Limitations section. Although some constraints can be inferred—such as computational overhead and dependence on multi-exposure data, explicitly acknowledging them would enhance the paper’s clarity and transparency.

### Questions
The paper thoroughly compares against HDR-GS but could be even stronger by including a direct ablation or comparison against a "vanilla" SDF integrated into 3D-GS (without the exposure-aware component). Table 6 compares to other SDF methods, but a controlled ablation within the Expo-GS framework (e.g., "Expo-GS w/ standard SDF loss") would more directly isolate the contribution of the exposure-aware part of the Expo-SDF.

### Soundness
2

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
The paper presents Expo-GS, a High Dynamic Range (HDR) novel view synthesis framework. The paper claims to remove radiometric bias and geometric inconsistency caused by varying exposures. A comprehensive qualitative and quantitative evaluation follows.

### Strengths
1. The paper is well written and clear.
2. The methodology is sound and the motivation to design a framework like this is great! This seems to be the first method to jointly model geometry and irradiance for 3D reconstruction. 
3. The results presented are impressive and quantitative evaluations seems to support the same.

### Weaknesses
1. No supplementary videos to verify the quality of the 3D reconstruction.

### Questions
1. Please cite PyTorch.

2. I would be willing to raise my rating if the 3D scene reconstruction videos are also provided as supplementary/shown as a sequence of images to further verify the fidelity of reconstructions.

### Soundness
4

### Presentation
4

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
This paper introduces a new framework for high-dynamic-range novel view synthesis (HDR-NVS). Specifically, HDR-NVS is decomposed into three stages: radiance field training, geometry training, and joint training. The key idea of this paper is to estimate the exposure reliability to modulate the geometry by lowering the weights for unstable regions. Besides, this paper also achieves exposure-aware density control in the joint control stage. Experiments are conducted to show the effectiveness of the proposed Expo-GS method.

### Strengths
- It's a good idea to consider the exposure reliability when reconstructing the geometry.
- This paper achieves superior performance, especially on real-world datasets.

### Weaknesses
- Eqs. 7-8 define an exposure score and apply an inverse weighting.
  This necessarily down-weights saturated regions but up-weights under-exposed, low-SNR areas, conflicting with the claim of suppressing both.
  The aggregation from the per-view quantity in Eq. 7 to the global quantity used in Eq. 8 is under-specified, leaving ambiguity in how per-view scores affect the 3D density.

- Eq. 9 uses a hard nearest-Gaussian/min-axis selection to construct the SDF surrogate.
  This introduces non-smoothness: gradients can switch discontinuously as the winning Gaussian/axis changes, especially near surface boundaries.
  Such discontinuities propagate to the normal consistency in Eq. 11 and to the growth/pruning triggers in Eqs. 13--14, risking training instability.

- Compared baselines are not extensive, especially lacking several recently published HDR methods (GaussHDR (CVPR'25), Mono-HDR-3D (ICML'25)).

- In Figure 4, only 3D-GS is compared, while visualizations of some SOTA baselines like HDR-NeRF and HDR-GS are provided.
  More importantly, there is a mislabeling in Table 2: HDR-NeRF performs better than the proposed Expo-GS in the LDR-NE ($t_2$, $t_4$) setting (PSNR).

- No limitations and failure cases are discussed in this paper.

### Questions
I choose to give a score of 4 because I think a score of 2 is unfair to this paper. On the other hand, I need to kindly remind the authors that the majority of my questions raised in the weakness part should be addressed to maintain this rating. Of course, it is still possible for me to improve my rating if the authors well address all my concerns. For detailed questions, please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
3
