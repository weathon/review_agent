# Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Generating a street-level 3D scene from a single satellite image is a crucial yet challenging task. Current methods present a stark trade-off: geometry-colorization models achieve high geometric fidelity but are typically building-focused and lack semantic diversity. In contrast, proxy-based models use feed-forward image-to-3D frameworks to generate holistic scenes by jointly learning geometry and texture, a process that yields rich content but coarse and unstable geometry. 
We attribute these geometric failures to the extreme viewpoint gap and sparse, inconsistent supervision inherent in satellite-to-street data.
We introduce Sat3DGen to address these fundamental challenges, which embodies a geometry-first methodology. This methodology enhances the feed-forward paradigm by integrating novel geometric constraints with a perspective-view training strategy, explicitly countering the primary sources of geometric error.
This geometry-centric strategy yields a dramatic leap in both 3D accuracy and photorealism. {\revisioncolor For validation, we first constructed a new benchmark by pairing the VIGOR-OOD test set with high-resolution DSM data. On this benchmark, our method improves geometric RMSE from 6.76m to 5.20m.} Crucially, this geometric leap also boosts photorealism, reducing the Fr\'echet Inception Distance (FID) from $\sim$40 to 19 against the leading method, Sat2Density++, despite using no extra tailored image-quality modules. We demonstrate the versatility of our high-quality 3D assets through diverse downstream applications, including semantic-map-to-3D synthesis, multi-camera video generation, large-scale meshing, and unsupervised single-image Digital Surface Model (DSM) estimation. The code will be released on https://github.com/qianmingduowan/Sat3DGen.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Sat3DGen, a framework designed to generate comprehensive street-level 3D scenes from a single satellite image. The method incorporates several techniques — a gravity-based density variation loss, spatial token padding, and a monocular relative-depth prior — to enhance the performance of Sat2Density++. However, the overall architecture remains almost identical to Sat2Density, leading to concerns about the lack of novelty.

### Strengths
1. The authors tackle a meaningful and challenging task: generating realistic ground-level 3D scenes from a single satellite image.
2. The proposed gravity-based density variation loss, spatial token padding, and monocular relative-depth prior improve upon the previous Sat2Density++ framework.

### Weaknesses
**1. The use of DINO-v3**

The model employs DINO-v3, which is computationally expensive. It is unclear how the inference speed and GPU memory consumption are affected. Moreover, it remains uncertain whether the performance gains primarily come from DINO-v3, rather than the proposed method itself.

**2. Unclear explanation of the Gravity-based Density Variation Loss (Lines 241–254)**

* The mathematical formulation is ambiguous. Is x defined in Line 197 as a 3D point?
* Why does δx (a scaled 3D point) represent “along gravity”?
* The statement “lower-altitude points usually have density that is no smaller than higher-altitude points” seems empirical — is there any theoretical justification?

**3. Depth estimation and supervision issues (Line 258–265)**
* How accurate is the depth estimated by Depth Anything v2 when applied to satellite imagery?
* What happens if the estimated depth is inaccurate?
* Why is depth only used as a loss term instead of being fused into the network representation?
* The supervision on spatial gradients (Line 265) might oversmooth regions that should exhibit sharp depth changes (e.g., building-ground boundaries).

**4. Inconsistent baseline comparison**

* In Figure 4(c), the comparison is made against Sat2Density instead of Sat2Density++. Why not compare with the most recent and stronger baseline?
* The paper lacks quantitative comparisons with related methods such as ControlNet, ControlS2S, or Canonical Image-to-3D.

**5. Limited novelty**

The overall architecture is nearly identical to Sat2Density++, with improvements mainly stemming from new loss terms and a stronger backbone.

### Questions
The problem corresponds one-to-one with the content of the weakness:
1. How does the use of DINO-v3 affect inference speed and memory consumption? Are the improvements mainly due to the stronger feature extractor?
2. Could the authors provide clearer mathematical formulations and theoretical justification for the Gravity-based Density Variation Loss?
3. How accurate is Depth Anything v2 on satellite imagery, and how sensitive is the model to potential depth estimation errors? Why not integrate depth directly into the model rather than using it as a loss? Does the spatial gradient supervision risk over-smoothing sharp depth transitions?
4. Why is Sat2Density used as the baseline in Figure 4(c) instead of Sat2Density++? Why are quantitative comparisons with ControlNet, ControlS2S, or Canonical Image-to-3D missing?
5. How do the authors justify the novelty of Sat3DGen given its high similarity to Sat2Density++?

### Soundness
2

### Presentation
1

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
This paper proposes Sat3DGen, a novel framework for generating high-fidelity, street-level 3D reconstructions from a single satellite image. The method introduces three geometry-focused contributions: (1) Gravity-based density variation loss, (2) Spatial token padding, (3) Monocular satellite-view depth regularization. It demonstrates substantial improvements over prior state-of-the-art approaches like Sat2Density++ and Sat2Scene, particularly in scene-level 3D geometry consistency, semantic fidelity, and rendered view realism, across both qualitative and quantitative benchmarks (e.g., FID, DINO similarity).

### Strengths
- Clearly motivated and well-structured paper with substantial methodological contributions.
- The paper achieves strong improvements in empirical results (e.g., FID drops from 40.8 to 19.2) and demonstrates practical utility in applications such as DSM estimation and multi-view video synthesis.
- Effective ablation studies and transparent discussion of limitations are provided.

### Weaknesses
- The paper lacks evaluation against metric 3D ground truth (e.g., DSM or city-scale LiDAR) and could be strengthened with controlled experiments on public datasets and analysis of DSM or mesh error.
- There is insufficient discussion of major failure modes (such as occlusions or challenging geometry), and additional metrics beyond FID/LPIPS—like multi-view photometric consistency or temporal flicker—would offer a fuller assessment of 3D and video realism.
- Robustness and generalization should be further explored, including evaluation on non-VIGOR data, handling of noisy or missing illumination inputs, and clarifications on methodological details and reproducibility (e.g., pretrained model release, citation updates, and clear diagram separation).

### Questions
1. Are the generated meshes watertight and suitable for downstream simulation tasks (such as physics simulation or driving simulation)?
2. How robust is the model to varying lighting conditions or the absence of panorama-derived illumination codes, and does it offer controllable rendering for different times of day?
3. Does the method generalize well to images outside the VIGOR dataset, including rural or non-urban areas?
4. It would be nice to see more analysis on the failure cases and the robustness of the method (e.g., non-planar surfaces, complex geometry, occlusion, etc.).

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
This paper proposes a method for efficiently generating high-quality street-level 3D scenes from a single satellite image. Specifically, this method designs an end-to-end framework from satellite image to 3D. This framework first uses DINO v3 to encode the features of the input satellite image, and then decodes the encoded features into a Triplane feature field. Subsequently, volume rendering of this feature field is performed using an MLP to reconstruct the 3D scene. To effectively address potential issues such as edge artifacts, geometric distortions, and roof errors during scene reconstruction, this paper introduces various optimization strategies, including physical constraints and depth constraints. Experimental results show that, compared to existing methods such as Sat2Density++, Sat3DGen can generate street-level 3D scenes with more accurate geometric information and more detailed rendering results.

### Strengths
1. Well-written: The paper has a clear organizational structure, is well-written, and has a logical flow.
2. Targeted improvements to geometric stability: Gravity-based Density Variation Loss is proposed, which modulates the volume density along the direction of gravity, significantly alleviating the common "floating layers/holes" problem, making the reconstruction more coherent and more renderable.
3. Simultaneous improvement in rendering quality and cross-view consistency: Combining depth prior with multi-view supervision of panorama/perspective, covering a wider field of view and strengthening geometric constraints, resulting in more stable reconstruction and higher rendering fidelity, especially more reliable in details such as boundaries and roofs.

### Weaknesses
1. The contribution is unclear: Based on my understanding of this article, its basic framework is quite similar to Sat2Density++, with the core contribution being the introduction of depth conditions as training constraints. The authors need to better clarify the differences between this and the Sat2Density++ framework.

2. The experimental validation is insufficient. The authors only conducted experiments on VIGOR-OOD. To my knowledge, VIGOR-OOD is mainly designed for urban scene acquisition. For scenes that are more suburban or rural (e.g.,CVACT[1]), will the authors' method still have significant robustness?

3. Lack of evaluation baseline: As I understand it, based on the contribution of the paper, the focus is on optimizing the 3D geometry generated by the baseline method. Therefore, this paper should add a comparison with general Image-3D methods [2, 3].

[1] Liu, Liu, and Hongdong Li. "Lending orientation to neural networks for cross-view geo-localization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
[2] Xiang, Jianfeng, et al. "Structured 3d latents for scalable and versatile 3d generation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[3] Hunyuan3D, Team, et al. "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material." arXiv preprint arXiv:2506.15442 (2025).

### Questions
1. Regarding the comparison of Figure 4(c), currently only Sat2Density is compared, excluding Sat2Density++. Given that Sat2Density++ clearly demonstrates a quality improvement over Sat2Density in its original paper, could you explain why this baseline was not included? Furthermore, since this paper achieves higher quantitative metrics, it is recommended to supplement the comparison with Sat2Density++ through parallel rendering, providing qualitative results under the same settings, to more comprehensively evaluate the differences between the two in terms of geometric consistency and texture fidelity.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Sat3DGen proposes a feed-forward framework to generate street-level 3D scenes from a single satellite image, addressing the trade-off between semantics and geometry in existing methods. Built on a tri-plane NeRF backbone, it introduces three key components: a gravity-based density variation loss to suppress floating artifacts and voids, spatial tokens to stabilize boundary geometry, and satellite-view depth regularization to resolve rooftop ambiguity. Additionally, it strengthens supervision by jointly training on panoramas and their projected perspective views. Experiments on VIGOR-OOD demonstrate superior performance and support downstream applications like DSM estimation and large-area mesh generation. The work’s core strength lies in targeted geometric optimizations, though it relies on combinations of existing techniques.

### Strengths
- **Clear framework design**: The pipeline (satellite encoding → tri-plane lifting → illumination-adaptive rendering) is logically coherent, with sufficient details for reproducibility.
- **effective geometric optimizations**: The gravity-based loss directly addresses volumetric field voids and floaters, which is a critical problem in scene-level NeRF-based generation, with clear qualitative and quantitative improvements.
- **Strong practical value**: Supports multiple downstream tasks (e.g. DSM estimation, multi-camera video generation, semantic-map-to-3D) without extra supervision, enhancing real-world applicability.
- **Comprehensive experimental validation**: Includes ablation studies for key components, cross-method comparisons, and qualitative/quantitative evaluations, ensuring result credibility.

### Weaknesses
- **relying on existing method combinations**: The core framework (tri-plane NeRF + 2D supervision) is borrowed from prior works (e.g., Sat2Density++). No breakthrough in methodology or framework design is presented.
- **Unclear motivation and lack of ablation for DINOv3 encoder**: The paper uses a frozen DINOv3 ViT encoder for satellite tokenization but provides no justification for choosing DINOv3 over other encoders. There is no ablation to verify whether DINOv3 contributes to performance gains, or if simpler encoders could achieve similar results, or if the model can generalize to out-of-distribution scenarios.
- **Lack of quantitative 3D geometric evaluation**: All geometric assessments are qualitative (mesh visualizations), with no quantitative metrics for 3D quality. This makes it hard to rigorously validate the claimed "superior geometric quality" compared to baselines like Sat2Density++.

### Questions
- What was the motivation for selecting DINOv3 as the satellite encoder? Have you conducted ablation studies comparing it with other encoders in terms of performance, computational cost, or token quality? If DINOv3 is replaced with a simpler encoder, how much performance degradation would occur? 
- Could you supplement quantitative 3D metrics to objectively validate geometric improvements?

### Soundness
3

### Presentation
3

### Contribution
3
