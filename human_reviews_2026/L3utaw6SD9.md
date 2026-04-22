# UrbanGS: Efficient and Scalable Architecture for Geometrically Accurate Large-Scene Reconstruction

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
While 3D Gaussian Splatting (3DGS) delivers high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments introduces critical challenges in geometric consistency, memory efficiency, and computational scalability. We present UrbanGS, a scalable reconstruction framework that effectively addresses these challenges for city-scale applications.
We propose a Depth-Consistent D-Normal Regularization module. In contrast to existing approaches that rely solely on monocular normal estimators—which effectively update rotation parameters but poorly optimize other geometric attributes—our method integrates D-Normal constraints with external depth supervision. This enables comprehensive updates of all geometric parameters. By further incorporating an adaptive confidence weighting mechanism based on gradient consistency and inverse depth deviation, our approach significantly enhances multi-view depth alignment and geometric coherence.
To improve scalability, we introduce a Spatially Adaptive Gaussian Pruning (SAGP) strategy, which dynamically adjusts Gaussian density based on local geometric complexity and visibility to reduce redundancy. Additionally, a unified partitioning and view assignment scheme is designed to eliminate boundary artifacts and optimize computational load. Extensive experiments on multiple urban datasets demonstrate that UrbanGS achieves superior performance in rendering quality, geometric accuracy, and memory efficiency, offering a systematic solution for high-fidelity large-scale scene reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The key contribution of this paper is a more effective geometry regularization and Gaussian pruning algorithms. The proposed techniques enable a state-of-the-art performance in both the rendering quality and geometric accuracy. It also benefits the reduction of the computation burden. Extensive experiments validated the effectiveness of the method.

### Strengths
- Originality: Simple yet effective technical innovation. 
- Quality: Solid and comprehensive experiments with subtle flaws.
- Clarity: Fine, but there is still space for improvement.
- Significance: A realistic yet effective large-scale scene reconstruction is an important problem. The paper effectively pushes the performance bound forward and provides a good solution.

### Weaknesses
1. There are many hyperparameters to manually tune, such as $\alpha$, $\beta$, and $\gamma$ of Eq.(15), $\gamma_d$ and $\tau$ in Eq.(11), making the application to custom datasets miserable.
2. There is some missing in the ablation. For SAGP ablation in Tab.3, the authors should compare with other candidates, like the one used in CityGaussianV2, instead of omitting it for comparison.
3. There are some flaws in the presentation. The font size of Fig.5 is too small, making it hard to read. Besides, the title for the x-axis of its left part seems to be missing. The highlighted difference in Fig.3 should also be zoomed in for better details.

### Questions
See weakness.

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
The paper proposes **UrbanGS**, a 3D Gaussian Splatting–based framework targeting large-scale urban scene reconstruction with accurate geometry, memory efficient, and scalability

Its main contributions are:

1. **Depth-Consistent D-Normal Regularization** — combines rendered normal supervision with depth guidance from a monocular estimator to jointly optimize Gaussian position, rotation, and scale.
2. **Geometry-Aware Confidence Weighting** — adaptively down-weights unreliable regions based on gradient consistency and inverse-depth deviation.
3. **Spatially Adaptive Gaussian Pruning (SAGP)** — prunes redundant Gaussians based on local geometric complexity and opacity.
4. **Partitioning strategy** for large-scale training with reduced boundary artifacts.

Experiments across Mill-19, UrbanScene3D, and GauU-Scene datasets show that UrbanGS improves novel-view synthesis and geometry reconstruction performance, and reduces memory/time costs compared to CityGS-v2,VCR-Gaus, and other NeRF-based and GS-based reconstruction methods.

### Strengths
- Consistent experimental setup and relatively solid quantitative evidence seen throughout the tables.
- Strong implementation efficiency; runs faster and more memory efficient than CityGS-v2
- Major ablation studies supporting participation of proposing mehtods.

### Weaknesses
1. **Limited novelty** – D-Normal regularization with depth cues closely resembles 2DGS[1]; little conceptual advancement beyond combining two existing signals.
2. **Unclear multi-view consistency claim** – depth anchors from monocular estimators cannot guarantee scale-aligned supervision across views; no metric (e.g., depth reprojection error) verifies this. Recent works apply learnable scaling factors[2] or video-diffusion models[3] to deal with the multi-view consistency, but current work lacks such effort.
3. **Geometry-aware confidence** – seems a heuristic way to use the reciprocal of the depth. No validation exists to justify the design of $L_{id}$
4. **Unclear pruning formulation** – lack definition; reproducibility depends on unspecified percentile $t$ of Eq.(14).
5. **Moderate increase in qualitative results** — View are not consistent (Fig.3, Fig B), hard to compare the qualitative results. Also, without ground truth, it is hard to say that current visualization show enhanced reconstruction (e.g., for Figure.3, UrbanGS result could be viewed as smoothing out details).
6. **Hyperparameter sensitivity** – too many manually tuned coefficients; unclear if improvements stem from architectural changes or parameter optimization.

[1] Huang, Binbin, et al. "2d gaussian splatting for geometrically accurate radiance fields." *ACM SIGGRAPH 2024 conference papers*. 2024.

[2] Tong, Jinguang, et al. "GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

[3] Liang, Ruofan, et al. "Diffusion Renderer: Neural Inverse and Forward Rendering with Video Diffusion Models." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

### Questions
1. How does the proposed method avoid scale ambiguity inherent to monocular depth? Can you show quantitative depth alignment across views (e.g., reprojection error or cross-view consistency metric)?
2. How does the proposed D-normal regularization differ from existing methods from 2DGS[1] and others[4][5]?
3. In Eq. (14), please define $v_i, v^{(t)}_{local}$, and explain how the percentile $t$ is chosen.
4. It would be great to show some justification (e.g.,ablation on design) for the $L_{id}$ of Eq.(8) and geometry-aware confidence of Eq.(11).
5. Sci-Art results underperform simpler baselines; it would be great if the authors analyze why.
6. It would be great if the authors to provide more qualitative results on benchmarks datasets, especially for GauU-Scene.
7. How sensitive is the method to the many hyperparameters? Could a fixed default generalize? More ablation studies (quantitative & qualitative) on these hyperparamaters may be necessary to support the scalability of the work.
8. Please clarify if the “position update along normal” proof was verified empirically (e.g., by measuring displacement vectors relative to normals).

[4] Liang, Zhihao, et al. "Gs-ir: 3d gaussian splatting for inverse rendering." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[5] Chen, Hongze, Zehong Lin, and Jun Zhang. "Gi-gs: Global illumination decomposition on gaussian splatting for inverse rendering." *The Thirteenth International Conference on Learning Representations*.

### Soundness
3

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
This paper addresses the challenge of scalable large-scale 3D reconstruction using 3D Gaussian Splatting (3DGS). The authors aim to improve both geometric accuracy and model efficiency. To this end, they propose two main technical contributions: 1) a Depth-consistent D-Normal Regularization term to enforce geometric fidelity, and 2) a Spatially adaptive Gaussian Pruning strategy to optimize the model's compactness. The authors present quantitative and qualitative comparisons against relevant baselines, supported by ablation studies, to demonstrate that their proposed method achieves more geometrically accurate reconstructions while maintaining a lightweight model.

### Strengths
- The paper is well-written and clearly organized, contributing to an effective presentation of the proposed method.
- The proposed method outperforms SOTA baselines across key metrics (memory consumption, geometric fidelity, novel view synthesis) and achieves visually cleaner mesh extractions on standard benchmarks (Figs. 3–4).

### Weaknesses
1. **Missing Related Work:** The related work section appears to omit a discussion of recent, relevant methods in large-scale 3D Gaussian Splatting, most notably CityGS-X [1]. Please discuss this work and position the current method in relation to it.
2. **Incremental Novelty:** The paper's conceptual novelty appears to be limited. The core components, such as the D-Normal regularizer and the partitioning strategy, seem to be extensions of prior art (e.g., VCR-Gaus [2] for consistency, CityGaussian/CityGS-v2 [3] for partitioning). The authors should more clearly articulate the core conceptual contributions of their work beyond the successful integration and extension of these existing ideas.
3. **Clarity of Depth Consistency:** The explanation for the depth consistency regularization requires clarification. To assess the method, please elaborate on how multi-view consistency is enforced. Specifically:
    - To which view(s) are the Gaussian depths rendered?
    - How is the depth loss calculated between these views?
4. **Motivation and Validation of SAGP:** The formulation of the SAGP component combines several known heuristics (e.g., volume, opacity). The specific weighting in Eq. 15 lacks theoretical motivation.
---
[1] Gao, Yuanyuan, et al. "Citygs-x: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction." arXiv preprint arXiv:2503.23044 (2025).

[2] Chen, Hanlin, et al. "Vcr-gaus: View consistent depth-normal regularizer for gaussian surface reconstruction." Advances in Neural Information Processing Systems 37 (2024): 139725-139750.

[3] Liu, Yang, et al. "Citygaussianv2: Efficient and geometrically accurate reconstruction for large-scale scenes." arXiv preprint arXiv:2411.00771 (2024).

### Questions
1. **Experimental Comparison:** Following the point on related work, please provide quantitative and qualitative comparisons against CityGS-X [1] on at least one shared dataset.
2. **Sensitivity to Priors:** How sensitive is the proposed method to the quality and noise level of the monocular priors? We request that the authors demonstrate the robustness of the confidence weighting (Eqs. 9–11), for instance by evaluating with different depth/normal estimators or by analyzing performance after injecting synthetic noise into the priors.
3. **Contribution of Depth-Consistency Term:** It is unclear if the performance gain from the depth-consistency term (Eq. 8) is due to the novel formulation itself or primarily due to the strength of the chosen prior (DepthAnything-v2 [2]). Please provide an ablation study 
 that uses alternative estimators (e.g., MiDaS[3], DPT[4]) to isolate the contribution of the regularization term.
4. **SAGP Hyperparameter Analysis:** To better understand the design of the SAGP component, please provide sensitivity analyses for its key hyperparameters (e.g., the weights $\alpha, \beta, \gamma$ in Eq. 15; $\kappa$ in Eq. 14; the cell size in Eq. 13; and the pruning schedule).

---

[1] Gao, Yuanyuan, et al. "Citygs-x: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction." arXiv preprint arXiv:2503.23044 (2025).

[2] Yang, Lihe, et al. "Depth anything: Unleashing the power of large-scale unlabeled data." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

[3] Birkl, Reiner, Diana Wofk, and Matthias Müller. "Midas v3. 1--a model zoo for robust monocular relative depth estimation." arXiv preprint arXiv:2307.14460 (2023).

[4] Ranftl, René, Alexey Bochkovskiy, and Vladlen Koltun. "Vision transformers for dense prediction." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Soundness
3

### Presentation
3

### Contribution
2
