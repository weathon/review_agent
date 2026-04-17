# Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Underwater 3D reconstruction remains challenging due to the intricate interplay between light scattering and environment dynamics. While existing methods yield plausible reconstruction with rigid scene assumptions, they struggle to capture temporal dynamics and remain sensitive to observation noise. In this work, we propose an Uncertainty-aware Dynamic Field (UDF) that jointly represents underwater structure and view-dependent medium over time. A canonical underwater representation is initialized using a set of 3D Gaussians embedded in a volumetric medium field. Then we map this representation into a 4D neural voxel space and encode spatial-temporal features by querying the voxels. Based on these features, a deformation network and a medium offset network are proposed to model transformations of Gaussians and time-conditioned updates to medium properties, respectively. To address input-dependent noise, we model per-pixel uncertainty guided by surface-view radiance ambiguity and inter-frame scene flow inconsistency. This uncertainty is incorporated into the rendering loss to suppress the noise from low-confidence observations during training. Experiments on both controlled and in-the-wild underwater datasets demonstrate our method achieves both high-quality reconstruction and novel view synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces UDF (Uncertainty-aware Dynamic Field), a novel framework for underwater 3D reconstruction that models dynamic geometry and medium properties while incorporating uncertainty. The framework builds on 3DGS and targets underwater scenes by 
1) Initializing a set of 3D Gaussians embedded in a volumetric medium. 
2) Encoding spatial-temporal features in a 4D neural voxel space via planar factorization.
3) Using a deformation network to model dynamic geometry and a medium offset network to capture evolving medium properties 
4) Incorporating uncertainty into the rendering loss guided by surface-view radiance ambiguity and inter-frame flow inconsistency. 

The authors evaluate the effectiveness of their method in multiple underwater datasets and showcase a significant improvement over prior methods.

### Strengths
The paper is very well written and the authors provide deep insights in their design choices. 
1) The integration of dynamic medium modeling with uncertainty-aware rendering is a substantial advancement over prior work 
2) The radiance ambiguity and flow inconsistency are physically motivated and well-integrated into the loss function
2) Strong results across all datasets 
3) The method achieves a good balance between rendering speed and memory usage.

### Weaknesses
1) In the experimental evaluation, the method improves over all metrics. The only metric that watersplatting outperforms the current is in SSIM. Can you comment on why that might happen?

2) While the paper is technically rich, some sections are very dense in information. Even figure 1. Is very technically dense. The paper could benefit from additional diagrams and moving details to supplementary material.

### Questions
As noted in Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes UDF, a dynamic scene representation that jointly models underwater structure (via 3D Gaussian primitives) and the participating medium (via a neural, view-conditioned medium field). A canonical scene (Gaussians embedded in a volumetric medium) is mapped into a 4D neural voxel space (K-Planes) to extract spatio-temporal features. Two heads then evolve the scene over time: a deformation network that predicts per-Gaussian offsets, and a medium-offset network that updates attenuation/backscatter conditioned on motion cues. The paper introduces a heteroscedastic rendering loss with per-pixel variance derived from surface-view radiance ambiguity and inter-frame flow inconsistency. Experiments on NUSR, DRUVA, and SeaThru show gains in PSNR/SSIM/LPIPS with qualitative  visualizations.

### Strengths
1. The pipeline overview is clear and visual results further aid understanding. These choices make the method and evidence easy to follow.

2. The combination of 3DGS geometry + learnable medium + 4D K-Planes provides a clean factorization of structure vs. medium over time. The two cues (surface-view ambiguity and flow-based inconsistency) are well-grounded and integrated in the NLL loss.

2. The paper reports consistent improvements on NUSR, DRUVA, and SeaThru with both quantitative and qualitative evidence (novel views, medium-free renderings, depth maps).

### Weaknesses
1. Eq. (6) uses a single σ_med inside T_med(s), yet later the paper separates σ_med into σ_att (structure) and σ_bs (backscatter). The derivation would benefit from writing the explicit separated transmittance and emission terms to avoid ambiguity, and clarifying wavelength-dependent parameterization (RGB-channel σ_med).

2. Consider adding a teaser in the introduction, e.g., a before/after comparison on a challenging underwater scene, to foreground the key difficulties and to show how your method addresses them. Without this, your advantages are not immediately clear to readers.

3. User study details. The 40-person, 1,500-image study is promising, but lacks description of the protocol (randomization, rating scale, inter-rater reliability).

4. You report VGGT vs. COLMAP with comparable results. Could you add experiments with noisy to quantify robustness to calibration inaccuracies typical in underwater capture?

5. Some typos:
- "Japanese Gradens" -> "Japanese Gardens"
- "a uncertainty-aware rendering loss" -> "an uncertainty-aware rendering loss"
- L226 "enotes the projection function" -> "denotes the projection function"
- "Zoomed-in regions shows" -> "Zoomed-in regions show"

### Questions
I hope the authors can address Weaknesses 1–5 in the rebuttal, and I remain positive about this work's contribution to the community.

### Soundness
4

### Presentation
4

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
This paper presents a method for reconstructing dynamic underwater scenes by using an uncertainty module to discard low-quality pixels in the measurements as well as dynamics modules to model the time-varying nature of both the scene and the medium. The authors demonstrate improved results on several benchmarks compared to baselines, and extensively validate components of their methods.

### Strengths
The problem is well-motivated because dynamics are almost unavoidable underwater, since currents and wildlife are commonplace in captured data. The qualitative results are compelling and show clear improvement, especially for dynamic objects and medium recovery, as can be seen in Figures 2 and 3, which show that both the scene and medium reconstructions are better than prior works. The quantitative metrics are also improved by relatively significant margins for most of the scenes. Finally, the evaluation is fairly comprehensive, covering pipeline as well as module design, and illustrates clearly some of the limitations of the method, shown in Figure 4.

### Weaknesses
About the videos in the supplement, the paper claims there are four of them but there are only three in the folder. Maybe this is a typo? Furthermore, the trajectories in IUI3RedSea and JapaneseGarden are relatively choppy. I know that the authors are just rendering at the positions provided by the original dataset, but I think it would be helpful to regenerate smooth trajectories with which to render the result.

### Questions
Could the authors rerender the supplement videos with smooth trajectories? I won't make my score conditional on the response to this question, since I think it's against the policies to ask authors to do excessive work or provide new results, particularly graphics, but I think it would just be nice.

### Soundness
4

### Presentation
3

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
This paper introduces Uncertainty-aware Dynamic Field (UDF) for 3D reconstruction of dynamic undetwater scenes. UDF jointly models scene structure and view-dependent medium properties over time. UDF initializes with 3D Gaussians in a volumetric medium field, maps them to a 4D neural voxel space, and employs deformation/medium offset networks to handle spatio-temporal dynamics. To suppress noise from low-confidence observations, UDF incorporates per-pixel uncertainty estimation based on surface-view radiance ambiguity and inter-frame flow inconsistency, and further integrates the uncertainty into the rendering loss. Experiments on controlled and in-the-wild underwater datasets demonstrate UDF achieves superior reconstruction quality and novel view synthesis for dynamic underwater scenes.

### Strengths
++ The shared 4D neural voxel space based on planar factorization extends the 3D canonical representation to a spatial-temporal space, which further enables the dynamics modeling of scene structure and medium in a physics-informed manner.

++ UDF estimates per-pixel uncertainty using physical cues and integrates it into the rendering loss to suppress noisy data during training. The effectiveness of this design is validated in the ablation study (Table 5).

### Weaknesses
-- The fundamental contributions and differentiating aspects of the underwater dynamic modeling approach proposed in this paper compared to existing methods require clearer articulation. The authors may need to emphasize these points in their writing.



-- Regarding the distorted colors shown in WaterSplatting and SeaThru-NeRF after medium removal: could the authors provide insight into the root cause? It would be helpful to know if UDF is immune to this artifact across the entire tested cases and, crucially, what specific aspect of its design confers this advantage.

-- From the supplementary videos (especially composite.mp4),  we can see significant artifacts around dynamic objects (fishes). This indicates that the method's performance in modeling moving objects is not satisfactory.

### Questions
-- The gradient-induced pseudo-normal, used in modeling surface-view radiance ambiguity, is particularly prone to significant errors at object edges and in rapidly moving regions. Consequently, failures in uncertainty modeling can occur, potentially adversely affecting the robustness of the entire approach. Can the authors provide some analysis on this aspect?

### Soundness
3

### Presentation
2

### Contribution
2
