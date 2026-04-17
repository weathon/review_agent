# Improving 2D Diffusion Models for 3D Medical Imaging with Inter‑Slice Consistent Stochasticity

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
3D medical imaging is in high demand and essential for clinical diagnosis and scientific research. Currently, diffusion models have become an effective tool for medical imaging reconstruction thanks to their ability to learn rich, high‑quality data priors.  However, learning the 3D data distribution with diffusion models in medical imaging is challenging, not only due to the difficulties in data collection but also because of the significant computational burden during model training. A common compromise is to train the diffusion model on 2D data priors and reconstruct stacked 2D slices to address 3D medical inverse problems. However, the intrinsic randomness of diffusion sampling causes severe inter‑slice discontinuities of reconstructed 3D volumes. Existing methods often enforce continuity regularizations along the $z$‑axis, which introduces sensitive hyper‑parameters and may lead to over-smoothing results. In this work, we revisit the origin of stochasticity in diffusion sampling and introduce Inter‑Slice Consistent Stochasticity (ISCS), a simple yet effective strategy that encourages inter‑slice consistency during diffusion sampling. Our key idea is to control the consistency of stochastic noise components during diffusion sampling, thereby aligning their sampling trajectories without adding any new loss terms or optimization steps.  Importantly, the proposed ISCS is plug‑and‑play and can be dropped into any 2D‑trained diffusion‑based 3D reconstruction pipeline without additional computational cost. Experiments on several medical imaging problems show that our method can effectively improve the performance of medical 3D imaging problems based on 2D diffusion models. Our findings suggest that controlling inter‑slice stochasticity is a principled and practically attractive route toward high‑fidelity 3D medical imaging with 2D diffusion priors. The code is available at: [https://github.com/duchenhe/ISCS](https://github.com/duchenhe/ISCS).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors exploit 2D diffusion models for 3D medical image reconstruction. The diffusion models are trained on xy-slices. 
To obtain continuity in z-axis the authors apply correlated noise during sampling. They call this method "Inter-Slice Consistent Stochasticitiy" (ISCS). ISCS is an easy addition to the sampling pipeline, requiring only a single change in the algorithm.

### Strengths
- The paper is well-written and contains careful experiments. Importantly, according to Table 1 ISCS performs better than adding an additional TV-regulariser in the z-axis (which is currently the predominant way of enforcing inter-slice consistency)
- ISCS is a simple method (i.e., it contains no hyperparameters) which can be easily incorporated into existing code-bases with a few extra lines and comes with only negligible additional computational cost 

Based on these points (better performance, simple implementation, no overhead computational cost) I would expect that this method will be well adapted in 3D-sampling tasks and code-bases.

### Weaknesses
- There are some important baselines missing in Table 1: using perpendicular 2d models [1] and batch-consistent sampling (i.e., using the same noise for all slices) 
- There is no comparison against the deterministic DDIM sampler (i.e. setting $\eta=0$ in Equation (8)). For the case of deterministic samplers, no additional noise has to be added during sampling. 
- With the adoption of flow-based models and deterministic samplers for diffusion models (e.g. PNDM [4]) the proposed method might no longer be relevant. For both flow-based models and deterministic sampler, the main argument for the lack of inter-slice consistency (starting at line 244) does no longer hold, because no noise is added. 
- The authors provide a heuristic motivation why their method should work, but not really any theory. In contrast, using a TV-regulariser on the z-axis ensures consistency between slices (albeit requiring an additional hyperparameter)
- In line 264 the authors write "recent studies have demonstrated that random samples from a high-dimensional standard normal distribution tend to concentrate on the surface of a hypersphere" and cite two works from 2024. However, this is a pretty standard concentration of measure phenomenon and known for a long time.

[4] Liu et. al "Pseudo Numerical Methods for Diffusion Models in Manifolds" (2022)



[1] Lee et al. "Improving 3d imaging with pre-trained perpendicular 2d diffusion models"

### Questions
- How does the method depend on the sampling of the initial noise vectors for the first and the last slice? 
- How does this method compare against the temporal correlated noise of [2] and the method in [3] ? 

[2] Liu and Vahdat "On Equivariance and Fast Sampling in Video Diffusion Models Trained with Warped Noise" (2025)

[3] Chang et al. "How I warped your noise: a temporally correlated noise prior for diffusion models"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the issue of inter-slice inconsistency that arises when solving 3D medical inverse problems using 2D diffusion model-based inverse solvers (DIS). The authors propose a simple yet effective approach in which the noise used during the reverse diffusion denoising process is made slice-wise correlated. Motivated by recent findings showing that spherical linear interpolation (Slerp) is more meaningful than linear interpolation for Gaussian noise, the method samples two anchor noise vectors for the first and last slices, and then generates noise for intermediate slices via Slerp interpolation between these anchors.

### Strengths
1. The main strength of this work lies in its ability to mitigate inter-slice inconsistency—a major challenge in 3D medical inverse problems—without introducing additional priors or complex  modifications. Since the method only modifies the noise structure and introduces no new hyperparameters, it is highly practical and easy to integrate into existing 3D medical reconstruction pipelines.
2. Experiments across multiple imaging modalities and tasks provide strong evidence that the proposed method can effectively replace the widely used total variation (TV) prior.
3. The comparison with batch-consistent sampling (BCS) clearly demonstrates the limitations of BCS, specifically the copying artifacts it introduces, and shows that the proposed approach effectively resolves these issues.

### Weaknesses
1. The novelty of the proposed method is somewhat limited because similar structural noise techniques have been explored in video diffusion models for enforcing temporal consistency and reducing flickering artifacts (e.g., [1,2,3]). For instance, [1] also samples noise with inter-frame correlations via noise interpolation. To clearly distinguish the contribution, it is necessary to directly compare the proposed method with these noise prior approaches in the same manner as the comparison against BCS.
2. The interpolated noise is only used during the re-noising steps. Thus, when using deterministic DDIM sampling ($\eta=0$), where pure noise is not used, the method cannot be applied. If $\eta=0$ is rarely used in inverse problems due to degraded performance, the authors should provide supporting evidence or references. Additionally, since the proposed method depends directly on the degree of stochasticity ($\eta$), an analysis of how $\eta$ influences reconstruction quality is required.
3. BCS applies its procedure to the initial noise $x_T$. It seems natural to also apply the interpolated correlated noise to the $x_T$ in the proposed method. The authors should clarify why this was not done.
4. The angle (distance) between the two anchor noise vectors is randomly determined. If there exists an optimal anchor spacing for a given target volume, performance variance may increase. An ablation analyzing reconstruction quality with respect to anchor angle is needed.
5. **Minor:** The authors attribute the copying artifacts in medical volumes to domain differences from videos, but this claim lacks supporting evidence. Since prior BCS work does not show temporal-slice artifacts in video, the absence of such artifacts is not demonstrated. If the authors wish to maintain the domain-difference argument, they should show that BCS does *not* produce similar artifacts in video inverse problems; otherwise, the proposed method may be better positioned as a domain-genera**l** solution. Of course, evaluating video inverse problems may be beyond the scope of this paper — this is only a suggested clarification and does not affect the overall contribution.
6. **Minor:** Line 114: “A is invertible” appears to be a typo; presumably it should be **non-invertible**.

[1] Ge, Songwei, et al. "Preserve your own correlation: A noise prior for video diffusion models." *ICCV*. 2023.

[2] Wu, Tianxing, et al. "Freeinit: Bridging initialization gap in video diffusion models." *ECCV.* 2024.

[3] Chang, Pascal, et al. "How i warped your noise: a temporally-correlated noise prior for diffusion models." ICLR (2024).

### Questions
1. While the method shows that it can replace the widely used TV prior, it would be informative to see how it compares with 3D priors such as TQDM. It is okay to the proposed method does not outperform them.
2. Implicit slice-wise regularization via noise correlation may underperform explicit regularizers (e.g., TV) under severe degradation where large portions of information must be reconstructed. Does the method maintain performance under such challenging forward models?
3. Since the proposed approach seems orthogonal to explicit volumetric priors such as 3D TV, can the two be combined for further performance improvement?
4. If the above comparisons and ablation studies (as suggested in the weakness section) are included, the contribution would be significantly strengthened, and I would be willing to raise my score accordingly.

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
5

### Summary
This paper addresses a key limitation of using 2D diffusion models (DMs) for 3D medical imaging: stochastic inconsistencies between independently sampled slices, leading to discontinuous 3D volumes. The authors propose Inter-Slice Consistent Stochasticity (ISCS)---a plug-and-play method that enforces inter-slice correlation in the diffusion sampling stage by introducing spherical linear interpolation (Slerp) between noise vectors, ensuring smooth stochastic transitions across slices. The method requires no retraining, extra loss terms, or additional computational cost, and can be integrated into existing diffusion-based inverse solvers (DDNM, DDS). Experiments on sparse-view CT, limited-angle CT, and MRI isotropic SR demonstrate consistent PSNR/SSIM/LPIPS improvements, outperforming TV-regularized baselines.

### Strengths
1. The paper clearly identifies uncorrelated stochasticity as the fundamental cause of inter-slice artifacts, a well-reasoned observation extending findings from video DMs (Kwon & Ye, 2025) to 3D medical imaging.

2. ISCS is lightweight and does not modify model training or inference pipelines. The Slerp-based correlated noise interpolation is mathematically sound and geometrically intuitive.

3. Strong empirical validations:
- Demonstrated across three inverse problems (SVCT, LACT, MRI-SR).
- Improves both fidelity (PSNR/SSIM) and perceptual quality (LPIPS).
- Outperforms both baseline DIS (DDNM/DDS) and TV-regularized counterparts, maintaining anatomical details without oversmoothing.

4. The plug-and-play property enables easy adoption in other 2D DM-based 3D reconstruction pipelines.

### Weaknesses
1. While geometrically motivated, there is no formal proof or quantitative analysis on how inter-slice correlation affects posterior sampling convergence or variance reduction.

2. Scalability and generalization:
- The Slerp interpolation assumes smooth anatomical transitions, which may fail in pathological cases (e.g., tumors, lesions) with abrupt structure changes.
- Performance under varying slice thickness or non-uniform z-spacing is not evaluated.

3. The paper lacks comparison with recent 3D-aware diffusion priors (e.g., DiffusionBlend [Song et al., NeurIPS '24]) or multi-plane consistency methods.

4. Only Slerp vs. identical noise is analyzed. Additional ablations (e.g., interpolation degree, anchor distance, or adaptive correlation strength) would strengthen claims.

### Questions
1. How does inter-slice correlation affect posterior sampling convergence or variance reduction? I expect formal proof or (at minimum) quantitative analysis.

2. How does ISCS interact with deterministic samplers (e.g., DDIM with $\eta = 0$)?

3. How does Slerp interpolation work if abrupt structure changes exist?

4. I strongly suggest including comparisons with recent 3D-aware diffusion priors (e.g., DiffusionBlend [Song et al., NeurIPS '24]) or multi-plane consistency methods.

5. Could Slerp interpolation be extended to spatially adaptive correlation where anatomical gradient drives $\alpha_i$?

6. How does ISCS perform with very thick slices or anisotropic resolutions?

7. Are there any cases where ISCS causes artifacts or over-correlation (e.g., repetitive patterns)?

8. Is the improvement consistent across random seeds, or does stochastic correlation introduce bias?

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
3

### Summary
The paper proposes Inter-Slice Consistent Stochasticity (ISCS) to improve 3D medical reconstructions obtained from 2D diffusion priors. The key idea is to synchronize the stochastic component of the reverse diffusion process across adjacent slices by initializing each slice’s noise via spherical linear interpolation (slerp) between two anchor noise maps. This smoothly correlates randomness across slices, aligning sampling trajectories without adding new loss terms, hyperparameters, retraining, or extra computation, and can be dropped into existing 2D-diffusion-based inverse-problem solvers (e.g., DDNM, DDS) as a plug-and-play component.

Experiments on three tasks: sparse-view CT, limited-angle CT, and MRI isotropic super-resolution—show consistent improvements in PSNR/SSIM and LPIPS over DDNM and DDS baselines, often rivaling or exceeding TV-regularized variants, while qualitatively reducing inter-slice artifacts in coronal/sagitta

### Strengths
- The method is simple, clear, and effective: replacing independent per-slice noise with a slerp-correlated noise volume improves inter-slice consistency without extra training or losses, and integrates directly into standard DDIM-style updates.

- Presentation is generally clear, with a helpful geometric interpretation (independent vs. identical vs. slerp noise) that motivates why smooth correlation avoids both uncorrelated flicker and over-rigid copying artifacts.

- Code is provided.

### Weaknesses
- Several related literature are missed. Di-Fusion (Wu etal., ICLR 2025) and DDM^2 (Xiang etal., ICLR 2023) and many more other papers on inverse problem solving for medical data. Those literature were published on ICLR and for a similar problem, even if not exactly the same. The authors are highly encouraged to discuss those related work in the paper.

- Lack of a dedicated inter-slice consistency metric in the main results. The quantitative tables report PSNR/SSIM/LPIPS only (per view). While the paper includes an “inter-slice difference” trajectory to illustrate coherence during sampling, this analysis is not integrated as a headline metric alongside PSNR/SSIM/LPIPS, making it hard to compare inter-slice consistency at a glance across methods and tasks. A specialized, standardized metric for inter-slice consistency would better highlight the core contribution.

- Insufficient theoretical and practical justification that slerp is the best interpolation choice. The current ablation contrasts identical noise (BCS) vs. slerp and shows slerp avoids “copying artifacts” and slightly improves quantitative metrics. However, other plausible interpolations (e.g., simple linear interpolation in noise space) are not evaluated, leaving open whether slerp is uniquely effective or merely one effective option.

- Comparative breadth. Although comparisons include strong diffusion-based solvers (DDNM, DDS), plus classical baselines (e.g., FDK, ADMM-TV), it would strengthen the evidence to compare against additional medical inverse-problem baselines that may (or may not) benefit from noise design, to more clearly position ISCS’s impact.

### Questions
- Training protocol parity: For the base diffusion models used with DDNM and DDS, were they derived from the same trained prior(s) (same data, architecture, and training protocol), with only the solver differing? A concise statement about identical priors across solvers would clarify fairness.

- Beyond inverse problems: Could ISCS plausibly extend to direct 3D volume generation (e.g., from very sparse or no views) by enforcing correlated stochasticity across slices during unconditional or weakly conditioned sampling? Any preliminary observations or limitations?

### Soundness
3

### Presentation
3

### Contribution
3
