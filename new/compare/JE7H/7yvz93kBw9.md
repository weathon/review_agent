---
job_id: 4956f030-12c8-4440-9979-0c4076fe82e6
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 7yvz93kBw9.pdf
paper: D²GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work focuses on sparse-view 3D Gaussian splatting, representation quality, and robustness metrics for learned 3D representations, which is squarely within ICLR’s scope (representation learning for vision, optimization, evaluation).

## Minimum Quality
Pass ✅.  
The paper is complete and well structured, with Abstract, Introduction, Related Work, Method (including several subsections), Experiments, Results (quantitative and qualitative), Discussion/Limitations, and Conclusion. The method is technically coherent, experiments are substantial on standard benchmarks, and the paper is written in clear English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing or hidden instructions targeting LLMs within the main paper content.

---

# Expected Review Outcome:

## Summary

The paper studies sparse-view 3D Gaussian Splatting (3DGS) and identifies two characteristic failure modes: overfitting in near-field regions with excessive Gaussian density and underfitting in far-field regions with insufficient Gaussian coverage. It proposes D²GS, which introduces (1) a Depth-and-Density Guided Dropout (DD-Drop) that adaptively drops Gaussians based on depth and local density and (2) a Distance-Aware Fidelity Enhancement (DAFE) loss that emphasizes supervision in distant regions using monocular depth masks. In addition, the paper defines an Inter-Model Robustness (IMR) metric based on an approximate Wasserstein distance between Gaussian mixtures to quantify stability of independently trained 3DGS models. Experiments on LLFF, MipNeRF360, and DTU show improved sparse-view novel view synthesis and lower IMR compared to several NeRF- and 3DGS-based baselines.

## Strengths

1. **Clear diagnosis of sparse-view 3DGS failure modes, supported by visual evidence.**  
   Figure 1 and the associated discussion (Pages 1–3) convincingly illustrate the overfitting / underfitting phenomena: in the green near-field region, the sparse-view DropGaussian baseline produces far more Gaussians than the dense-view model (11,450 vs. 6,112), along with visible artifacts, while in the red far-field region it has significantly fewer Gaussians (3,082 vs. 5,224) and visibly blurred structure. This figure is an effective and concrete motivation for why “uniform” dropout is problematic in sparse-view settings.

2. **Reasonable and well-motivated formulation of depth- and density-aware dropout.**  
   The DD-Drop mechanism in Section 3.2 combines a local continuous score and global discrete depth layering. Equation (1) defines a dropout score \(S_i = \omega_{\text{depth}}\hat d_i + \omega_{\text{density}}\hat \rho_i\) using min–max-normalized depth and k-NN density, and Equation (2) modulates this by stratified attenuation factors \(\lambda_{\text{middle}}, \lambda_{\text{far}}\) for different depth bands. This is a clean, interpretable design that directly encodes the intuition that near, dense regions require stronger regularization, whereas middle/far regions should be preserved.

3. **Complementary far-field supervision via DAFE with explicit loss formulation.**  
   The DAFE module (Section 3.3) uses monocular depth estimation to construct a binary distant-region mask via Equation (4), and then computes a masked L1 loss (Equation (5)) that is integrated into the overall objective (Equation (6)). By explicitly specifying how pixels are selected and how the loss is normalized by \(\sum M_{\text{dis}}\), the paper makes it straightforward to understand and reproduce how far-field regions receive extra training signal.

4. **A principled robustness metric for 3DGS models, grounded in optimal transport over Gaussians.**  
   The IMR metric (Section 3.4) abstracts each trained 3DGS as an opacity-weighted Gaussian mixture (Equation (9)) and compares models through a mixture Wasserstein distance solved via entropic OT (Equations (12)–(13)). The derivation of the Bures distance approximation in Appendix A, leading from Equation (10) to Equation (11) and then to Equation (24), is mathematically coherent and provides a clear argument for the chosen approximation term \(\frac{1}{4}\mathrm{tr}((\Sigma_1 - \Sigma_2)\Sigma_2^{-1}(\Sigma_1 - \Sigma_2))\). This is a reasonably principled way to quantify structural differences between 3DGS reconstructions.

5. **Strong quantitative gains across multiple benchmarks, with solid ablations.**  
   - On LLFF 3-view (1/8 res.), Table 1 shows D²GS reaching 21.35 dB PSNR and AVGE 0.087, outperforming DropGaussian (20.76 / 0.097) and other strong sparse-view baselines such as CoR-GS and LoopSparseGS.  
   - On MipNeRF360, Table 2 reports D²GS achieving 20.09 dB PSNR, clearly above CoR-GS (19.52) and DropGaussian (19.74).  
   - The method also generalizes to 6-view LLFF and 24-view MipNeRF360 (Table 8) and to the DTU dataset (Table 9), where it consistently improves PSNR/SSIM/LPIPS over 3DGS-based and NeRF-based methods.  
   - Table 4’s ablation shows monotonic improvements as each of density score, depth score, depth layering, and DAFE are added, both in PSNR and in lower IMR, supporting the claim that each component contributes.

6. **Evidence that the method improves robustness, not only average accuracy.**  
   The IMR results in Table 3 indicate that D²GS yields lower dispersion across independent runs than 3DGS, CoR-GS, and DropGaussian under both 3-view and 6-view LLFF (e.g., IMR 3.039 vs. 3.205 for DropGaussian in 3-view). Figure 3 (left) further visualizes PSNR oscillations over 10 runs for the baseline, highlighting the instability D²GS aims to mitigate. This dual focus on quality and stability is a useful addition to the 3DGS literature.

7. **Qualitative results support the quantitative claims.**  
   Figures 4, 5, and 6 show side-by-side comparisons with 3DGS, CoR-GS, DropGaussian, and ground truth on LLFF and MipNeRF360. In the highlighted red-box regions (e.g., stair railings, office railings, foliage textures), D²GS tends to exhibit less aliasing and fewer floaters than DropGaussian and sharper structures compared to CoR-GS. Together with Figure 2’s pipeline diagram, the figures make the method’s intuition and its effects on reconstructions tangible.

8. **Implementation details and ablations are reasonably thorough.**  
   Appendix B provides concrete training hyperparameters, k-NN density computation frequency, and SH schedule. Table 5 systematically explores sensitivity to \(r_{\min}, r_{\max}\), \(\omega_{\text{depth}}, \omega_{\text{density}}\), depth threshold \(\tau\), and \(\lambda_{\text{DAFE}}\), with results that are mostly stable around the chosen default. Table 6 further shows that DAFE yields gains with three different monocular depth estimators, suggesting the method is not tightly coupled to a specific depth backbone.

## Weaknesses

1. **IMR’s connection to perceptual or task-relevant robustness is underdeveloped.**  
   While the IMR metric is mathematically grounded (Equations (9)–(14)) and Table 3 shows lower IMR for D²GS, the paper does not convincingly demonstrate that lower IMR correlates with better rendering quality or user-visible stability. For example, there is no analysis of the correlation between IMR and PSNR/LPIPS across runs or scenes, nor any visualization linking high IMR to visibly different Gaussian distributions or artifacts. Figure 3 (left) only shows baseline PSNR variability, not how IMR tracks that variability. As a result, it is unclear whether IMR captures something substantially different or more informative than simply measuring variance in image-space metrics across runs. This weakens the practical significance of the IMR contribution.

2. **Heuristic and hand-crafted design choices in DD-Drop lack deeper justification.**  
   DD-Drop relies on several heuristic decisions: min–max normalization of depth and density, linear combination \(S_i\) in Equation (1), fixed stratification into three depth bins via tertiles, fixed attenuation factors \(\lambda_{\text{middle}}=0.7,\lambda_{\text{far}}=0.3\), and a linear schedule for \(r(t)\) in Equation (3). Table 5 explores some of these hyperparameters, but the design remains heavily hand-tuned. There is no analysis of failure modes when the depth distribution is highly non-uniform or multi-modal (e.g., outdoor scenes with large open spaces and isolated foreground objects) and no ablation on the number of depth layers or the choice of tertiles versus other partitioning strategies. This raises concerns about how robust DD-Drop is across more diverse scenes than LLFF/MipNeRF360/DTU.

3. **Dependence on monocular depth priors and potential error amplification in DAFE.**  
   DAFE hinges on per-view depth maps from an off-the-shelf monocular estimator to choose far pixels via Equation (4). Although Table 6 compares three depth estimators and shows small performance differences, there is no analysis of failure cases where depth is severely wrong (e.g., reflective surfaces, textureless regions, outdoor scenes with sky). Since Equation (5) normalizes by the number of far pixels, any systematic depth bias could over-emphasize the wrong regions, potentially harming reconstruction. There is also no discussion on how depth scale ambiguity (common in monocular depth) is handled beyond using \(D_{\max}\), nor whether depth is ever inconsistent across views. This may be acceptable in the current benchmarks but limits confidence in harder in-the-wild settings.

4. **IMR computation details and scalability tradeoffs are not fully specified in the main text.**  
   Section 3.4 mentions importance sampling ~10k Gaussians and using Sinkhorn OT, but the choice of sampling distribution, entropic regularization \(\varepsilon\), and stopping criteria are only described at a high level. It is unclear how sensitive IMR is to these choices. Moreover, while Appendix A provides a good derivation for the Bures approximation, there is no empirical study quantifying the approximation error (Equation (11) vs. exact Equation (10)) for typical Gaussian covariances in 3DGS. Without such analysis, it is hard to judge whether the IMR scores in Table 3 are numerically reliable or how much they depend on these approximations.

5. **Limited comparison set for robustness and stability claims.**  
   IMR comparisons in Table 3 are reported only for 3DGS, CoR-GS, DropGaussian, and D²GS; other relevant sparse-view methods evaluated elsewhere in the paper (e.g., FSGS, LoopSparseGS) are absent. For instance, LoopSparseGS explicitly targets sparse-view robustness, but its IMR is not reported. This makes it difficult to assess whether D²GS’s stability advantage is specific to the chosen baselines or holds generally. Similarly, Figure 3 (left) only visualizes PSNR fluctuations for a single prior method rather than multiple baselines including D²GS.

6. **Missing discussion and comparison with several recent sparse-view 3DGS methods.**  
   The related work covers some important 3DGS-based sparse-view approaches (CoR-GS, LoopSparseGS, PixelSplat, MVSplat, HiSplat), but omits several highly relevant recent methods that also target sparse-view settings with different priors or architectures (see “Potentially Missing Related Work”). Some of these (e.g., LM-Gaussian, InstantSplat, SparseGS-W, S2Gaussian) are directly comparable in problem setup and design philosophy. Their absence weakens the positioning and may overstate the claimed “state-of-the-art” status.

7. **Evaluation scope is still relatively narrow given the stated goals.**  
   Despite using three datasets and multiple view-count settings, the experiments remain close to the established benchmarks (LLFF, MipNeRF360, DTU) that already have well-behaved COLMAP poses and reasonably clean imagery. The paper’s motivation stresses instability and over/underfitting in realistic sparse-view scenarios, yet there is no evaluation on more challenging cases such as unbounded outdoor scenes “in the wild”, noisy poses, or strong occlusions. Given that both DD-Drop and DAFE assume reasonably accurate geometry and depth priors, it is unclear how the method behaves under more realistic capture errors.

8. **IMR and AVGE as custom metrics are introduced with limited interpretability and validation.**  
   AVGE is described as the geometric mean of transformed PSNR/SSIM/LPIPS, but there is no justification that this combination is more informative than reporting the three metrics separately (as is already done). For IMR, Equation (14) uses a log ratio \(\ln(\sum S_{ij}^2 / \sum S_{ij})\); while this amplifies larger distances, the choice feels ad hoc and there is no exploration of alternative aggregations (e.g., mean or variance). Without empirical or theoretical backing for these particular functional forms, the community may find these metrics hard to interpret or adopt.

9. **Some implementation and cost tradeoffs deserve more scrutiny.**  
   Table 7 reports that D²GS training is 1.46× slower than DropGaussian on LLFF 3-view (82s vs. 56s), due to density computation, depth estimation, and dropout overhead. While this is “relatively small” in the authors’ wording, the slowdown may be non-negligible in practice, especially given that the main benefit over DropGaussian in Table 1 is about 0.6 dB PSNR and modest SSIM/LPIPS gains. There is no clear discussion on memory overhead from storing depth/density scores, nor any profile of IMR computation time (even though IMR is not used during training, it is relevant if IMR is proposed as a standard evaluation tool).

10. **Some aspects of the math and notation could be clearer.**  
   - In Equation (2), the authors implicitly rely on the fact that \(S_i\in[0,1]\) (as a convex combination of normalized scores) and \(\lambda_{\text{middle}},\lambda_{\text{far}}<1\) to ensure \(P_i\le 1\), but this constraint is not stated explicitly; a brief note or clipping operator would clarify that \(P_i\) is a valid Bernoulli parameter.  
   - In Equation (13), the minimization is written without explicit marginal constraints, which are specified earlier in Equation (12); a short statement that the same constraints apply under entropic regularization would avoid ambiguity.  
   - The sampling scheme for selecting 10k Gaussians mentions “depth-stratified importance sampling” but does not define the exact probabilities; concrete notation or pseudo-code (even brief) would help others reproduce IMR.

Taken together, these issues do not invalidate the method or results, but they temper the strength of the claimed robustness contribution and raise questions about generality and reproducibility outside the tested settings.

## Potentially Missing Related Work

The following directly relevant works on sparse-view 3D Gaussian splatting or closely related settings are not cited or discussed in the paper and should be included:

1. **InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds (Fan et al., 2025).**  
   - Relevance: Proposes a fast, SfM-free 3DGS reconstruction pipeline from few images, directly addressing sparse-view constraints and robustness.  
   - Suggested integration: Discuss in Section 2 (Novel View Synthesis with Sparse Views) alongside PixelSplat and MVSplat, contrasting DD-Drop/DAFE’s training-time regularization with InstantSplat’s architectural and initialization strategies. It could also be added as a baseline in future work if training-time rather than feed-forward methods are compared.

2. **S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting (Wan et al., 2025).**  
   - Relevance: Targets high-quality reconstruction from sparse, low-resolution views with 3DGS, sharing the same sparse-view goal but focusing on super-resolution.  
   - Suggested integration: Mention in Related Work when discussing sparse-view 3DGS-based methods that tackle underconstrained geometry with additional priors. The discussion around DAFE (Section 3.3) could contrast S2Gaussian’s super-resolution approach with depth-guided supervision of far-field regions.

3. **SparseGS-W: Sparse-View 3D Gaussian Splatting in the Wild with Generative Priors (Li et al., 2025).**  
   - Relevance: Uses generative priors to reconstruct complex outdoor scenes from very few images, addressing robustness and generalization in more challenging settings than LLFF/MipNeRF360.  
   - Suggested integration: Add to Section 2 as a complementary line that leverages strong priors rather than dropout or depth-guided losses. Discussion in Section 4 could mention that D²GS currently focuses on standard benchmarks and does not yet compete in such “in the wild” scenarios.

4. **LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors (Yu et al., 2024).**  
   - Relevance: Improves sparse-view 3DGS by injecting large-model priors (e.g., depth or semantic cues), conceptually related to using external depth estimators in DAFE.  
   - Suggested integration: Discuss in Related Work as another approach that leverages external priors, and compare with DAFE’s simpler monocular depth supervision. This would help contextualize D²GS as a lighter-weight alternative or complementary module to heavier prior-based systems.

Including and briefly comparing against these works would strengthen the paper’s positioning in the fast-evolving sparse-view 3DGS literature.

## Questions

1. **Correlation between IMR and image quality:**  
   Can you provide quantitative evidence that lower IMR correlates with better reconstruction metrics (e.g., PSNR, LPIPS) across runs and scenes? A scatter plot or correlation coefficient between per-scene IMR and the variance of PSNR/LPIPS over runs would significantly increase my confidence that IMR is measuring meaningful robustness.

2. **Approximation quality of the Bures distance:**  
   Have you empirically checked the error between the approximated \(\hat W_2^2\) in Equation (11)/(24) and the exact Bures-based 2-Wasserstein distance (Equation (10)) on typical covariance matrices from trained 3DGS models? Even a simple experiment on a subset of Gaussians would help justify that the approximation does not materially distort IMR ordering.

3. **Behavior under severe depth estimation errors:**  
   How sensitive is DAFE to large, structured depth errors from the monocular estimator? For example, if the estimator consistently misclassifies mid-distance regions as very far (or vice versa), does the model over-regularize or under-regularize these areas? Any qualitative failure cases or an experiment with artificially perturbed depth maps would clarify the robustness of DAFE.

4. **Generalization to more challenging capture conditions:**  
   Have you tried D²GS on scenes with noisy COLMAP poses, motion blur, or more extreme sparsity (e.g., 2 views)? If so, how does DD-Drop behave when the initial point cloud is substantially wrong? Even if not included in the main paper, some qualitative results or preliminary metrics in the rebuttal would be useful to assess generalizability.

5. **IMR computation specifics and cost:**  
   Could you detail the sampling strategy and Sinkhorn settings used for IMR (e.g., sampling probabilities across depth strata, value of \(\varepsilon\), number of Sinkhorn iterations), and report approximate compute time for IMR on a typical LLFF scene and ten models? This will help readers judge whether IMR is practical as a routine evaluation metric.

6. **Potential to learn dropout parameters:**  
   Given that many hyperparameters in DD-Drop (weights \(\omega_{\text{depth}},\omega_{\text{density}}\), \(\lambda_{\text{middle}},\lambda_{\text{far}}\), stratification tertiles) are hand-crafted, have you considered or attempted any scheme to learn or adapt them (e.g., via meta-learning, reinforcement, or simple grid search per-scene)? A discussion of why fixed values suffice, or what you observed when trying to learn them, would be valuable.

7. **Comparison to LoopSparseGS and similar methods in robustness:**  
   Why were methods like LoopSparseGS or FSGS not included in the IMR comparison in Table 3, given they are evaluated elsewhere? Including their IMR scores (even in supplementary) would give a more complete picture of where D²GS stands in terms of stability.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core techniques (DD-Drop, DAFE) and IMR metric are logically constructed and the derivations (especially for the Wasserstein approximation) are sound at the level claimed. Experiments are comprehensive on standard benchmarks with appropriate baselines and ablations. Remaining concerns center on heuristic choices, missing robustness analyses, and unvalidated aspects of the new metrics, rather than outright methodological flaws.

## Presentation Rating

3: good.  
The paper is generally clear, well structured, and readable. Figures such as Figure 1 (failure modes) and Figure 2 (overall architecture) are particularly helpful. A few mathematical and implementation details around IMR and dropout probability could be clarified, and the related work could better cover recent sparse-view 3DGS methods, but overall the exposition meets ICLR standards.

## Contribution Rating

3: good.  
The paper addresses an important problem (sparse-view 3DGS) and offers a combination of practical modules that yield consistent performance gains, as well as a conceptually interesting robustness metric. The ideas are incremental rather than fundamentally new, and some parts (especially IMR) could be better validated, but the overall contribution is solid and likely to be useful to practitioners.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work offers a well-motivated and carefully implemented set of improvements for sparse-view 3DGS, demonstrating consistent gains over strong baselines on established datasets and introducing a principled, if still somewhat under-validated, robustness metric. The main reservations concern the heuristic nature of DD-Drop, incomplete empirical validation of IMR and AVGE, limited evaluation on more challenging real-world conditions, and missing related work. With clearer positioning and deeper analysis of the new metric, the paper would comfortably merit acceptance; as it stands, it is a solid and useful contribution that marginally clears the bar.

## Reviewer Confidence

4: confident.  
I am familiar with NeRF/3DGS literature and sparse-view reconstruction, have carefully checked the core equations and ablations, and do not see major hidden pitfalls, though I cannot fully assess all aspects of the optimal transport approximation without additional experiments.