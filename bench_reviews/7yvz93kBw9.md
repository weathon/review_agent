## Summary

The paper proposes D²GS, a framework for improving sparse-view 3D Gaussian Splatting by addressing two identified failure modes: near-field overfitting (excessive Gaussians) and far-field underfitting (insufficient Gaussians). The method introduces a Depth-and-Density Guided Dropout (DD-Drop) mechanism that probabilistically drops Gaussians based on local density and depth scores, and a Distance-Aware Fidelity Enhancement (DAFE) module that reinforces supervision in distant regions using monocular depth priors. Additionally, the paper proposes an Inter-Model Robustness (IMR) metric based on 2-Wasserstein distance and optimal transport to quantify the stability of learned Gaussian distributions across independent training runs.

## Strengths

- **Clear and well-validated failure mode diagnosis.** The observation that sparse-view 3DGS suffers from spatially imbalanced Gaussian distributions—overfitting near the camera and underfitting far away (Figure 1, Section 3.1)—is specific, well-illustrated, and directly motivates the two complementary modules. The quantitative evidence (e.g., 11,450 vs. 6,112 Gaussians in the near field) grounds the motivation concretely rather than relying on vague claims.

- **Novel evaluation metric with theoretical grounding.** The IMR metric (Section 3.4) addresses a genuine gap in the literature: 2D image metrics cannot assess the stability of the 3D representation itself. The use of Wasserstein distance between Gaussian mixture distributions, with the Bures metric approximation and Sinkhorn solver, provides a principled formulation. Table 3 demonstrates that D²GS achieves the lowest IMR, and this is a fresh perspective for evaluating 3DGS robustness.

- **Consistent and meaningful improvements across multiple benchmarks and settings.** D²GS achieves the best results on LLFF (3-view, 6-view, both resolutions), MipNeRF360, and DTU across PSNR, SSIM, LPIPS, and AVGE. The gains over strong baselines like DropGaussian (+0.59 dB PSNR on LLFF 1/8, +0.55 dB on LLFF 1/4) and CoR-GS (+0.9 dB) are non-trivial. The ablation studies (Tables 4, 5, 6) systematically validate each component.

- **Principled soft dropout design that avoids DropGaussian's hard selection pitfalls.** The discussion in Appendix C clearly articulates why hard top-k dropout causes persistent suppression and over-suppression of detail-rich regions, and how DD-Drop's probabilistic mechanism avoids these issues. This design insight—*how* guidance signals are applied matters more than *what* signals are used—is valuable for the community.

## Weaknesses

### Major:

- **No analysis of sensitivity to depth estimation errors.** The DAFE module relies critically on monocular depth estimates to construct far-field masks. While Table 6 shows DAFE works with different depth estimators, all three estimators (MiDaS, DPT, DepthAnything V2) are strong modern models likely producing qualitatively similar depth maps. What happens when depth estimates are systematically wrong—e.g., for reflective surfaces, textureless regions, or scenes with depth inversions? The paper does not include any experiment with injected depth noise, corrupted depth maps, or documented failure cases. Since the paper positions itself around "stability" and "robustness," this gap is significant: the method could be introducing a brittleness that is not tested.

- **The IMR metric's correlation with perceptual/rendering quality is not validated.** The paper introduces IMR as a novel metric and shows D²GS achieves the best IMR (Table 3) and best PSNR (Table 1). However, there is no systematic analysis demonstrating that lower IMR *correlates* with higher rendering quality across methods and scenes. A scatter plot of IMR vs. PSNR/LPIPS across the 10 independent runs and across baselines would establish whether IMR is a meaningful proxy for quality, or whether it simply happens to agree for D²GS. Without this, IMR risks being a metric that is theoretically motivated but practically unvalidated.

### Minor:

- **Incomplete isolation of guidance signals vs. dropout softness.** The ablation in Table 4 progressively adds components, and Table 1 compares against DropGaussian (random dropout). However, there is no "soft random dropout" baseline—i.e., probabilistic dropout with the same time-varying schedule but without depth/density guidance. This makes it hard to fully separate the contribution of the *guidance mechanism* from the contribution of switching from hard top-k to soft probabilistic dropout. The comparison with DropGaussian provides indirect evidence, but a direct ablation would be more conclusive.

- **Computational cost of IMR is not reported.** While training time is reported (Table 7), the time required to compute IMR for a single scene (10 pairwise Wasserstein distances between Gaussian mixtures, each requiring Sinkhorn optimization over ~10k Gaussians) is not quantified. This matters because if IMR takes hours per scene, its utility for routine benchmarking is limited. The depth-stratified sampling to 10k Gaussians is mentioned but its effect on accuracy is not analyzed.

- **The Taylor approximation for the Bures metric may be inaccurate under high variance.** The first-order Taylor expansion (Eq. 11, Appendix A) assumes small deviations Δ between covariance matrices. Under sparse-view conditions where independently trained models can diverge significantly, this assumption may be violated. The paper does not discuss the approximation error empirically (e.g., comparing the approximate vs. exact Bures distance on a subset of Gaussians where exact computation is feasible).

### Trivial:

- **DropGaussian baseline reproducibility.** Appendix E notes difficulty reproducing DropGaussian's reported results. While transparency is appreciated, this raises a fair concern about whether the re-implemented baseline was given equal hyperparameter tuning effort. However, the authors' implementation appears to be used consistently, and the improvements are substantial enough that this is unlikely to be the sole explanation.

## Nice-to-Haves

- Comparison with feed-forward sparse-view methods (PixelSplat, MVSplat, HiSplat) to contextualize where optimization-based approaches stand against generalizable ones, though these are fundamentally different paradigms.
- A smooth alternative to the discrete depth layering (near/middle/far tertiles with hard attenuation factors λ_middle, λ_far), which could reduce the heuristic feel of the global mechanism and potentially improve generalization across scene types.
- Reporting rendering FPS to confirm the method preserves 3DGS's real-time advantage, since the added modules only affect training.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: DTU results relegated to appendix.** This is a space/formatting concern, not a scientific weakness. The results are present and complete.
- **Weakness: Missing broader impact discussion.** While sometimes expected at ICLR, the absence of a broader impact statement is not a scientific weakness of the method.
- **Weakness: No variance (±std) reported for PSNR/SSIM.** Reporting single-run results is the norm in 3DGS papers. The paper already goes beyond this by running 10 models for the IMR metric. Demanding variance for all metrics when it's not community standard is unreasonable.
- **Weakness: Min-max normalization may require hyperparameter retuning across scenes.** The cross-dataset results (LLFF, MipNeRF360, DTU) with very different scales already demonstrate that the method generalizes. The min-max normalization is precisely what makes it scale-invariant.
- **Weakness: L1 loss in DAFE insufficient for high-frequency details; should use perceptual loss.** This is speculative—the ablation shows DAFE works with L1. Suggesting alternative losses is a nice-to-have, not a weakness.
- **Weakness: Why is equal weighting (ω_depth=0.5) optimal?** This is a curiosity about the ablation outcome, not a methodological flaw. The ablation itself answers the question by showing the method is not overly sensitive to this parameter.

## Novel Insights

The paper's most underexploited insight is the distinction between *how* guidance signals are applied versus *what* signals are used for dropout. The discussion in Appendix C makes a compelling case that DropGaussian's failure with selective dropout stems not from the depth/density signals being uninformative, but from the hard top-k selection mechanism causing persistent suppression. This suggests that the community's current practice of comparing dropout strategies by their *signal* (random vs. gradient vs. distance) may be missing the more important axis of *application mode* (hard vs. soft). This insight generalizes beyond this paper and could inform regularization design in other optimization-based 3D reconstruction methods.

## Suggestions

- Add an experiment with synthetic depth noise (e.g., Gaussian noise, scale perturbations, or random pixel dropout on the depth maps) to quantify DAFE's sensitivity and demonstrate the method's robustness bounds. Even a simple analysis showing that D²GS degrades gracefully (rather than catastrophically) would significantly strengthen the robustness narrative.
- Provide a scatter plot correlating IMR with PSNR across the 10 training runs for each method to validate IMR as a meaningful quality proxy. If the correlation is strong, this validates the metric; if weak, it is important to report and discuss.
- Report the wall-clock time for computing IMR per scene so users can assess its practical utility as a benchmarking metric.