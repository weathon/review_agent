Now I have sufficient context. Let me write the final consolidated review.## Summary

The paper identifies two failure modes of 3D Gaussian Splatting (3DGS) under sparse-view conditions: near-field overfitting (excessive Gaussians near the camera) and far-field underfitting (insufficient Gaussians at distance). To address these, the proposed D²GS framework introduces (1) a Depth-and-Density Guided Dropout (DD-Drop) that adaptively drops Gaussians based on depth and density, and (2) a Distance-Aware Fidelity Enhancement (DAFE) module that reinforces supervision in distant regions using monocular depth-derived masks. Additionally, the paper proposes an Inter-Model Robustness (IMR) metric based on Wasserstein/optimal-transport distances between learned Gaussian distributions to evaluate 3DGS robustness beyond image-space metrics.

## Strengths

- **Well-motivated problem diagnosis.** The identification of two complementary failure modes (near-field overfitting, far-field underfitting) in sparse-view 3DGS is clearly articulated and empirically supported by Gaussian primitive visualizations (Fig. 1) with specific counts (11,450 vs. 6,112 near-field primitives, 3,082 vs. 5,224 far-field). This provides a concrete foundation for the proposed approach.

- **Consistent quantitative improvements over strong baselines.** D²GS improves over the best competing 3DGS-based methods by 0.3–0.9 dB PSNR across LLFF and MipNeRF360 datasets (e.g., +0.59 dB over DropGaussian on LLFF 1/8 res, +0.35 dB on MipNeRF360), with accompanying SSIM/LPIPS gains. The improvements are systematic across all tested settings.

- **Comprehensive ablation studies.** Table 4 systematically validates each component (density score, depth score, depth-based layering, DAFE), showing consistent incremental gains. Tables 5–6 study sensitivity to key hyperparameters and robustness to different monocular depth estimators.

- **Thoughtful local-global design in DD-Drop.** Combining a continuous local scoring function (Eq. 1) with discrete depth-layer attenuation (Eq. 2) is a reasonable way to balance fine-grained and scene-level regularization, with clear motivation from the different visibility patterns across depth ranges.

## Weaknesses

### Major:

- **The IMR metric is insufficiently validated as a general-purpose robustness measure.** IMR is presented as a core contribution—claiming to assess "robustness and fidelity beyond conventional 2D evaluations." However, (a) no correlation analysis is provided between lower IMR and better/more stable PSNR/SSIM across independent runs; (b) the metric uses opacity-weighting (Eq. 9) and depth-stratified oversampling of far-field Gaussians, which embeds the authors' own design biases about important regions—the same biases that DAFE explicitly optimizes for—creating a partial circularity; (c) a degenerate method producing nearly identical but useless distributions across runs would achieve low IMR, yet there is no sanity check against such pathologies; (d) IMR is reported for only 4 methods (3DGS, CoR-GS, DropGaussian, D²GS) on one dataset (LLFF), and not on MipNeRF360. Without demonstrating that IMR correlates with meaningful robustness properties that image metrics cannot capture, this metric remains an internal diagnostic rather than a validated evaluation tool.

- **Narrow experimental scope for general sparse-view claims.** The paper claims contributions for "sparse-view reconstruction" and "real-world scenarios" broadly, but experiments are limited to 3-view settings on LLFF and MipNeRF360. No evaluation at 6, 9, or 12 views (standard in the sparse-view literature) is provided for rendering metrics—6-view IMR is shown, but no 6-view PSNR/SSIM/LPIPS. It is unknown whether DD-Drop's near-field suppression hurts performance as views increase, or whether DAFE's far-field emphasis remains beneficial. This gap between the breadth of claims and narrowness of evaluation is notable.

- **No variance/confidence intervals reported for core metrics despite motivation emphasizing instability.** The paper itself motivates IMR by showing that 3DGS has high run-to-run variance (Fig. 3), yet Tables 1–2 report only per-dataset averages without standard deviations or confidence intervals across scenes. This is particularly striking when the PSNR gains over baselines are modest (~0.3–0.9 dB)—whether these gains are statistically meaningful is unclear. Moreover, despite training 10 independent models for IMR, the variance of PSNR/SSIM/LPIPS across those same runs is not reported, which would have been natural and informative.

### Minor:

- **Incremental methodological contribution of individual components.** DD-Drop combines a weighted sum of normalized depth and density scores with three-layer depth-based attenuation—a straightforward extension of DropGaussian's uniform dropout. DAFE applies a binary depth-masked L1 loss restricted to far-field pixels. While the combination is effective, each component individually is a relatively simple heuristic, and the paper does not explore more principled alternatives (e.g., uncertainty-guided dropout, learned attention weights).

- **Several design choices lack principled justification.** The tertile-based depth partitioning (Eq. 2), specific attenuation factors (λmiddle=0.7, λfar=0.3), and τ=5% depth threshold for DAFE are set by "experimental experience" or ablation. Table 5 shows non-trivial sensitivity (e.g., τ=5% vs. 15% gives ~0.15 dB PSNR difference), but no guidance is offered for how these might transfer to different scene types.

- **Monocular depth estimation as a dependency with limited failure analysis.** DAFE relies on monocular depth maps from DepthAnything V2. While Table 6 shows compatibility with three estimators, the paper does not analyze how systematic depth errors (scale ambiguity, boundary artifacts in far regions) affect DAFE mask quality or how failure cases propagate—particularly concerning since DAFE specifically targets regions where monocular depth is least reliable.

- **No computational overhead analysis.** DD-Drop requires computing k-nearest-neighbor density at training time, DAFE requires running a monocular depth estimator, and IMR requires solving optimal transport over ~10K Gaussians for C(10,2)=45 pairs. None of these costs are quantified, making it difficult to assess the practical trade-off relative to baselines.

### Trivial:

- None worth including.

## Nice-to-Haves

- Report PSNR/SSIM/LPIPS variance across the 10 independent training runs to directly complement the IMR analysis.
- Evaluate on a broader range of sparsity levels (6, 9 views) and/or additional datasets (DTU, RealEstate10K) to validate the generality claim.
- Provide a correlation analysis between IMR and image-metric variance across methods/runs to validate IMR as a meaningful metric.
- Discuss failure cases or scene types where the near/far-field assumption may not hold (e.g., indoor scenes with uniform depth).
- Consider soft depth-weighting in DAFE instead of the hard τ threshold, which could reduce sensitivity to the threshold parameter.

## Removed Points

- **"No comparison with feed-forward sparse-view methods (PixelSplat, MVSplat, HiSplat)."** These operate under a fundamentally different paradigm (cross-scene generalization with feed-forward inference) compared to the per-scene optimization setting of D²GS. The paper's baselines are appropriately chosen within the same optimization-based paradigm. This is a scope mismatch, not a valid weakness.

- **"Unfair comparison because DAFE uses monocular depth estimation as external prior."** The paper explicitly discloses using monocular depth and shows compatibility across three estimators (Table 6). Many recent sparse-view 3DGS methods (CoR-GS, SparseNeRF, DNGaussian) also leverage depth priors—this is standard in the field. The comparison is within the same class of methods.

- **"AVGE metric is introduced without motivation."** While terse, AVGE (geometric mean of MSE, √(1-SSIM), LPIPS) is a composite that balances different error types; this is a minor presentation issue, not a substantive weakness.

- **"Implementation details missing (k for k-NN, frequency of density recomputation, etc.)."** These are implementation-level details. The paper provides sufficient information to reproduce the method in the main text and appendix. This is a standard reproducibility nitpick for an empirical paper.

- **"Unclear whether baselines were re-implemented with same training budget."** The paper states "Our implementation is built on DropGaussian, with 10k training iterations." The comparisons follow standard protocol for this research area, where published numbers from prior work are typically used when available and settings match. This is not a strong fairness concern given the consistent improvements.

## Novel Insights

The observation that sparse-view 3DGS has a systematic spatial imbalance—Gaussians pile up in near-field regions while far-field regions remain under-populated—is genuinely insightful and well-supported with quantitative evidence. The coupling of this diagnosis with a dropout/growth strategy that differentially treats near vs. far regions is a logical response. However, the IMR metric, while an interesting conceptual direction, reveals a subtle challenge in 3DGS evaluation: standard image metrics cannot capture representation-level stability, but distributional metrics like IMR risk measuring self-consistency of priors rather than genuinely useful robustness properties. This tension deserves more honest acknowledgment.

## Suggestions

- Validate IMR by correlating it with PSNR/SSIM variance across the 10 independent runs you already have. Even a scatter plot of (IMR, std(PSNR)) across methods/scenes would be informative and directly address the metric's utility.
- Report per-scene results and variance, not just dataset-level means. This is especially important given the modest PSNR gains and the stated goal of robustness.
- Add results at 6-view settings for PSNR/SSIM/LPIPS (you already train 6-view models for IMR, so the rendering metrics should be available).
- Report training time and memory overhead relative to the DropGaussian baseline.

## Score and Decision Calibration

**Calibration papers considered:**
- **SplatFormer** (Accept-Spotlight, scores 8/8/8/6): Novel architecture (point transformer on 3DGS), extensive OOD evaluation, SOTA by large margins. Much stronger novelty and empirical evidence.
- **Geo-3DGS** (Reject, scores 5/5/5/5): 3DGS + SDF combo with moderate improvements. Similar profile of incremental methodological contribution on a popular topic.
- **RAIN-GS** (Reject, scores 6/6/6/5): Simple modifications to 3DGS initialization with moderate gains. Similar pattern of effective but incremental heuristics, lacking theoretical novelty.
- **MoDGS** (Accept-Poster, scores 6/5/8/8): Uses depth priors for dynamic GS, moderate novelty, but good gains and novel problem setting. Similar reliance on external priors.
- **MutualNeRF** (Reject, scores 5/5/5/3): Proposes a new metric (mutual information-based) alongside method improvements, but metric validation is weak. Similar profile to this paper's IMR issue.

D²GS sits in a similar space to Geo-3DGS and RAIN-GS: effective but incremental modifications to 3DGS with moderate PSNR improvements, plus a proposed metric that is under-validated. The narrow evaluation (3-view only) and the unvalidated IMR metric are meaningful weaknesses. However, the problem diagnosis is cleaner and more empirically grounded than many comparable papers, and the quantitative gains, while modest, are consistent.

**Score: 5.0**

This reflects: a useful problem diagnosis and consistent improvements at the tested settings, counterbalanced by (1) an IMR metric presented as a main contribution but insufficiently validated, (2) a narrow experimental scope that does not support the breadth of claims, and (3) individually incremental technical components. The paper would be substantially stronger if IMR were properly validated (showing it correlates with practical robustness) and experiments were extended to more view counts.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>