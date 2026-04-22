# Horseshoe Splatting: Handling Structural Sparsity for Uncertainty-Aware Gaussian-Splatting Radiance Field Rendering

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
We introduce Horseshoe Splatting, a Bayesian extension of 3D Gaussian Splatting (3DGS) that jointly addresses structured sparsity in per-splat covariances and delivers calibrated uncertainty. While neural radiance fields achieve high-fidelity view synthesis and 3DGS attains real-time rendering with explicit anisotropic Gaussians, existing pipelines do not explicitly encode structural sparsity in the covariance—e.g., axis-wise variances or pairwise correlations—leaving noise-dominated components insufficiently regularized. Uncertainty is likewise essential for trustworthy and robust novel-view prediction, yet most 3DGS variants remain deterministic. We place a global-local Horseshoe prior on the covariance scales, whose spike-at-zero and heavy-tails adaptively shrink irrelevant directions while preserving the salient structure. We fit the model with a factorized variational inference scheme that mirrors the Horseshoe's inverse-Gamma augmentation, enabling Monte Carlo rendering and pixel-wise posterior uncertainty with minimal overhead. Theoretically, we establish posterior contraction rates for the scale parameters and transfer them to the rendered image via a local Lipschitz mapping, providing guarantees that estimation error and predictive uncertainty diminish with data. Empirically, Horseshoe Splatting produces high-quality uncertainty maps while matching state-of-the-art 3DGS visual fidelity and runtime, yielding a practical, uncertainty-aware renderer that is robust to structured sparsity in the radiance field.
The code is available at https://github.com/HKU-MedAI/Horseshoe-Splatting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Horseshoe Splatting, a Bayesian extension of 3DGS for radiance field rendering. The key idea is to introduce a global–local Horseshoe prior over per-splat covariance scales, allowing the model to encode structured sparsity in the Gaussian footprints while also providing pixel-wise uncertainty estimates.  Empirically, the proposed method achieves state-of-the-art image quality and well-calibrated uncertainty on the LF and LLFF benchmarks, outperforming both prior NeRF- and 3DGS-based uncertainty modeling methods, while maintaining real-time rendering speed.

### Strengths
- The paper introduces a conceptually clear global–local Horseshoe for 3DGS uncertainty estimation while keeping sufficient efficiency.

- Theoretical derivations are provided to support the method. 

- Strong performance is shown on LF and LLFF.

### Weaknesses
- It is unclear whether the observed benefits are specific to the Horseshoe, or simply due to introducing any sparsity prior. Also, the ablation only reports improved metrics on the datasets, not how much structural sparsity was actually induced. Thus, the paper lacks quantitative evidence that the prior truly enforces the claimed structural sparsity. It should be further verified whether Horseshoe is necessary and essential.

- The assumption of scaling sparsity places priors on covariance scales, which has not been well verified whether this is sufficiently generalizable for various scenes. Also, as it does not rely on color or opacity, the effectiveness may be limited in scenes dominated by photometric ambiguity rather than geometric uncertainty, such as non-Lambertian surfaces.

- The experiments regarding active view selection may be too ideal. Considering there initially does not contain too many views in the LF dataset, starting with 10% of views may be too sparse and seems to be a toy study that can hardly match the real-world application. Would like to see if the method can still work in more scenarios.

### Questions
See the weaknesses.

### Soundness
2

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
4

### Summary
This paper introduces Horseshoe Splatting, a Bayesian variant of 3D Gaussian Splatting (3DGS) that explicitly models structural sparsity and uncertainty in radiance field rendering. The method places a global–local Horseshoe prior on per-splat covariance scales to suppress noisy or redundant directions while retaining key anisotropic structures. A variational inference framework enables Monte Carlo rendering and pixel-wise uncertainty estimation with little computational cost. Experiments on Light Field and LLFF datasets show that the approach achieves state-of-the-art image quality and well-calibrated uncertainty, while maintaining real-time rendering performance.

### Strengths
1. Innovative Bayesian extension of 3DGS that meaningfully incorporates structural sparsity and uncertainty estimation.
  2. Solid theoretical grounding, with posterior contraction guarantees linking model uncertainty to data consistency.
  3. Strong empirical validation, outperforming prior methods on fidelity and uncertainty metrics without slowing inference.
  4. Clear writing and presentation, with convincing visual and quantitative results.

### Weaknesses
1. Limited Conceptual Novelty Beyond Integration. While the Horseshoe prior is applied creatively to 3DGS, the overall contribution is more of a principled integration of known Bayesian shrinkage techniques rather than a fundamentally new rendering paradigm. The work’s novelty thus lies mainly in adaptation, not invention.
  2. Dependence on Heavy Computational Infrastructure. The approach relies on large-scale Gaussian Splatting models and variational inference, which may limit reproducibility and accessibility. It’s unclear how well the method scales down to smaller datasets or lightweight hardware settings.
  3. Lack of Diversity in Experimental Scenarios. Experiments are limited to static and relatively clean datasets (LF, LLFF). The paper does not demonstrate how the model behaves under dynamic scenes, noisy inputs, or severe view sparsity — conditions where uncertainty modeling would be most critical.
  4. Insufficient Analysis of Practical Utility of Uncertainty. While uncertainty maps are visually convincing, the paper provides little quantitative evidence of how uncertainty benefits downstream tasks, beyond one active view selection experiment. More demonstration of practical value would strengthen the impact.

### Questions
1. Generalization and Robustness – How does Horseshoe Splatting perform on dynamic or large-scale outdoor scenes where scene sparsity and motion are more complex?
  2. Computational Cost – What is the training time and memory overhead compared to vanilla 3DGS? Could this method realistically run on commodity GPUs?
  3. Ablation on Horseshoe Hierarchy – How sensitive is performance to the hierarchical prior choice? Would simpler priors (e.g., Laplace or Gaussian scale mixtures) achieve similar uncertainty quality?
  4. Downstream Use of Uncertainty – Beyond visualization, have the authors tested whether the uncertainty maps improve robustness in active learning or out-of-distribution detection?
  5. Scalability and Model Release – Are there plans to release pretrained models or a lighter implementation for reproducibility?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Horseshoe Splatting, which applies a global–local Horseshoe shrinkage prior to the covariance scales of each Gaussian in 3D Gaussian Splatting (3DGS) to encode structured sparsity. Using an inverse-gamma augmentation equivalent to the half-Cauchy prior, the method builds a factorized variational family for Monte Carlo rendering and pixel-wise uncertainty estimation. The authors also provide theoretical guarantees on posterior contraction under Lipschitz assumptions and validate the method on LF/LLFF datasets with improved view and uncertainty quality.

### Strengths
1. The paper’s exploration of uncertainty modeling in 3DGS is valuable. Incorporating both uncertainty and structured sparsity is novel, as most existing 3DGS uncertainty modeling focuses on geometry, semantic fields, or Fisher information approximations, with little direct regularization on covariance structures.
2. The experiments are comprehensive, jointly evaluating RGB and depth uncertainty (e.g., NLL, AUSE) and active view selection scenarios, with ablations (Gaussian vs. Horseshoe) demonstrating the benefit of the proposed prior.

### Weaknesses
1. The local Lipschitz assumption may be fragile since 3DGS rendering involves depth sorting and α-blending, which can cause discontinuities around occlusion or visibility changes. More clarification or evidence on how local neighborhoods avoid these discrete transitions would strengthen the theory section.
2. The active-view selection strategy lacks details — the paper mentions adding one view every 500 steps until 30%, but the convergence criterion, acquisition function, and budget alignment across methods are not clearly defined.

### Questions
It would be helpful if the authors could validate the method on more wild or challenging scenes, such as MipNeRF or Tanks and Temples, to demonstrate broader applicability.

### Soundness
3

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
This paper introduces Horseshoe Splatting, a Bayesian extension of 3D Gaussian Splatting (3DGS) that addresses structural sparsity in per-splat covariances while providing calibrated uncertainty estimates. The key innovation is the application of a global-local Horseshoe prior on covariance scales, which adaptively shrinks noise-dominated directions while preserving salient anisotropic structure. The authors develop a factorized variational inference scheme that enables Monte Carlo rendering with pixel-wise posterior uncertainty estimation. Theoretically, they establish posterior contraction rates for scale parameters and propagate these guarantees to rendered images via local Lipschitz mapping. Empirically, the method achieves state-of-the-art visual fidelity on standard benchmarks while producing high-quality uncertainty maps with minimal computational overhead.

### Strengths
- The paper addresses a genuine limitation of existing 3DGS methods: the lack of explicit structural sparsity encoding in per-splat covariances and the absence of principled uncertainty quantification
- The observation that noise-dominated components in covariance structures remain insufficiently regularized is well-articulated and demonstrated through visualizations (Figure 1)
- The choice of Horseshoe prior is well-justified by its spike-at-zero and heavy-tail properties, which are precisely what is needed for adaptive sparsification
- The method achieves state-of-the-art uncertainty estimation (Table 1: average AUSE of 0.18 on depth, NLL of -0.74 on RGB for LF dataset)
- Novel view synthesis quality is not sacrificed for uncertainty: PSNR of 30.05 on LF dataset, outperforming baselines by significant margins

### Weaknesses
- The paper addresses a genuine limitation of existing 3DGS methods: the lack of explicit structural sparsity encoding in per-splat covariances and the absence of principled uncertainty quantification
- The observation that noise-dominated components in covariance structures remain insufficiently regularized is well-articulated and demonstrated through visualizations (Figure 1)
- The choice of Horseshoe prior is well-justified by its spike-at-zero and heavy-tail properties, which are precisely what is needed for adaptive sparsification
- The method achieves state-of-the-art uncertainty estimation (Table 1: average AUSE of 0.18 on depth, NLL of -0.74 on RGB for LF dataset)
- Novel view synthesis quality is not sacrificed for uncertainty: PSNR of 30.05 on LF dataset, outperforming baselines by significant margins

### Questions
- Only covariance scales are stochastic; opacity, color (SH), and positions appear deterministic. Are there failure modes where scale uncertainty alone is insufficient?

### Soundness
3

### Presentation
3

### Contribution
3
