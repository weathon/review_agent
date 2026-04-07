## Summary

This paper proposes Temporal Score Rescaling (TSR), a training-free method for controlling sampling diversity in diffusion and flow matching models. The approach applies a time-dependent scaling factor to learned score functions, derived analytically for isotropic Gaussian mixtures. TSR is compatible with both deterministic and stochastic samplers, requires no additional inference compute, and is empirically validated across five diverse domains: image generation, protein design, depth estimation, pose prediction, and robotic manipulation.

## Strengths

- **Training-free and computationally efficient**: TSR requires only a scalar multiplication per denoising step, making it immediately applicable to any pre-trained diffusion or flow model without retraining, distillation, or additional inference computation.
- **Broad empirical validation**: The method is tested across five distinct domains with different data modalities (images, proteins, depth maps, poses, robot actions), demonstrating that the approach generalizes beyond a single application area.
- **Compatibility with deterministic samplers**: Unlike Constant Noise Scaling (CNS) which requires stochastic samplers, TSR works with ODE-based solvers that are increasingly common for flow-matching models. The paper shows CNS performs poorly on SD3 with stochastic sampling (Section A.1), making TSR the only practical option for modern flow models.
- **Clear theoretical grounding for Gaussian case**: The derivation in Section 3.2 correctly shows that for isotropic Gaussians, temperature scaling of the data distribution corresponds to a time-dependent rescaling of scores. The extension to well-separated Gaussian mixtures (Appendix B) provides bounds on approximation error, even if limited to idealized settings.
- **Mode preservation demonstrated**: The toy experiments (Figures 2, 3) convincingly show that TSR preserves multimodal structure while reducing local variance, whereas CNS exhibits mode collapse on checkerboard and swiss roll distributions.

## Weaknesses

- **Marginal or inconsistent empirical improvements in several domains**: In depth estimation (Table 2), TSR improves AbsRel on ETH3D from 6.82 to 6.68 (2% relative), but ties with CNS on NYUv2. In pose prediction (Table 3), CNS with k=1600 actually outperforms TSR (k=7.0, σ=0.5) on every metric. The paper's framing that TSR "yields performance gains" should be moderated to acknowledge that CNS can match or exceed TSR in some settings.

- **No numerical results for protein design**: Section 5.2 presents only a scatter plot (Figure 6) for protein generation, with no quantitative table. The absence of numerical values makes it impossible to assess the magnitude of claimed improvements in designability and FID.

- **Hyperparameter selection lacks principled guidance**: The parameters (k, σ) appear to be selected via grid search, but the paper does not clarify whether a held-out validation set was used. Different tasks require very different values (k≈0.93 for image generation, k≈7.0 for pose prediction, k≈1.25 for robotics), and there is no guidance for new domains beyond trial-and-error.

- **Statistical significance not reported for small improvements**: The robotic manipulation improvement (81.7% → 82.8% average success rate) represents a 1.1 percentage point gain over 150 rollouts per task. At this sample size, individual task differences of 1-2 percentage points are not statistically significant, yet the paper presents them as meaningful improvements without confidence intervals.

- **Theoretical guarantees limited to idealized setting**: The mixture-of-Gaussians analysis (Appendix B) assumes well-separated modes with bounds depending on N (number of components) and d (dimensionality). For real high-dimensional data, these bounds provide limited quantitative guidance, and the paper offers no analysis of when the approximation degrades.

## Nice-to-Haves

- A principled heuristic or automated procedure for selecting (k, σ) would reduce the practical burden of deployment
- Failure mode characterization: The paper notes TSR hurts Tasks 2 and 8 in robotics, attributing this to low base success rates, but deeper analysis of when and why TSR degrades performance would be valuable
- Extended theoretical discussion on why the Gaussian assumption works sufficiently well in practice despite real data being non-Gaussian

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Comparison against Langevin MCMC correctors**: The spark finder suggests this comparison, but the paper explicitly discusses in Section 2 that Langevin correction "increases the computational cost at inference by an order of magnitude." Since TSR's contribution is efficiency alongside training-free operation, comparing to a method that requires ~10× more compute is not a fair like-for-like comparison.

- **Standard image benchmarks (ImageNet, COCO)**: While broader evaluation would strengthen credibility, the paper evaluates on SD3 and Flux.1, which are current state-of-the-art text-to-image models. Requesting ImageNet/COCO evaluation may be scope creep for a methods paper focused on sampling control rather than image generation benchmarking.

- **Citation of Hinton et al. (2015) for temperature sampling**: The harsh critic notes this refers to knowledge distillation, but the context (temperature on softmax outputs) is actually appropriate for the concept, even if not the canonical citation. This is a minor citation quibble.

- **Claims about CNS comparison fairness**: The harsh critic suggests the paper should be more balanced about CNS comparisons. Upon examination, the paper does acknowledge in Section 5.4 that CNS achieves slightly better performance on SYMSOL, so this criticism is partially addressed in the text.

## Novel Insights

The key insight from the paper is that temperature scaling of the data distribution corresponds to a time-dependent rescaling of learned scores, with the rescaling factor r_t(k,σ) = (η_t σ²/k + 1)/(η_t σ² + 1) depending on the signal-to-noise ratio η_t. This explains why constant scaling (CNS) fails: it cannot account for the varying noise-to-signal balance across timesteps. The observation that TSR achieves similar effects to "global" temperature while preserving mode weights (local temperature) rather than reweighting them is genuinely novel and has practical implications for controlled generation in multimodal distributions.

## Suggestions

- Report confidence intervals or statistical significance tests for all quantitative comparisons, particularly where improvements are small
- Include a numerical table for protein design results with specific (k, σ) values and best TSR configuration numbers
- Acknowledge explicitly in the abstract and introduction that TSR is not uniformly superior to CNS, and characterize the conditions under which each method is preferable
- Provide validation set details for hyperparameter selection to address potential overfitting concerns