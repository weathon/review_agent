## Summary
This paper proposes D²GS, a method to improve 3D Gaussian Splatting (3DGS) for sparse-view novel view synthesis. It identifies and addresses two key failure modes: overfitting (excessive Gaussian density) in near-field regions and underfitting in far-field regions. The solution combines a Depth-and-Density Guided Dropout (DD-Drop) module and a Distance-Aware Fidelity Enhancement (DAFE) module. The paper also introduces a new distribution-based metric, Inter-Model Robustness (IMR), to evaluate the stability of the learned 3D Gaussian representation.

## Strengths
- **Clear, data-driven problem analysis and well-motivated solution.** The paper effectively diagnoses distinct spatial failure modes (Figure 1) and designs two complementary components (DD-Drop for near-field, DAFE for far-field) that directly address them.
- **Novel and thoughtful evaluation metric.** The proposed IMR metric, based on Wasserstein distance between Gaussian mixture distributions, moves beyond standard 2D image metrics to directly assess the robustness and stability of the 3D representation itself, offering a new lens for analysis in the field.
- **Extensive experimental validation.** The method is evaluated on multiple standard datasets (LLFF, Mip-NeRF360, DTU) under various sparse settings, consistently showing improved performance over strong baselines. Ablation studies are thorough and validate the contribution of each component.

## Weaknesses
- **Dependence on external depth priors.** Both core modules (DD-Drop's depth layering and DAFE's supervision mask) rely on monocular depth estimates. While an ablation shows consistent gains across different estimators (Table 6), the method's performance is inherently tied to the quality and generalizability of this external prior, which may fail in challenging scenes (e.g., textureless or reflective surfaces).
- **The utility and validation of the IMR metric are under-explored.** While novel, the paper does not establish the practical significance of IMR for end-users. For instance, it does not show a correlation between IMR and perceptual quality or training instability, nor does it compare IMR scores for key baselines (e.g., 3DGS, DropGaussian) to substantiate the claim that D²GS yields "more robust" distributions.
- **Incomplete ablation baseline.** The primary ablation study (Table 4) uses vanilla 3DGS as the baseline. Since the method is built upon and directly improves DropGaussian's dropout strategy, a direct ablation comparing guided dropout (DD-Drop) against uniform random dropout (DropGaussian) is necessary to cleanly isolate the benefit of the proposed guidance mechanism.

## Nice-to-Haves
- A more detailed breakdown of the computational overhead introduced by each new component (density computation, depth estimation, IMR calculation) to better assess the practicality of the added cost.
- An exploration of simple adaptive schemes for the hand-crafted thresholds (e.g., depth tertiles, DAFE masking ratio τ) to improve generalization across diverse scenes.
- A brief discussion correlating IMR scores with more intuitive measures of instability (e.g., variance in PSNR across runs) to help the community interpret the metric.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Insufficient methodological details for reproducibility (density computation, IMR sampling)."** The paper provides key details: density is computed via k-NN (k=6 stated in Appendix B), IMR uses depth-stratified importance sampling. Further implementation specifics are standard for the community.
- **Weakness: "Lack of statistical validation for quantitative improvements (no standard deviations)."** Single-run evaluation for PSNR/SSIM is the standard in this field for large-scale benchmarks. The stability claim is separately addressed by the IMR metric over multiple runs.
- **Weakness: "Missing comparison with recent feed-forward methods (PixelSplat, MVSplat)."** The paper's scope is improving optimization-based sparse-view 3DGS. Feed-forward methods represent a different paradigm (test-time feed-forward prediction) and are not standard baselines for this line of work.
- **Weakness: "No validation of the Bures distance approximation."** The approximation is derived to improve numerical stability and efficiency. Demanding an error analysis for a derived approximation used in a novel metric is an arbitrary rigor requirement beyond community standards for an empirical paper.
- **Weakness: "Omitted discussion of broader impact."** While good practice, its absence is not a technical weakness of the contribution.
- **Strength: "The paper is well-written."** This is generic and applies to any competently written paper.

## Novel Insights
The paper provides a systematic analysis revealing that sparse-view 3DGS fails in spatially distinct ways: it overfits by placing too many Gaussians in texture-rich near-field regions and underfits by placing too few in far-field regions. This insight directly motivates a unified solution with two spatially complementary components. Furthermore, it introduces a novel distribution-based robustness metric (IMR) that shifts evaluation from 2D image space to the stability of the 3D representation itself, a conceptual advance for assessing 3D reconstruction methods.

## Suggestions
- Compute and report the IMR metric for key baseline methods (e.g., 3DGS, DropGaussian) to provide direct, quantitative evidence supporting the claim of improved robustness.
- Include DropGaussian (uniform random dropout) as an ablation baseline in Table 4 to directly demonstrate the advantage of the proposed depth-and-density guidance over a naive dropout strategy.
- Add a brief analysis or discussion on how the method might perform when the monocular depth prior is particularly noisy or unreliable, to better characterize its limitations.