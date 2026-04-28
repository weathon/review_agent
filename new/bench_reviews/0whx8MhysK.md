## Summary
This paper proposes Influence-Guided Diffusion (IGD), a framework for dataset distillation that steers diffusion model sampling using trajectory influence functions to generate training-effective synthetic data. The method achieves state-of-the-art results on ImageNet-1K (60.3% at IPC=50) and demonstrates strong cross-architecture generalization without requiring diffusion model retraining.

## Strengths
- **State-of-the-art empirical performance on ImageNet-1K**: Table 2 shows Minimax-IGD achieves 60.3% accuracy at IPC=50, surpassing RDED (56.5%) and the fine-tuned Minimax baseline (58.6%). This is a genuine SOTA result on the full ImageNet-1K benchmark, which is the primary evaluation standard in the field.

- **Robust cross-architecture generalization**: Table 3 demonstrates the distilled datasets generalize well across four unseen architectures (ResNet101, MobileNet-V2, EfficientNet-B0, Swin Transformer), with Minimax-IGD outperforming RDED by an average of 5.0% at IPC=50. This addresses a common failure mode in dataset distillation where methods overfit to the surrogate architecture.

- **Modular, training-free integration**: The IGD framework plugs into both pretrained DiT and fine-tuned Minimax models without retraining the diffusion backbone. Table 1 shows DiT-IGD improves raw DiT by 5.8% on ImageNette and 6.6% on ImageWoof at IPC≥50, validating that the guidance mechanism provides value independent of the base model.

## Weaknesses

### Fatal
None

### Major
- **Missing generation time/cost data despite efficiency claims**: The abstract and Section 3.2 claim the method is "efficient" and avoids "prohibitive cost," yet the paper provides no wall-clock time or GPU-hour measurements for dataset generation. Section 4.1 states only that "All the experimental results of our method can be obtained on a single RTX 4090 GPU" without quantifying hours required. For ImageNet-1K at IPC=50, IGD requires backpropagation through a surrogate model for every generated image during guided sampling (Algorithm 1, Line 9), involving 50,000 images × ~15 guided steps × backprop cost. Without comparing this against baselines like DM, IDC, or Minimax (which optimize pixels over many iterations), the efficiency claim is unverifiable. This is a significant gap for a method positioning efficiency as a core contribution.

- **Theoretical overclaiming about "optimal equivalence"**: Section 3.2 states "Replacing the checkpoints θ_e^S with θ_e^T_c ... is an optimally equivalent target." This claim is not mathematically rigorous. The equivalence holds only under the idealized condition that synthetic data provides identical training dynamics as the full dataset, which is precisely what dataset distillation attempts to achieve—not an assumption that can be made. The paper later acknowledges this is "an approximation that bypasses the inner loop" when extended to mini-batch updates, but the initial framing overstates the theoretical grounding. Additionally, the use of cosine similarity instead of dot product (standard in influence functions) is acknowledged as a design choice "to stabilize the magnitude of the guidance signal," but this fundamentally changes the metric from influence (which depends on gradient magnitude) to gradient direction matching. The paper should more accurately frame the method as a heuristic approximation rather than claiming equivalence to trajectory influence functions.

### Minor
- **Hyperparameter sensitivity requires per-dataset tuning**: Figure 2c shows validation accuracy drops significantly when the influence factor k≥10 with entire guidance. Section 4.1 states hyperparameters (k, γ_t, guided range [A,B], checkpoint similarity threshold) are "empirically preset" with details deferred to the appendix. For a method proposed as a general framework, the lack of dataset-agnostic guidelines for setting these parameters undermines practical utility. If k and the guidance range require per-dataset tuning to avoid the abnormalities shown in Figure 2a, the "training-free" advantage is partially offset by tuning overhead.

- **Guidance signal quality degrades in later diffusion steps**: The early-stage guidance analysis (Section 4.4, Figure 2) reveals that applying guidance throughout the entire generation process with large k degrades performance despite reducing influence loss. This suggests the guidance signal becomes noisy or misaligned in later diffusion steps, which contradicts the premise of a stable influence metric. The paper interprets this as overfitting to the surrogate, but does not analyze why the influence signal would become less reliable at later timesteps.

### Trivial
- **Figure 3 interpretation ambiguity**: The figure shows Minimax-IGD has higher Wasserstein distance than DiT-IGD but better accuracy. The authors interpret this as support for their "conditional distribution" hypothesis, but the figure caption and main text could more clearly explain why a higher Wasserstein distance correlates with better performance in this case.

## Nice-to-Haves
- **Influence metric correlation analysis**: Plotting the correlation between computed influence scores (Eq. 7) and actual test accuracy of models trained on generated samples would verify whether the metric predicts training utility.
- **Failure case visualization**: Figure 2a shows abnormalities with high k for one class; providing more failure examples across different semantic concepts would help understand which categories are harder to distill.
- **Generation cost breakdown**: Even if total time is comparable to baselines, reporting the breakdown (surrogate training time, guided sampling time, vanilla sampling time) would clarify where computational bottlenecks lie.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Critic's claim about "checkpoint substitution being theoretically unsound"**: The paper does acknowledge this is an approximation (Section 3.2: "This adjustment mitigates the mismatch caused by the discrepancy between synthetic and real trajectories"). The criticism is valid but the paper does address it, albeit briefly. Moved to Major weakness about theoretical overclaiming.

- **Critic's claim about cosine similarity "decoupling from actual definition of influence"**: The paper explicitly states this is a design choice for stability (Section 3.2: "we use cosine similarity instead of the dot product to stabilize the magnitude of the guidance signal"). This is acknowledged, not hidden. The weakness is retained but reframed as the method being more accurately described as "gradient direction guidance" rather than strict influence guidance.

- **Strength about "efficient checkpoint selection reduces computation overhead"**: Table 6 shows the gradient-similarity method with 4 checkpoints achieves 82.0% vs. 79.8% for regular selection with 10 checkpoints. However, this compares different checkpoint counts, not equal-count efficiency. The strength is partially supported but the efficiency gain claim is not fully substantiated. Moved to Nice-to-Have.

- **Strength about "explicit diversity mechanism enhances training efficacy"**: Table 5 ablation shows G_D adds value (76.5% → 81.0% for DiT-IGD at IPC=50). This is concrete and retained in the main review implicitly through the modular design strength.

## Novel Insights
The paper's observation that Minimax-IGD achieves better accuracy than DiT-IGD despite having higher Wasserstein distance to the original dataset (Figure 3) is genuinely interesting. This suggests that for dataset distillation, matching the authentic data distribution may be less important than identifying a "pivotal conditional distribution" optimized for training effectiveness. This finding challenges the common assumption in generative dataset distillation that distribution alignment (lower FID/Wasserstein) correlates with better downstream performance, and warrants further investigation into what distributional properties actually matter for distillation.

## Suggestions
1. **Add generation time comparisons**: Include a table comparing wall-clock hours to generate distilled datasets for IGD vs. DM, IDC, and Minimax at comparable IPC settings. Even single-run measurements would substantiate the efficiency claims.

2. **Reframe theoretical claims**: Revise Section 3.2 to accurately describe the method as a heuristic approximation of trajectory influence based on real-data gradients and gradient direction matching, rather than claiming "optimal equivalence." This would align the theoretical framing with the actual implementation.

3. **Provide hyperparameter guidelines**: Add a subsection or appendix with heuristics for setting k and the guidance range [A,B] based on dataset properties (resolution, class count, IPC). Even rough guidelines (e.g., "for ImageNet-1K at IPC=50, we found k∈[0.5,2] works well") would improve practical utility.

4. **Analyze guidance signal stability**: Investigate why the influence guidance becomes less effective in later diffusion timesteps. Is it because the predicted clean data z̃_0|t becomes less accurate at early timesteps, or because the surrogate model's gradients are less informative for fine-grained details?

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Comparison to IGD |
|-------|-----------|-------------------|
| OD³ (W6gbWvvovB.md) | 6.00 | Similar SOTA claims with clear efficiency data (4.7 hours reported); IGD lacks this |
| DAP (Hvge3NzkJN.md) | 5.50 | Similar diffusion-based DD on ImageNet; IGD has slightly better results but similar weakness on missing cost analysis |
| CoDA (6ycBM1nsS3.md) | 4.50 | Reports 60.4% on ImageNet-1K (similar to IGD's 60.3%); rejected due to incremental method and scope concerns |
| Flash-DD (JRfZuc6zqI.md) | 5.50 (Reject) | Strong efficiency claims WITH data (843× speedup); IGD claims efficiency without data |
| GSDD (S77KqxyEEj.md) | 4.00 | Claims efficiency without full ImageNet-1K evaluation; rejected |
| "Why Less is More" (8KcjEygedc.md) | 7.50 | Theoretical paper with strong empirics; much higher bar for theory |
| 5hfa2itwGz.md | 5.50 (Accept) | Strong empirics with theory clarity weaknesses; similar profile to IGD |

**Reasoning:** The paper achieves genuine SOTA results on ImageNet-1K (60.3%), which is a significant empirical contribution. Compared to anchors:
- OD³ (6.0) has similar SOTA claims but provides explicit efficiency data; IGD does not
- DAP (5.5) has similar diffusion-based approach and missing cost analysis; IGD has better results
- CoDA (4.5) has similar ImageNet-1K results but was criticized as incremental; IGD is more novel
- Flash-DD (5.5, rejected) claims efficiency WITH data; IGD claims without

The missing efficiency data is a notable weakness but not fatal given the strong empirical results. The theoretical overclaiming is concerning but the method works in practice. This aligns with the 5.5-6.0 range for papers with strong empirics but some methodological gaps. I score this at **6.0** because the SOTA ImageNet-1K results are genuine and the cross-architecture generalization is robust, similar to OD³ which also scored 6.0 with comparable strengths and weaknesses.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>