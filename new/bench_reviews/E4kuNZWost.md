## Summary
This paper proposes TULiP, a post-hoc uncertainty estimation method for OOD detection that derives an upper bound on epistemic uncertainty from linearized training dynamics (NTK theory) and implements it via weight perturbation with variance matching. The method achieves strong empirical performance on OpenOOD benchmarks, particularly in near-OOD settings, without requiring training data access.

## Strengths
- **Strong empirical performance on OpenOOD benchmark**: Table 1 shows TULiP achieves top-1 or top-2 AUROC scores on near-OOD tasks across CIFAR-10 (89.67), CIFAR-100 (81.29), and ImageNet-200 (83.84), outperforming methods that require training data access like ViM and MDS on several metrics.
- **Comprehensive evaluation across architectures and OOD scenarios**: Figure 3 demonstrates consistent performance gains over baselines (MLS, ODIN) across MobileNet V3, VGG 16, and RegNet Y 16GF on ImageNet-1K. Table 2 shows competitive results on covariate shift OOD (ImageNet-C, ImageNet-R).
- **Theoretical derivation connecting training dynamics to uncertainty**: Unlike purely heuristic post-hoc methods, Section 3 derives an upper bound on training fluctuation (Theorem 3.1, Eq. 5) based on gradient distance to training data, providing a principled foundation for the uncertainty score.

## Weaknesses

### Fatal
None

### Major
- **Theory-implementation gap undermines the "theoretically-driven" claim**: Section 4.1 explicitly states "Lazy training often fails to capture the full characteristics of practically trained neural networks" and acknowledges "significant changes in the empirical NTK throughout the training process." The proposed layer-wise scaling (Eq. 12) is admitted to be "highly heuristic" (line 174). This means Contribution (i) claiming a "theoretical framework... which is empirically verified" is overstated—the theory's core assumption (constant NTK in lazy regime) is acknowledged to be false in the experimental setting, and the method relies on heuristic patches rather than theoretical derivation. This is similar to wKkKkFteiO (score 3.0) where reviewers criticized unjustified NTK use, though here the empirical results are stronger.

- **No justification for variance matching to upper bound**: Section 4.3 constructs surrogate posterior samples by forcing their variance to match the theoretical upper bound (Eq. 9) via scaling factor γ (Alg. 1, lines 14-16). However, Figure 2 shows the bound is strictly loose (light orange region significantly exceeds ground-truth ensemble). There is no theoretical or empirical justification for why matching variance to a *loose upper bound* yields better OOD discrimination than using raw perturbation variance. This renders the bound a heuristic motivation rather than a functional driver, similar to the theory-experiment disconnect in HksswvbYIp (score 3.5) where the bound was found "largely vacuous."

### Minor
- **Missing naive weight perturbation baseline**: The method's core mechanism is perturbing weights at test time (Alg. 1, line 8). While TULiP introduces specific scaling (Γ) and variance matching (γ), there is no baseline comparing to simple "Post-hoc Weight Noise" (e.g., computing variance of f(z; θ_T + N(0, ε)) without theoretical scaling). MCD uses dropout-based perturbation, not Gaussian weight noise. Without this ablation, it is unclear if the theoretical components provide benefit over naive weight perturbation, a known uncertainty estimator. This is comparable to eH9Wlahibz (score 4.0) where missing baselines weakened claims about the proposed technique's value.

- **Efficiency claims not fully substantiated**: The paper claims TULiP is "efficient" (Abstract, Contribution ii) and takes "3× less time" than ViM (line 311). However, TULiP requires M=10 forward passes with weight modifications per sample, while methods like MLS, GEN, and ASH are single-pass. No wall-clock latency comparison against these single-pass post-hoc baselines is provided. The efficiency advantage is only relative to feature-space methods requiring training data access, not to the actual single-pass competitors in Table 1.

### Trivial
None

## Nice-to-Haves
- Report calibration metrics (ECE, Brier Score) to assess whether forcing variance to match a loose upper bound produces well-calibrated uncertainty estimates, not just good ranking (AUROC).
- Include inference latency breakdown across different hardware to better substantiate efficiency claims for resource-constrained deployment scenarios.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim about θ_Init ≈ 0 approximation**: The critic notes Lemma 3.2 assumes access to θ_{t_s} but Alg. 1 substitutes θ_0 = 0. However, line 158 explicitly addresses this: "we take t_s = 0 and substitute θ_{t_s} with E[θ_0] = 0 (or other mean specified by initialization schemes)." This is acknowledged in the paper, not a hidden limitation.

- **Harsh Critic's claim about efficiency being "misleading"**: While valid as a Minor weakness, the claim that it's "orders of magnitude slower" is slightly overstated—M=10 forward passes is more costly than single-pass but not orders of magnitude. The paper does acknowledge computational tradeoffs.

- **Strength Finder's claim about theoretical bound being "empirically validated" in Figure 2**: This is partially misleading—Fig. 2 shows the bound *captures the pattern* of uncertainty but is visibly loose. The strength is kept but tempered in the final review.

- **Harsh Critic's claim about missing appendix/proofs**: The parser strips appendix sections; the paper references Appendix A.3, A.4, A.5, A.6, A.7, B.1, B.2, C, C.2, C.3, C.4 which exist in the original submission.

## Novel Insights
The paper's core tension—using NTK theory as motivation while explicitly acknowledging its assumptions are violated in practice—reflects a broader pattern in theoretically-grounded ML papers where elegant theory provides intuition but requires heuristic adaptation for practical use. Unlike papers that silently ignore assumption violations (which reviewers penalize heavily, e.g., wKkKkFteiO), this paper's transparency about the gap is commendable but simultaneously undermines its theoretical contribution claims. The empirical success despite theoretical mismatch suggests the weight perturbation mechanism itself is effective, independent of the NTK justification.

## Suggestions
1. Reframe Contribution (i) to present the theory as *motivation* rather than *derivation*—e.g., "We provide theoretical intuition from linearized training dynamics that motivates our weight perturbation approach."
2. Add a "Naive Weight Perturbation" baseline in Table 1 or ablation study to isolate the value of Γ scaling and γ variance-matching over simple Gaussian weight noise.
3. Include wall-clock inference time comparison to single-pass methods (MLS, GEN, ASH) to clarify the computational tradeoff.
4. Discuss whether the loose bound's conservativeness might actually benefit OOD detection by amplifying the signal, rather than treating it as a limitation.

## Score and Decision

**Calibration anchors compared:**
- **GEtOzC4MIi (6.0, Accept)**: Strong OOD empirical results with theoretical framework that Reviewer 1 noted "does not fit coherently" with the final method. This paper is similar but more transparent about assumption violations, which paradoxically weakens its theoretical claim more.
- **7rvMexIZA1 (5.6, Accept)**: NTK-based OOD detection with unclear algorithm implementation details. Very similar pattern—theory motivates method but implementation has gaps.
- **HksswvbYIp (3.5, Reject)**: Empirical NTK paper where reviewers found the bound "vacuous" and disconnected from experiments. This paper is stronger—Fig. 2 shows the bound does capture uncertainty patterns, and empirical results are solid.
- **wKkKkFteiO (3.0, Reject)**: NTK-guided continual learning where the NTK-forgetting link was unjustified with no empirical validation. This paper has much stronger empirical validation.
- **eH9Wlahibz (4.0, Reject)**: Flat-minima optimization missing strong baselines. Similar weakness pattern but this paper's empirical results are more comprehensive.

This paper sits between the 5-6 range anchors (GEtOzC4MIi, 7rvMexIZA1) and the 3-4 range anchors (HksswvbYIp, wKkKkFteiO). The empirical results are as strong as the 6.0 anchor, but the explicit admission that core theoretical assumptions are violated is more damaging than GEtOzC4MIi's implicit gap. Compared to 7rvMexIZA1 (5.6), this paper has clearer implementation but more severe theoretical overclaiming. The appropriate score is **5.0**—the method works well empirically and the theoretical transparency is honest, but the "theoretically-driven" claim cannot stand as written.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>