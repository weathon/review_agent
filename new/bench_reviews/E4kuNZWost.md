## Summary

TULiP proposes a post-hoc uncertainty estimator for OOD detection inspired by linearized (lazy-regime) training dynamics. The paper derives a bound on epistemic uncertainty (Theorem 3.1 and Proposition 3.3) that relates test-time sensitivity to the distance from a test point to the training set in gradient-embedding space, and introduces a weight-perturbation algorithm to estimate this bound without training data or explicit Jacobians. Empirically, TULiP achieves competitive or state-of-the-art near-OOD detection performance on the OpenOOD benchmark across multiple datasets and architectures.

## Strengths

- **Novel theoretical bound on training-fluctuation uncertainty.** Theorem 3.1 (Eq. 5) bounds the discrepancy between perturbed and unperturbed converged networks by the gradient-embedding distance to the training set, and Proposition 3.3 converts this into a tractable test-time quantity. This addresses a genuine gap in post-hoc OOD detection by connecting uncertainty to training dynamics rather than logits alone.
- **Strong near-OOD empirical results.** Table 1 shows TULiP achieves top-1 or top-2 near-OOD AUROC on OpenOOD for CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K (e.g., best FPR@95 33.80 and AUROC 89.67 on CIFAR-10). Figure 3 further demonstrates consistent gains across MobileNet-V3, VGG-16, and RegNet-Y on ImageNet-1K.
- **Efficient, training-data-free post-hoc design.** The practical algorithm (Alg. 1) avoids explicit Jacobian computations via finite-difference Jacobian-vector products (Eq. 13) and Hutchinson-style trace estimation (Proposition 4.1), yielding roughly a 3× speedup over training-data-dependent methods such as ViM (Sec. 5.2).

## Weaknesses

### Fatal
None.

### Major
- **Structural disconnect between lazy-regime theory and practical algorithm.** Theorem 3.1 and Proposition 3.3 are derived under infinite-width, constant-NTK, gradient-flow assumptions, whereas TULiP is evaluated on finite ResNets, VGGs, and MobileNets trained with SGD. Figure 1a shows that the empirical NTK evolves significantly during training, directly violating the constant-NTK assumption. The layer-wise scaling heuristic (Eq. 12) is introduced precisely because the empirical NTK at convergence differs from the theoretical one, yet the paper provides no theoretical justification for why scaling by $1/\sqrt{|\theta_l|}$ preserves the bound. Consequently, the theoretical framework does not rigorously transfer to the evaluated setting, and claims that the method is “theoretically-driven” for practical architectures are overstated.
- **The closeness assumption (Eq. 8) is critical but only coarsely validated.** Proposition 3.3 relies on this pointwise geometric inequality for every test point $\mathbf{z}$. The paper’s only empirical check is Figure 1d, which aggregates over a sample of points from a single ImageNet-1K ResNet-18. Because OOD detection must be valid for each input individually, a dataset-level average is insufficient to guarantee that Eq. 8 holds pointwise. If the assumption fails for even a subset of points, the tractable bound in Eq. 9 ceases to upper-bound the uncertainty, undermining the theoretical bridge to the implementation.

### Minor
- **Missing statistical dispersion in benchmark results.** Table 1 reports averages over only three runs with no standard deviations or confidence intervals. This makes it impossible to judge whether small AUROC differences are statistically meaningful (e.g., CIFAR-100 near-OOD: TULiP 81.29 vs. GEN 81.31).
- **Large finite-difference perturbations lack bias analysis.** Proposition 4.1 and Eq. 13 assume $\epsilon, \delta \to 0$, but the implementation uses $\epsilon=2.0$ and $\delta=2$—far from the asymptotic regime—without analyzing the finite-$\epsilon$ bias of the Jacobian-vector product or trace estimators.
- **Generality beyond classification is unsubstantiated.** The abstract claims TULiP is “not limited to classification problems,” yet Algorithm 1 is classification-specific (softmax + entropy) and all large-scale experiments are on image classification. Only a synthetic regression task (Fig. 2a) is shown, with no real-world regression or structured-prediction benchmark.

### Trivial
- **Proof sketch clarity for Theorem 3.1.** The proof sketch mentions bounding the fluctuation “with an arbitrarily chosen pivot point $\mathbf{x}^*$,” while the theorem statement uses $\inf_{\mathbf{x}\in X}$. The relationship between the proof argument and the infimum could be spelled out more clearly.

## Nice-to-Haves
- Ablation comparing the raw tractable bound (Eq. 9) directly against the surrogate ensemble + entropy score (Alg. 1, lines 14–18) to isolate whether empirical gains stem from the theoretical quantity or from the ad hoc ensemble construction.
- Real-world regression or dense-prediction benchmark to substantiate claims of applicability beyond classification.
- Finite-$\epsilon$ error analysis for the Jacobian and trace estimators to justify the large perturbation magnitudes used in practice.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **Strawman about synthetic experiments:** The critic claims Sec. 5.1 “does not validate TULiP’s heuristic approximations on real networks.” The paper never makes this claim; synthetic validation is explicitly scoped to the lazy-regime bound, so this criticism is unfounded.
- **Speculative hyperparameter tuning on OOD data:** The critic suggests semantic-shift hyperparameters in Table 1 may have been tuned on a near-OOD validation set. The paper states it follows the OpenOOD protocol and conducts hyperparameter search “on a small validation set whenever possible.” There is no evidence that OOD data were used for tuning; this is pure speculation.
- **Surrogate posterior samples as a strength:** Moved here because the claim that TULiP generalizes beyond classification is only supported by a synthetic task (Fig. 2a), conflicting with the verified weakness that no real-world non-classification task is shown.
- **Layer-wise scaling as a supporting strength:** Moved here because, while Fig. 1 provides limited empirical support, the scaling remains a highly heuristic, load-bearing component with theoretical justification only on a single CIFAR-10 ResNet-18 run.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Explicitly qualify the scope of the theoretical claims: state that the lazy-regime bound motivates the practical heuristic rather than guaranteeing it for finite networks, or develop theoretical justification for the layer-wise scaling and finite-difference approximations under realistic training conditions.
- Report standard deviations or confidence intervals across the three benchmark runs to strengthen the evidential case.
- Include at least one standard non-classification benchmark (e.g., a UCI regression task) if the paper wishes to maintain the claim of broad applicability.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/xUO1HXz4an.md` (NegLabel, avg human score 7.50): Broader domain evaluation and stronger theoretical grounding; TULiP is below this because of the significant theory–practice gap.
- `/home/wg25r/review_agent/human_reviews/N6ba2xsmds.md` (HamOS, avg human score 6.75): Similar profile (novel method, strong empirical results, some overclaim); TULiP is slightly weaker due to a more fundamental disconnect between its lazy-regime theory and its finite-network implementation.
- `/home/wg25r/review_agent/human_reviews/ym0ubZrsmm.md` (SSOD, avg human score 5.33): Accepted poster with strong experiments but missing comparisons and practical concerns; TULiP has stronger benchmark coverage and fewer deployment flaws.
- `/home/wg25r/review_agent/human_reviews/am7BPV3Cwo.md` (ImOOD, avg human score 5.75): Rejected due to limited scope and missing large-scale validation; TULiP has broader experiments and stronger results, placing it above this band.
- `/home/wg25r/review_agent/human_reviews/Oo5spZRpH6.md` (HAct, avg human score 3.67): Rejected over serious methodological and evaluation issues; TULiP is substantially above this low anchor.

TULiP delivers real empirical contributions—most notably state-of-the-art near-OOD detection on OpenOOD—and introduces an interesting theoretical perspective. However, the transfer from lazy-regime theory to the practical algorithm is not rigorously established, and key assumptions (e.g., Eq. 8) are only coarsely validated. These issues keep it below the strongest accepted papers in the area but well above the rejected band. Relative to the calibration cluster, it sits near the lower end of the accepted range.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>