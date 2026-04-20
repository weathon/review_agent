## Summary

This paper proposes a framework for comparing multiple image representations by synthesizing "principal distortions" — a pair of stimulus distortions that maximize the variance of log-sensitivity ratios across $N$ models, extending pairwise generalized eigen-distortions (Zhou et al., 2023) to arbitrary model sets. Applied to early visual system models and DNNs (AlexNet vs. ResNet50, with adversarial and stylized training variants), the method reveals previously undocumented qualitative differences in local geometry between architectures, and shows that adversarial training — unlike stylized ImageNet — substantially shifts local sensitivities.

## Strengths

- **Principled $N$-model extension of pairwise distortion analysis.** The mathematical framework (Section 3) is well-motivated: the metric in Equation 3 is shown to be a pseudometric with scaling invariance, and the variance-maximization objective (Equation 4) provides a clean, closed-form generalization of Zhou et al.'s pairwise generalized eigenvalue approach that avoids the $O(N^2)$ pairwise combinatorics.
- **Novel empirical findings on architecture-level local geometry differences.** Figure 3E shows a robust, consistent separation between AlexNet and ResNet50 across 100 base images, with AlexNet more sensitive to textured/high-variability regions and ResNet50 to smooth/constant regions. This is genuinely new — the authors note that "as far as we know, this qualitative difference... has not been documented" (line 196) — and it persists across training variants (Figure 4).
- **Clear visual interpretability of local geometry differences.** The principal distortions are visually intelligible (Figures 2–5), making abstract FIM geometry concrete without resorting to opaque representational similarity metrics that often find these architectures equivalent.
- **Successful isolation of training-procedure effects from architectural effects.** Figures 4 and 5 convincingly show that adversarial training shifts local geometry more than architectural changes, while SIN training preserves the AlexNet/ResNet50 architectural signature.
- **Well-calibrated scope and limitations.** The paper explicitly acknowledges the local linear approximation, the Gaussian noise assumption (Section 5), and frames the human psychophysics application as a direction for future work (line 186, 242) rather than claiming current empirical validation.

## Weaknesses

### Fatal

None.

### Major

- **The iterative pruning / $O(\log N)$ efficiency claim is structurally unsound.** Section 4.1 (line 188) proposes that after collecting human perceptual data on the principal distortions for $N$ models, "models whose sensitivities are far from human sensitivities can be discarded and this procedure can be repeated to best differentiate the remaining models." However, the principal distortions are optimized jointly for a specific model set. Removing models and re-optimizing produces entirely new distortion vectors $(\epsilon_1, \epsilon_2)$ that are not comparable to the distortions used in earlier rounds. Previously collected human data cannot be reused, and the claimed $2 \log_2(N)$ scaling assumes transferability of distortions across model subsets — which contradicts the joint-optimization premise of the method itself. This argument overstates practical utility.
- **Absence of quantitative baselines for the $N$-model distortion selection objective.** The paper demonstrates that principal distortions separate models, but does not compare them against simpler alternatives: for example, top eigenvectors of an averaged FIM across models, or random distortions. The random distortion control in Figure 2A for the early visual model case is reasonable, but the DNN experiments (Sections 4.2) lack any baseline showing that the variance-maximization approach actually finds better differentiating distortions than simpler pooling strategies. Without this, it is unclear how much the principled objective adds over basic aggregations of pairwise methods.

### Minor

- **No discussion of FIM conditioning or optimization stability in deep architectures.** The paper computes FIMs as $I(s) = J_f(s)^T J_f(s)$ (line 164) for networks with pooling, batch normalization, and non-linearities, where Jacobians are frequently rank-deficient. No regularization, eigenvalue thresholding, or numerical stability measures are described. While the empirical results appear stable across 100 base images (Figure 3E) and random seeds (Supp. Fig. SI.5), these engineering details should be documented for reproducibility.
- **Limited model zoo restricts the generality of findings.** The DNN experiments focus exclusively on AlexNet and ResNet50, which the authors acknowledge as "not currently state-of-the-art" (line 194). While Supp. Fig. SI.2 is cited for EfficientNet and ViT, and Supp. Fig. SI.5 for random initializations, these are supplementary rather than main-text claims. The generalizability to transformer architectures and modern vision models is demonstrated only peripherally.

### Trivial

- None beyond minor notation and presentation preferences that are adequately handled in the paper.

## Nice-to-Haves

- A quantitative comparison of principal distortions against pairwise generalized eigen-distortions averaged across all model pairs, to show the $N$-model extension is doing something beyond pairwise averaging.
- Fourier spectra or spatial frequency analysis of $\epsilon_1$ and $\epsilon_2$ to more rigorously characterize the texture/smooth bias observed qualitatively in Figures 3B and 4B.
- Confidence intervals (bootstrapped across base images) for the log-sensitivity ratio results rather than standard deviation bars alone, to more sharply characterize statistical separation.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. *"Core utility claims for human perception and interpretability are unvalidated — zero human psychophysical data or functional/behavioral validation."* — The paper explicitly states this is future work (lines 186, 242) and does not claim to have conducted psychophysical experiments. The abstract frames this as an application direction ("suggest how it could be used to compare model representations with human perception"), not a demonstrated result. The qualitative comparison to human thresholds in Figure 2C is presented as preliminary and consistent with prior work (Berardino et al., 2017). Removing as a weakness would be unreasonable scope creep.

2. *"FIM conditioning leads to numerical instability or divergence in log-ratio calculations — no regularization or rank-handling."* — While the paper does not discuss this in detail, the empirical results are empirically stable (100 images, consistent patterns). This is a reproducibility concern worth noting but not a methodological flaw — it's a missing detail that can be addressed in revision. Downgraded to Minor.

3. *"No functional validation (classification degradation, neural predictivity) to prove distortions capture behaviorally-relevant geometry."* — The paper is a method paper about comparing local geometries, not a benchmark paper about predicting human behavior. Asking for downstream task validation is outside the stated scope. Moved to Nice-to-Have.

4. *"The claimed logarithmic scaling... is mathematically incoherent... invalidates a key practical claim."* — The O(log N) claim is indeed structurally flawed as noted above, but this is one speculative paragraph about a future application workflow, not a core mathematical result. Kept the substance as a Major weakness but weakened its scope from "invalidates the paper" to "overstates practical utility."

5. *"Missing appendix proofs — the parser strips those sections."* — Removed per hard rule regarding parser artifacts.

6. *"Missing optimization protocol details (learning rate, constraints, etc.) in main text."* — Appendix B is referenced for the gradient-based optimization algorithm (line 158), which is the appropriate location. Removed as scope-appropriate deferral.

## Novel Insights

The principal distortions framework elegantly reframes the question "which models differ in their local geometry?" as a variance-maximization problem over log-sensitivity ratios, producing exactly two interpretable distortion directions regardless of $N$. This is a genuinely useful theoretical contribution that avoids the combinatorial explosion of pairwise comparison. The finding that AlexNet and ResNet50 consistently differ in their sensitivity to textured vs. smooth image regions — across architectures, training regimes, and base images — provides a concrete, previously undocumented geometric signature of these architectures that global representational similarity measures cannot detect. This suggests that local geometry (as captured by FIM curvature in stimulus space) carries complementary information to global RSA and may be a fruitful direction for diagnosing why different architectures with equivalent benchmark performance behave differently in deployment.

## Suggestions

1. **Remove or substantially revise the $O(\log N)$ iterative pruning argument.** Replace it with a statement that, for large $N$, the two principal distortions provide a fixed-size stimulus set for initial discrimination, and note that re-optimization on model subsets is future work.
2. **Add a quantitative baseline experiment** comparing principal distortions against averaged pairwise generalized eigen-distortions or top eigenvectors of pooled FIMs, to demonstrate the value of the variance-maximization objective.
3. **Include a brief discussion of numerical stability** (eigenvalue regularization, or observed condition numbers) in the main text or a clearly referenced appendix section.

## Score and Decision

I calibrated against several papers from the corpus:

- **High-scoring anchors (7–8):** Papers like `9Cu8MRmhq2.md` (novel framework with extensive experiments) and `TwJrTz9cRS.md` (strong method with thorough ablation) scored 8s from humans for papers that combined novel methodology with strong, comprehensive empirical validation. Papers in the neuroscience/visual perception space like `rmg0qMKYRQ.md` (generative classifiers with human psychophysics) scored 8s partly because they included actual human behavioral validation.

- **Borderline anchors (5–6):** `wFPfYccHJ1.md` (method paper, missing experiments, scores 5,5,5,3) and `dgb4rfPzaw.md` (geometric framework, incomplete validation, scores 5,5,5,5) show that method papers without full empirical validation land in the 5 range. The psychophysics comparison paper `4GfEOQlBoc.md` scored 5,5,6,5 — similar pattern of theoretical interest but incomplete validation. Papers with qualitative claims lacking quantitative baselines tend to cluster around 5–6.

This paper has genuine mathematical novelty and visually compelling results that go beyond incremental extension. The core theoretical contribution (the metric and variance-maximization objective) is sound, and the empirical findings about architecture and training differences are real and interesting. However, the lack of quantitative baselines and the overclaimed efficiency argument bring it below the strong-accept tier. It is comparable to or slightly stronger than the borderline anchors because of the mathematical coherence and the genuinely novel empirical observations, but below the high-scoring anchors because it lacks comprehensive empirical validation and includes the structurally flawed efficiency claim.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>