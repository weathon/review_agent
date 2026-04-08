=== CALIBRATION EXAMPLE 31 ===

# Final Consolidated Review
## Summary

This paper establishes eigenvector overlaps between training and test Hessians as a fundamental object in local loss geometry, arguing that spectra alone are insufficient to characterize generalization. The authors derive a universal fluctuation law (Theorem 1) decomposing the expected test-loss increment under training perturbations into spectral and alignment components, a transfer law for overlap functions under noise (Theorem 2), and apply these to ridge regression with arbitrary covariate shift—obtaining closed-form overlap decompositions that explain both covariate shift effects and multiple descent. Scalable estimators (Overlap-KPM) are developed and applied to a ResNet-20 on CIFAR-10, revealing that class imbalance reshapes train–test geometry through induced misalignment.

## Strengths

- **Theoretical framework is genuinely novel and well-motivated.** The paper identifies a precise mathematical gap—the missing eigenvector alignment ingredient in two-loss local geometry—and fills it with rigorous results. Theorem 1's decomposition of E[ΔL] into a double integral over spectral measures weighted by the overlap kernel O(λ₁, λ₂) is clean and informative, making precise the intuition that "which directions matter" is as important as "how curved they are." The isospectral covariate shift experiment (Fig. 1) elegantly isolates overlap effects by rotating Hessians while fixing spectra, providing a compelling proof of principle that alignment per se governs generalization.

- **Unification of multiple descent and covariate shift under a single overlap lens.** The demonstration that multiple descent peaks arise from eigenspace misalignment (rather than Hessian ill-conditioning alone) corrects a common simplification in prior work. The analysis in Section 3.2.2 and Fig. 3 shows how the overlap function's approximately block-diagonal structure governs which training modes route error into sensitive test directions—a concrete, mechanistic explanation that is genuinely new.

- **Overlap-KPM algorithm is a practical contribution.** Combining subspace iteration for outliers with Hutchinson + Chebyshev polynomial approximation for bulk overlaps enables estimation in models with millions of parameters using only matrix-vector products (standard Hessian-vector products via autograd). The complexity is O(PK²md), which is linear in both model size and data size—genuinely scalable compared to forming Hessians explicitly.

## Weaknesses

- **The quadratic approximation regime is not characterized.** The entire theoretical edifice (Theorems 1–3) rests on a local quadratic approximation to the train and test losses. The MLP experiments (Fig. 4a,b) show visible deviations between theory and measurement as perturbation magnitude grows, yet the paper provides no formal or even empirical characterization of the approximation's domain of validity. For the framework to be actionable, it matters greatly whether the quadratic regime applies under realistic SGD noise magnitudes, realistic distribution shifts, or only under infinitesimal perturbations. The surrogate-free formulation (Appendix B.2.1) replaces H_train with an effective Hessian H_train^eff, but this does not resolve the issue—it merely shifts the question to whether the resulting perturbation remains small.

- **Quantitative validation of the fluctuation law does not extend beyond tiny MLPs.** The predicted vs. measured ΔL plots (Fig. 4a,b) validate the theory on MLPs with layer widths (5,5,5,1)—networks with ~75 parameters. The ResNet-20 experiment (Section 3.4) demonstrates that overlap estimation is *feasible* at scale and reveals qualitative structure, but does not quantitatively confirm the fluctuation law. Without a predicted-vs-actual ΔL plot at ResNet-20 scale, the claim that overlaps "govern" generalization in modern networks rests on extrapolation rather than evidence.

- **The class-imbalance experiment is descriptive rather than predictive.** Section 3.4 shows that class imbalance reduces train–test Hessian alignment (Fig. 5), but does not demonstrate that overlap metrics predict generalization error more informatively than standard spectral measures or simple class-conditional accuracy. The observed misalignment is consistent with what is already known (imbalance hurts), and without a quantitative comparison to baselines, it is unclear whether the overlap perspective adds predictive leverage beyond existing diagnostics. This is the natural next step to establish practical significance.

- **Closed-form overlap results are restricted to stylized covariance models.** The explicit overlap decompositions in Section 3.2 rely on the k-level covariance model (Eq. 12) and Gaussian inputs, leveraging the free transfer law (Theorem 2) which requires asymptotic freeness. While the *framework* (Theorem 1) is general, the *analytically tractable results* that give the paper much of its explanatory power—resolving multiple descent, quantifying covariate shift—hold only for this specific family. Whether the overlap perspective yields new quantitative predictions for real data distributions (where no such closed forms exist) remains an open question.

## Nice-to-Haves

- **Comparison of overlap-based prediction against spectral baselines.** A plot showing that overlap-augmented predictions of ΔL outperform spectrum-only predictions (e.g., trace of H_test · C_train without the overlap kernel) would directly substantiate the "spectra are not enough" claim.

- **Quantitative characterization of the quadratic regime's validity.** Even an empirical curve showing theory-vs-measurement error as a function of perturbation magnitude across multiple architectures would clarify when the framework applies in practice.

- **Experiments on established distributional shift benchmarks** (e.g., CIFAR-10-C, DomainNet) rather than only class imbalance, to test whether overlap structure is diagnostic of shift severity in realistic settings.

- **An overlap-aware optimization experiment.** The Discussion mentions "alignment-aware optimization" as future work; even a preliminary regularizer encouraging train–test alignment and its effect on generalization would demonstrate the framework's prescriptive value.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Freeness not verified in neural network settings"** — Theorem 2 (transfer law) is applied only in the ridge regression analysis (Appendix C), where freeness holds asymptotically by construction. The MLP and ResNet experiments estimate overlaps *directly* without invoking Theorem 2. The concern about freeness in NNs is therefore misdirected: the paper's empirical pipeline does not depend on freeness holding in finite networks.

- **"Algorithm 1 is O(PK³) not O(PK²)"** — The dominant cost is matrix-vector products, not dot products. Algorithm 1 requires O(PK²) matrix-vector products at O(md) each, for total O(PK²md). The O(PK³d) cost of dot products is negligible. The paper's complexity claim is correct.

- **"Multiple solutions for the self-consistent equation (14)"** — Proposition 4 in Appendix D proves that r(z) is unique, holomorphic, and obtainable by fixed-point iteration under the stated conditions. This concern is explicitly addressed.

- **"Notation switches between H_train and Σ̂_train"** — The paper explicitly states (Section 3.2) "we will loosely refer to Σ̂_train as the train Hessian" since H_train = Σ̂_train + λI and they consider the ridgeless limit λ → 0. This is a deliberate, explained choice.

- **"No broader impact statement"** — ICLR does not require broader impact statements; this is a formatting/standard nitpick.

- **"Code availability"** — Reproducibility concerns about implementation details fall under the hard rule against nitpicks about reproducibility.

- **"E[∆w] = 0 is restrictive"** — The paper acknowledges this assumption and notes it holds for symmetric noise (including label noise under MSE, which is the primary setting analyzed). The general formulation with E[∆w] = 0 is standard in perturbation analysis; removing it would require a first-order term that the paper already writes explicitly.

- **"First-order term vanishing is only asserted"** — In the ridge regression setting, E[z] = 0 follows directly from E[ξ] = 0 (Gaussian label noise). In the general case, the first-order term is explicitly written in Eq. (5); the paper notes it vanishes for natural cases without claiming it always vanishes. This is not an unsupported assertion.

- **"Consolidate Contributions 5 and 7"** — Formatting/style suggestion, removed per hard rules.

- **"Missing related work on NTK eigenvector structure"** — Removed per hard rules on missing related works.

## Novel Insights

The most striking conceptual insight is that multiple descent—an effect previously attributed to Hessian spectral phase transitions—is actually governed by *which test directions* the near-zero training modes overlap with, not merely by their existence. Fig. 3c,d crystallizes this: between critical sampling densities α, the minimum training eigenvalue continues to decrease (which a spectrum-only analysis would predict to increase error), yet error *decreases* because the low-curvature training modes begin overlapping predominantly with the flat test subspace. This reframing turns a purely spectral puzzle into a geometric one, and suggests that interventions targeting alignment (rather than just flatness) could be a productive direction for generalization improvement.

## Suggestions

- **Add predicted vs. measured ΔL at ResNet-20 scale**, even for a single perturbation magnitude, to bridge the gap between MLP validation and large-scale application. This is the single most impactful addition.

- **Include a variance decomposition** quantifying what fraction of generalization-gap variance is explained by spectral changes vs. overlap changes, to substantiate the claim that overlaps are the "missing ingredient" rather than a secondary correction.

- **Explicitly state the domain of validity of the quadratic approximation**—even a rough rule of thumb (e.g., "the theory is accurate when ‖Δw‖ is small compared to the smallest curvature radius") would help practitioners assess applicability.

- **Report a quantitative alignment metric** (e.g., average normalized diagonal overlap) for the ResNet-20 experiment with error bars, so the "alignment largely disappears" claim in the class-imbalance comparison is statistically supported rather than purely visual.

---

**Novelty:** Strong. The formalization of eigenvector overlaps as the central missing object in two-loss geometry, with rigorous theorems and a transfer law, is a genuine conceptual advance. The perspective shift from "spectra determine geometry" to "spectra × overlaps determine geometry" is clean and productive.

**Technical soundness:** Good overall. Theorems 1 and 2 are correctly proved in the appendices. The ridge regression analysis is rigorous. The main gap is the uncharacterized validity of the quadratic approximation in practical settings.

**Empirical support:** Mixed. Theoretical predictions are beautifully confirmed in the linear and small-MLP settings. The large-scale experiments demonstrate feasibility and reveal qualitative structure, but lack quantitative validation of the core prediction (the fluctuation law) at scale.

**Significance:** High potential. If overlap metrics prove predictive of generalization beyond spectral baselines, this could shift how the community thinks about loss geometry. The current evidence is promising but incomplete—the bridge from theory to practice at modern scale needs one more plank.

**Clarity:** Good. The paper is well-organized with clear contributions enumerated upfront. The theoretical machinery (free probability) is appropriately deferred to appendices. Some notation could be more consistent, but this does not impede understanding.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 2.0, 4.0]
Average score: 5.6
Binary outcome: Accept
