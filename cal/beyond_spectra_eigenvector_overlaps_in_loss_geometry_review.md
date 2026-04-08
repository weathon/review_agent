=== CALIBRATION EXAMPLE 36 ===

# Final Consolidated Review
## Summary

This paper introduces a two-loss framework for local loss geometry, arguing that eigenvector overlaps between train and test Hessians—not just their spectra—are fundamental to understanding generalization. The authors derive a local fluctuation law (Theorem 1) expressing expected test-loss increments as a trace combining spectral data with an explicit overlap kernel, a transfer law (Theorem 2) for how overlaps transform under noise using free probability, and apply these to ridge regression to show that multiple descent and covariate shift effects are governed by eigenspace (mis)alignment. Scalable estimators (Overlap-KPM) are developed and demonstrated on ResNet-20/CIFAR-10.

## Strengths

- **The two-loss framing is a genuine conceptual advance.** The paper identifies a precise structural gap: most Hessian analyses treat spectra as synonymous with geometry, but as soon as one compares train and test losses, eigenvector alignment becomes essential. The isospectral covariate shift experiment (Figure 1) cleanly isolates this: identical spectra, rotated eigenspaces, and substantially different generalization error. This is more than rhetoric—it is a controlled demonstration that spectra alone are insufficient.

- **The overlap fluctuation law (Theorem 1) is clean and mechanistically insightful.** The decomposition E[ΔL] = ½∫λ₁λ₂O(λ₁,λ₂)μ_test(dλ₁)μ_train(dλ₂) directly shows that test error is large precisely when high-variance displacement directions overlap high-curvature test directions. This is sharper than the heuristic "flat minima generalize better" because it specifies *which* flat directions matter and *when*.

- **The multiple descent explanation via overlaps is compelling.** The paper identifies a regime (between lines 3–4 in Figure 3) where the minimum training eigenvalue decreases (which spectral logic predicts should increase error) yet error actually *decreases*, because the near-null train eigenspaces shift to overlap the flat test subspace. This is a concrete case where spectral reasoning gives the wrong prediction and overlap reasoning gives the right one.

- **The transfer law (Theorem 2) is a non-trivial technical contribution.** Factoring O_{A,B̂} = O_{A,B} · O_{B,B̂} under freeness provides an overlap calculus that decomposes complex operator relationships into tractable pieces. The proof in Appendix B.3, while dense, is rigorous and uses operator-valued free probability correctly.

- **Overlap-KPM makes the theory practically computable.** The combination of Chebyshev polynomial approximation with Hutchinson trace estimation and subspace iteration for outliers is well-designed, with essentially linear scaling in model size and data examples (O(PK²md) runtime, O(Kd) memory with the streaming implementation described in the text).

## Weaknesses

### Major:

- **The quadratic approximation regime is the theory's foundation, and its limits are insufficiently explored.** The entire framework assumes the loss is locally well-approximated by its second-order Taylor expansion. The paper acknowledges this in Appendix B.2.1 via an "effective Hessian" that absorbs higher-order terms, but does not provide bounds on *when* or *how badly* the quadratic approximation breaks down. For deep networks trained with large learning rates, SGD trajectories can traverse non-convex regions far from any minimum. Without understanding the radius of validity (e.g., how perturbation magnitude ϵ relates to local curvature), practitioners cannot know whether overlap analysis applies to their setting. A perturbation-magnitude ablation—even on the MLP—would significantly strengthen the paper by delineating the valid operating regime.

- **The gap between theoretical validation and claims about "modern neural networks" is substantial.** The fluctuation law is validated quantitatively on ridge regression (exact) and tiny MLPs (width 5,5,5,1). The ResNet-20 experiment demonstrates that overlaps *can be computed* at scale and that class imbalance correlates with misalignment, but it does *not* validate whether the fluctuation law actually predicts generalization in this setting. The paper's strongest claims ("universal," "fundamental missing ingredient") are calibrated to the theoretical and small-scale evidence, not to the large-scale setting. Either the empirical section needs to validate the fluctuation law on ResNet-20 (e.g., by adding label noise and measuring ΔL vs. the overlap prediction), or the claims need to be scoped to "theoretical foundations and diagnostic tools."

- **The paper claims to "correct" prior spectral interpretations of multiple descent without clearly demonstrating that spectral explanations give wrong predictions in a setting where overlap explanations succeed.** The key argument (Section 3.2.2, around Figure 3) is that error decreases between lines 3–4 despite decreasing minimum eigenvalue. However, existing spectral explanations (Chen & Mei, 2022; Mel & Ganguli, 2021) already account for multiple descent via spectral phase transitions—the issue is whether they attribute it to the *wrong* mechanism or whether overlaps simply provide a *more refined* mechanistic picture. The paper would be stronger with an explicit example where two models have identical Hessian spectra (including eigenvalue multiplicities) but different generalization due to different overlap structure, analogous to the isospectral covariate shift experiment but in a multiple descent setting.

### Minor:

- **Freeness assumptions have unclear finite-dimensional consequences.** The transfer law (Theorem 2) requires X to be free from A,B, which holds asymptotically. For finite d, deviations from freeness will distort overlap estimates. No guidance is given on how large d must be, or how to diagnose freeness violations empirically. This matters because the ridge regression formulas are presented as asymptotically exact, but applied to simulations with d=5000 (Figure 2) and d=100 (Figure 1).

- **The Overlap-KPM algorithm lacks theoretical error bounds.** The paper states that the Chebyshev approximation error decays "exponentially fast in K" and variance decays as O(1/√P), but no formal error bound is provided for the combined estimator. The smoothing kernel width σ also lacks clear selection criteria—too narrow and the overlap function resolves individual eigenvalues (noisy), too wide and fine-grained structure is lost. A principled guideline for σ selection would improve reliability.

- **The surrogate-free formulation (Appendix B.2.1) is important for neural network applicability but is relegated to the appendix.** This extension replaces the fixed H_train with an effective Hessian H^eff_train that accounts for non-quadraticity along the displacement path. Since the MLP and ResNet experiments are precisely the settings where this matters, promoting this discussion to the main text would strengthen the connection between theory and practice.

### Trivial:

- **Minor memory complexity discrepancy.** The text claims O(K) vectors in memory via streaming, but Algorithm 1 stores v_{j,k,μ} for j up to 2K and k up to K. The text's implementation description resolves this, but the pseudocode and prose are inconsistent.

## Nice-to-Haves

- **Causal intervention experiment for class imbalance.** The ResNet-20 experiment shows correlation between class imbalance and misalignment. An experiment that explicitly regularizes for train-test alignment and shows improved generalization would establish a causal role for overlaps, not just a diagnostic one.

- **Overlap evolution during training.** Tracking O(λ₁,λ₂) across epochs would reveal whether misalignment develops gradually or abruptly, and whether it is a cause or consequence of the optimization trajectory. This would bridge the static analysis (at a minimum) with the dynamic reality of training.

- **Validation of the fluctuation law on ResNet-20.** Adding controlled noise to the CIFAR-10 training set and measuring whether ΔL matches the overlap prediction would test whether the quadratic theory holds beyond the tiny MLP setting.

- **Discussion of how overlap analysis could inform optimization.** The "alignment-aware optimization" mentioned in the Discussion is intriguing but entirely undeveloped. Even a sketch of how one might differentiate through the overlap functional to form a regularizer would substantially increase practical impact.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Literature positioning incomplete—Wu et al. (2022) and May et al. (2019) not differentiated."** Removed per hard rule against missing related works criticisms; cannot confirm what other works exist or whether they address this exact framing.

- **"The claim that spectra alone don't explain generalization is asserted rather than demonstrated."** Removed as factually wrong—Figure 1 explicitly demonstrates this with an isospectral shift experiment.

- **"First-order term vanishes only for specific noise models—caveat should be explicit."** Removed as factually wrong—the paper explicitly states this caveat: "in several natural cases (e.g., label noise under MSE, analyzed below), vanishes exactly in expectation" (Section 3.1.1).

- **"Connection to TIC not developed."** Removed as scope creep—the paper explicitly positions TIC as a limiting case of the broader framework; developing it further is outside the paper's stated scope.

- **"H_train notation inconsistent (sometimes Σ̂_train + λI, sometimes Σ̂_train)."** Removed as factually wrong—the paper explicitly states the convention: "we will loosely refer to Σ̂_train as the train Hessian" since they focus on the ridgeless limit λ→0.

- **"No error bars or confidence intervals in Figure 4."** Removed per soft rule—requesting confidence intervals for small-scale theory-validation plots is not standard practice in this area.

- **"2D slice in Figure 4(d) is cherry-picking."** Removed as unreasonable—choosing a slice through the relevant points is standard visualization practice for loss landscapes.

- **"Number of trials not stated for Figure 1 experiments."** Removed per hard rule against reproducibility nitpicks about trivial implementation details.

- **"No comparison to baselines for Overlap-KPM."** Removed as unfair—there is no prior method for computing overlap functions between two implicit matrices; the comparison baseline (explicit eigendecomposition) is infeasible at scale, and the paper does validate on synthetic data (Figure 8).

- **"Presentation complexity / accessibility of free probability."** Removed as style/formatting nitpick per hard rules.

## Novel Insights

The paper's most striking observation is the **asymmetry between train and test roles in the overlap decomposition**: the training Hessian acts as a *variance filter* (inflating displacement along flat directions, suppressing along sharp ones), while the test Hessian acts as a *cost function* (penalizing displacement along its sharp directions). Generalization error is then determined not by either role in isolation, but by how the filter's output directions route into the cost function's sensitive directions. This filter-routing picture unifies phenomena that look unrelated from a single-loss perspective: covariate shift is harmful when it rotates the filter's high-variance output into the cost's high-sensitivity input; multiple descent peaks occur when spectral phase transitions cause the filter's high-variance modes to suddenly switch alignment between the test's flat and sharp subspaces. The conceptual payoff is that "flat minima generalize better" is not universally true—it depends on *which* flat directions and *what they overlap with*.

## Suggestions

- **Run a perturbation-magnitude ablation on the MLP.** Systematically increase noise amplitude and measure where the quadratic prediction breaks down. This would define the theory's valid operating regime and is a minimal addition that addresses the most important limitation.

- **Validate the fluctuation law on ResNet-20.** Even a single experiment adding controlled label noise to CIFAR-10 training, measuring ΔL, and comparing to the overlap prediction would bridge the gap between "overlaps can be computed at scale" and "the theory predicts generalization at scale."

- **Add a dedicated Limitations paragraph.** Discuss: (i) when the quadratic approximation fails, (ii) finite-d freeness violations, (iii) computational cost for very large models, and (iv) that the current experiments establish correlation (class imbalance ↔ misalignment) but not yet causation. This is standard practice and would strengthen credibility.

- **Promote the surrogate-free formulation (Appendix B.2.1) to the main text.** It directly addresses the "quadratic approximation" concern by showing the fluctuation law holds with an effective Hessian, and readers evaluating the neural network experiments need this context.

---

**Quality Assessment:**

- **Novelty:** High. The two-loss overlap framework identifies a genuine gap in the Hessian analysis literature and provides a precise mathematical formalism for filling it. The transfer law is a non-trivial contribution from free probability.

- **Technical soundness:** Strong for the theoretical contributions (Theorems 1–3, proofs in appendices). Weaker for the empirical bridge: the fluctuation law is validated in controlled settings but not at the scale where the paper's strongest claims are directed.

- **Empirical support:** Moderate. Ridge regression and MLP validation are thorough. The ResNet-20 experiment demonstrates computational feasibility but does not validate the theory's quantitative predictions in this setting. The multiple descent explanation is compelling but would be stronger with an isospectral demonstration analogous to Figure 1.

- **Significance:** Potentially high if the framework is adopted for analyzing generalization under distribution shift. Currently limited by the gap between diagnostic insight and actionable methodology.

- **Clarity:** Good for a technically dense paper. The main results are clearly stated. The free probability machinery in the appendices is necessarily complex, but the main text effectively conveys the key ideas. Some calibration of language ("universal," "resolve," "fundamental") would better match the evidence.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 2.0, 4.0]
Average score: 5.6
Binary outcome: Accept
