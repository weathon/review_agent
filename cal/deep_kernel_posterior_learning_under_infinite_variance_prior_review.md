=== CALIBRATION EXAMPLE 34 ===

# Final Consolidated Review
---

## Summary

This paper proposes Deep α-stable Kernel Processes (Dα-KP), derived as the infinite-width limit of deep Bayesian neural networks whose weights are elliptically distributed with infinite variance. The central theoretical result (Theorem 1) establishes that such networks converge to an α-stable process admitting a conditionally Gaussian representation, and that the resulting stochastic covariance kernels can be computed via a recursive formula in the spirit of Cho & Saul (2009). This formulation enables MCMC-based posterior inference while avoiding the O(n^{I+2}) exponential-in-dimension complexity of the prior shallow stable method (Loria & Bhadra, 2024), and is theoretically shown (Proposition 3) to support genuine feature learning—i.e., data-dependent posteriors over the kernel—unlike standard deep GP limits. Experiments on discontinuous synthetic functions and small UCI regression datasets support the computational and predictive advantages over GP baselines and DIWP.

---

## Strengths

- **Principled emergence of stochastic kernels:** Unlike Aitchison et al. (2021), which artificially injects noise to produce random kernels, stochasticity here arises *organically* from the infinite-width limit under infinite-variance priors. This is a conceptually cleaner and architecturally motivated design, directly traceable to the Gaussian scale-mixture representation of stable random variables (Eq. 1 and Theorem 1).

- **Elimination of exponential computational complexity:** The kernel-space recursion developed in Theorem 1 sidesteps the combinatorial feature-space enumeration of Loria & Bhadra (2024) (O(n^{I+2})), enabling application to 10-dimensional inputs and UCI datasets with I=6–13 where the prior method is entirely infeasible. This is a concrete, quantified improvement over the most closely related predecessor.

- **Formal characterization of feature learning via Proposition 3:** The paper provides a clean mathematical demarcation: for α < 2, the posterior of the features z_j^{(ℓ)} genuinely depends on observations y through the posterior of the mixing scales; for α = 2 (Gaussian), this dependence vanishes. This is not merely asserted but proved, connecting the stochastic kernel structure to the broader representation-learning critique of deep GPs (Yang et al., 2023).

- **Unification and extension of classical results:** Theorem 1 simultaneously generalizes Neal (1996), Cho & Saul (2009), and the shallow stable result of Loria & Bhadra (2024) into a single coherent multi-layer framework. The reduction to the Gaussian case when α → 2 (since S_{α/2→1}^+ degenerates to a point mass at 1) provides an elegant consistency check.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **No experiments on smooth functions — the key trade-off is unquantified.** All synthetic experiments are explicitly constructed with step/sign discontinuities, which are by design maximally favorable to stable priors. There is no evaluation on any smooth regression benchmark. Without this, it is impossible to assess whether Dα-KP degrades relative to GP methods when the truth is smooth — a critical practical concern. The paper scopes itself to the discontinuous case as motivation, but the *cost* of using a heavy-tailed prior on smooth data is a central question that any practitioner must answer, and the paper provides no guidance.

- **Depth provides no benefit — a serious negative finding that is underexplored.** Table 2 shows that RMSE is essentially identical across L = 3, 6, 11, 16 layers (e.g., 0.56→0.58 in 1D, 0.82→0.88 in 2D, 8.09→8.11 in 10D). The paper's conjecture that "one hidden layer is rich enough" is plausible but not investigated. If this is uniformly true, it raises the question of whether the "deep" architecture provides any actual benefit. Is this due to kernel collapse, signal propagation dynamics specific to stable marginals, or the particular test functions? The authors gesture at Schoenholz et al. (2017) but do not provide analysis. This is an important negative result that deserves more than a paragraph.

- **The nature of "feature learning" via global scale mixing deserves scrutiny.** In Dα-KP, the stochasticity of the kernel at each layer ℓ > 1 arises from a single scalar s_+^{(ℓ)} shared across *all* i,j weight pairs in that layer. This means the kernel is globally rescaled by one latent variable per layer — the kernel adapts its overall *magnitude* but not its *shape* locally in response to data geometry. True representation learning typically implies local adaptation of the feature space (e.g., different parts of the input space mapped differently). Whether global scale mixing constitutes the kind of feature learning that is claimed — and how it compares to, say, learned GP lengthscales — is not discussed. The Q-Q plot evidence of non-Gaussianity (Figure 3) confirms the mathematical prediction of Proposition 3 but does not demonstrate that this stochasticity translates into richer functional representations.

- **Missing modern competing methods.** The comparison pool — GP Bayes, GP MLE, DIWP, NNGP, and Stable — omits deep kernel learning (DKL; Wilson et al., 2016) and sparse variational GP methods (e.g., SVGP). These are natural competing approaches for flexible kernel learning in the deep setting and are standard in the literature the paper aims to advance beyond. Their absence makes it difficult to calibrate the magnitude of the empirical advantage.

### Minor

- **Asymmetric prior specification across layers is unexplained.** In Theorem 1, layer 1 assigns I individual scales (one per input dimension, akin to ARD), while layers ℓ > 1 each receive a single shared scalar s_+^{(ℓ)}. This asymmetry — plausibly motivated by input-dimension heterogeneity vs. hidden-unit exchangeability — is not stated or justified anywhere. If it is a modeling choice, it should be discussed; if it is a mathematical necessity for the derivations, that should be noted.

- **MCMC convergence diagnostics are entirely absent.** The correctness of the method's inference depends on the MCMC chain mixing well. No trace plots, R-hat statistics, or effective sample size estimates are provided anywhere for any experiment. For a method whose primary claim over variational alternatives is "gold standard" full posterior inference, this is a notable gap.

- **Computational complexity of the proposed method is not stated in the main text.** Computational efficiency over Loria & Bhadra (2024) is listed as Contribution 4, but the paper never explicitly states the complexity of Algorithm 1. The dominant cost is presumably O(n³) per MCMC iteration for the kernel matrix inversion, plus O(n²L) for the recursive kernel computation. A single clear statement of this would substantiate the efficiency claim.

- **Scalability to large n is unaddressed.** Experiments are limited to n ≤ 769 observations. The paper acknowledges inducing points and variational inference as future work, but does not bound the practically feasible n or provide any scaling experiments. For a paper targeting the ICLR community, this is a meaningful omission.

### Tiny

- **σ² sampling is not visible in Algorithm 1.** The paper mentions a half-Cauchy prior on σ² as part of the "fully-Bayesian" specification, but no sampling step for σ² appears in Algorithm 1. This is presumably handled in Algorithms 2/3 in the appendix, but should be made explicit.

- **The justification of kernel stochasticity via feature non-Gaussianity is slightly circular.** Section 4.2 argues: features are non-Gaussian, therefore Σ^{(ℓ)} is stochastic. But stochasticity of Σ^{(ℓ)} is already established by Theorem 1; a more direct empirical verification (e.g., posterior variance of kernel entries) would be cleaner than the indirect argument via non-Gaussianity.

---

## Nice-to-Haves

- **Ablation on α in the main text.** Currently fixed at α=1 in all experiments with ablation deferred to Appendix H.3.3. Since α is the most important hyperparameter and practitioners must choose it, even a brief summary of the ablation (or guidance on selection) in the main text would help.

- **Learnable α.** Fixing α as a modeling choice limits adaptability. Incorporating α as a learned or inferred parameter would make the model more flexible and remove a manual hyperparameter.

- **Visualization of stochastic kernel realizations.** Plotting samples of Σ^{(L)} (or its eigenvalue distribution) near discontinuities vs. smooth regions would directly illustrate whether the kernel structure adapts locally or merely scales globally — a targeted experiment that speaks directly to the feature-learning claim.

- **Calibration plots.** Coverage statistics are in Appendix H.2/I.2, but reliability diagrams would more transparently verify that the 90% predictive intervals actually achieve 90% coverage, supporting the UQ claims made in the abstract.

- **Activation function generalization.** Theorem 1 is stated for g_δ(ζ) = ζ^δ 1_{ζ>0}. A brief discussion or proof sketch on whether the recursive kernel derivation extends to non-homogeneous activations (tanh, GELU, sigmoid) would clarify the method's scope.

- **Finite-width convergence experiment.** The theory relies on the infinite-width limit, but no experiment validates how quickly finite-width BNNs approximate the limiting kernel process. A plot of prediction error vs. network width would ground the theoretical results in practice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Removed] No statistical significance testing demanded.** The harsh reviewer requests formal tests of significance on the RMSE differences. However, the paper reports mean and standard deviation over 20 splits, which is the standard practice for these benchmark comparisons. Formal hypothesis tests are not expected in this community, and the information provided is sufficient to assess overlap.

- **[Removed] DIWP baseline unfair comparison.** The harsh reviewer suggests DIWP may be undertuned on discontinuous tasks. However, using DIWP's own published hyperparameters is asymmetrically conservative toward the baseline — it cannot inflate Dα-KP's advantage. This is appropriate scientific practice.

- **[Removed] Running times hidden in appendix.** Table 1 captions explicitly direct readers to "Supplementary Section H.1" for timing results. Appendix placement is a style preference and does not constitute a substantive gap.

- **[Removed] Non-Gaussianity demonstration is "not surprising."** The harsh reviewer calls Figure 3 unsurprising because it follows from Proposition 3. While theoretically expected, the numerical confirmation on real data still provides value as an empirical sanity check and is not misleading.

- **[Removed / Weakened] Comparison against Deep Ensembles/SWAG/MC Dropout.** These methods do not arise from infinite-width BNN limits and are based on finite-width approximate inference. Including them is interesting but out of scope for a paper specifically about kernel process theory. At most a nice-to-have.

- **[Removed] Claiming second-best to Stable is a major weakness.** The paper is transparent that Dα-KP is second to Stable in 1D/2D while being dramatically faster and the only scalable option in higher dimensions. The trade-off is acknowledged and is the paper's core computational contribution.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the question of whether the "feature learning" achieved by Dα-KP corresponds to the kind of adaptive representation learning claimed. The stochasticity of the kernel at each hidden layer ℓ > 1 is entirely captured by a single scalar mixing variable s_+^{(ℓ)}, which rescales the entire covariance matrix globally but does not permit the shape of the kernel to adapt locally to data geometry. In contrast, the first layer does enjoy I dimension-specific scales (akin to ARD), providing genuine input relevance weighting. This structural asymmetry means the "feature learning" of deep layers is quantitatively different from — and arguably weaker than — the ARD-type adaptation at layer 1, and certainly different from the local adaptation mechanisms in standard deep networks. The paper proves that posterior features depend on data (which is formally correct), but does not characterize *how* or *how much* the kernel adapts. A clearer articulation of what kind of representation learning is and is not achieved — and a comparison against the ARD-GP baseline as a proxy for the layer-1 effect — would significantly sharpen the paper's claims.

---

## Suggestions

1. **Add a smooth-function benchmark.** Include at least one experiment where the true function is smooth (e.g., a sinusoid or squared-exponential draw) to characterize the price paid for heavy-tailed priors when they are unnecessary. This is the single highest-value addition to the experimental section.

2. **Investigate the null depth effect.** In Table 2, provide an analysis of whether the lack of benefit from depth is due to (a) kernel collapse across layers (check the spectral structure of Σ^{(ℓ)} as a function of ℓ), (b) the specific test functions used, or (c) a structural property of stable marginals. Even a short Appendix experiment would transform a puzzling negative result into an informative finding.

3. **State the computational complexity explicitly.** Add a single sentence to the main text stating the per-iteration cost of Algorithm 1 (e.g., O(n³) for kernel inversion plus O(n²L) for the recursion) and the practical upper bound on n under current implementation.

4. **Discuss and justify the layer-1 vs. layer-ℓ>1 scale asymmetry.** In Section 2 or a remark following Theorem 1, explain why layer 1 uses I per-dimension scales while layers ℓ > 1 use one shared scalar, and what modeling implications this has (e.g., only input-space relevance determination, not hidden-layer adaptation).

5. **Report or reference MCMC diagnostics.** Include effective sample size and/or R-hat for at least one representative experiment, or add a brief statement to the appendix confirming chain mixing.

6. **Provide guidance on α selection.** The ablation in Appendix H.3.3 is a start, but a brief practical recommendation (e.g., α=1 is robust; sensitivity is low for α ∈ [0.5, 1.5]) would help practitioners.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 5.0, 8.0, 5.0]
Average score: 6.8
Binary outcome: Accept
