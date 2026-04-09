## Summary

This paper presents a streaming framework for adaptive stimulation-response modeling of latent neural dynamics. The core contributions are: (1) a novel streaming jPCA algorithm (sjPCA) for real-time rotational subspace identification, (2) a nonparametric kernel regression estimator $\hat{S}$ that models stimulus-response mappings as a function of latent state, stimulus, and time (enabling adaptation to non-stationarities), and (3) a constrained optimization problem for designing high-dimensional stimulation patterns that drive low-dimensional latent dynamics along desired directions under feasibility constraints (sparsity, non-negativity). The pipeline integrates multiple streaming dimensionality reduction methods and dynamical models in parallel, with adaptive selection of the best predictor at each timepoint. All components run in under 100ms end-to-end.

## Strengths

- **Unified streaming pipeline with real-time benchmarking**: The integration of streaming latent space construction, dynamical prediction, stimulus-response learning, and constrained optimization into a single real-time pipeline is a genuine engineering contribution. Concrete hardware specifications and per-component timing breakdowns (Appendix H) are provided—this level of benchmarking is rare in the closed-loop neuroscience literature and directly addresses the community's latency concerns.

- **Adaptive nonparametric stimulus-response modeling**: The kernel regression estimator $\hat{S}$ (Eq. 7) incorporates time as a regression feature, explicitly enabling the model to adapt to non-stationarities such as plasticity, probe drift, or photobleaching. The demonstration of recovery from abrupt mapping changes (Fig. 2e, "Flip" and "Rotate" conditions) shows this is not just a design feature but a functional capability.

- **Parallel latent space evaluation with adaptive selection**: Running proSVD, sjPCA, and mmICA in parallel and selecting the best predictor at each timepoint (Fig. 1c) mitigates the risk of committing to a single manifold hypothesis. The reported improvement in average log predictive probability (−1.72 with best single space to −1.01 with adaptive selection) provides quantitative evidence this mechanism adds value.

- **Validation on real stimulation data in Appendix C**: While the main experiments use simulated stimulations, Appendix C validates the stimulus-response regression on two datasets with actual photostimulation events (Daie et al., 2021; Draelos et al., 2025), showing lower prediction error than the blind model. This partially addresses concerns about biological validity, though it remains offline analysis.

## Weaknesses

### Major:

- **Main experiments use simulated stimulations on pre-recorded data**: The primary validation on calcium imaging and electrophysiology datasets (Section 4.1) injects synthetic stimulation effects via an autoregressive function ($y_t = r_t + a_t$, $a_t = 0.8 \cdot a_{t-1} + u_t$). The "closed-loop" is simulated: the biological system does not actually respond to the computed $u^*$. While the Abstract technically says "demonstrate our approach on both simulated and real neural data," the distinction between real recordings with simulated perturbations and true biological closed-loop is obscured. The core scientific claim—that the method can adaptively drive latent dynamics—remains validated only in silico. Appendix C's offline analysis of real stimulation data provides evidence that $\hat{S}$ can learn real stimulus-response relationships, but does not test the full optimization pipeline in a closed-loop biological setting. This gap between the framing ("adaptive stimulation of latent neural activity") and the evidence is the paper's most significant limitation.

- **Insufficient baselines for the stimulus-response mapping**: The primary comparison throughout is against a "blind model" that withholds stimulation information from the dynamical predictor. This is a minimal baseline that any stimulation-aware model should beat. The paper does not compare $\hat{S}$ against parametric alternatives (e.g., a linear model $S(u) = Wu$, or a simple affine mapping), which would establish whether the nonparametric kernel regression is necessary or whether a simpler model suffices. Given the computational overhead of kernel regression, this comparison is needed to justify the design choice. Notably, Appendix D shows that the closed-loop (kernel regression) estimator actually performs *worse* than the open-loop (linear assumption) estimator on the trivial stimulus-response mapping (Fig. D.1), suggesting the nonparametric approach may overfit when the true mapping is simple—a tradeoff that is not discussed.

- **No ablation studies**: The method composes multiple non-trivial components (three latent spaces, three dynamical models, kernel regression with state/stimulus/time features, constrained optimization with L1 regularization). No ablations are provided to determine which components are essential. For example: Does the time kernel $K_3$ actually improve performance over a stationary kernel? Does state-dependent kernel $K_1$ help beyond what stimulus-only regression provides? Does adaptive space selection improve stimulation outcomes, or only prediction accuracy? Without ablations, it is unclear whether the system's complexity is justified or whether simpler alternatives would perform comparably.

### Minor:

- **Optimization formulation clarity (Eq. 8)**: The term $\lambda_1(\|u\|_{0,\max} - \|u\|_1)$ is described as encouraging "a solution with the number of non-zero elements close to $n$." Under minimization with box constraints $[0,1]^N$, this term effectively *maximizes* $\|u\|_1$, which under these constraints pushes entries toward 1 (dense, high-power solutions) rather than promoting sparsity in the traditional L1-regularization sense. The intent appears to be: set $\|u\|_{0,\max}$ as a target, and maximize $\|u\|_1$ so that under near-binary solutions, the number of active neurons approximates this target. However, this design choice and its interaction with the alignment objective deserve clearer justification, as it inverts the standard LASSO-type relaxation of L0 constraints.

- **Kernel regression long-term scalability**: Eq. 7 sums over all $N$ past stimulation events. The paper notes that $N$ grows slowly (at the rate of stimulation events) and that the time kernel $K_3$ discounts old samples, but no explicit pruning or fixed-buffer mechanism is described. In a prolonged experiment, unbounded growth of the kernel dictionary could eventually violate the <100ms timing guarantee, particularly on the hardware specified. A discussion of practical mitigation strategies (e.g., discarding samples with negligible kernel weight, or a fixed memory budget) would strengthen the real-time feasibility claims.

- **sjPCA time derivative estimation**: The streaming jPCA formulation (Eq. 1) requires $\dot{X}_t$, but the paper does not detail how time derivatives are estimated causally in the streaming regime. If finite differences are used, the resulting delay in the feedback loop should be discussed. This is an implementation detail that could affect convergence or introduce bias.

### Trivial:

- The runtime claims are hardware-dependent, but the paper provides sufficient benchmarking (Appendix H) for readers to extrapolate.

## Nice-to-Haves

- **Comparison with existing stimulation optimization methods**: Experimental comparison against Bayesian optimization (Minai et al., 2024) or active learning (Wagenmaker et al., 2024) approaches for stimulation design would establish the relative merits of this constrained optimization approach.

- **Behavioral outcome measurements**: The paper motivates causal testing of how latent variables encode behaviors, but no behavioral metrics are reported. The authors acknowledge this scope limitation (Section 5); demonstrating that optimized stimulation changes behavior would significantly strengthen the neuroscience contribution.

- **Theoretical convergence analysis for sjPCA**: Empirical convergence to offline fits is shown (Fig. 1a), but stability bounds for the Sherman-Morrison-based update of the skew-symmetric matrix would increase confidence in the streaming estimator.

- **Uncertainty quantification on $\hat{S}$ predictions**: For safety-critical in vivo applications, confidence intervals or predictive uncertainty on the stimulus-response mapping would be valuable. The kernel regression framework naturally admits variance estimates.

- **Optimization landscape analysis**: No analysis of whether the constrained optimization (Eq. 8) has local minima that could trap solutions, which affects reliability for experimental deployment.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Missing related works / references not cited** — The spark finder suggested the paper fails to compare with Minai et al. (2024) and Wagenmaker et al. (2024). However, both are cited and discussed in the Introduction as prior work. The concern is actually about experimental comparison (baselines), not missing citations. Per rules, I do not flag missing related works.

- **Weakness: Reproducibility concerns about undisclosed hyperparameters** — The harsh critic flagged insufficient detail on kernel bandwidth tuning, RBF scaling initialization, etc. Per rules, nitpicks about reproducibility such as undisclosed hyperparameters are removed.

- **Weakness: Generalization across brain regions/behavioral states** — The spark finder questioned whether the method generalizes across brain regions or stimulus types. This is a generic concern that could be raised about any method; the paper tests on two modalities and multiple dynamical regimes, which is adequate for an initial demonstration.

- **Weakness: Scalability to thousands of neurons (whole-brain imaging)** — The neutral reviewer flagged this. The current experiments use 130–592 neurons, which matches the scale of holographic optogenetics applications the paper targets. Demanding whole-brain scale is scope creep.

- **Weakness: Cross-dataset generalization (train on one, test on another)** — This is not a standard evaluation for adaptive/stimulation methods, which are inherently experiment-specific. Removed as an unreasonable demand.

## Novel Insights

The paper reveals an interesting asymmetry in the feasibility landscape of latent perturbations: some directions in the latent space are naturally easy to drive via excitation-only constraints (e.g., along the first principal component), while others are structurally infeasible (e.g., population-wide inhibition). This feasibility structure is a property of the neural population and its embedding, not the optimization method, and could itself serve as a tool for characterizing the geometry of neural manifolds. The observation that closed-loop (nonparametric) stimulus-response estimation can outperform open-loop (linear assumption) design on nontrivial mappings but underperforms on simple ones (Appendix D) highlights a fundamental bias-variance tradeoff in adaptive stimulation that the community should attend to: adaptive methods are most valuable precisely when the mapping is unknown, but they require sufficient exploration to avoid overfitting.

## Suggestions

- **Reframe the contribution explicitly as a simulation-validated framework**: Adjust the Abstract and Introduction to clearly state that the method is validated via simulated stimulations on real neural recordings and offline analysis of real stimulation datasets, with in vivo closed-loop deployment as future work. This honest framing would strengthen rather than weaken the paper.

- **Add a linear stimulus-response baseline**: Compare $\hat{S}$ against $S(u) = Q^\top u$ (the open-loop/linear assumption) on the main real-data experiments. The toy model comparison in Appendix D already hints at the tradeoff; extending this to the calcium and electrophysiology datasets would clarify when the nonparametric approach is warranted.

- **Perform targeted ablations**: At minimum, ablate (a) the time kernel $K_3$ to test the adaptive/non-stationary benefit, and (b) the state-dependent kernel $K_1$ to test whether location-dependent responses matter. These would directly validate the two most distinctive features of the stimulus-response model.

- **Clarify the optimization formulation**: Either provide a more detailed justification for why maximizing $\|u\|_1$ approximates an L0 constraint under the stated conditions, or revise the formulation to use a more standard sparsity penalty and explain the design tradeoffs.

- **Add a pruning mechanism for kernel regression**: Describe even a simple strategy (e.g., discarding samples where all kernel weights fall below a threshold) to guarantee bounded memory and computation for long experiments.