## Summary
This paper introduces a real-time framework for adaptive stimulation and response modeling of latent neural dynamics. It integrates streaming latent space construction (including a novel streaming jPCA method), a nonparametric stimulation-response map, and a constrained optimization that designs high-dimensional stimuli to drive low-dimensional dynamics along desired directions under experimental sparsity and non-negativity constraints.

## Strengths
- **Novel streaming jPCA (sjPCA):** Proposes a new streaming variant of jPCA for real-time identification of rotational latent dynamics, demonstrated to converge to offline fits (Section 2.1, Fig. 1a).
- **Adaptive, nonparametric stimulation-response modeling:** Uses kernel regression to map stimulations to latent effects, accounting for state-dependence and temporal non-stationarity; shown to robustly adapt to drifts and discontinuities (Section 2.3, Eq. 7, Fig. 2e).
- **Practical optimization with experimental constraints:** Formulates an optimization problem for high-dimensional stimulus design with non-negativity and sparsity penalties, directly addressing limitations of tools like holographic optogenetics (Section 2.4, Eq. 8, Fig. 4).
- **Real-time performance on diverse data:** Validates the integrated framework on simulated data and two real neural datasets (calcium imaging, electrophysiology), with end-to-end runtimes under 100 ms, enabling future *in vivo* applications (Sections 3, 4, Appendix H).

## Weaknesses
- **No validation with real neural perturbations:** Experiments on real data use simulated stimulation effects (autoregressive model), not actual optogenetic or electrical stimulation responses. This leaves the core claim of causally driving latent dynamics unproven for real biological systems (Sections 4.1, 4.2).
- **Insufficient comparison to state-of-the-art baselines:** The optimization is compared only to random strategies and a naive model; it lacks comparison to established stimulation design methods like Bayesian optimization or active learning cited in related work, limiting assessment of its advancement (Fig. 4a, no comparison to methods such as Minai et al. 2024).
- **Approximate sparsity constraint handling:** The optimization uses an L1 penalty to approximate L0 sparsity, but the paper does not analyze how closely this enforces exact neuron counts or the trade-offs involved, which is critical for experiments with hard target limits (Eq. 8, no analysis of achieved sparsity vs. constraint).
- **Incomplete evaluation of adaptive latent space selection:** The framework runs multiple latent spaces in parallel, but the benefit and mechanism of adaptively selecting the best representation for stimulation design are not thoroughly quantified or explained (Fig. 1c, Appendix A.4).
- **Lack of statistical rigor:** Key quantitative results (e.g., alignment angles in Fig. 4, prediction errors in Figs. 2e, 3c) are presented without measures of variance or statistical significance across multiple runs, reducing confidence in performance claims.

## Nice-to-Haves
- Ablation study on the components of the kernel regression model (e.g., state, stimulus, and time kernels) to justify its complexity.
- Visualization of optimized high-dimensional stimulus patterns to illustrate what the method designs.
- Analysis of optimization failure modes, such as for infeasible directions, to delineate method limits.
- Quantification of adaptation speed to non-stationary stimulus-response mappings.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Insufficient methodological details for sjPCA and kernel regression:** The paper provides algorithmic descriptions and promises code release; missing derivations do not invalidate the contributions.
- **Demand for exploration of nonlinear manifolds:** The paper explicitly scopes to affine latent spaces; nonlinear methods are outside its stated contributions.
- **Formatting or writing style nitpicks:** None were substantive enough to include.

## Novel Insights
The paper's primary novel insight is the integration of streaming latent space estimation, adaptive stimulation-response modeling, and constrained high-dimensional optimization into a single real-time framework for causal neural dynamics interrogation. Beyond this, the reviews highlight that the method's efficacy on real perturbations and comparisons to existing approaches are critical gaps that, if addressed, would significantly strengthen its impact.

## Suggestions
- Apply the framework to a publicly available dataset with recorded neural responses to real optogenetic or electrical stimulations (e.g., from Daie et al. 2021 or Draelos et al. 2025) to validate the stimulation-response learning and optimization.
- Benchmark the optimization against state-of-the-art baselines like Bayesian optimization or active learning for stimulation design, at least in simulation.
- Include error bars or confidence intervals in key figures to provide statistical context for performance metrics.
- Elaborate on the adaptive latent space selection mechanism and evaluate its benefit for stimulation design in the experiments.