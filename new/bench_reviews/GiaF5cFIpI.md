Now I have enough context to synthesize the final review. Let me carefully verify the key criticisms against the actual paper content.

## Summary

This paper presents a streaming framework for adaptively modeling and designing neural stimulations that drive low-dimensional latent dynamics in desired directions. The method integrates three components: a novel streaming dimensionality reduction method (sjPCA), an adaptive kernel-regression model $\hat{S}(x,u,t)$ for mapping high-dimensional stimuli to their effects on latent dynamics, and a constrained optimization procedure to select feasible stimulation patterns. The approach is validated on simulated data and on real neural recordings (calcium imaging and electrophysiology) with simulated stimulation effects, achieving sub-100ms computation times.

## Strengths

- **Addresses an important and underexplored problem:** Designing high-dimensional stimulation patterns to control low-dimensional neural dynamics in real time is a significant open challenge for causal neuroscience experiments. The paper provides one of the first end-to-end frameworks for this problem.

- **Principled and modular framework:** The pipeline cleanly separates latent space construction, dynamical modeling, stimulus-response estimation, and optimization (Algorithm 1). Each component can be swapped out, which is practically useful.

- **Adaptive stimulus-response modeling:** The kernel regression model for $\hat{S}$ that jointly conditions on latent state $x$, stimulus $u$, and sample age $t$ (Eq. 7) is well-motivated for handling non-stationary, state-dependent responses. The recovery from discontinuities and drift in the toy model (Fig. 2e) is a meaningful demonstration of adaptivity.

- **Real-time feasibility rigorously demonstrated:** Sub-10ms average runtimes on commodity hardware are reported, which directly addresses the practical constraint enabling closed-loop experiments.

- **Consideration of experimental constraints:** The optimization explicitly incorporates excitation-only constraints ($\mathbf{0} \preceq u \preceq \mathbf{1}$) and sparsity penalties, reflecting realistic limitations of optogenetic stimulation.

## Weaknesses

### Major

- **All validation on real data uses simulated stimulations, not actual neural perturbations.** The paper states "we simulated stimulations using an autoregressive function" ($y_t = r_t + a_t$, $a_t = 0.8 \cdot a_{t-1} + u_t$), which is a very simple linear injection that does not capture key properties of real optogenetic responses (nonlinear recruitment, state-dependent effects, sublinear summation, heterogeneous latencies). While the authors acknowledge this in the Discussion, the abstract and introduction frame the contribution as "a novel real-time method for designing neural stimulations that perturb latent dynamics" and state the method "enables the next generation of experiments." The core claim of adaptively driving neural dynamics can only be validated with real stimulation data. The paper demonstrates that $\hat{S}$ can learn a *known, artificially injected* mapping, but not that it can learn the *unknown, biologically complex* mapping from stimuli to neural responses. This is the most important gap between claims and evidence.

- **Baselines for both response modeling and stimulation optimization are insufficient.** For response modeling, the sole comparison is a "blind" model that ignores stimulation entirely — trivially, any method that accounts for known perturbations should outperform it. There is no comparison to simple parametric alternatives (e.g., linear regression in $u$ and $x$, input-output linear dynamical systems, or GLMs). For stimulation optimization, only random strategies (single neurons, random groups, shuffled designed stimuli) are compared against, with no comparison to principled baselines such as greedy neuron selection based on latent-space loadings (trivial under the open-loop map $S(u) = Q^\top u$), or existing methods like MiSO (Minai et al., 2024) or Bayesian optimization approaches (Wagenmaker et al., 2024) that are extensively cited but never benchmarked. This makes it impossible to assess whether the proposed approach offers any advantage over much simpler alternatives.

- **The optimization formulation is under-specified and insufficiently validated.** The objective in Eq. (8) uses $\|u\|_0^{\text{max}}$, which is never formally defined in the text — the surrounding prose says "offset by N to encourage a solution with the number of non-zero elements close to n" but the relationship between $\|u\|_0^{\text{max}}$, the parameter $n$, and $N$ is unclear. There is no analysis of how well the L1 relaxation approximates the L0 sparsity constraint, no ablation varying $\lambda_1$, and no discussion of optimization convergence (the solver, step size, initialization, and number of iterations are not specified). Since the optimization is the paper's key mechanism for designing stimuli, these gaps matter.

### Minor

- **The streaming latent space contributions (sjPCA, multi-space comparison) are loosely connected to the main stimulation results.** The stimulation experiments primarily use proSVD as the latent space. Figure 1c shows predictive-utility heatmaps for adaptive model selection, but no closed-loop stimulation experiment demonstrates that switching between latent spaces improves control. This makes the sjPCA and multi-space machinery feel like an add-on rather than an integrated contribution.

- **The assumption of at most one pending stimulus at a time** (Section 2.3) is acknowledged but not analyzed. In fast closed-loop experiments, overlapping stimulus effects could be common.

- **Scalability of kernel regression is unaddressed.** Eq. (7) sums over all $N$ previously observed stimulus-response pairs. As experiments proceed, this grows without bound. The paper does not discuss pruning, capping, or computational scaling, though references to online kernel regression literature (Cesa-Bianchi et al., 2015; Li & Liao, 2023) are cited without detailing how those advances are incorporated.

### Trivial

- The "10–20 stimulations" claim for convergence of $\hat{S}$ is only demonstrated on low-dimensional toy models; its generalization to realistic high-dimensional settings is not analyzed.

## Nice-to-Haves

- Comparison against even one existing stimulation design method (e.g., input-output LDS, Bayesian optimization) on the same simulated data.
- Analysis of how optimization performance scales with population size and latent dimensionality.
- A closed-loop demonstration with real stimulation (even in a reduced biological preparation) that validates the end-to-end pipeline.
- Ablation of the L1 sparsity penalty, showing what fraction of designed stimuli actually achieve the target sparsity.

## Removed Points

These points were flagged for removal or significant softening:

- **"No behavioral relevance evaluation" (from human finder):** The paper explicitly states in the Discussion that behavioral effects are out of scope: "We did not include any explicit consideration of the effects of stimulations on behavior." Criticizing an absent behavioral analysis is scope creep.

- **"Assumption 3.4 violations, identifiability concerns" (from iSSM review, human finder):** This criticism was imported from a different paper (iSSM) and does not apply here.

- **"Low correlation coefficients, sample sizes not clear" (from iSSM review, human finder):** Again, this is from a different paper's review and not applicable.

- **Formatting/style nitpicks:** Removed per instructions.

- **Reproducibility concerns about hyperparameters, optimization solver details:** While the solver is not specified (a valid minor concern), demanding complete training logs etc. is disproportionate for a methods paper.

## Novel Insights

The paper's insight that a stimulus-response map $\hat{S}(x,u,t)$ conditioning jointly on latent state, stimulus pattern, and sample age can track non-stationary perturbation effects in real time is genuinely useful. The combination of time-adaptive kernel regression with state-dependent response modeling addresses a real practical need in optogenetic experiments where response properties can drift. However, this insight remains largely unvalidated under realistic biological conditions.

## Suggestions

- **Most critical:** Run the framework on data with actual optogenetic or electrical stimulation perturbations, even from publicly available datasets, to validate that $\hat{S}$ can learn real stimulus-response mappings.
- **Add at least one principled baseline** for stimulation optimization — even a simple greedy baseline under the open-loop mapping would dramatically strengthen the evaluation.
- **Define $\|u\|_0^{\text{max}}$ formally** and provide an ablation showing how the L1 penalty affects the sparsity and quality of designed stimuli.
- **Discuss or implement memory management** for the kernel regression (e.g., a cap on stored observations, reservoir sampling, or forgetting factors) to address the inevitable growth of the observation set in long experiments.

## Score and Decision

**Calibration:** I compared with the following papers:
- **iSSM (FwW3jqchtY.md)**: Scores 3,5,6,6 → rejected. Had similar issues (simulated perturbations, no baselines, strong assumptions) but also theoretical identifiability flaws. This paper is somewhat stronger because it has a more complete pipeline and real-time implementation.
- **Closed-loop EEG stimulation (4ltiMYgJo9.md)**: Scores 3,6,8,6 → rejected. Similar problem of claiming "closed-loop" without real closed-loop experiments.
- **BRAID (3usdM1AuI3.md)**: Scores 6,8,8,3 → accepted poster. Stronger empirical validation on real data with real inputs, clear methodology.
- **realSEUDO (b7DsNJYmeo.md)**: Scores 3,5,6 → rejected. Incremental software contribution without sufficient novelty.
- **FCCA/controllable subspaces (4AlNpszv66.md)**: Scores 3,3,5,8 → rejected. Had theoretical derivation issues and limited empirical validation.

This paper addresses an important problem and proposes a coherent framework, but the empirical validation falls short of supporting the core claims. The combination of simulated-only stimulations (with a simplistic AR model), no principled baselines for the two main technical contributions, and an under-specified optimization makes it difficult to assess whether the proposed approach would work in real closed-loop experiments — the very application it targets. The paper is above the weakest calibration papers (which had more fundamental flaws) but below accepted papers like BRAID.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>