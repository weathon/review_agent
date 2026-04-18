## Summary

This paper proposes a nonparametric causal identification framework for functional longitudinal data—where treatments, confounders, and outcomes are observed as continuous-time stochastic processes—using measure theory and stochastic process theory to handle uncountably infinite treatment-confounder feedbacks. The framework generalizes classical g-computation, inverse probability weighting, and doubly robust identification formulas to this setting, accommodates time-varying outcomes subject to mortality and censoring, and establishes a "near-nonparametric" property (density of the identified model class in the space of all observed-data laws). The framework is evaluated through simple Monte Carlo simulations.

## Strengths

- **Addresses a genuine and important gap.** Extending causal identification from discrete-time or counting-process longitudinal settings to continuous-time functional data is a substantive theoretical challenge. The paper correctly identifies that existing frameworks cannot handle uncountably infinite treatment-confounder feedbacks, and the motivation from MIMIC-IV and CGM data is compelling.

- **Principled theoretical development.** The use of measure theory, martingale processes, and Radon-Nikodym derivatives to define the g-computation process $H_{\mathbb{G}}(t)$, IPW process $Q_{\mathbb{G}}(t)$, and doubly robust formula provides a mathematically rigorous generalization of classical results. The martingale structure naturally captures the sequential adjustment inherent in longitudinal causal inference.

- **Nonparametric identification result.** Theorem 4 proving that the model class satisfying Assumptions 1–4 is dense in total variation in the set of all observed data distributions is a meaningful theoretical contribution, showing the framework does not impose parametric restrictions. The paper honestly acknowledges the piecewise-continuous path space restriction.

- **Handles practical complications.** The framework accommodates mortality and right censoring (Assumption 2, time-to-event notation), which are ubiquitous in medical applications.

## Weaknesses

### Fatal
None.

### Major

- **The simulation study is essentially disconnected from the core theoretical contribution.** The main text simulation (Section 4) uses a data-generating process with no mortality, no censoring, and no measured confounders beyond the outcome itself. Treatment is drawn directly from the target regime $\mathbb{G}$ with no treatment-confounder feedback, so the g-computation formula reduces to a simple sample average. Verifying that this sample average converges to the analytically known value (zero) is a check of Monte Carlo integration and the law of large numbers, not of the causal identification machinery. The paper acknowledges that a more complex scenario is in Appendix D, but the main numerical contribution does not exercise the framework under the conditions (confounding, censoring, mortality) that motivate it. This is a significant evidential gap: for a paper whose primary contribution is theoretical, the numerical validation should at minimum test identification under confounding in continuous time.

- **No estimation framework or practical guidance.** The paper explicitly scopes out estimation, stating it is "beyond the scope of this study and left for future research." The identification formulas involve conditional expectations and Radon-Nikodym derivatives on infinite-dimensional path spaces; estimating these from finite samples is deeply non-trivial. Without even a sketch of how estimation could proceed, the practical impact of the identification results remains unclear. This gap is especially pronounced given that Ying (2024b) apparently develops related ideas further but also lacks numerical examples, suggesting this line of work remains far from application.

- **Assumption 1 is novel and lacks sufficient justification.** The "full conditional randomization" assumption (eq. 9) encodes no-unmeasured-confounding as a total-variation bound over infinitesimal time windows. This formulation is non-standard and differs from the usual sequential randomization or coarsening-at-random conditions. The paper provides an informal interpretation ("approximately, $(T_{\bar{a}}, L_{\bar{a}}) \perp \bar{A}(t+\eta)|\mathcal{F}_t$") but no equivalence proof to standard causal conditions, no examples demonstrating that the bound $\varepsilon(t,\eta)$ exists in practical settings (not even in the Gaussian process simulation), and no discussion of its stringency or testability. Since all identification results hinge on this assumption, this is a meaningful gap.

### Minor

- **The "doubly robust" label in Theorem 3 is somewhat overstated for the current stage of development.** The result shows that at the population level, a particular representation equals the target parameter if either the outcome model or the propensity model is correctly specified. This is the standard identification-level definition of double robustness, but without an estimator or analysis of misspecification in finite samples, calling it "doubly robust" may mislead readers into thinking a practical robust estimator is available. The additional technical condition (eq. 22, interchange of limit and expectation) also lacks concrete sufficient conditions.

- **Presentation is extremely dense.** The paper proceeds rapidly through heavy measure-theoretic notation with limited intuitive exposition. Key objects like $\mathcal{G}_t$ (used in Definitions 1–2 and Theorem 3) are not explicitly defined. The transition between informal motivation ("we loosely have the following decomposition") and formal results (Proposition 1) relies heavily on appendix material, making the main text difficult to parse for non-specialists.

- **Minor inconsistencies in the simulation.** In Step 1, the covariance kernel for $Y_{\bar{a}}$ is specified as $e^{-3|t-s|}$, but in Step 3 the covariance matrix entries use $e^{-|t_i - t_j|}$ (exponent $-1$ instead of $-3$). Since the identification test only involves the mean, this does not affect the numerical result but suggests a lack of care in the simulation description.

### Trivial
- The term "functional longitudinal data" is used before being defined; the definition appears in the introduction but a forward reference would help.

## Nice-to-Haves

- Include a simulation with genuine treatment-confounder feedback, censoring, and mortality to test whether the identification formulas recover causal effects under the conditions the paper is designed for.
- Provide at least a brief discussion of how estimation could approach the identified quantities (e.g., functional regression, sieve methods, or discretization-based approximations).
- Add a concrete example or sufficient condition showing that Assumption 1 holds, ideally in the Gaussian process setting used for the simulation.
- Compare identification results under continuous-time vs. discrete-time approximations at varying grid sizes, to demonstrate when the continuous-time framework offers practical advantages over simply discretizing (which is what the simulation actually implements).

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Claim that the paper's nonparametric density result (Theorem 4) "trivializes" the assumptions.** The harsh critic argued that since any observed-data law can be approximated by laws satisfying Assumptions 1–4, the assumptions lack substantive causal content. However, this is precisely the point of nonparametric identification results in the coarsened data literature (Gill et al., 1997, cited by the paper): showing that the identification assumptions are compatible with essentially any observable data pattern. The paper acknowledges the piecewise-continuous restriction. This is a philosophical concern, not a substantive flaw.

- **Claim that $\mathbb{P}_\mathbb{G}$ is "not fully defined" and Proposition 1 is "opaque."** The paper states that formal decomposition and proofs are in Section A of the appendix. Having technical details in appendices is standard practice in theoretical statistics and does not constitute a structural flaw. The informal intuition is provided in the main text for readability.

- **Demand for real data analysis.** This is a theoretical identification paper. While real data would strengthen the work, demanding it goes beyond the paper's stated scope and community norms for purely theoretical contributions in causal inference methodology.

- **Claim of a "sign error" invalidating the simulation.** The derivation in eqs. (29)–(30) yields $\int_0^1 (t-0.5)\,dt = 0$, and since $\mathbb{E}(Y_{\bar{a}}(t)) = -a(t)$, the expected value is also 0. Any sign error is inconsequential since the answer is symmetric around zero.

## Novel Insights

The paper's use of net convergence (discrete-time approximations converging as mesh → 0) to construct the intervened measure $\mathbb{P}_\mathbb{G}$ from the observed measure $\mathbb{P}$ is an elegant way to handle the continuous-time intervention problem. This approach naturally parallels the discrete-time g-computation formula while providing a rigorous limiting justification. However, this insight also highlights the core tension: the practical implementation inevitably discretizes, and the paper does not yet establish whether the continuous-time machinery yields identification advantages over careful discretization at sufficiently fine grids.

## Suggestions

- Redesign the simulation to include at least one setting with time-varying confounders, censoring, and mortality, so that the g-computation formula does genuine causal adjustment rather than reducing to a sample average.
- Add a paragraph of intuitive explanation before each major definition (especially Assumption 1 and the intervened measure construction), connecting the measure-theoretic objects to their discrete-time analogs.
- Provide one concrete sufficient condition for Assumption 1, even if restrictive, to demonstrate its logical consistency and give readers a starting point.

## Score and Decision

**Calibration:** I compared against papers with similar profiles. The ODE Discovery paper (avg ~6.8) had strong theoretical contributions with limited experiments. The Stable Survival Causal Effects paper (avg ~4.7) had heavy notation and limited validation. The Incremental Causal Effect paper (avg ~5.75) had novel identification results with somewhat limited numerical evaluation. This paper makes a genuine theoretical contribution but with substantially weaker numerical evidence than any of these: the main simulation does not test the core causal identification claim (no confounding, no censoring, no mortality), there is no estimation framework, and Assumption 1 needs significantly more justification. These weaknesses are more severe than those in the comparison papers.

**Score: 4.0** — The theoretical framework is novel and addresses an important gap, but the weak numerical validation (which does not exercise the core causal identification machinery), the absence of estimation guidance, and the insufficient justification of the key identifying assumption significantly limit confidence in the contribution at this stage. The paper reads more like a theoretical development note than a complete methodological contribution.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>