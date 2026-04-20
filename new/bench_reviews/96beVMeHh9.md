## Summary

This paper proposes a nonparametric causal identification framework for functional longitudinal data—specifically, continuous-time, infinite-dimensional settings with time-varying outcomes subject to mortality and censoring. It generalizes g-computation, inverse probability weighting (IPW), and doubly robust formulas to this setting using a measure-theoretic path-space approach, and validates the g-computation formula via Monte Carlo simulation. The core challenge addressed is handling uncountably infinite treatment-confounder feedbacks, which breaks discrete-time causal identification approaches.

## Strengths

- **Addresses a genuinely challenging and relevant methodological gap.** Functional longitudinal data (e.g., continuous glucose monitoring, ICU vital sign streams) are increasingly common, and the problem of continuous-time treatment-confounder feedback is real and technically thorny. Proposition 1 (Eq. 12) provides a concrete bridge from finite partitions to continuous time via total variation convergence, resolving a key theoretical step for this setting.
- **Generalizes all three classical identification formulas to continuous time.** Theorems 1–3 give explicit g-computation (Eq. 17), IPW (Eq. 20), and doubly robust (Eq. 23) formulas for path-space settings with censoring and mortality. This is a non-trivial extension beyond prior work that was limited to discrete-time or parametric assumptions.
- **Measure-theoretic approach is appropriate.** Using measures on path spaces rather than density-based formulations (Footnote 1) is a sound technical choice for infinite-dimensional processes with potential jumps or irregular paths, where densities are ill-defined.
- **Explicitly separates identification from estimation.** The paper honestly scopes itself to population-level identification, deferring finite-sample estimation and inference to future work (Section 5).

## Weaknesses

### Fatal
None

### Major

- **The simulation does not evaluate causal identification — it only verifies that a Monte Carlo average approximates a known expectation under direct sampling from the target regime.** In Section 4, the authors simulate treatment trajectories $A_i(t)$ directly from the targeted stochastic regime $\mathbb{G}$ and generate outcomes conditionally (Section 4, Step 3, Eq. 27-28). They then compute a sample mean of a discretized integral to approximate the target estimand (Eq. 30). This validates the law of large numbers and Riemann integration, not that counterfactuals under $\mathbb{G}$ can be recovered from an *observational* distribution $\mathbb{P}$ that operates under a different treatment mechanism with time-varying confounding. No observational distribution with confounding is ever simulated or used; no weighting or g-computation adjustment is actually applied to recover the counterfactual from confounded data. The paper states the justification: "inverse probability weighting cannot be directly approximated without estimation or computation" (Section 4, point 1), which is incorrect — population-level IPW $\mathbb{E}[\frac{d\mathbb{P}_{\mathbb{G}}}{d\mathbb{P}}\nu]$ can be simulated when both $\mathbb{P}$ and $\mathbb{G}$ are known. Without a simulation that generates confounded observational data and then applies the identification formula to recover the counterfactual, the paper provides zero empirical evidence that the proposed framework actually solves the identification problem it claims to address. This is a critical gap for a paper that uses simulation to "empirically assess how the identification works."

### Minor

- **The "nonparametric" framing conflates density in TV norm with nonparametric estimation properties.** Theorem 4 (Section 3.4) states that the set of full-data distributions satisfying Assumptions 1–4 is dense in the observed-data distributions under the TV norm, and uses this to claim the framework is "nonparametric." However, in causal inference, "nonparametric" typically means that identification does not restrict nuisance function forms or that estimation can proceed without parametric modeling. Showing density merely states that any observed law can be approximated by some compatible full law — it does not address the ill-posedness, efficiency bounds, or boundedness of the identification functional required for nonparametric estimation in infinite-dimensional spaces. The claim that "achieving this is the best one can hope for" (Section 3.4) is also not justified with respect to what estimation practitioners actually require.

### Trivial
None

## Nice-to-Haves

- Formally connect Assumption 1 and Proposition 1 to established continuous-time causal inference frameworks (e.g., Rytgaard et al.'s intensity-based formulations, martingale orthogonality conditions). This would clarify whether the TV-bound assumption is equivalent to or generalizes existing conditions, and would specify the predictability/integrability conditions needed for the mesh limit to hold.
- Include a diagnostic visualization showing how the bias in the recovered target estimand scales with grid density *and* the degree of time-varying confounding, to reveal whether the identification mechanism works as the partition refines.
- The doubly robust formula's limit condition (Eq. 22) would benefit from additional context on why it holds under the proposed measure convergence, or a brief sketch connecting it to discrete-time DR theory.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- *Criticism that the paper lacks theoretical connections to established continuous-time survival/martingale frameworks like Rytgaard et al. (2022).* The paper does cite Rytgaard et al. and positions its contribution against this work in Section 2. While the technical contrast could be sharper, the paper does make a structural distinction: it uses measure-theoretic path-space convergence rather than intensity-based counting processes. This is a presentational gap, not a fundamental missing comparison. (Weakness 3 from Harsh Critic)
- *Criticism about notation being "overly abstract" or "mathematical language inconsistent between Assumption 1 and Assumption 2."* The paper explicitly addresses this: "Note that how Assumption 2 is given on the intensity process whereas Assumption 1 is on the conditioning event. This is because one does not have intensity process for a general stochastic process" (Section 3.2, paragraph after Eq. 11). This is a deliberate design choice, not a flaw. (Section-by-section notes from Harsh Critic)
- *Criticism that the paper should have included IPW and DR in the simulation alongside g-computation.* The paper explicitly states it evaluates only g-computation because IPW/DR require estimation of Radon-Nikodym derivatives that cannot be empirically evaluated without first fitting these quantities, which is outside the paper's identification-focused scope. Requesting full estimation machinery in an identification paper is scope creep. (Missing Experiments from Harsh Critic)

## Novel Insights

The paper's most novel contribution is the formalization of continuous-time causal identification through a path-space measure-theoretic lens rather than an intensity-based approach. By framing treatment-confounder feedback using total variation convergence of intervened measures (Proposition 1), it provides a potentially more general foundation than counting-process methods, which require restrictive stepwise/jump assumptions. However, this novelty is undercut by a simulation that fails to demonstrate that the framework actually enables identification from confounded observational data — the core claim of the paper. The gap between the theoretical ambition and the empirical validation is the paper's defining tension.

## Suggestions

- Add a simulation where data is generated under an observational distribution $\mathbb{P}$ with a distinct treatment assignment mechanism that induces time-varying confounding (e.g., treatment depends on past outcomes), and then apply the discretized g-computation formula to recover the counterfactual estimand. This would directly validate that the identification framework works in the intended setting.
- Clarify in the Theorem 4 discussion that "nonparametric" in this context refers to identification-level flexibility (density of compatible full-data models) rather than the stronger estimation-level nonparametric properties (bounded influence functions, efficiency bounds). This prevents mischaracterization and sets appropriate expectations for downstream estimation work.

## Score and Decision

I compared this paper against several calibration anchors:

- **High-scoring (7-8):** `3cuJwmPxXj.md` (8/8/8/8) and `2efNHgYRvM.md` (8/8/8) — these theoretical identification papers had both rigorous proofs *and* meaningful empirical evaluation or strong technical depth.
- **Borderline (5-6):** `xbUlKe1iE8.md` (3, 3, 6, 6, 6) — theoretically grounded causal identification with limited empirical evaluation. Our paper shares this profile but has an even weaker simulation that does not test the actual identification claim.
- **Low-scoring (3-5):** `or8wkKoBP4.md` (3, 5, 3, 5) — rejected for vague assumptions and no experiments. Our paper is stronger than this because it has a coherent mathematical framework and at least a Monte Carlo simulation.

The paper has a sound theoretical framing and a meaningful conceptual contribution, but the core simulation does not validate what the paper claims to evaluate (identification from observational data). This prevents it from reaching the 6+ range. However, it avoids the fatal mathematical flaws and complete empirical absence of the low-scoring anchors. I position it slightly below the borderline papers due to the severity of the simulation gap relative to the paper's own stated goal ("to empirically assess how the identification works").

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>