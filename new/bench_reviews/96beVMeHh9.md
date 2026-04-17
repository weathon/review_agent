Now I have a thorough understanding of the paper and the reviewers' claims. Let me write the final review.

## Summary

This paper proposes a nonparametric causal identification framework for functional longitudinal data—settings where outcomes, treatments, and confounders are continuously measured over time, producing infinite-dimensional stochastic processes. Using measure theory, stochastic process theory, and net convergence, the authors generalize classical g-computation, inverse probability weighting (IPW), and doubly robust identification formulas to this continuous-time setting, accommodating time-varying outcomes subject to mortality and censoring. A density result (Theorem 4) is offered to argue that the framework is "almost nonparametric," and a Monte Carlo simulation examines the g-computation formula.

## Strengths

- **Addresses an important and genuinely under-developed problem.** Causal inference for functional longitudinal data with continuous-time treatment–confounder feedback is a real gap. The motivating examples (MIMIC-IV, CGM) are well-chosen and the problem is practically significant.

- **Mathematically ambitious and formally rigorous.** The formalization of interventions via limiting discretizations (Proposition 1) and the extension of g-computation/IPW/DR to infinite-dimensional path-space settings using martingale and Radon–Nikodym machinery is technically non-trivial. The measure-theoretic foundation is appropriate for the setting.

- **Complete generalization of the three classical formulas.** Extending all three identification strategies—g-computation, IPW, and doubly robust—to the functional longitudinal setting provides a comprehensive identification toolkit, paralleling the discrete-time theory.

- **The density result (Theorem 4) is conceptually interesting.** Showing that the set of observed laws induced by some full-data model satisfying the assumptions is dense in all observed-data laws (for piecewise-continuous path spaces) connects this work to the CAR literature and demonstrates that the assumption class is not trivially restrictive.

## Weaknesses

### Major:

- **The simulation is trivially simple and does not test the core framework challenges.** The main experiment sets T = C = ∞ (no mortality, no censoring), has no confounding process beyond the outcome itself, and draws treatment directly from the target regime G—completely eliminating treatment–confounder feedback, the defining challenge motivating the paper. The g-computation formula then reduces to a simple sample average of a Gaussian process integral, and the simulation merely verifies Monte Carlo convergence (law of large numbers + Riemann sum convergence). While a "more complicated scenario" is referenced in the appendix, the main text presentation validates nothing about the identification framework under the conditions it is designed to handle. This is a significant evidential gap.

- **The "nonparametric" framing is overstated.** The section title "No Restrictions on the Observed Data Distribution" and the abstract's claim of a "nonparametric framework" are misleading. Theorem 4 establishes that the model class M is *dense* in P (under TV norm)—not that every observed law belongs to M. Denseness means nearby compatible models exist, but for a single observed law, there is no guarantee it is itself in M. The paper partially acknowledges this ("Technically, we have not achieved full nonparametric paradigm"), but the surrounding rhetoric ("no restrictions") directly contradicts this concession. Assumptions 1–4 do impose structural constraints; Theorem 4 shows these constraints are topologically mild, not absent.

- **The identification formulas remain at an abstract level without connection to estimable quantities.** In discrete-time g-computation, the identification formula decomposes into iterated conditional expectations of L_t and A_t given the past—quantities directly estimable from data. Here, Theorem 1 expresses the target as H_G(0−) = E_G[ν(X,Ȳ)|G_0−], an expectation under the intervened law P_G. While Proposition 1 establishes P_G as the limit of discretized observed-data interventions, the paper never shows what this limit looks like concretely in terms of observable conditional distributions or hazard processes. The IPW formula (Theorem 2) involves the Radon–Nikodym derivative dP_G/dP, and Theorem 3's doubly robust formula involves the limit Ξ(H,Q) of complex stochastic integrals. Without a bridge to conditional distributions or intensity processes that practitioners could model, the formulas are formal identification results but not usable identification recipes.

- **Assumption 1 (full conditional randomization) is formulated at the level of counterfactual path-space measures, making it opaque and effectively unverifiable.** The assumption bounds the TV distance between P(dT_ā dL_ā | Ā(t+η), F_t) and P(dT_ā dL_ā | F_t) across all treatment paths. These are distributions of *all future counterfactual trajectories* conditional on treatment in a small interval—quantities that are never observed. The paper provides intuition ("approximately, (T_ā, L_ā) ⊥ Ā(t+η)|F_t"), but does not connect this to more standard, modelable conditions (e.g., treatment intensity processes depending on observed history, as in Lok/Røysland/Rytgaard). Since this assumption is the core device enabling Proposition 1 and the entire identification chain, its lack of operational interpretability is a real limitation.

### Minor:

- **No comparison with discrete-time baselines.** Even at the identification level, demonstrating how the continuous-time formulas reduce to or improve upon discrete-time g-computation applied to discretized versions of the same process would clarify the framework's advantage over the natural practitioner alternative.

- **The positivity assumption (Assumption 4: P_G ≪ P) is extremely demanding for infinite-dimensional treatment paths** and receives insufficient discussion. For stochastic regimes G on function space, absolute continuity requires that every continuous treatment trajectory under G has positive probability under the observed distribution—essentially impossible with infinite-dimensional paths without strong parametric restrictions. The paper acknowledges this briefly in Section 5 but does not explore implications within the identification framework itself.

- **Heavy self-citation pattern.** The paper's primary related-work comparison and extensions rely on Ying (2024a,b,c), and the incremental novelty over Ying (2024a) (which already developed the core limiting-intervention approach for single endpoints) could be more sharply delineated.

### Trivial

- Notation inconsistencies: $\tilde A$ and $\tilde L$ appear in Section 3.4 without earlier introduction; $\mathcal G_t$ is used before formal definition.

## Nice-to-Haves

- Apply the framework to a real dataset (e.g., MIMIC-IV) as a proof-of-concept, even just showing how observed data maps to the notation and what the identified estimand would represent.
- Provide any preliminary estimation strategy or discussion of computational pathways for $H_{\mathbb G}(t)$ and $Q_{\mathbb G}(t)$ under parametric or semiparametric working models.
- Add a notation table given the heavy measure-theoretic notation.
- Evaluate the IPW and doubly robust formulas numerically, even with oracle estimators, to demonstrate the doubly robust property.

## Novel Insights

The paper's identification framework, while formally complete, highlights a fundamental tension in extending causal inference to functional longitudinal data: the "no unmeasured confounding" assumption becomes substantially harder to formulate and verify in continuous time, and the natural measure-theoretic formulation (Assumption 1) operates at a level of abstraction that removes it from empirical scrutiny. This suggests that practical progress in this area may require either (a) restricting to specific process classes (e.g., counting processes with predictable intensities) where confounding assumptions can be stated in terms of observable intensities, or (b) developing partial identification bounds rather than point identification under such strong regularity conditions.

## Suggestions

- **Design a simulation with genuine treatment–confounder feedback, mortality, and censoring.** Even a simple continuous-time Markov model with treatment depending on a time-varying confounder that is itself affected by past treatment would demonstrate that the identification formulas recover the correct causal quantity under the conditions the framework is designed to handle.
- **Clarify the "nonparametric" framing** by changing the section title to something like "Near-Nonparametric Framework" and explicitly discussing what the denseness result does and does not guarantee.
- **Connect Assumption 1 to more interpretable conditions.** For example, discuss under what classes of stochastic process models (diffusions, counting processes, etc.) Assumption 1 would hold, and provide worked examples where the TV bound ε(t,η) can be computed.
- **Clarify the relationship to Ying (2024a)** with a specific paragraph delineating what is new (time-varying outcomes, mortality/censoring, nonparametric proof, numerical results) versus what is inherited.

## Calibration

I compared against the following papers:

1. **Twinned Interventional Flows** (Reject, scores 3/3/6, avg ~4): Continuous-time causal inference paper with overclaimed identifiability and weak/unclear experiments. Similar pattern of ambitious theory + weak validation, but this paper has more coherent (if abstract) theory.

2. **Doubly Robust Structure Identification from Temporal Data** (Reject, scores 3/3/6/6/6, avg ~4.8): Theoretical claims for temporal causal discovery with weak/minimal experiments. Similar disconnect between theory and validation.

3. **On the Recoverability of Causal Relations from Temporally Aggregated Data** (Reject, scores 6/6/6/6/5, avg ~5.8): Stronger theoretical contribution with correct (non-overclaimed) results but limited experiments. Better positioned than this paper because its claims are more carefully scoped.

4. **Incremental Causal Effect for Time to Treatment Initialization** (Accept poster, scores 6/6/6/5, avg ~5.75): Clean identification + estimation + simulation + real data. Shows what a more complete contribution looks like in continuous-time causal inference.

5. **Generator Matching** (Accept oral, scores 8/8/8/8): Exceptional theoretical contribution with strong empirical validation—high end anchor.

6. **Red Pill or Blue Pill** (Reject, scores 1/3/5, avg ~3): Trivially incremental contribution with unsurprising results—low end anchor.

This paper sits below the "Recoverability" paper (~5.8) because its claims are more overstated relative to what is proven, and its simulation is even less informative. It sits somewhat above "Twinned Interventional Flows" (~4) because the theoretical framework is more coherent and less contradictory. The core issue is that the identification results, while formally non-trivial, do not bridge to estimable quantities, the primary selling point ("nonparametric") is overclaimed, and the simulation provides no meaningful validation of the framework's handling of the problem it is designed to solve.

## Score and Decision

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>