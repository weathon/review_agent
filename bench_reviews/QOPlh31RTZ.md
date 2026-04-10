## Summary

This paper proposes CB-RLHF, a constrained bi-level optimization framework for reinforcement learning from human feedback (RLHF). The method jointly learns a reward function, a cost function, and a policy from trajectory-level human feedback that includes separate labels for optimality preference, constraint-satisfaction preference, and binary constraint violation. A dual formulation handles the non-convex constrained RL lower-level problem, and a Clarke subdifferential approximation addresses non-differentiability in the bi-level gradient computation. The authors provide an \(O(1/\sqrt{K})\) convergence guarantee and demonstrate performance on four MuJoCo tasks with synthetic feedback.

## Strengths
- **Novel problem formulation:** The paper cleanly formalizes the joint learning of reward, cost, and policy from multi-dimensional human feedback as a constrained bi-level optimization problem. This explicitly addresses the identified limitations of constraint inference and policy misalignment within a single framework.
- **Theoretical grounding:** The paper provides non-trivial theoretical analysis, including a convergence rate for the proposed algorithm (Theorem 1) and a suboptimality bound for the learned policy (Theorem 2). The use of the dual formulation and Clarke subdifferential is technically sound.
- **Empirical demonstration:** Experiments on four MuJoCo environments show CB-RLHF can achieve a favorable balance of high cumulative return and low constraint violation rate compared to selected baselines (PEBBLE, Safe RLHF, PARL), providing initial evidence for its practical efficacy.

## Weaknesses

### Major:
- **Critical algorithmic oversight in gradient computation:** The upper-level loss \(F(\phi, \psi, \lambda^*(\phi, \psi))\) (Eq. 5) depends on the policy \(\pi_{\phi,\psi,\lambda^*(\phi,\psi)}\) because the expectation in \(L_r\) and \(L_c\) is taken over the trajectory distribution \(\rho(\tau; \pi_{\phi,\psi,\lambda^*(\phi,\psi)})\). The hypergradient derivations (Eq. 7, 8) and the statement that \(F\) is "continuously differentiable (as shown in Appendix A.7)" appear to consider only the direct dependence through \(\lambda^*\) and the explicit appearance of \(\phi, \psi\) in \(J_{r_\phi}(\tau)\), ignoring the crucial path through the policy-dependent data distribution. If the gradient does not propagate through this sampling dependency, the claimed coupling that resolves misalignment is broken. This is a fundamental issue that undermines the core algorithmic contribution and its theoretical analysis.
- **Insufficient empirical validation of core claims:** The experiments do not convincingly isolate the benefit of the bi-level mechanism.
    1.  **Misalignment claim:** The improvement over PEBBLE/Safe RLHF could stem from the dual-objective setup rather than the on-policy feedback coupling. A necessary ablation is a version of CB-RLHF trained on a fixed, off-policy dataset (breaking the bi-level loop) to demonstrate the necessity of the iterative coupling.
    2.  **Constraint inference claim:** Comparisons with Safe RLHF (which also learns a cost) are mixed. Safe RLHF fails in HalfCheetah, but CB-RLHF does not consistently outperform it elsewhere (e.g., similar violation rates in Walker2d/HalfCheetah, lower return in Hopper). Without analysis of why Safe RLHF fails and a clearer, consistent advantage, the claim that CB-RLHF uniquely solves constraint inference is weakly supported.
- **Unrealistic and unvalidated human feedback model:** The method requires humans to provide three separate, clean labels \((y_r, y_c, z)\) per trajectory pair. This decomposes the very "ambiguous" holistic judgment the paper aims to address. The experiments use synthetic labels generated from ground-truth functions, completely sidestepping the practical challenge of obtaining such fine-grained, separable feedback from real humans. The method's robustness to noisy, correlated, or holistic feedback is untested.

### Minor:
- **Strong and limiting theoretical assumptions:** Theorem 2's suboptimality bound relies on **Assumption 2**, which states the true human reward/cost functions belong to the same parametric class \(\mathcal{F}\) as the learned models. This is a strong and often unrealistic assumption in RLHF, limiting the practical relevance of the guarantee.
- **Lack of computational analysis:** The double-loop algorithm involves inner-loop constrained RL optimization and potential subdifferential approximation. No analysis of wall-clock time, sample complexity, or computational cost relative to baselines is provided, leaving the practical efficiency trade-off unknown.
- **Sparse experimental details:** Critical details are deferred to the appendix: the exact design of ground-truth cost functions, the specific algorithm (e.g., PPO-Lagrangian) for the inner-loop constrained RL update (Algorithm 1, line 5), and the procedure/cost for checking the differentiability conditions in Proposition 1. This hinders reproducibility and a full understanding of the empirical setup.

### Trivial:
- **Narrative selectively highlights favorable results:** The conclusion states CB-RLHF "achieves strong performance," but in Hopper, its return (292) is lower than both Safe RLHF (329) and PARL (304). A balanced presentation is needed.

## Nice-to-Haves
- A small-scale user study with real human feedback on a simple task to test the practicality of the three-label feedback scheme.
- Visualization or quantitative analysis of the learned cost function versus the ground-truth cost in a simple environment to show what constraint is being captured.
- Analysis of how often the subdifferential approximation is triggered in practice and its effect on training stability.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

**Strengths Removed:**
- *"The topic is important."* (Generic strength, applies to any paper in the area).
- *"The paper is well-written."* (Generic strength).
- *"The experiments are extensive."* (Judgment call; four MuJoCo tasks is a standard, not extensive, evaluation).

**Weaknesses Removed:**
- *"The algorithm may not scale to high-dimensional state spaces."* (This is a speculative weakness not grounded in evidence from the paper or experiments. The method uses function approximators, and scalability is not tested or discussed as a limitation).
- *"Missing comparison to more recent RLHF methods."* (The paper justifies its baselines as representatives of key limitations. Demanding an ever-expanding set of comparisons is a generic, one-size-fits-all weakness).
- *Criticisms about the existence or availability of MuJoCo or cited models/tools.* (All cited resources are assumed to exist and be available per the hard rules).
- *Nitpicks about undisclosed hyperparameters or large artifacts.* (These are standard reproducibility details deferred to the appendix/supplement, not core flaws).
- *"The derivation for the classification loss (Eq. 4) is confusing."* (The paper cites Dai et al. (2023) for this formulation, and it is mathematically defined in Section 3 and 4. A request for expanded explanation is a clarity nitpick, not a substantive weakness).

## Suggestions
- The most critical action is to **re-derive the hypergradients (Eq. 7, 8) to explicitly account for the dependence of the trajectory distribution \(\rho(\tau; \pi_{\phi,\psi,\lambda^*(\phi,\psi)})\) on \(\phi\) and \(\psi\)**. The theoretical results (Theorems 1, 2) and the algorithm's validity depend on this correction. If the gradient through the sampling distribution is non-existent or intractable, the paper's core contribution is severely compromised and must be reframed.
- **Add the proposed ablation study:** Compare CB-RLHF against a variant trained on a fixed, initial dataset to isolate the effect of the iterative bi-level feedback loop on mitigating misalignment.
- **Strengthen the constraint inference analysis:** Include a baseline where a standard constrained RL algorithm (e.g., PPO-Lagrangian) uses the *ground-truth* cost function. This would clearly show the performance gap attributable solely to cost function learning.

---

**Overall Assessment:**
The paper presents an elegant and theoretically grounded formulation. However, a **potentially fundamental error in the gradient computation** raises serious doubts about whether the proposed algorithm functions as intended to solve misalignment. Combined with empirical validation that does not convincingly isolate the benefits of the bi-level mechanism and relies on an unrealistic feedback model, the paper currently **does not provide robust support for its core claims**. The contribution is promising but requires major revisions to address these critical issues before it could be considered acceptable.