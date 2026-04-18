## Summary

This paper studies global convergence of bilevel optimization, a problem that has previously only seen results for stationary points or local minima. The authors propose two sufficient conditions—joint PL and blockwise PL—on the penalized bilevel objective $L_\gamma(u,v)$ that guarantee global convergence of the penalty-based bilevel gradient descent (PBGD) algorithm. They then verify these conditions along PBGD's optimization trajectory for two specific applications with linear/bilinear models: representation learning (two-layer linear network with MSE) and data hyper-cleaning (one-layer linear model with sigmoid-weighted MSE), establishing almost-linear convergence to global optima in both cases.

## Strengths

- **Novel and important problem formulation.** Global convergence for bilevel optimization has been essentially unaddressed in the modern gradient-based bilevel literature. The paper correctly identifies that even when both levels satisfy PL conditions, the nested bilevel objective $F(u)$ can be non-PL (Example 1, Figure 1), motivating the penalty reformulation approach. This is a genuine conceptual contribution.

- **Well-structured theoretical framework.** The joint PL and blockwise PL conditions (Definition 1) are natural generalizations of the standard PL condition to the bilevel penalty setting, and their correspondence to Jacobi vs. Gauss–Seidel PBGD updates is well-motivated by the isomorphic vs. heterogeneous structure of the two applications. Theorem 1 provides a clean conditional convergence result under these conditions.

- **Non-trivial trajectory-dependent analysis.** A notable technical contribution is the induction-based proof showing that local PL and smoothness constants remain bounded along PBGD's trajectory (challenge T2), using acute matrix perturbation theory. This overcomes the difficulty that only local, iteration-dependent PL conditions hold even in the linear model cases, and goes beyond simply assuming global PL. The paper is transparent about this: Section 1.3 explicitly states "only local PL and smoothness conditions are satisfied, with constants that vary along the optimization trajectory of PBGD."

- **Useful landscape insight.** The observation that the penalized reformulation $L_\gamma(u,v)$ can have a more benign landscape than the nested formulation $F(u)$ is illustrated clearly (Figures 1–2) and provides a concrete reason to prefer penalty-based approaches for global optimization. Observation 2 on PL additivity under linear composition, while mathematically classical, is clearly stated and well-leveraged.

## Weaknesses

### Fatal

None.

### Major

- **The scope of the global convergence results is very narrow, and the gap between the paper's ambitions and its actual deliverables is significant.** The title positions the paper as "Unlocking Global Optimality in Bilevel Optimization," and the introduction motivates the work with policy-making, energy systems, healthcare, and RLHF. However, the actual results apply only to two highly structured linear/bilinear model problems with restrictive assumptions: overparameterized two-layer linear networks with full-rank data (representation learning) and one-layer models with diagonal Gram matrices (data hyper-cleaning). While the "pilot study" subtitle provides some mitigation, the motivational text does not sufficiently acknowledge this gap. The results represent a first step but are far from "unlocking" global optimality for bilevel optimization at large.

- **The structural assumptions for data hyper-cleaning are extremely restrictive.** Theorem 3 requires $[X_{\text{trn}}; X_{\text{val}}][X_{\text{trn}}; X_{\text{val}}]^\top$ to be diagonal, meaning training and validation feature vectors are mutually orthogonal. This condition almost never holds in practice, and the paper provides no analysis of how results degrade under approximate orthogonality, nor empirical validation beyond exactly orthogonal synthetic data. For representation learning, Assumption 2 (existence of near-optimal full-rank bilevel solutions) is a strong compatibility condition between training and validation objectives whose sufficient conditions (deferred to Appendix F.1) are not discussed intuitively in the main text. These assumptions limit the applicability of the theory to carefully curated settings.

### Minor

- **The "benign landscape" framing suggests algorithm-agnostic geometric conditions, but the verification is algorithm-dependent.** The paper motivates C2 as "benign properties of the penalty reformulation ensure convergence to global optimum," suggesting intrinsic landscape properties. While Theorem 1 provides exactly such a conditional result, the actual application proofs (Theorems 2–3) rely on trajectory-dependent PL constants maintained by PBGD's specific dynamics. This is acknowledged in Section 1.3 (T2) and Section 3.3, but the high-level narrative could more clearly distinguish between the general framework (Theorem 1) and the algorithm-specific trajectory analysis (Theorems 2–3). A reader could overestimate the generality of the landscape claims.

- **The penalty parameter $\gamma = \mathcal{O}(\epsilon^{-0.5})$ creates a practical coupling between accuracy and condition number.** Since the PL constant scales as $\mathcal{O}(\gamma)$ and the stepsizes scale as $\mathcal{O}(\gamma^{-1})$, achieving high accuracy requires large $\gamma$ and correspondingly small stepsizes. The experiments (Figure 3) confirm that larger $\gamma$ yields slower convergence. The paper does not discuss whether $\gamma$ could be annealed or whether there are regimes where the bounds become vacuous.

### Trivial

- The relationship between joint PL and blockwise PL conditions is stated as "cannot imply each other" but no formal proof or separating example is provided.

## Nice-to-Haves

- Experiments on nonlinear models (even small neural networks) for representation learning or data hyper-cleaning, testing whether the global convergence behavior of PBGD persists beyond linear models, would substantially strengthen the paper's practical relevance.
- A sensitivity analysis for the data hyper-cleaning result when the diagonal Gram matrix assumption is approximately (but not exactly) satisfied.
- Comparison with nested/implicit-differentiation-based bilevel methods (AID/ITD) on the same problems to empirically validate the "more benign landscape" narrative of the penalty formulation.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"No experiments on nonlinear models or real datasets"**: While the experiments are indeed limited to linear models, the paper is explicitly scoped as a pilot study on linear models and devotes Section 1.1 to justifying this choice ("even analyzing the linear models can capture the essence of the problem structure"). Requesting realistic neural network experiments goes beyond the paper's stated scope, though it remains a nice-to-have.

- **"No stochastic/mini-batch experiments"**: The paper's theory and analysis are purely deterministic. Stochastic extensions would broaden impact but fall outside the paper's scope and are not standard in this style of optimization convergence analysis.

- **"No comparison with non-penalty bilevel methods"**: The paper's focus is on landscape analysis and global convergence, not on method comparison. The comparison with F²SA and BOME (both penalty-based) serves to validate that the landscape insight extends beyond PBGD specifically. Comparing with AID/ITD methods would test a different claim (landscape quality) and is a reasonable nice-to-have rather than a weakness.

- **"Incremental algorithmic contribution since PBGD is from prior work"**: The paper's contribution is explicitly about landscape analysis and global convergence proofs, not about a new algorithm. The authors acknowledge this in Remark 1. The novelty lies in the conditions and proofs, not the algorithm.

- **"Claim that penalty reformulation always yields better landscapes"**: Reading more carefully, the paper says in C1 that "the constrained formulation is easier to yield a benign landscape"—with "easier," not "always"—and provides the example and Figure 1 to illustrate, not claim universality. The discussion of PL non-additivity (Appendix C.2) further acknowledges limitations.

- **"Assumption 2 is opaque":** The paper defers sufficient conditions to Appendix F.1 but does state in the main text what Assumption 2 means. The sufficient conditions are indeed technical, but this is a minor presentation issue, not a fundamental problem.

## Novel Insights

The key insight that emerges from combining the reviewers' perspectives is the tension between the paper's landscape-level narrative (benign conditions on $L_\gamma$ enabling global convergence) and the reality that the conditions are verified algorithm-dependently along PBGD's trajectory. This suggests a useful distinction for future work: "static" benign landscape conditions that hold globally and guarantee convergence for any reasonable algorithm, versus "dynamic" benign conditions that are maintained only by specific algorithmic dynamics. The paper's Theorem 1 represents the former, while Theorems 2–3 represent the latter. Making this distinction explicit would sharpen both the presentation and the research agenda, as it clarifies what remains to be shown: whether the penalty reformulation has *intrinsic* benign landscape properties for broader problem classes, or whether benignity is predominantly an artifact of PBGD's trajectory.

## Suggestions

- Reframe the title and abstract to better reflect the pilot/specialized nature of the results. For example, "Global Convergence of Bilevel Optimization: A Pilot Study on Linear Models" would set more accurate expectations.
- Add a brief discussion in the main text summarizing the sufficient conditions for Assumption 2 (currently in Appendix F.1), providing intuition for when they hold.
- Include a small experiment or analysis showing how the data hyper-cleaning result degrades as $X_{\text{trn}} X_{\text{trn}}^\top$ moves away from diagonality, even if theoretical guarantees can only be given for the diagonal case.
- Clearly delineate static landscape results (Theorem 1) from dynamic/trajectory-dependent results (Theorems 2–3) in the main text to avoid overinterpretation of the conditions.

## Score and Decision

**Calibration:** I compared against bilevel optimization papers with similar patterns. CvYBvgEUK9 (penalty methods for bilevel, Accept spotlight, scores 5–8) had stronger practical relevance. O0FOVYV4yo (local PL for linear models, Reject, scores 3–6) had similarly narrow scope. v49jqgmGtM (generalized PL conditions, Reject, scores 5–6) had analogous limitation to toy settings. i6EtCiIK4a (Moreau envelope for bilevel, Reject, scores 5–8) also had concerns about reformulation equivalence and strong assumptions. The current paper has genuine novelty (first global convergence results for bilevel gradient methods) and a clean conceptual framework, but its results are restricted to linear models with strong structural assumptions, and the gap between motivational claims and actual scope is substantial. It falls between the stronger bilevel papers (which were borderline accept/spotlight) and the weaker theoretical papers (which were rejected).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>