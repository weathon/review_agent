## Summary

This paper proposes the first framework for establishing global (rather than merely stationary-point) convergence of first-order bilevel optimization methods. It introduces two novel sufficient conditions—joint PL and blockwise PL—for the penalized bilevel objective, proves that the PBGD algorithm achieves $\mathcal{O}(\log^2(1/\epsilon))$ global convergence under these conditions, and verifies the conditions along the optimization trajectory for two bilevel applications (representation learning and data hyper-cleaning) with linear models and least-squares loss. The key technical contribution is a trajectory-dependent induction analysis using matrix perturbation theory to uniformly bound local PL constants that vary across iterations.

## Strengths

- **Novel conceptual framing of the bilevel landscape challenge**: The paper identifies and concretely demonstrates (Example 1, Figure 1) that the nested bilevel objective $F(u)$ can violate the PL condition even when both $f(u,v)$ and $g(u,v)$ jointly satisfy it—a non-trivial negative result that justifies the penalty reformulation over nested approaches. The landscape visualizations (Figures 1–2) provide tangible evidence.

- **First global convergence conditions for bilevel optimization via PL generalization**: Definition 1's joint PL and blockwise PL conditions (eq. 2 and eqs. 3a–3b) extend the standard PL condition to two-variable and block-separable bilevel settings. Theorem 1 cleanly proves $\mathcal{O}(\log^2(1/\epsilon))$ global convergence under either condition, mapping isomorphic bilevel problems → Jacobi update and heterogeneous problems → Gauss-Seidel update in a principled way.

- **Trajectory-dependent verification via technically non-trivial proofs**: Theorems 2 and 3 overcome the difficulty that only *local, non-uniform* PL conditions hold with constants varying along the PBGD trajectory. The induction-based proofs (Lemma 1 for representation learning, Lemma 2 for data hyper-cleaning) leveraging acute matrix perturbation theory to bound $\mu_k \geq \mu$ and $L_k \leq L$ uniformly along the trajectory constitute the most technically substantial contribution.

- **Useful structural result (Observation 2)**: The additivity property that $h_1(Az) + h_2(Bz)$ satisfies PL with constant $\min\{\mu_1\sigma_*(A), \mu_2\sigma_*(B)\}$ is a clean structural fact that enables verifying the bilevel PL conditions and has independent value for landscape analysis.

## Weaknesses

### Fatal
None

### Major

- **Restrictive assumptions narrow theoretical applicability**: The analysis of data hyper-cleaning (Theorem 3, Lemma 2) requires $[X_{\text{trn}}; X_{\text{val}}][X_{\text{trn}}; X_{\text{val}}]^\top$ to be diagonal—a condition rarely satisfied in real-world data where features exhibit correlations. This assumption is used to characterize $\ell_{\text{trn}}^*(u)$ in closed form and to show that $\arg\min_W \ell_\gamma(u,W)$ is independent of $u$ (line 339). Without it, the blockwise PL derivation over $u$ does not carry through. Similarly, the overparameterization assumption ($m \geq \max\{N, N'\}$, $h \geq \max\{m, n\}$) and full row-rank data matrices are standard linear-network idealizations but constrain the scope of the results. The paper acknowledges this is a "pilot study," but the diagonal covariance assumption is a non-trivial restriction that prevents the data hyper-cleaning result from generalizing to realistic feature distributions.

### Minor

- **Penalty parameter scaling limits practical precision**: The analysis sets $\gamma = \mathcal{O}(\epsilon^{-0.5})$ and requires stepsizes $\alpha = \mathcal{O}(\gamma^{-1}) = \mathcal{O}(\epsilon^{0.5})$ (Theorems 1–3). As target accuracy tightens ($\epsilon \to 0$), $\gamma$ grows large and stepsize vanishes, creating a conditioning barrier. The paper does not quantify this trade-off or discuss whether true high-precision global optimality is practically reachable on finite-precision hardware. While this is inherent to penalty reformulations (citing Shen et al., 2023), a condition number analysis of $\nabla^2 L_\gamma$ as $\gamma$ increases would strengthen the practical claims.

- **Experimental scope is strictly synthetic and linear**: Sections 4–5 and experiments (Figures 3–4) are confined to two-layer linear networks with least-squares loss on small synthetic setups. While the paper explicitly motivates applications in RLHF, healthcare, and robotics (lines 33–36, citing Modares 2015; Biyik 2022), no evaluation on non-convex objectives, deeper architectures, classification losses, or real-world data is provided. The experiments verify that what the theory already proves—convergence on the exact setups used in the proofs—rather than stress-testing the framework beyond its assumptions.

### Trivial
None identified.

## Nice-to-Haves

- Plotting empirical PL constants ($\mu_k$) along optimization trajectories for rank-deficient or correlated data would reveal whether the inductive bounds actually hold or degrade, directly addressing whether the theory extends beyond idealized settings.
- A comparative table of complexity results across existing bilevel methods (stationary vs. local vs. global convergence) would help position the $\mathcal{O}(\log^2(1/\epsilon))$ rate in context.
- Relaxing the diagonal covariance assumption in Theorem 3 to block-diagonal or general positive-definite forms, or proving that blockwise PL fails without it, would clarify the necessity of this restriction.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **Overclaiming experimental relevance**: The critic argues that motivational claims about RLHF/healthcare/robotics are "completely unsupported." While technically true that experiments don't cover these domains, the paper explicitly labels itself "A PILOT STUDY" and states (lines 77–81): "it is evident that even analyzing the linear models can capture the essence of the problem structure and exclude other confounding factors." The paper does set appropriate expectations for its scope. Moved to weakness and nice-to-have rather than a major concern.
- **Penalty scaling makes convergence "practically unattainable"**: The critic claims $\gamma \propto \epsilon^{-0.5}$ with $\alpha \leq \mathcal{O}(\gamma^{-1})$ makes convergence "practically unattainable in finite-precision arithmetic." This is overstated—penalty-based methods inherently have this trade-off (as acknowledged by citing Shen et al., 2023), and the paper targets $\epsilon$-solutions, not exact convergence. The concern is real but should be weakened to a practical discussion point rather than a fundamental flaw.
- **Zero train loss assumption**: The critic says $L_{\text{trn}}^*(W_1) = 0$ is "explicitly baked into the analysis" and breaks with realistic representation learning. The paper addresses this in Assumption 2 (line 297) by allowing $(\epsilon_1, \epsilon_2)$ solutions where the lower-level residual is bounded by $\epsilon_2$, not requiring exact zero. The critique partially misreads the assumption.
- **Comparisons with F²SA and BOME are "uninformative"**: The critic says comparing penalty-based methods on the same benign landscape doesn't demonstrate superiority. However, the paper's claim (Figure 4c–d, Section 6) is that "our local PL-based analysis can be extended to other penalty reformulation-based algorithms"—i.e., the PL conditions provide a general characterization of when *any* penalty-based method achieves global convergence, not that PBGD outperforms these baselines. The critic misunderstands the purpose of this experiment.

## Novel Insights

The paper's central contribution—showing that the bilevel landscape's distortion under nested composition ($F(u) = \min_{v \in \mathcal{S}(u)} f(u,v)$) breaks the PL condition even when both $f$ and $g$ are well-behaved, and that penalty reformulation restores it—is a genuinely useful conceptual insight for the bilevel optimization community. The trajectory-dependent induction technique for bounding local PL constants uniformly along the optimization path is technically noteworthy and could be repurposed for other non-convex problems where global PL-like conditions only hold locally. The distinction between isomorphic and heterogeneous bilevel couplings (mapped to joint vs. blockwise PL and Jacobi vs. Gauss-Seidel updates) provides a clean taxonomy that future work can build on. However, the paper's self-described scope as a "pilot study" on linear models means these insights remain to be validated in more expressive, realistic settings.

## Suggestions

- In the data hyper-cleaning section, add a brief discussion (or appendix lemma) clarifying whether the diagonal covariance assumption is necessary for the blockwise PL derivation over $u$, or whether it could be relaxed to structured covariance (e.g., block diagonal, sparse) without breaking the proof.
- Add a short discussion in the conclusion on the $\gamma$–$\alpha$ trade-off and its practical implications for high-precision convergence, referencing the conditioning of $\nabla^2 L_\gamma$.
- Consider a small experiment (even synthetic) with mildly correlated features to empirically probe whether the global convergence behavior degrades gracefully when the diagonal covariance assumption is slightly violated.

## Score and Decision

I calibrated against several anchor papers:
- **Accepted strong-theory bilevel papers**: A4aG3XeIO7 (6,8,6,6 — accepted poster) and cyPMEXdqQ2 (6,6,8,6 — accepted poster) both feature rigorous convergence analysis for bilevel optimization with experiments; they score in the 6–8 range.
- **Rejected narrow-scope theoretical papers**: O0FOVYV4yo (6,6,3,5 — rejected) and AM4AT2MyXQ (3,3,3,5 — rejected) were rejected for limited novelty or overly restricted theoretical scope with no/weak experiments.
- **Borderline PL-theory papers**: x45vUUY4nT (5,3,6,5,6 — rejected) and O2GBkHujdP (3,3,5,6 — rejected) analyzed PL convergence but were rejected due to methodological limitations.
- **Accepted with restrictive assumptions**: UZ893n8FXr (8,8,6,6,6 — accepted) had strong restrictive assumptions but was accepted due to genuine novelty.

The paper under review sits between these clusters. It is stronger than the rejected papers (O0FOVYV4yo, x45vUUY4nT) because it has a genuinely novel framework (first global convergence conditions for bilevel via PL generalization), well-executed trajectory-dependent proofs, and experiments confirming theory. However, it is weaker than accepted anchors (A4aG3XeIO7, UZ893n8FXr) because the diagonal covariance assumption in data hyper-cleaning is a structural limitation, experiments don't test beyond the theory's exact assumptions, and the overall scope is narrow by design ("pilot study"). The paper is honest about its scope, which mitigates some overclaiming concerns. I position it slightly above the borderline rejected PL-theory papers but below the fully accepted strong-theory bilevel papers.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>