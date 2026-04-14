=== CALIBRATION EXAMPLE 27 ===

# Final Consolidated Review
## Summary

OBCD proposes a Block Coordinate Descent method for nonsmooth composite optimization (with both smooth $f$ and nonsmooth $h$) under orthogonality constraints (Stiefel manifold). In each iteration, $k \geq 2$ rows of the solution matrix are updated via a constraint-preserving scheme $X^{t+1}(B,:) \leftarrow V X^t(B,:)$ with $V \in \text{St}(k,k)$, ensuring feasibility throughout without retraction. The paper introduces Block-$k$ Stationary (BS$_k$) points, formally proven to be stronger than standard critical points, establishes ergodic $\mathcal{O}(1/\varepsilon)$ convergence plus KL-based non-ergodic rates, and provides a Breakpoint Searching Method (BSM) for exactly solving the $k=2$ subproblem even for the nonsmooth $\ell_0$ case.

---

## Strengths

- **Novel constraint-preserving row-wise update**: The scheme $X^+ = X + U_B(V - I_k)U_B^\top X$ with $V \in \text{St}(k,k)$ exactly preserves Stiefel feasibility at every iterate, requiring no retraction or polar decomposition. This is structurally distinct from existing column-wise BCD (Shalit & Chechik 2014) and Riemannian BMM methods that require a retraction step after each update. Lemma 2.1 and Theorem 3.1 together establish that every point in $\text{St}(n,r)$ is reachable from any starting point via this scheme, a non-trivial completeness property.

- **BS$_k$ stationarity hierarchy with separation examples**: Theorem 3.6 formally establishes $\{\text{critical points}\} \supseteq \{\text{BS}_2\text{-points}\} \supseteq \{\text{BS}_{k+1}\text{-points}\} \supseteq \{\text{global optima}\}$, with part (d) demonstrating that all inclusions are strict via concrete examples. This is a substantive theoretical contribution, not merely definitional—it identifies a new class of stronger fixed points that existing Riemannian subgradient/proximal methods cannot claim to converge to.

- **Exact global solver for the nonsmooth $k=2$ subproblem (BSM)**: Using the parameterization of $\text{St}(2,2)$ via Givens rotations and Jacobi reflections (Lemma 2.5), the subproblem reduces to a 1D problem. For $h = \lambda\|\cdot\|_0$, BSM identifies all $2r + 4$ breakpoints (including quartic roots via Ferrari's method), guaranteeing a globally optimal solution. This is non-trivial: related BMM/BCD methods explicitly resort to approximate subproblem solvers, and the inclusion of Jacobi reflections (not just rotations) is shown to be necessary by explicit counterexample (Remark 2.6).

- **Comprehensive convergence theory**: The paper provides both ergodic ($\mathcal{O}(1/\varepsilon)$) and, under KL, non-ergodic last-iterate rates (finite steps for $\sigma=0$, linear for $\sigma \in (0,\tfrac{1}{2}]$, sublinear otherwise), in both deterministic (cyclic) and stochastic (random) working set regimes. This matches the state of the art for nonconvex optimization on manifolds.

---

## Weaknesses

### Fatal
None.

### Major

- **Baseline coverage critically insufficient**: The related work explicitly names Riemannian BCD/BMM methods (Li et al. 2023, 2024; Breloy et al. 2021; Gutman & Ho-Nguyen 2023; Cheung et al. 2024) as the most closely related algorithmic family, yet none appear in the experiments. Only LADMM and SPM (operator splitting methods) are compared. Since the paper's primary claim over the related work is exactly that it avoids the approximate subproblem solving and retraction steps required by Riemannian BMM/BCD, the absence of these competitors makes the empirical claims unverifiable. Without these comparisons, the performance advantage cannot be attributed to the proposed mechanism.

- **Abstract-algorithm inconsistency on global optimality**: The abstract states the method "globally solves a small nonsmooth optimization problem," but Algorithm 1 (Step S3) explicitly states "global **or local** optimal solution," and Remark 2.4(b) acknowledges that for general $k$ and $h(\cdot) \neq 0$, "strong optimality may be compromised" and only critical point convergence is guaranteed. The BS$_k$ stationarity guarantee—the paper's main theoretical selling point—is thus fully realized only when the subproblem is solved globally, which in practice occurs only for $k=2$ (via BSM). This scope is narrow and should be stated transparently in the abstract and introduction, not buried in a remark.

### Minor

- **Missing ablation on block size $k$**: All experiments use $k=2$ (necessitated by BSM). The BS$_k$ hierarchy predicts that larger $k$ yields stronger stationary points. No experiment validates whether $k > 2$ with approximate subproblem solutions actually produces better or worse solutions empirically. This ablation is necessary to understand the practical value of general $k$.

- **No evaluation of greedy working set strategies**: The paper lists greedy strategies for working set selection as a contribution (Section 1.2, item iii) and devotes Appendix D to their analysis. Yet only OBCD-R (random strategy) is evaluated in the experiments. The performance implications of the greedy variants—and whether they are worth the added complexity—are entirely empirical questions that the paper leaves unanswered.

- **OBCD not tested with random initialization**: In Table 1, OBCD-R is initialized only from the identity matrix, while baselines include both identity and random initializations. To substantiate the claim that OBCD is robust to initialization and escapes poor local minima, OBCD-R with random initialization should also be reported.

- **Notation error in the $h(\cdot)$ decomposition**: In Equation (10), the paper writes $h(\mathcal{X}_B^t(V)) = h(U_B U_B^\top X^t + U_B V U_B^\top X^t)$, but the correct decomposition of $\mathcal{X}_B^t(V) = U_B V U_B^\top X^t + U_{B^c} U_{B^c}^\top X^t$ yields $h(\mathcal{X}_B^t(V)) = h(V U_B^\top X^t) + h(U_{B^c}^\top X^t)$. The first factor $U_B$ in the first term should be $U_{B^c}$, and the constant $\tilde{c}$ as stated is missing the term $h(X^t(B^c,:))$. The algorithm and descent inequality are unaffected (all constants can be absorbed), but the derivation as written is incorrect and should be fixed.

### Tiny

- **Numerical stability of Ferrari's method**: BSM for $\ell_0$ uses Ferrari's method to find all real roots of a quartic. No discussion of numerical conditioning or treatment of near-zero leading coefficients is provided; this can matter in floating-point arithmetic.

- **Parameter sensitivity of $\alpha = 10^{-5}$**: No ablation on $\alpha$ is provided, leaving it unclear how sensitive convergence behavior is to this regularization.

- **Convergence guarantees under Lemma 4.4's bounded-subgradient assumption not verified for $\ell_0$**: Remark 4.5 notes $\ell_1$ satisfies this assumption, but the primary experimental case ($\ell_0$) is left unaddressed.

---

## Nice-to-Haves

- **Experiment on a deep learning task with orthogonality constraints**: The introduction motivates DNNs (Cogswell et al., Bansal et al., Huang & Gao), but no DNN experiment is conducted. Even a small-scale experiment (e.g., orthogonal weight regularization in an RNN or CNN) would meaningfully broaden the demonstrated applicability for the ICLR audience.

- **Visualization of BS$_k$ measure vs. Riemannian gradient norm**: Plotting the proposed $\epsilon$-BS$_k$ measure (Definition 4.1) against the standard Riemannian gradient norm across methods would visually confirm whether competing methods converge to strictly weaker stationary points, providing empirical evidence for the theoretical hierarchy.

- **Release of C++ BSM code**: The breakpoint searcher is central to the reproducibility claim and currently a black box in the submission.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Insufficient baselines—comparison is circular because F_min = min over compared methods"** *(Harsh critic)*: The self-referential nature of the F_min metric is a presentation concern, not a methodological flaw. The real issue is the weak baseline set, which is already captured above as a Major weakness. Framing OBCD "winning by construction" mischaracterizes the metric.

- **"Implementation unfairness: C++ vs MATLAB"** *(Harsh critic)*: The paper directly addresses this in footnote 2, noting that competing BLAS/LAPACK routines for matrix multiplication and SVD are also highly optimized compiled code. The BSM involves element-wise loops that are inherently slow in MATLAB regardless of optimization level. The paper also provides iteration-count comparisons (convergence curves). This criticism is not substantive given the paper's explicit justification.

- **"Single run, no error bars"** *(Harsh critic)*: For large-scale benchmarks in Stiefel manifold optimization, single-run evaluation with time-budgeted comparison is the community norm, consistent with prior work (Wen & Yin 2013; Chen et al. 2020). Requiring multi-run statistics imposes a non-standard expectation for this setting.

- **"Missing related works"**: Per meta-review policy, no criticisms about missing related works are retained.

- **"Dense tensor product notation in Lemma 2.2 inaccessible to ML audience"** *(Review 2)*: This is a pure style nitpick; the notation is mathematically standard and the paper's audience for an optimization submission is expected to be comfortable with it.

- **"Comparison with BMM/BCD methods is unfair if subproblems are solved approximately there"** *(implicit in related work)*: Since approximate subproblem solutions in BMM methods would only hurt those baselines (i.e., the asymmetry favors the baselines, not OBCD), such comparisons would strengthen, not weaken, the paper's claims. This is not a fairness concern.

---

## Novel Insights

The most genuinely novel theoretical contribution—beyond what any of the reviews highlight in isolation—is the combination of **Theorem 3.1 (basis representation)** with **Definition 3.5 (BS$_k$ points)**: the same row-wise update scheme that achieves computational feasibility also enables an optimality notion that is provably strictly stronger than critical points used by every existing competitor. Specifically, Theorem 3.1 shows that the proposed update can reach *any* point on the Stiefel manifold from any starting point using $k=2$ updates, making the update scheme both complete and globally expressive. This connects the algorithmic mechanism directly to why BS$_k$ points are non-trivially stronger: the BS$_2$ condition checks that no improvement is possible in any 2-row subspace, and since any orthogonal transformation can be decomposed into such moves, a BS$_2$ point is genuinely hard to escape. This structural argument—feasibility mechanism ↔ optimality strength—is underemphasized in the paper and deserves to be foregrounded.

---

## Suggestions

1. **Add Riemannian BCD/BMM baselines** (Li et al. 2024, Cheung et al. 2024, or equivalent): This is the single highest-priority revision. Without it, the empirical section cannot support the paper's claims.

2. **Fix the abstract**: Replace "globally solving a small nonsmooth optimization problem" with language that accurately reflects that global solving is achieved for $k=2$ via BSM, and that for general $k$ and $h(\cdot)$, local solutions are used with convergence to critical points.

3. **Add OBCD-R with random initialization** to Table 1 to match the baseline evaluation protocol.

4. **Report an ablation over $k \in \{2, 5, 10\}$**, even on a synthetic problem, to empirically validate the BS$_k$ hierarchy's practical implications.

5. **Fix the notation in Equation (10)**: Correct the $h$-decomposition formula ($U_B$ → $U_{B^c}$) and include $h(X^t(B^c,:))$ in $\tilde{c}$ explicitly.

6. **Report numerical constraint violation $\|X^\top X - I\|_F$** over iterations to empirically substantiate the "feasible method" claim relative to projection-based alternatives.

---

**Overall evaluation axis summary:**

- *Novelty*: High. Row-wise BCD on the Stiefel manifold with exact nonsmooth subproblem solving and a new, formally stronger stationarity notion is original.
- *Technical soundness*: Moderate–high. The convergence theory is rigorous; the notation issue in Eq. (10) and the global/local discrepancy need fixing but do not undermine the proofs.
- *Empirical support*: Weak. The missing Riemannian BCD/BMM baselines are a serious gap; the narrow experimental scope (ℓ₀-SPCA only in main text, two baselines, single run, one initialization for OBCD) is insufficient for the claims made.
- *Significance*: Moderate. The theoretical framework is clean and broadly applicable; significance is currently limited by the narrow empirical validation.
- *Clarity*: Moderate. The mathematical exposition is well-organized but contains the notation error noted above, and the scope of the global-solving guarantee is understated.

# Actual Human Scores
Individual reviewer scores: [6.0, 3.0, 3.0]
Average score: 4.0
Binary outcome: Reject
