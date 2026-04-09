=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
##Summary

The paper proposes Gradient Order Combination (GOC), a new optimization method for unconstrained convex quadratic problems. By analyzing the reciprocal step-length parameter $r_k = g_k^T A g_k / (2g_k^T g_k)$, the authors interpret steepest descent as a "first-order" method and CBB as a "second-order" method, then extend this framework to construct "third-order" (and higher) updates using combinations of gradient, $Ag$, and $A^2g$ terms, approximated via finite differences of gradients.

## Strengths

- **Geometric/eigenvalue interpretation of SD and CBB:** The paper provides a coherent lens for understanding how the parameter $r_k$ oscillates between large and small eigenvalue regions, and how CBB's repeated-step structure accelerates convergence over SD in ill-conditioned settings. The analysis in Section 2 connecting the symmetry structure of CBB to the $Ag_0$ direction is an insightful geometric observation.
- **Hessian-free formulation for higher-order information:** The method approximates $Ag$ and $A^2g$ via finite differences of gradients (Algorithm 1, lines computing $Ag_k = (g_k - g_k^1)/d$ and $A^2g_k = (g_k - g_k^2)/d^2$), avoiding explicit Hessian storage. This is a practical design choice for high-dimensional problems where forming $A$ is infeasible.

## Weaknesses

### Major:

- **The efficiency claim is illusory when measured by true computational cost.** The paper's central claim is that GOC offers "faster convergence rates" and is "more efficient." However, each GOC iteration requires 3 gradient evaluations (computing $g_k$, $g_k^1$, $g_k^2$ in Algorithm 1), whereas BB and CBB require only 1 per iteration. When accounting for this, the reported results actually show GOC is *less* efficient: in Figure 3a, GOC uses ≈1864 × 3 = 5592 gradient evaluations vs. CBB's 3194, and in Figure 3b, GOC uses ≈2163 × 3 = 6489 vs. CBB's 3515. The paper never reports total gradient evaluations or wall-clock time, making the efficiency claim misleading. This undermines the core contribution.

- **No formal convergence proof for the GOC method.** The paper provides descriptive analysis of $r_k$ dynamics (Section 3) and geometric intuition, but no convergence theorem with rate guarantees for the proposed algorithm. For a theory-oriented optimization paper at ICLR, the absence of a rigorous convergence result (even for the quadratic case) is a significant gap. The analysis in Section 3 remains qualitative ("$r$ value will seesaw between larger eigenvalue area and smaller eigenvalue area generally") rather than providing quantitative bounds.

- **Non-standard "order" terminology conflicts with established optimization taxonomy.** The paper labels SD as "first-order" and CBB as "second-order," but in standard optimization, "order" refers to the highest derivative information used (first-order = gradients, second-order = Hessians). CBB is a first-order method. Redefining "order" to mean the polynomial degree applied to the gradient is confusing and risks misleading readers about computational properties. The paper does not adequately motivate or justify this departure from standard terminology.

- **Missing engagement with polynomial acceleration and Krylov subspace literature.** The GOC update (Eq. 24) constructs a cubic polynomial of $A$ applied to $g$, which is precisely the structure exploited by Chebyshev semi-iterative methods and Conjugate Gradient (CG). CG is the optimal Krylov subspace method for quadratic problems and converges in at most $n$ iterations. The paper does not acknowledge, compare against, or differentiate GOC from these well-established methods, making the novelty claim difficult to assess. Notably, CG also avoids explicit Hessian storage and is guaranteed optimal for this problem class.

### Minor:

- **Finite difference step size $d$ is unanalyzed.** The approximation $Ag_k \approx (g_k - g_k^1)/d$ is numerically sensitive: too-small $d$ amplifies floating-point noise, too-large $d$ yields inaccurate curvature estimates. The compounding error in $A^2g$ (which squares the approximation error) is particularly concerning. The paper provides no guidance, sensitivity analysis, or adaptive strategy for choosing $d$.

- **Experimental scope is extremely narrow.** Only a single synthetic quadratic problem with $n = 100{,}000$ and two initial condition settings (Figure 3a, 3b) are tested. No variation in condition number, dimension, or problem structure is explored. For the specific domain of quadratic optimization, the absence of CG as a baseline is notable.

### Trivial:

- Some notation is introduced without clear definition (e.g., $C_m^k$ in Eq. 22 appears to denote binomial coefficients but is not explicitly stated).

## Nice-to-Haves

- Evaluation on non-convex or stochastic objectives (e.g., neural network training) to demonstrate relevance to the ICLR community, even though the paper's current scope is quadratic optimization.
- Comparison with Conjugate Gradient on the quadratic problems, since CG is the standard optimal solver in this setting.
- A cost-adjusted convergence plot (x-axis = total gradient evaluations) to honestly represent computational efficiency.
- Adaptive or theoretically motivated selection of the finite difference step $d$.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Title typo "GRAD## ENTS"**: This is a parser artifact per the review instructions, not a paper error.
- **Inconsistent citation styles ("YH (2003)", "Z (2015)")**: Formatting nitpick; the references section contains identifiable entries (e.g., Dai YH 2003, Kalousek Z 2015).
- **Specific spelling/grammar error listings**: While the paper does have writing quality issues, enumerating individual typos ("recipprocal," "quisi-Newton") is a style nitpick.
- **Demand for comparison with Adam/AdamW on ML tasks**: While ICLR relevance is a concern, the paper explicitly targets convex quadratic optimization; demanding modern adaptive ML optimizers is scope creep. The more appropriate missing baseline is CG.
- **Demand for non-convex convergence theory**: The paper is explicitly scoped to convex quadratic problems. The limitation this creates for ICLR relevance is noted above, but demanding the paper solve non-convex problems is scope creep.
- **Stochastic extension demand**: Same reasoning—the paper does not claim to handle stochastic gradients.
- **Claim that the paper's "Hessian-free" label is misleading**: The method genuinely does not form or store the Hessian matrix; it approximates Hessian-vector products via finite differences. This is consistent with how "Hessian-free" is used in the literature (e.g., Martens 2010).

## Novel Insights

The eigenvalue-oscillation perspective on $r_k$ — that SD stabilizes $r_k$ near two fixed values while CBB allows wider oscillation, and that this oscillation between large and small eigenvalue regions is the mechanism for acceleration — is an intuitive and potentially pedagogically useful way to understand why methods like CBB outperform SD on ill-conditioned problems. However, this insight does not appear to yield a method that is practically superior to existing approaches when computational cost is honestly measured.

## Suggestions

1. **Re-evaluate using total gradient evaluations as the cost metric.** Re-plot Figure 3 with the x-axis scaled by gradient evaluations per iteration. If GOC is indeed slower on this basis, the framing of the paper must change from "more efficient" to a theoretical contribution about the order structure.
2. **Add CG as a baseline.** For the convex quadratic setting, CG is the gold standard. Comparing against it will clarify whether GOC offers any practical advantage over the optimal method for this problem class.
3. **Provide a formal convergence theorem** for the GOC method on convex quadratics, even if only for the exact-arithmetic case with exact Hessian-vector products (before finite-difference approximation).
4. **Consider reducing per-iteration cost.** If the 3 gradient evaluations per step could be reduced (e.g., via reusing gradients across iterations or automatic differentiation for Hessian-vector products), the method might become practically competitive. This would be a more compelling contribution.
5. **Clarify the relationship to Krylov/polynomial methods** in a related work or discussion section, explaining how GOC differs from Chebyshev acceleration or CG.

---

**Axis assessments:**

- **Novelty:** Low. The polynomial filtering of the gradient is a well-studied concept (Krylov methods, Chebyshev acceleration). The specific "order" framing and the finite-difference implementation have some novelty, but the core idea is not new, and no comparison with prior polynomial methods is provided.
- **Technical soundness:** Moderate concerns. The quadratic analysis is internally consistent but lacks a convergence proof. The efficiency claim is undermined by the per-iteration cost analysis. The finite-difference stability question is unaddressed.
- **Empirical support:** Weak. Only synthetic quadratics are tested, the comparison metric (iterations) is misleading, and critical baselines (CG) are absent.
- **Significance:** Low for ICLR. No demonstration of relevance to learning tasks, and for the stated quadratic domain, the method appears computationally worse than existing approaches when honestly measured.
- **Clarity:** Below average. Non-standard terminology, qualitative rather than quantitative analysis, and writing issues impede full understanding.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject
