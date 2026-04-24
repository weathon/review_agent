Now let me search for calibration papers.Now I have enough information to write the consolidated final review.

---

## Summary

OBCD is a Block Coordinate Descent algorithm for nonsmooth composite optimization on the Stiefel manifold. The key contributions are: (1) a row-wise, feasibility-preserving update scheme using orthogonal transformations of k rows at a time; (2) a novel "block-k stationarity" hierarchy that is provably stronger than standard critical points; (3) ergodic O(1/ε) and KL-based non-ergodic convergence guarantees; and (4) a Breakpoint Searching Method (BSM) that exactly and globally solves the k=2 subproblem for the ℓ₀-norm case. Experiments on ℓ₀-SPCA across 10 datasets show OBCD consistently dominates two operator-splitting baselines from the early 2010s.

---

## Strengths

- **Novel block-k stationarity hierarchy (Theorem 3.6):** Definition 3.5 introduces BS_k-points, and Theorem 3.6 establishes the strict hierarchy: {critical points} ⊇ {BS₂-points} ⊇ {BS_{k+1}-points} ⊇ {global optima}, with reverse inclusions not always holding (part (d)). This provides a meaningfully stronger notion of optimality tied directly to the BCD structure—a genuine conceptual contribution that goes beyond prior work.

- **Exact BSM for k=2 (Section 5):** The reduction of the St(2,2) optimization subproblem to a 1D problem with at most 2r+4 breakpoints (Lemma 5.1, closed-form quartic via Ferrari's method) gives a provably exact global solver at negligible cost. Unlike prior BCD/BMM methods on manifolds that solve subproblems approximately, this is a concrete technical advantage.

- **Inclusion of Jacobi reflection matrices (Lemma 2.5, Remark 2.6, Theorem 3.1):** Theorem 3.1(b) proves any X ∈ St(n,r) is reachable from any X⁰ using both Givens rotations and Jacobi reflections, which is necessary since reflections cannot be represented by rotations alone. Remark 2.6(ii) provides concrete 2×2 examples where using only rotations misses the global optimum—a non-obvious but important technical point absent from prior work (e.g., Shalit & Chechik 2014).

- **Comprehensive convergence theory (Theorems 4.2, 4.6, 4.10, 4.11):** The paper provides the full spectrum: O(1/ε) ergodic convergence to ε-BS_k points (Theorem 4.2), ergodic convergence to ε-critical points (Theorem 4.6), a finite-length property under KL (Theorem 4.10), and last-iterate rates with finite/linear/sublinear convergence depending on the KL exponent σ (Theorem 4.11). This is comprehensive by the standards of the optimization community.

---

## Weaknesses

### Fatal
None.

### Major

- **Omission of the most relevant contemporary baselines.** The paper's own related work (Section 1.1) explicitly identifies Chen et al. (2020), Cheung et al. (2024), and Li et al. (2024) as proximal/subgradient/BMM methods for the nonsmooth+orthogonality problem. None of these appear in Section 6. The only comparisons are against LADMM and SPM from 2012–2014. While it is true that the main experiment is ℓ₀-SPCA (where L1-proximal methods like Chen et al. 2020 do not apply directly), the paper also claims to include ℓ₁-SPCA experiments in Appendix J—yet the main body never compares against Chen et al. (2020) where it would be directly applicable. The claim of "superior performance across various tasks" (Abstract, Section 7) cannot be fully established without these comparisons.

- **Self-referential F_min metric significantly limits the informativeness of Table 1.** F_min is defined as "the smallest objective among all compared methods" (Section 6). OBCD-R achieves 0.00e+00 on 10/10 datasets by construction—because OBCD always achieves the minimum and the gap is measured from that minimum. This tells us OBCD is better than these two specific baselines, not how good the solutions actually are in absolute terms. The convergence curves in Figure 1 are more informative, but Table 1's "perfect" performance is an artifact of the metric design. An external reference value (e.g., from a global solver on small instances, or the best known bound) would strengthen the empirical claim.

- **Bounded subdifferential assumption (Lemma 4.4) is not verified for the main experiment.** Lemma 4.4 and Theorem 4.6 require ‖∂h(X)‖_sp ≤ l_h for all X ∈ St(n,r). Remark 4.5 explicitly states that ℓ₁ satisfies this—but the main experiment uses h(X) = λ‖X‖₀. The paper never addresses whether the ℓ₀ subdifferential satisfies this assumption (the indicator of {0} makes this non-trivial). This creates a gap between the theory and the main experimental application.

### Minor

- **C++/MATLAB asymmetry in wall-clock comparisons.** The BSM is implemented in C++ and integrated into MATLAB (footnote 2), while LADMM and SPM are in pure MATLAB. The paper's claim of fairness ("We expect the overall speed of OBCD will remain similar even after adapting LADMM and SPM to C++") is speculative. For a comparison grounded in wall-clock time, this asymmetry is a real concern.

- **Global subproblem solvability only guaranteed for k=2 with BSM; the general case falls back to standard critical points.** Remark 2.4(b) acknowledges that for general k and h≠0, only a locally stationary solution satisfying K(V̄;·) ≤ K(I_k;·) is guaranteed, and the convergence guarantee reduces to standard critical points. Since the stronger BS_k guarantee is precisely what differentiates OBCD, and it holds only in the k=2 case experimentally, the practical scope of the main theoretical claim should be stated more prominently.

- **Greedy working-set strategies (Appendix D) are never experimentally validated.** These are listed as a "side contribution" in Section 1.2(iii), but Section 6 only tests OBCD-R (random strategy). No ablation against greedy variants is provided anywhere in the main body. Without experimental evidence, this contribution is unsubstantiated.

### Trivial

- The convergence bound for the cyclic strategy in Theorem 4.2(c) includes a C_n^k factor that grows combinatorially with n. The practical implications for scalability (large n, k>2) are not discussed.

---

## Nice-to-Haves

- Compare against Chen et al. (2020) on ℓ₁-SPCA (apparently in Appendix J), and explicitly move that comparison to the main body or provide a clear pointer in the experimental section.
- Provide an ablation: random vs. greedy working-set strategy on at least one dataset to validate the greedy strategies contribution.
- For Table 1, add a small-scale experiment with known global optimum (e.g., synthetic data with ground truth) so the absolute quality of solutions can be verified.
- Clarify whether the bounded subdifferential assumption holds for ℓ₀ or explicitly state this as an open assumption in the main theorem statements.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**From Harsh Critic:**

- *"Equation (10) contains a typographical error writing U_B where U_{B^c} is needed"* — Removed. The paper's Section 2 derivation correctly uses the coordinate-wise separability of h(·) to split h(X^{t+1}) = h(U_B terms) + h(U_{B^c} terms), with the B^c rows constant. This is a parser artifact, not a paper error.

- *"Working-set strategies in Appendix D have no convergence guarantees"* — Partially removed. This is a mild concern (kept as minor), but demanding convergence proofs for heuristic working-set strategies goes beyond what is standard for this type of paper.

- *"Per-epoch convergence cost C_n^k makes cyclic OBCD impractical"* — Downgraded to trivial. The paper correctly includes this factor in the formal bound; this is an expected property of cyclic BCD, not a flaw.

**From Strength Finder:**

- *"Consistent empirical superiority on Table 1"* — Partially kept (convergence curves in Figure 1 are informative) but weakened; the Table 1 "perfect" results are an artifact of metric design, and the baselines are not state-of-the-art.

---

## Novel Insights

The BS_k stationarity hierarchy is a genuinely novel and clean theoretical construct that cleanly connects BCD's block-minimization structure to manifold optimization quality. The insight that Jacobi reflections (det = -1 matrices) are necessary—not just rotations—for completeness of the update scheme and for escaping certain local minima is underappreciated in prior work and represents a concrete, portable observation. The BSM's reduction of a 2D constrained nonsmooth problem to a 1D breakpoint enumeration is technically elegant and could find application beyond this specific paper.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg score | Comparison |
|---|---|---|---|
| BCD for Neural Networks Global Minima | `n2RIkaf1S4.md` | 4.00 (Low) | Also BCD theory paper; rejected for weak motivation, questionable assumptions, thin experiments. Less novel than OBCD. |
| Retraction-free Stiefel + LoRA | `c2OtbtZXFC.md` | 4.75 (Low-Medium) | Stiefel manifold optimization, missing baselines, insufficient novelty; weaker contribution than OBCD. |
| Optimization without retraction on generalized Stiefel manifold | `5mtwoRNzjm.md` | 6.50 (High) | Also Stiefel optimization; had missing baselines but accepted by one reviewer with a 10. More incremental than OBCD's contributions. |
| ADMM for Structured Fractional Minimization | `DcZpQhVpp9.md` | 6.67 (High) | Similar profile: novel nonsmooth optimization algorithm, strong theory, decent experiments. Accepted. |
| Riemannian optimization on generalized Stiefel (CNN) | `6w9qffvXkq.md` | 2.60 (Very Low) | Rejected for negligible contribution; clearly below OBCD's level. |

**Assessment relative to anchors:**

OBCD has substantially stronger theoretical contributions than `c2OtbtZXFC` (4.75) and `n2RIkaf1S4` (4.00): the BS_k hierarchy, BSM, and comprehensive KL convergence analysis are all genuine novelties. Compared to `5mtwoRNzjm` (6.50) and `DcZpQhVpp9` (6.67)—both accepted-quality papers with similar profiles—OBCD has comparable theoretical depth but weaker experimental evaluation. The missing contemporary baselines and self-referential metric are real gaps that prevent a strong-accept. The L0 assumption gap is also a real mismatch between theory and experiment.

Balancing the clear theoretical merits (BS_k theory, BSM, comprehensive convergence) against the limited and partially flawed experimental section, the paper sits slightly below the anchor cluster centered at ~6.5. A score of **5.5** reflects: genuine algorithmic and theoretical contributions above borderline, but an experimental section that is not up to the standard of the theoretical claims.

**Decision: Borderline Reject** — The theory is meritorious and publishable, but the paper needs (1) at least one comparison against a method from Chen et al. (2020) or Cheung et al. (2024) and (2) clarification of the L0 assumption gap before it would be ready.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>