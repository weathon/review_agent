=== CALIBRATION EXAMPLE 81 ===

# Harsh Critic Review
Now I have sufficient material to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate and the abstract is generally well-written, but it contains a claim that requires scrutiny: "matching the theoretical bounds established by the complex algorithm of Adil-Kyng-Peng-Sachdeva." Adil et al. (2019a; 2024) achieve O(p² n^{(3p−2)/(p−2)} log(n/ε)) total linear system solves. Theorem 1.2 states O(p² log n · log(n/ε)) subproblems each requiring O(n^{(3p−2)/(p−2)}) linear system solves, giving a total of O(p² n^{(3p−2)/(p−2)} **log n** · log(n/ε)). This is a log n factor **worse** than the stated target. The proof of Theorem 1.2 (Appendix E) invokes κ = O(log n) (in the regime where p ≤ log n/(log log n − 1)) or κ = p − 2 otherwise, and the outer loop count is O(p² κ log(n/ε)), not O(p² log(n/ε)). The authors should reconcile this discrepancy or clarify what "matching" precisely means — is the comparison made for a fixed p regime, or asymptotically in n? This is the most critical technical claim in the paper and deserves more precision.

---

### Introduction & Motivation

The motivation is clear and well-placed: the gap between provably fast but impractical algorithms (Adil et al. 2019a) and efficient but theoretically suboptimal IRLS algorithms (Adil et al. 2019b) is a genuine and important open problem. The historical narrative is well-organized.

**Missing comparison with Jambulapati-Liu-Sidford (2022).** The paper cites this work in the related work section but provides no substantive discussion. Jambulapati et al. achieve O(p^p · d^{(3p−2)/(p−2)} polylog(n/ε)) for *overconstrained* problems (n ≫ d, minimizing ‖Ax − b‖_p with A ∈ ℝ^{n×d}). For large n, small d, their bound can be far superior to n^{(3p−2)/(p−2)}, which can be enormous. The paper's setting is underdetermined (A ∈ ℝ^{d×n}, minimizing ‖x‖_p subject to Ax = b), which is different, but the relationship between these results should be clarified. The introduction gives the impression that this paper achieves the "state of the art" without adequately scoping that claim.

**The problem formulation** is restricted to p ≥ 2 in the main theorems, with 1 < p < 2 handled by a duality reduction in Appendix C that simply invokes Adil et al. (2019a) for the full proof ("We refer the reader to Adil et al. (2019a) for the detailed error analysis"). This means the 1 < p < 2 case is not fully self-contained and the contribution there is essentially a note, not a new result.

---

### Method / Approach (Sections 2–3)

The core idea — maintaining a primal-dual invariant (eq. 1) that links the ratio of energy increase to ℓq-norm increase to M² — is elegant and conceptually novel. The connection to Ene-Vladu (2019) for ℓ₁/ℓ∞ regression is clearly stated, and the extension to general ℓp is non-trivial because the objective lacks coordinate-separable structure.

**Algorithmic presentation (Algorithms 1–4):** The pseudocode is presented clearly, though the γ update in Algorithm 2 (lines for Cases 1, 2, 3) is somewhat dense. The case structure — Case 1 (primal solution from termination condition), Case 2 (averaged primal solutions from slow dual steps), Case 3 (dual infeasibility certificate from fast dual growth) — makes intuitive sense, but the distinction between T_hi and T_lo iterations and how they interact is only fully explained in the appendix.

**The high-precision algorithm (Section 3, Algorithm 3–4):** This section is notably less self-contained. The iterative refinement framework is credited to and heavily relies on Adil et al. (2019a), invoking it as a black-box. The novelty here is the new residual solver (Algorithm 4), which adapts Algorithm 2 to handle a mixed ℓp + ℓ2 objective. The key challenge — that the ℓ2 regularization breaks scale-freeness — is correctly identified and handled, but the presentation jumps quickly to the invariant without spelling out why the initialization to r^{(0)} = 2^{q-1}/n^{1/q} (vs. n^{1/q} in Algorithm 2) is necessary. This is a subtle but important point.

**Complexity of Algorithm 4 (Lemma E.1):** The claimed O(n^{1/(2q+1)}) iteration bound for the residual solver — and how this equals n^{(p−2)/(3p−2)} via q = p/(p−2) — is critical to the O(n^{(3p−2)/(p−2)}) total per subproblem. The derivation goes through Lemmas E.9 and E.11, and while the high-level structure (T_hi + T_lo decomposition) closely mirrors Section 2, the T_lo bound (Lemma E.11) is O(S^{1/2}/ln S · ln n + ln n) with S = n^{2/(2q+1)}, yielding O(n^{1/(2q+1)}) — this arithmetic should be verified carefully since the garbled PDF makes absolute certainty difficult.

**The small-p regime (p ≤ log n/(log log n − 1)):** Algorithm 4 handles this with a single O(1) linear system solve (Lemma E.3), exploiting that q ≥ log n so the ℓq norm of a uniform vector concentrates. This is a clean trick. However, this regime implies very small p (e.g., p ≈ log n/log log n ≈ constant for typical n), and it is unclear how useful this case is in practice.

---

### Experiments & Results (Section 4)

The experimental section has several weaknesses.

**Narrow parameter range tested.** Virtually all experiments in Figures 1–2 use p = 8 or vary p in {3,...,50} and n up to ~2500 (or graph size up to 10000). The theoretically interesting regime for the gap between this algorithm and p-IRLS is for large p and large n. While some evidence is shown for large p (Figures 1d, 1h with p up to 50), the matrix sizes are modest. For Table 1 (real datasets), only p = 8 and ε = 10^{−10} are tested — a single configuration.

**The 1–2.6× speedup claim is modest and imprecise.** The paper says "our algorithm is 1–2.6 times faster than p-IRLS." This is a wide range. For some configurations (e.g., Table 1, "KEGG Metabolic" dataset: 42 vs 50 iterations, 1.7 vs 2.5s), the improvement is ~16% in iterations and ~32% in time. For larger instances or higher p the gap grows, but the overall impression is that the empirical gain, while real, is not dramatic.

**Extremely high precision target (ε = 10^{−10}).** This is an unusual setting for ML regression. In practice, ε = 10^{−4} or 10^{−6} is more than sufficient. It would be valuable to see results at moderate precision (ε = 10^{−4}) to understand whether the theoretical iteration improvement materializes at practically relevant precisions, especially since the low-precision regime (Theorem 1.1 with poly(1/ε) dependence) is *not* evaluated at all.

**CVX comparison is limited.** CVX/SDPT3/SeDuMi is an interior point solver not designed for high precision ℓp regression; using it as a baseline mainly shows that IRLS methods are faster than general-purpose semidefinite solvers at high precision, which is already known from Adil et al. (2019b). The interesting comparison is between this algorithm and p-IRLS. The CVX numbers are missing for large instances precisely because CVX is too slow there.

**No ablation or stress tests.** There is no evaluation of: (a) sensitivity to initialization; (b) behavior at the boundary p = log n/(log log n − 1) where the algorithm switches regimes; (c) performance on ill-conditioned or structured problems; (d) comparison against warm-started baselines.

**Low-precision solver not evaluated.** Theorem 1.1 (poly 1/ε, the genuinely novel low-precision solver) has no experimental evaluation. All experiments use the high-precision Algorithm 3 which relies heavily on the Adil et al. iterative refinement framework. The claim that the Algorithm 2 sub-solver is the key new practical ingredient is not directly demonstrated.

---

### Proofs & Technical Correctness

The proof structure is sound at a high level. The Lemma 2.1 (invariant), 2.2 (Case 1), and 2.3 (Case 3) form a complete correctness argument. The convergence analysis via the T_hi/T_lo decomposition is well-conceived. The lower bound Lemma F.1 (for bounding product lower bounds) is useful and correctly proven using the concavity of log.

**One technical subtlety worth flagging:** In the proof of Lemma 2.3 (Case 3), the final step concludes E(r^{(T)})/||r^{(T)}||_q ≥ M²/(1+ε)² ≥ M²/(1+ε)² from ||r^{(T)}||_q ≥ 1/ε. This is valid when ε is small, but the bound (1−ε) ≥ 1/(1+ε)² requires ε ≤ 1/2 or similar; it should be noted that ε < 1 is assumed.

**The analogous result in Lemma E.8 (Algorithm 4, Case 3)** has r^{(0)} initialized to have ||r^{(0)}||_q = 2^{q-1}/2^q = 1/2 (not 1 as in Algorithm 2), but the stopping condition is ||r^{(T)}||_q ≥ 1 (not 1/ε). This leads to a different telescoping and a constant approximation (M²/q² factor), as needed for the residual solver. The logic is consistent but the difference from Algorithm 2 should be highlighted more clearly in the main text.

---

### Venue Fit & Broader Impact (ICLR Specific)

This is the most serious concern for the submission. The paper is a **theoretical algorithms paper** in the tradition of STOC, SODA, and the FOCS/SODA theory community. The applications to machine learning (semi-supervised learning, graph clustering) are cited as motivation but are not the subject of any experiment. The experiments test abstract random regression problems on MATLAB. ICLR's audience and review criteria center on representation learning, neural networks, generalization, and empirically impactful methods. While ICLR does accept optimization theory papers, they typically have either (a) strong empirical significance to neural network training, or (b) broad conceptual novelty applicable to ML problems. This paper is primarily a complexity improvement for a classical continuous optimization problem.

The paper would be a strong submission to **NeurIPS or ICML** (which have more theory-friendly programs) and potentially excellent at **SODA or STOC** where this line of work (Adil et al., Ene-Vladu, Jambulapati et al.) has been published. At ICLR, it is unclear what the target audience is.

There is no Broader Impact section. Given the purely theoretical nature of the work (faster convex optimization algorithms), broader impact concerns are minimal, but the omission is notable for ICLR norms.

---

### Overall Assessment

This paper makes a genuine and technically sound contribution to the theory of ℓp regression: it presents a new IRLS-based algorithm that, under a primal-dual invariant framework, achieves provably improved iteration complexity over p-IRLS (Adil et al., NeurIPS 2019b) and matches (or nearly matches — see the log n concern above) the best known theoretical algorithm (Adil et al., JACM 2024). The algorithmic ideas, particularly the coordinate-wise multiplicative update derived from an energy-rate invariant without mirror maps or smoothing, are novel and cleanly explained. However, the paper has several substantive issues: (1) the central "matching" claim needs more careful quantification given the potential log n discrepancy in the total iteration count of Theorem 1.2; (2) the experimental evaluation is narrow, testing only p = 8 at ε = 10^{−10} with modest-sized instances, and the low-precision solver (Theorem 1.1) is not empirically evaluated at all; (3) comparison with Jambulapati et al. (2022) is inadequate; and (4) the venue fit for ICLR is poor — this is fundamentally a theory paper that would be more at home at SODA, ICML, or NeurIPS. The contribution stands on its theoretical merits, but the paper as submitted does not meet ICLR's threshold without substantially strengthening the experimental section or the connection to modern ML practice.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes a new Iteratively Reweighted Least Squares (IRLS) algorithmic framework for $\ell_p$ regression that achieves state-of-the-art theoretical iteration complexity while maintaining practical efficiency. By leveraging a primal-dual approach with specific energy invariants, the authors present lightweight solvers that match the theoretical bounds of previous complex algorithms (Adil et al.) and outperform existing practical methods ($p$-IRLS) and interior-point-based solvers (CVX). The contributions span both low-precision (polylogarithmic $\epsilon$) and high-precision (logarithmic $\epsilon$) regimes, effectively bridging the gap between theoretical constructiveness and empirical usability.

### Strengths
1.  **Bridging Theory and Practice:** The primary strength is the demonstration of an algorithm that theoretically matches the best-known iteration complexity (e.g., matching Adil et al. 2024 bounds) but uses a simpler, more practical lightweight scheme compared to the complex subroutines required by previous optimal theoretical algorithms.
2.  **Empirical Performance:** The experimental evaluation is extensive, covering synthetic data (random matrices, graphs) and real-world UCI datasets. The results (Table 1 and Figures 1-2) consistently show significant speedups (1-2.6x faster) and fewer linear system solves compared to the closest prior work ($p$-IRLS) and general-purpose solvers (CVX).
3.  **Primal-Dual Framework:** The algorithmic design using a simple invariant on the dual energy function (Section 2.2) is theoretically elegant. It avoids the need for mirror maps or smoothing regularizers typical in standard IRLS or multiplicative weight updates, allowing for larger update steps.
4.  **High-Precision Extension:** The integration of an iterative refinement framework (Algorithm 3) to handle high-accuracy requirements is a significant addition, ensuring the algorithm is applicable beyond low-precision regimes, which is crucial for many applications.

### Weaknesses
1.  **Scope of $\ell_p$ Regimes:** The paper primarily focuses on $p \ge 2$, reducing the $1 < p < 2$ case via duality (Appendix C). While explained, a unified treatment or analysis specific to the geometry of $p \in (1, 2)$ could provide deeper insights and potentially tighter constants for that specific regime.
2.  **Comparison Baselines:** While comparisons against $p$-IRLS and CVX are rigorous, the landscape of fast $\ell_p$ solvers has evolved. There is limited comparison against other modern, practical optimization libraries or recent flow-based linear system solvers that might compete on the "linear system solve" cost, which dominates the runtime.
3.  **Parameter Sensitivity:** The text mentions that prior theoretical algorithms require complex hyperparameter tuning. However, the sensitivity analysis of the hyperparameters in the proposed algorithm (e.g., the threshold $S$, initialization steps in Algorithm 4) is not explicitly detailed. Users might face difficulties if the proposed parameters require tuning based on matrix condition numbers or dimensions.
4.  **Linear System Complexity:** The runtime bounds rely heavily on the cost of solving linear systems. While the *iteration* complexity is improved, the paper does not fully quantify how the condition number of the weighted systems $A^\top D A$ behaves throughout the iterations, which is critical for the practical stability of the method.

### Novelty & Significance
**Novelty:** The work is novel in its application of a specific primal-dual energy invariant to IRLS for $\ell_p$ regression. While IRLS is well-studied, achieving the theoretical SOTA iteration complexity without the complexity of width-reduced multiplicative weights or complex Newton steps is a meaningful contribution to the optimization literature.

**Significance:** $\ell_p$ regression is a fundamental primitive in robust machine learning, clustering, and statistics. A practical algorithm that guarantees convergence rates previously only available in theory is highly significant. It enables faster deployment of robust regression techniques in high-dimensional settings where CVX is infeasible and previous IRLS methods were either too slow or lacked convergence guarantees.

### Suggestions for Improvement
1.  **Enhance Baseline Comparisons:** Include comparisons with other recent iterative solvers or libraries that handle $\ell_p$ norms (beyond CVX and the specific $p$-IRLS implementation). This would better contextualize the performance gains in the broader ecosystem.
2.  **Clarify Hyperparameters:** Provide a discussion on the robustness of the algorithm's hyperparameters (e.g., the choice of $S$) across different datasets. If they depend on $n$ or condition numbers, provide practical guidelines or a tuning strategy.
3.  **Condition Number Analysis:** Add a brief discussion or experiment regarding the condition number of the systems solved at each iteration. If the condition number deteriorates significantly, it could impact the practical runtime despite favorable iteration counts.
4.  **Unified $\ell_p$ Treatment:** If possible, extend the core analysis to the $1 < p < 2$ case more directly or explain the overhead introduced by the reduction (Appendix C) more explicitly in relation to total runtime.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with Theoretical SOTA (Adil et al. 2024):** The paper claims to match the theoretical bounds of Adil et al. (2024) while being practical, yet only compares empirically against the slower Adil et al. (2019b). An implementation or runtime comparison with the 2024 algorithm is needed to validate the claim that it is impractical.
2. **High-Dimensional Scalability ($d \gg 1$):** Experiments primarily use small $d$ (e.g., Power dataset has $d=11$) or $d \approx n \le 10,000$. Since the algorithm solves $d \times d$ linear systems, performance on high-dimensional data ($d > 10^4$) must be shown to support scalability claims.
3. **Modern First-Order Baselines:** Comparisons are restricted to IPM and prior IRLS. Competitive modern first-order solvers (e.g., ADMM, specialized proximal gradient) are missing, leaving uncertainty about the method's advantage in large-scale regimes where second-order steps are costly.
4. **Comprehensive $p$ Evaluation:** The main experimental section focuses on $p \ge 2$, with $1 < p < 2$ relegated to the appendix. Given the abstract claims applicability for "all values of $p$," the main evaluation must cover the full range to substantiate the contribution.

### Deeper Analysis Needed (top 3-5 only)
1. **Condition Number Stability:** IRLS methods often suffer from ill-conditioning as weights update. An analysis of the condition number of the matrix $ADA^\top$ across iterations is required to verify that linear system costs do not explode in practice.
2. **Sensitivity to Hyperparameters:** Algorithm 3 introduces parameters like $\kappa$ without ablation studies. Analysis is needed to show how performance degrades if these are not optimally tuned for specific problem structures.
3. **Constant Factor Dependence on $p$:** The theoretical bounds hide constants exponential in $p$. An empirical analysis of how runtime scales with large $p$ (e.g., $p > 20$) is needed to assess practical utility beyond small integer $p$.
4. **Memory Complexity:** The paper reports runtime but omits memory usage. For the claimed "large instances," a memory footprint analysis is critical to determine if the method fits within standard hardware constraints.

### Visualizations & Case Studies
1. **Convergence Curves:** Plot objective value versus iteration number to verify if the empirical convergence rate aligns with the theoretical poly($1/\epsilon$) or log($1/\epsilon$) guarantees.
2. **Weight Distribution Evolution:** Visualize the histogram of IRLS weights over iterations to demonstrate numerical stability and rule out weight explosion or collapse.
3. **Failure Mode Case Study:** Provide a specific instance where the baseline p-IRLS (Adil 2019b) stalls or degrades to isolate the mechanism behind the proposed method's improvement.

### Obvious Next Steps
1. **Preconditioning Strategy:** Integrate and evaluate specific preconditioners for the weighted least squares subproblems to guarantee stable linear solver performance on ill-conditioned data.
2. **Downstream ML Application:** Evaluate the solver within a downstream ML task (e.g., robust regression training) to demonstrate end-to-end utility rather than isolated optimization performance.
3. **Distributed Implementation:** Since linear solves are the bottleneck, a discussion or prototype of a parallel/distributed version is necessary to claim true scalability beyond single-machine limits.

# Final Consolidated Review
## Summary

This paper presents a new IRLS-based algorithm for ℓp regression that achieves improved iteration complexity over prior practical methods (p-IRLS by Adil et al., NeurIPS 2019) while matching (or nearly matching) the best known theoretical bounds (Adil et al., JACM 2024). The key technical innovation is a primal-dual framework where coordinate updates are derived from a maintained invariant linking energy increase to dual norm increase, enabling large multiplicative steps without mirror maps or smoothing. The paper provides algorithms for both low-precision (poly(1/ε) convergence) and high-precision (log(1/ε) convergence) regimes, with empirical validation on synthetic and real-world datasets.

## Strengths

- **Theoretical contribution:** The paper successfully bridges the gap between theoretically optimal but impractical algorithms and practical but theoretically suboptimal IRLS methods. The primal-dual invariant approach (Equation 1) is elegant and yields provable convergence guarantees without complex width-reduction machinery.
- **Improved iteration complexity:** The low-precision solver (Algorithm 2) achieves O(poly(1/ε)) convergence with iteration bounds improving on p-IRLS's O(poly(n)·poly(1/ε)), while the high-precision solver achieves O(p² log n · log(n/ε)) subproblems—improving the p-IRLS guarantee while maintaining practicality.
- **Clear algorithmic presentation:** Algorithms 1–4 are well-structured with explicit pseudocode. The case analysis (Case 1: direct primal solution, Case 2: averaged primal from slow iterations, Case 3: dual certificate) provides intuition for the convergence mechanism.
- **Empirical improvement over p-IRLS:** Across all tested configurations, the algorithm shows consistent 1.2–2.6× speedup over the prior practical IRLS implementation, with fewer linear system solves and lower wall-clock time.

## Weaknesses

- **Potential log n factor discrepancy:** Theorem 1.2 states the high-precision algorithm uses O(p² log n · log(n/ε)) subproblems, each requiring O(n^{(3p−2)/(p−2)}) linear system solves, yielding total complexity O(p² n^{(3p−2)/(p−2)} log n · log(n/ε)). The paper claims this "matches" Adil et al. (2024)'s O(p² n^{(3p−2)/(p−2)} log(n/ε)), which differs by a factor of log n. The authors should clarify whether this comparison accounts for some constant factor or if the log n factor is absorbed differently in the analysis.
- **No empirical evaluation of the low-precision solver:** Theorem 1.1 presents a genuinely novel contribution—the low-precision Algorithm 2 with poly(1/ε) convergence—but all experiments in Section 4 use the high-precision Algorithm 3. Without empirical validation, it remains unclear whether this theoretically novel component is practical.
- **Narrow experimental parameter range:** Real-world experiments (Table 1) test only p = 8 and ε = 10^{−10}. The theoretical results are most interesting for large p and varying ε, yet the empirical evaluation is limited. Testing p ∈ {3,...,50} on modest n ≤ 2500 matrices does not stress-test the n^{(3p−2)/(p−2)} dependence for large n.
- **Missing comparison with overconstrained setting:** The paper cites Jambulapati-Liu-Sidford (2022) in related work but does not discuss the overconstrained regression setting (n ≫ d) where that work achieves dimension-dependent bounds. While the underdetermined setting is different, readers would benefit from a clearer discussion of when each algorithm is preferable.
- **No condition number analysis:** IRLS methods can suffer from ill-conditioning in the weighted linear systems ADA^T. The paper does not analyze or empirically measure how condition numbers evolve across iterations, which affects practical runtime beyond iteration counts.

## Nice-to-Haves

- Comparison with modern first-order methods (ADMM, proximal gradient) for ℓp regression, even if these lack iteration complexity guarantees.
- Empirical validation of Algorithm 2 at moderate precision (e.g., ε ∈ {10^{−4}, 10^{−6}}) to demonstrate practicality of the low-precision solver.
- Ablation study on hyperparameter sensitivity, particularly the threshold S and initialization constants.

## Removed Points

These points are flagged to be removed, treat them with caution:

*Removed: Venue fit criticism (ICLR vs. SODA).* This is subjective and not a substantive technical flaw. ICLR has published theoretical optimization papers, and the relevance to ML applications is reasonably argued.

*Removed: Request for comparison with unimplemented theoretical algorithms.* The paper correctly notes that Adil et al. (2024)'s algorithm is not practically implementable due to complex subroutines and hyperparameters. Demanding empirical comparison with an unavailable baseline is unreasonable.

*Removed: Demand for comprehensive 1 < p < 2 treatment.* Appendix C provides a reduction and Theorem C.1 gives the iteration complexity. While a unified analysis would be elegant, the current treatment is adequate for a conference paper.

## Novel Insights

The primal-dual energy invariant (Equation 1: E(r^{(t+1)}) − E(r^{(t)}) ≥ M² · (‖r^{(t+1)}‖_q − ‖r^{(t)}‖_q)) provides a clean geometric interpretation for why large multiplicative coordinate steps are permissible in IRLS. Unlike standard mirror descent which regularizes to ensure small steps, this approach shows that maintaining the energy-to-norm increase ratio directly controls convergence. The decomposition into "high-gain" iterations (T_hi) and "low-gain" iterations (T_lo) with separate convergence arguments is a useful proof technique that may apply to other ℓp optimization settings.

## Suggestions

- Clarify the exact relationship between the claimed bound and Adil et al. (2024), addressing whether the log n factor is absorbed in hidden constants or represents a genuine gap.
- Add experiments with Algorithm 2 (low-precision solver) at ε ∈ {10^{−2}, 10^{−4}, 10^{−6}} to validate its practical utility.
- Include an analysis or empirical measurement of the condition numbers of the weighted systems ADA^T across iterations to assess numerical stability.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 6.0]
Average score: 6.5
Binary outcome: Accept
