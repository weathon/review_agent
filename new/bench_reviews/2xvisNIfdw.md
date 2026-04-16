Now I have a good understanding of the paper and the calibration papers. Let me synthesize my final review.

Key calibration anchors:
- CvYBvgEUK9 (Penalty methods for bilevel optimization, Accept-spotlight): Scores 5-8, avg ~6.2. This is directly related work on penalty-based bilevel optimization, got accepted with solid but incremental contributions.
- O0FOVYV4yo (Local PL for overparameterized linear models, Reject): Scores 3-6, avg ~5. Very similar methodology (local PL analysis for linear models), rejected due to limited novelty and scope.
- b9ZG7cI8ic (PL conditions for FedAvg with single-layer NN, Reject/Withdrawn): Scores 3-6, avg ~4.25. Similar pattern of PL conditions verified only in very restricted settings.
- 2fSyBPBfBs (Bilevel without lower-level strong convexity, Reject): Scores 3-6, avg ~4.2. Related bilevel work that was rejected partly due to limited scope of examples.
- NbbsRnPBoS (Faster GD in deep linear networks, Reject): Scores 1-3, avg ~2.33. Very restricted setting (scalar), rejected as too narrow.

The paper under review has a similar methodology to O0FOVYV4yo (local PL for linear models with trajectory-dependent constants) but applies it to bilevel optimization. It's better scoped than NbbsRnPBoS but similarly restricted to linear settings. The harsh reviewer makes valid points about scope overclaiming but is too dismissive of the genuine theoretical contribution. The neutral reviewer correctly identifies the strengths and weaknesses in a balanced fashion.

Let me write my final review now.</think>## Summary

This paper studies global convergence (as opposed to convergence to stationary points or local minima) in bilevel optimization, an important open problem. The authors propose two "benign landscape" conditions—joint PL and blockwise PL—on the penalized reformulation of bilevel problems, and prove that the penalty-based bilevel gradient descent (PBGD) algorithm globally converges under these conditions. The conditions are rigorously verified along the optimization trajectory for two specific applications: representation learning with two-layer linear networks and data hyper-cleaning with one-layer linear models and least-squares losses.

## Strengths

- **Novel and important problem framing.** The distinction between global convergence and stationary-point convergence in bilevel optimization is substantively important. The paper clearly identifies a gap—prior work almost exclusively targets stationarity or local minima—and the observation that the penalty reformulation $L_\gamma(u,v)$ can yield a more benign landscape than the nested objective $F(u)$ (illustrated through Example 1 and Figure 1) is insightful and cleanly motivates the approach.

- **Technically non-trivial trajectory-based analysis.** The key technical challenge (T2) is that only local, iterate-dependent PL and smoothness constants hold along the PBGD trajectory, not global ones. The authors use induction-based proofs combined with matrix perturbation theory to establish uniform bounds on singular values, yielding constant lower/upper bounds on $\mu_k$ and $L_k$. This is genuine technical work, not a routine application of standard PL theory.

- **Clean conceptual contribution via Observation 2.** The identification that strongly-convex-composite-with-linear functions preserve PL additivity is a useful structural insight that connects bilevel global convergence to known landscape tools and could potentially be reused for other quadratic/linear bilevel problems.

- **Two distinct structural patterns well-matched to update schemes.** The joint PL / Jacobi pairing (for isomorphic variables in representation learning) and blockwise PL / Gauss-Seidel pairing (for heterogeneous variables in hyper-cleaning) is a thoughtful division that reflects genuine structural differences in bilevel problems.

## Weaknesses

### Major:

- **Significant gap between broad framing and narrow scope of validated examples.** The title ("Unlocking Global Optimality in Bilevel Optimization") and introduction suggest a broad advance, including references to high-stakes applications in policy-making, energy, healthcare, and robotics. However, the actual theory rigorously covers only two highly structured linear-model settings: (1) a two-layer linear network with least-squares loss, overparameterized, full-row-rank data under Assumption 2; and (2) data hyper-cleaning where $[X_{\text{trn}}; X_{\text{val}}][X_{\text{trn}}; X_{\text{val}}]^\top$ is *diagonal*. The diagonal Gram matrix assumption in Theorem 3 is particularly restrictive—it requires orthogonal training samples, which essentially never holds for real data. While the authors argue that "global convergence analysis for single-level optimization also starts with linear models" (§1.1), single-level PL results for linear models led to widely-applicable tools; here, the PL conditions depend on Observation 2, which applies only to strongly-convex-composite-with-linear structure, and the trajectory-based verification appears difficult to extend beyond these constructed examples. The paper would be stronger if it honestly scoped its claims or provided even partial evidence (e.g., negative results, discussion of barriers) beyond these two cases.

- **The "benign landscape conditions" are verified only algorithm-dependently, not as intrinsic function properties.** In both applications, the joint PL / blockwise PL conditions hold with iterate-dependent constants ($\mu_k$ depends on $\sigma_{\min}(W_1^k), \sigma_{\min}(W_2^k)$ in representation learning; the PL constant over $u$ depends on $c(W)$ in hyper-cleaning). The paper verifies these conditions *along the PBGD trajectory* via induction, not as global properties of the objective. This means these are not intrinsic landscape properties of the function class—they are algorithm-relative invariant region arguments. The conceptual claim "penalty reformulations yield genuinely more benign landscapes" is established only in the narrow sense that "for these specific problems, running PBGD keeps iterates in a region where PL holds." This distinction matters because it limits the generality of Theorem 1: its hypotheses (global joint/blockwise PL) are not actually verified for any problem; only algorithm-dependent local versions are verified.

- **Experiments are limited to confirming theoretical predictions on toy problems.** All experiments use synthetic data matching the exact theoretical setups (linear models, least-squares losses, synthetic matrices). There are no experiments on real datasets, no experiments with any nonlinearity, no experiments under perturbed assumptions (e.g., slightly non-diagonal Gram matrices), and no stochastic/mini-batch experiments. The comparison in Figure 4 only tests other penalty-based methods (F²SA, BOME) on the same linear problems with tuned stepsizes, merely confirming that gradient methods converge on these simple landscapes—unsurprisingly, all methods converge globally.

### Minor:

- **Assumption 2 is stated abstractly.** It requires the existence of an $(\epsilon_1, \epsilon_2)$ bilevel solution $W_1^*$ that also minimizes $L_{\text{trn}}$ globally, which is a non-trivial structural condition. While sufficient conditions are given in Appendix F.1, a more intuitive discussion in the main text about when it holds would help readers assess its reasonableness.

- **The exact correspondence between penalized and original optima is not fully addressed for general $\gamma$.** The paper shows an $(\epsilon, \mathcal{O}(\epsilon))$ approximation, but whether the global minimizer of $L_\gamma$ exactly coincides with the bilevel global optimum (or whether spurious minimizers of $L_\gamma$ can exist that are not near any bilevel solution) is not discussed for general settings. This gap affects whether "convergence of PBGD on $L_\gamma$" truly means "convergence to the bilevel global optimum."

### Trivial:

- Minor typo: "almost linear onvergence" in §6 should be "almost linear convergence."

## Nice-to-Haves

- Even a negative result or informal discussion about whether the joint/blockwise PL conditions can hold for neural networks with ReLU activations (even shallow ones) would significantly clarify the boundary of the approach.
- Empirical testing under perturbed assumptions (e.g., add small noise to the diagonal Gram matrix) to assess robustness of the global convergence behavior.
- A trajectory visualization of $\mu_k$ over iterations (e.g., $\sigma_{\min}^2(W_1^k)$ and $\sigma_{\min}^2(W_2^k)$) to build intuition about how tight the inductive bounds are.

## Removed Points

- **"No experiments on real datasets" (from Spark reviewer #1).** While the experiments are indeed toy-only, demanding real-dataset experiments from a theory paper focused on proving global convergence for linear models is scope creep. The paper's contribution is theoretical; confirming convergence on synthetic data matching the theory is appropriate. Removed as a major weakness demand but retained as a nice-to-have.

- **"No comparison with nested bilevel methods" (from Spark reviewer #2).** The paper's central claim is about the penalty reformulation's landscape advantage, so comparing with nested methods would be reasonable. However, this is an experimental suggestion for a primarily theoretical paper, not a fundamental flaw. Moved to nice-to-have.

- **"No stochastic/mini-batch experiments" (from Spark reviewer #4).** The theory is for full-batch GD. Demanding stochastic analysis is outside the paper's stated scope. Removed from main weaknesses.

- **"Limited novelty of the algorithm since PBGD is from prior work" (Neutral reviewer #3).** The paper explicitly acknowledges this (Remark 1) and frames its contribution as the landscape analysis, not the algorithm. This is not a real weakness.

- **"Incomplete discussion of penalized vs. original optima gap" (Neutral reviewer #4).** The paper does address this through the $(\epsilon, \mathcal{O}(\epsilon))$ correspondence (Definition 2 and Theorem 1), which is the standard penalty approximation. The exact correspondence for general settings has been studied in prior work. Downgraded to minor.

- **"Incremental contribution relative to prior penalty-based analysis" (Human Finder weakness #3).** This comparison is unfair. While penalty-based bilevel methods exist, none proved *global* convergence. The step from stationary-point guarantees to global convergence—even for restricted settings—is a qualitative advance, not incremental. Removed.

- **"Clarify gap between pilot study framing and title claims" (Spark suggestion).** The subtitle *is* "A Pilot Study," which already tempers the title. This is a presentation preference, not a substantive weakness. Removed.

## Novel Insights

The most distinctive conceptual contribution is the explicit demonstration (via Example 1 and Figure 1) that PL conditions on both individual levels do not transfer to the nested bilevel objective $F(u)$, even with a unique and linear solution mapping—this is a genuine structural barrier that justifies working with the penalty reformulation. The trajectory-dependent PL analysis, while limiting the generality of the "landscape conditions" framing, is a pragmatic proof strategy that may inspire similar inductive approaches for other structured bilevel problems where global PL fails.

## Suggestions

- **Temper the framing.** Add a candid paragraph in §1 or §6 explicitly acknowledging that the validated examples are heavily structured (linear networks, diagonal Gram matrices) and that extending to nonlinear models is a major open challenge. This would prevent the overclaiming concern without diminishing the genuine theoretical contribution within scope.
- **Add a "limitations" discussion** of when the diagonal Gram matrix assumption and Assumption 2 fail, and what the barriers are to relaxing them.
- **Consider adding even one small experiment** with a slightly nonlinear model (e.g., a 2-layer network with ReLU) in representation learning to empirically test whether global convergence still holds or where it breaks.

## Score and Decision

**Calibration analysis:**

| Paper | Topic | Scope | Score Range | Key Issue |
|-------|-------|-------|------------|-----------|
| CvYBvgEUK9 | Penalty methods for nonconvex bilevel | Broader, general convergence theory | 5–8 (avg 6.2, Accept-spotlight) | Many assumptions but clear landscape theory |
| O0FOVYV4yo | Local PL for linear networks | Local PL for overparameterized linear models | 3–6 (avg 5, Reject) | Incremental over prior work, narrow setting |
| b9ZG7cI8ic | PL + FedAvg + single-layer NN | PL conditions verified only in 1-hidden-layer NN | 3–6 (avg ~4.25, Reject) | Proof issues, restricted setting |
| NbbsRnPBoS | GD in deep linear networks, width-1 | Extremely narrow (scalar) | 1–3 (avg ~2.3, Reject) | Too restricted, limited novelty |

This paper is more substantial than O0FOVYV4yo (it opens a new direction—global convergence for bilevel—rather than extending PL analysis within an established line) and more rigorous than b9ZG7cI8ic (no proof errors identified). However, it shares the core weakness of being limited to very structured linear settings with assumptions that are far from practical (especially the diagonal Gram matrix). The "pilot study" subtitle acknowledges this, but the title and introduction overclaim. Compared to CvYBvgEUK9 (which also studies penalty-based bilevel methods but with broader scope and got 6.2 avg, spotlight), this paper has narrower applicability and stronger framing-vs-content mismatch, placing it below that anchor.

Score: **4.5** — The theoretical contribution of establishing trajectory-based PL conditions and proving global convergence for bilevel problems (even restricted ones) is genuine and the first of its kind. However, the extremely narrow scope of verified applications (linear models with diagonal Gram matrices), the algorithm-dependent nature of the "landscape conditions," and the overclaiming in framing substantially limit the impact.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>