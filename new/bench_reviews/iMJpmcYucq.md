Now I have enough information to write the consolidated review. Let me systematically assess each claim.

## Summary

This paper proposes a variance-reduced estimator for the intractable Bures–Wasserstein (BW) gradient in Gaussian variational inference. The key idea is to use the Stein score of the variational distribution, $\Sigma_k^{-1}(X_k - m_k)$, as a control variate for $\nabla V(X_k)$ — motivated by the observation that as $\mu_k$ approaches $\pi$, $\nabla V$ and $-\Sigma_k^{-1}(x - m_k)$ approximately coincide. The resulting algorithm (SVRGVI) is a single-line modification of SGVI (Diao et al., 2023) that adds only $O(d^2)$ overhead per iteration. Theoretically, Lemma 1 gives an exact variance decomposition, Theorems 1–2 guarantee variance reduction in meaningful regimes, and Theorems 3–4 show improved convergence bounds over SGVI. Empirically, the method achieves orders-of-magnitude improvement in KL divergence on synthetic targets.

## Strengths

- **Principal, elegant, and minimal idea**: Using $\Sigma_k^{-1}(X_k - m_k)$ as a control variate is directly motivated by the VI objective ($\mu_k \approx \pi$) and has zero expected value by construction. The algorithm change from SGVI to SVRGVI is a single line (Algorithm 1), and the overhead is $O(d^2)$ dominated by the $O(d^3)$ Cholesky decomposition already required (Section 3, "Minimal extra computational cost").

- **Strong unconditional guarantee at the optimum**: Lemma 1 shows the variance of $\tilde{b}_k$ equals the MC variance plus $c^2 \mathrm{Tr}(\Sigma^{-1}) - 2c\mathrm{Tr}(\mathbb{E}\nabla^2 V)$. At the optimum $\hat{\pi}$, this extra term is $c(c-2)\mathrm{Tr}(\hat{\Sigma}^{-1}) < 0$ for $c \in (0,2)$, minimized at $c=1$. This is a clean, unconditional result — variance reduction is *guaranteed* at convergence.

- **Propagation into convergence theory**: Theorems 3–4 show that variance reduction improves the convergence bounds of SGVI, specifically the noise term scales with $(1+\tau_{\max,E})$ instead of a fixed constant, and the contraction rate improves (Remark 3). While the rates remain the same, the constants measurably improve.

- **Dramatic empirical improvements, scaling with dimension**: Figure 3 shows the gap between SVRGVI and SGVI/BWGD grows from negligible at $d=10$ to 5 orders of magnitude at $d=200$ (KL $\approx 10^{-2}$ vs. $\approx 10^3$). This is consistent with the theory since variance reduction of the mean gradient matters more in high dimensions. The method also outperforms EVI at higher dimensions.

## Weaknesses

### Fatal
None.

### Major

- **Hessian variance is not addressed, potentially bounding improvement in some regimes**: The proposed control variate reduces only the mean gradient $\mathbb{E}_\mu \nabla V$ variance, while the Hessian estimator $S_k = \nabla^2 V(X_k)$ remains an unreduced single-sample MC estimator over a $d \times d$ matrix. The paper acknowledges this explicitly in Section 3 ("$W_k = \Sigma_k^{-1}$ is deterministic... the control variate does not affect $\tilde{S}_k$; we keep the standard estimator") and Remark 3 ("the noise terms... would not disappear because of another source of randomness coming from $S_k$"). In very high dimensions or for non-quadratic targets, Hessian noise could dominate, and without an analysis of the relative magnitudes, the claim that mean-gradient variance is the *primary* bottleneck needs more empirical justification. The empirical results suggest this is not yet a problem at $d=200$, but it could limit scaling further. This limits the completeness of the solution rather than invalidating it.

- **All experiments use synthetic targets**: Gaussian targets, Student's $t$ targets, and Bayesian logistic regression on simulated data ($X_i \sim \mathcal{N}(0, I_d)$, no real datasets). While appropriate for a methodological paper establishing convergence properties, practical deployment on real data (e.g., UCI classification) would significantly strengthen the contribution. The paper acknowledges this in Section 6: "our experiments focused on synthetic targets."

### Minor

- **Convergence improvements are in constants, not rates**: Theorems 3–4 improve the bounds over SGVI (Diao et al., 2023, Theorems 5.7–5.8) by replacing noise coefficients with smaller ones and tightening the contraction rate, but the asymptotic convergence rates remain unchanged. The paper should more explicitly state this distinction between constant-level and rate-level improvement.

- **The "orders of magnitude" accuracy improvement at $\eta=1$ conflates two effects**: SVRGVI benefits from both variance reduction per se and the ability to use $\eta=1$ (the step size that variance reduction enables). If SGVI with a carefully tuned smaller step size converges to similar accuracy but more slowly, the "5 orders of magnitude" gap partly reflects divergent baselines rather than a fundamental accuracy gap. The paper does study varying step sizes in Appendix B.7, which partially addresses this, but the headline results at $\eta=1$ could be misleading without qualification.

- **Fixed $c=0.9$ rather than adaptive $c_k^*$**: Remark 1 derives the optimal $c_k^* = \mathrm{Tr}(\mathbb{E}_{\mu_k}\nabla^2 V)/\mathrm{Tr}(\Sigma_k^{-1}) \approx 1$ near $\hat{\pi}$, justifying $c=0.9$. However, during early iterations far from the optimum, $c^*$ may differ, and the paper provides no empirical comparison between fixed and adaptive $c$.

### Trivial
None.

## Nice-to-Haves

- Experiments on real-data Bayesian logistic regression (e.g., UCI datasets) to demonstrate practical relevance beyond synthetic targets.
- A variance decomposition experiment quantifying the relative contribution of mean gradient noise vs. Hessian noise across dimensions, providing evidence for whether mean gradient variance is indeed the dominant bottleneck.
- Exploration of Hessian variance reduction (e.g., minibatching $\nabla^2 V$ or second-order control variates), which could further push the method's scalability.

## Removed Points

- **"Comparison with tuned step sizes for baselines"**: This is partly addressed (Appendix B.7 studies varying step sizes). The remaining concern (that the "orders of magnitude" headline conflates divergence and accuracy) is kept as a minor weakness. The demand for a fully separate best-step-size comparison is removed as it is partially addressed and largely a presentation concern.

- **"The variance reduction guarantee is local/conditional, leaving a potential gap"**: The paper itself notes in Remark 2 that even for convex $V$, variance reduction holds for sufficiently small $c_k$ at every iteration. The "gap" scenario (small variance but far from optimum) can be handled by choosing $c_k$ adaptively per Remark 2. This concern is partially addressed.

- **"EVI comparison is incomplete"**: The paper explains that EVI has different per-iteration cost, so a convergence-curve comparison is not apples-to-apples. Reporting final accuracy for a carefully optimized EVI is a reasonable choice. Removed.

- **Reproducibility concerns about hyperparameters or missing appendix**: Appendix experiments study varying $\eta$ and $c$. Code is available. Per our rules, nitpicks about undisclosed details are removed.

- **Formatting/typo nitpicks**: Removed per instructions.

## Novel Insights

The observation that the Stein score of the variational distribution serves as a natural control variate for the target score is both simple and powerful — it leverages the VI objective itself ($\mu \to \pi$) to construct the variance reduction. The asymmetry this creates is noteworthy: the mean gradient admits a stochastic but correlated control variate, while the Hessian control variate $W_k = \Sigma_k^{-1}$ is deterministic and thus provides no variance reduction. This is a fundamental structural difference between the two components of the BW gradient, and addressing it would require fundamentally different techniques (e.g., minibatching or antithetic sampling for the Hessian). The paper's contribution is therefore best understood as solving the variance problem for the dominant noise source, rather than the complete solution.

## Suggestions

- Add a real-data experiment (e.g., UCI Bayesian logistic regression) to demonstrate practical applicability beyond synthetic settings.
- Include a plot showing the empirical variance of the mean gradient vs. Hessian over iterations, quantifying which noise source actually dominates and justifying the focus on $\tilde{b}_k$.
- Qualify the "orders-of-magnitude" improvement language in the abstract and Section 5 to note that at $\eta=1$, some baselines diverge, and the improvement partly reflects the stability that variance reduction enables at aggressive step sizes.

## Assessment by Axis

**Originality**: The control variate construction is novel and well-motivated. Using the Stein score of the variational distribution as a control variate for the target score, and deriving Lemma 1 via Stein's lemma, is a clean contribution that has not appeared in the BW-VI literature.

**Importance**: Making BW-space VI practical is significant — this line of work (Lambert et al., 2022; Diao et al., 2023) had strong theory but impractical algorithms due to gradient noise. The dramatic empirical improvements demonstrate real progress.

**Claims support**: The core theoretical claims (variance reduction, convergence improvement) are well-supported by Theorems 1–4 and Lemma 1. The "orders-of-magnitude" empirical claim is supported but should be qualified regarding step-size effects.

**Soundness of experiments**: Experiments are sound but limited to synthetic targets. The comparison protocol (same $\eta=1$ for all BW methods, different per-iteration cost for EVI) is fair and explained clearly.

**Clarity**: The paper is very well-written. The motivation, derivation, and theoretical results flow logically. The minimal change from SGVI to SVRGVI is clearly presented.

**Value to community**: This is a practical, principled improvement to a method family (BW-VI) that needed it. The code is available and the algorithm is easy to implement.

## Score and Decision

**Calibration anchors**:

- **High (7–8)**: Path gradient estimators for normalizing flows (avg 7.5, Accept poster) — similar variance-reduction idea but for NFs; OCM paper (avg 8, Oral) — simple elegant modification with moderate empirical gains; CMCD (avg 7.2, Accept poster) — connects OT and VI.
- **Medium (4–6)**: SteinDreamer (avg 5.0, Reject) — Stein control variate for score distillation, but weaker theory and less compelling experiments.
- **Low (<3)**: Unnormalized VI (avg 3, Reject) — fundamental technical flaws.

This paper sits solidly above the medium anchors. It has a cleaner theoretical foundation than SteinDreamer (Lemma 1 + Theorems 1–4 vs. less formal analysis), and dramatically stronger empirical results. Compared to the path gradient paper (7.5), it has comparable theory strength but slightly narrower experimental scope (synthetic only). Compared to the OCM paper (8), it has a similarly elegant idea but less broad experiments. The Hessian noise limitation and synthetic-only experiments prevent it from being quite at the 8 level, but the clear theory + dramatic empirical gains place it well above borderline.

MY FINAL SCORE: <pineapple>7</pineapple>
MY FINAL DECISION: <orange>Accept</orange>