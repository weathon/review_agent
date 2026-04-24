## Summary

This paper proposes a simple control-variate estimator for Bures–Wasserstein Gaussian variational inference that dramatically reduces the variance of the single-sample Monte Carlo gradient estimate $\tilde b_k = \nabla V(X_k) - c_k\Sigma_k^{-1}(X_k - m_k)$, reusing the variational Gaussian’s Stein score. The authors prove local and strongly-convex variance-reduction guarantees (Theorems 1–2), show that variance reduction tightens the SGVI convergence bounds (Theorems 3–4), and demonstrate orders-of-magnitude empirical improvements over BWGD and SGVI, especially in high dimensions.

## Strengths

- **Minimal, effective estimator.** The control variate requires no extra samples and only $O(d^2)$ overhead by reusing the Cholesky factor needed for sampling (Sect. 3, Alg. 1). It is a true drop-in modification of SGVI.
- **Clean local theory.** Lemma 1 gives an exact variance decomposition, and Theorem 1 rigorously establishes variance reduction in a well-defined Wasserstein neighborhood of the optimum. Theorem 2 extends this globally for strongly convex $V$ with sufficiently diffuse covariance.
- **Dramatic empirical gains.** On Gaussian targets the method achieves KL divergences of $10^{-2}$ versus $10^3$ for SGVI/BWGD at $d=200$ (Fig. 3c), and it remains the best-performing method on Student’s $t$ and Bayesian logistic regression targets (Fig. 4).
- **Adaptive coefficient.** The paper derives a cheap adaptive rule $c_k \approx \mathrm{Tr}(S_k)/\mathrm{Tr}(\Sigma_k^{-1})$ grounded in the first-order optimality condition, avoiding manual tuning (Remark 1).

## Weaknesses

### Fatal
None.

### Major
None. The core contribution—a practical variance-reduced mean-gradient estimator with provable local guarantees and strong empirical improvements—is sound and well supported.

### Minor
- **Framing overstates the scope of the resolution.** The introduction states “We resolve this fundamental limitation” (p. 2) and the conclusion says the technique “completely resolves this issue” (p. 9), referring to the high-variance MC problem in the forward step. However, the forward step involves *two* intractable expectations, $\mathbb{E}\nabla V$ and $\mathbb{E}\nabla^2 V$, and the proposed control variate only reduces variance for the former; the Hessian estimator $S_k=\nabla^2 V(X_k)$ is left untouched (Sect. 3: “the control variate does not affect $\tilde S_k$”). The paper is honest about this limitation in the technical sections and in Remark 3 (“another source of randomness coming from $S_k$”), but the abstract-level framing is broader than what is delivered. The claims should be scoped to mean-gradient variance reduction.
- **Convergence bounds are conditional.** Theorems 3–4 assume per-iterate variance reduction (Eq. 7) but do not prove that the practical algorithm—with the fixed $c=0.9$ or the noisy adaptive $c_k$ used in experiments—remains in a variance-reduction regime throughout execution for general convex $V$. The paper provides Theorems 1–2 to partially justify this, but they do not give unconditional global guarantees for the algorithm as implemented. The bounds are still valuable constant improvements under stated assumptions, but the practical algorithm lacks an end-to-end guarantee.
- **Empirical results lack statistical dispersion.** All convergence curves report means over 10 runs but provide no error bars, confidence intervals, or quantile bands (Sect. 5: “bold line showing the average performance”). For methods described as unstable, the mean can obscure run-to-run variability, making the claimed “orders of magnitude” gap harder to assess.
- **Adaptive $c_k$ uses a noisy numerator.** Remark 1 proposes $c_k \approx \mathrm{Tr}(S_k)/\mathrm{Tr}(\Sigma_k^{-1})$, but $S_k=\nabla^2 V(X_k)$ is itself a high-variance single-sample estimator. The paper does not analyze how this approximation affects the variance-reduction guarantee in practice.

### Trivial
None.

## Nice-to-Haves
- Add error bars or median–IQR bands to all convergence curves to verify run-to-run stability.
- Include a brief main-text discussion or small table summarizing the step-size sensitivity and minibatch-comparison results that the paper already provides in the appendix.
- A small ablation keeping $\tilde b_k$ single-sample but estimating $S_k$ with multiple samples would help clarify how much the remaining Hessian noise limits gains on non-Gaussian targets.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **“Missing minibatch comparison in main text.”** The paper explicitly states that a minibatch comparison is provided in Appendix B.5 (p. 8). Appendix material is stripped by the parser; this is not a missing main-text requirement.
- **“Missing step-size sensitivity in main text.”** The paper explicitly refers to step-size experiments in Appendix B.7 (p. 8).
- **“Hessian variance is the fundamental unaddressed bottleneck.”** While true that the Hessian estimator is untouched, the paper does test non-Gaussian targets (Student’s $t$, logistic regression) where $\nabla^2 V$ is not constant, and the method still dominates baselines. Calling this a “structural” flaw overstates the case; it is a scope limitation honestly noted in Remark 3.
- **Typo/formatting complaints.** Per instructions, these are parser artifacts and are removed.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Rephrase the abstract and introduction to say the paper resolves the *mean-gradient* variance bottleneck rather than the full forward-step bottleneck.
- Add error bars to the empirical figures.
- Include a short paragraph in the main text analyzing why reducing $b_k$ variance appears sufficient even when $S_k$ noise remains, perhaps citing the empirical variance traces from Appendix B.4.

## Score and Decision

**Calibration anchors used:**
- **High:** PP1rudnxiW (avg 7.2, Accept poster): Unifying VI/OT framework with extensive experiments and strong technical depth. The present paper is narrower in scope and has fewer experiments, so it sits below this.
- **High:** ywFOSIT9ik (avg 6.8, Spotlight): Clean theory for zeroth-order optimization with limited practical experiments and some wording issues. The present paper is comparable in theory-experiment tradeoff but has stronger empirical gains on its target problem.
- **Medium:** Pf85K2wtz8 (avg 5.75, Accept poster): Wasserstein gradient flow with decent theory, limited datasets, and some performance concerns. The present paper has cleaner theory and much stronger empirical improvements, so it sits above this.
- **Medium:** OwNoTs2r8e (avg 6.0, Accept poster): Primarily conceptual, limited practical guidance. The present paper has stronger experiments and practical relevance.
- **Low:** ohHtdp3jDi (avg 4.0, Reject): Subpar performance, shaky justification, ignored related work. The present paper is vastly superior.
- **Low:** pu7a7JHW20 (avg 3.0, Reject): Fundamental technical errors and very weak experiments. Not comparable.

The paper under review has a clear, practical contribution with sound local theory and striking empirical results. Its weaknesses—framing that slightly oversells the scope, conditional convergence bounds, and missing error bars—are real but do not undermine the core claims. It is stronger than the 5.75 poster anchor and comparable to the 6.0–6.8 cluster of accepted theory-and-experiment papers.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>