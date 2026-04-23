Now I have all the information needed. Let me write the final review.

## Summary

This paper studies global convergence in bilevel optimization, identifying that the Polyak-Łojasiewicz (PL) condition does not compose through bilevel structure (i.e., PL on both levels does not guarantee PL on the nested bilevel objective). It proposes using a penalty reformulation of the bilevel problem instead, defines two PL-type conditions (joint PL and blockwise PL) on the penalized objective, and proves that penalty-based bilevel gradient descent (PBGD) converges globally under these conditions. The conditions are verified for two specific linear-model bilevel problems—representation learning and data hyper-cleaning—using trajectory-based induction arguments.

## Strengths

- **Clean demonstration that PL does not compose through bilevel structure.** Example 1 (Section 2.1) provides an explicit counterexample where $f(u,v) = \frac{1}{2}(u-2\sin v)^2$ and $g(u,v) = \frac{1}{2}(u-v)^2$ are both jointly PL, yet $F(u) = \frac{1}{2}(u-2\sin u)^2$ has saddle points and violates PL. This is a genuine and non-obvious insight that the community should be aware of, and it rigorously justifies why the penalty reformulation route is necessary.

- **The penalty reformulation landscape comparison is well-constructed and informative.** The argument that $L_\gamma(u,v)$ can yield a more benign landscape than $F(u)$, illustrated through Examples 1–2 and Figures 1–2, provides concrete intuition. Figure 2 showing PBGD trajectories converging to the global optimum across different $\gamma$ values makes the conceptual point clearly.

- **The trajectory-based local-to-global proof strategy is a meaningful technical contribution.** Rather than assuming global PL conditions, the paper verifies local, non-uniform PL conditions along PBGD's trajectory using induction and acute matrix perturbation theory (as highlighted in challenge T2, Section 1.3). For representation learning, Lemma 1 establishes a trajectory-dependent PL constant $\mu_k = (\sigma_{\min}^2(W_1^k) + \sigma_{\min}^2(W_2^k))\sigma_*^2(X_\gamma)$, and Theorem 2 bounds $\mu_k \geq \mu = \mathcal{O}(\gamma)$ uniformly. This trajectory-based approach could inspire similar analyses in other bilevel settings.

- **Observation 2 on PL additivity under linear composition (Section 3.3)** is a concrete technical tool that enables the landscape analysis for both applications, and addresses the general non-additivity of PL functions noted with counterexamples in Appendix C.2.

## Weaknesses

### Fatal
None.

### Major

- **The diagonal Gram matrix assumption in Theorem 3 severely restricts the practical relevance of the data hyper-cleaning result.** Theorem 3 requires $[X_{\text{trn}}; X_{\text{val}}][X_{\text{trn}}; X_{\text{val}}]^\top$ to be diagonal, meaning all data points (training and validation) must be mutually orthogonal. While this can be satisfied in overparameterized settings with $m \geq N + N'$, it is essentially never satisfied on real datasets. The paper does not discuss whether this assumption can be relaxed or provide any path toward relaxing it. A global convergence result under an assumption that rules out practically all real data is of limited value and undermines the claim of "unlocking global optimality" in data hyper-cleaning.

- **Gap between the broad framing and the narrow scope of verified results.** The title "Unlocking Global Optimality in Bilevel Optimization" and the abstract's claim of presenting "two sufficient conditions for global convergence" create an expectation of broadly applicable insight. The conditions themselves are natural extensions of standard PL to the bilevel setting (joint PL is just PL on $(u,v)$; blockwise PL is PL per block). The real intellectual content lies in *verifying* when these conditions hold, and this verification covers only two linear-model problems with heavily restrictive assumptions. The paper does acknowledge studying "certain (not all)" applications and justifies linear models by analogy with single-level theory (Section 1.1), but the framing still overshoots the delivered scope. The "pilot study" framing in the original title partially mitigates this, but the abstract and introduction do not.

### Minor

- **The PL conditions themselves are straightforward generalizations of standard PL.** The joint PL condition (Definition 1, Eq. 2) is exactly the standard PL condition treating $(u,v)$ as a single variable, and the blockwise PL condition (Eqs. 3a–3b) is the natural block-coordinate extension. The paper does not claim these are deeply novel, but the contribution weighting should favor the verification effort rather than the condition definitions.

- **The $c(W)$ constant in Lemma 2 requires every training point to have positive mismatch.** The blockwise PL constant over $u$ involves $c(W) = \min_i\{\|y_i^\top - x_i^\top W\|^2 - \|y_i\|^2 \mathbb{1}(\cdot)\}_{>0}$, which requires at least one training point to have strictly positive mismatch. If any training point is perfectly fit, $c(W)$ could be zero, weakening the PL constant. The paper uses perturbation theory to bound this along the trajectory, but the dependency is not discussed as a limitation.

- **Experiments only confirm theory on the same toy settings it was derived for.** The experiments (Figures 3–4) are conducted on synthetic linear-model data matching the theory, and while they validate the convergence rates, they provide no evidence that insights extend to nonlinear models, non-least-squares losses, or realistic data. A negative result on a nonlinear bilevel problem would have been informative.

### Trivial
None.

## Nice-to-Haves

- Extending the verification to at least one setting with a nonlinear model component (e.g., fixed feature extractor + linear head) to demonstrate the framework's applicability beyond pure linearity.
- Characterizing when the diagonal Gram matrix assumption can be relaxed for data hyper-cleaning, even partially.
- Explicit discussion of the tension between solution quality ($\gamma$ large) and convergence speed ($\alpha = \mathcal{O}(\gamma^{-1})$ small) as $\gamma$ grows.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Assumption 2 is insufficiently justified because sufficient conditions are deferred to Appendix F.1"** — The rule states that criticisms about missing appendix content should be removed, as the parser strips appendices. The appendix exists in the original submission.

- **"The sufficient conditions are essentially tautological"** — While the conditions are natural extensions of PL, calling them "tautological" overstates the case. The paper's contribution is not merely defining conditions but verifying they hold for specific bilevel problems via non-trivial trajectory analysis. The joint PL condition applied to $L_\gamma$ is not the same as PL applied to $f$ or $g$ individually—it involves the interplay of the penalty structure. Still, the point that the conditions are straightforward generalizations is retained as a minor weakness above.

- **"Observation 2 is essentially a linear algebra exercise"** — This is dismissive. While the proof may be straightforward, the result is a useful technical tool that enables the subsequent landscape analysis and addresses a real gap (PL non-additivity). The contribution is in identifying and applying it, not in the difficulty of the proof.

- **"Circularity concern in the induction proof for representation learning"** — The paper explicitly addresses this through its induction structure: it bounds $\sigma_{\min}(W_1^k)$ away from zero along the trajectory using the induction hypothesis, not by assuming it. The induction resolves the potential circularity. This is a feature of the proof technique, not a flaw.

- **"Comparison with $F^2$SA and BOME is shallow"** — The paper explicitly states in Remark 1 that these methods could also have global convergence, and the experiments (Figure 4c–d) show they converge on the tested problems. The paper's primary contribution is the theoretical framework, not an algorithmic comparison. Criticizing the comparison depth on a toy problem that all methods solve is not substantive.

- **"Missing experiments on nonlinear models or non-synthetic data"** — This is retained as a minor weakness above (experiments only confirm theory), but the demand for experiments beyond the theory's scope is partially a scope-creep criticism. The paper explicitly scopes itself to linear models.

- **"Discussion of the gap between penalized problem optimum and original bilevel problem optimum"** — The paper addresses this through the result from Shen et al. (2023) and through application-specific approximate equivalence (challenge T3, Section 1.3). While more discussion would be valuable, this is a nice-to-have rather than a flaw.

## Novel Insights

The paper reveals a fundamental structural insight about bilevel optimization: the landscape distortion introduced by the lower-level solution mapping $\mathcal{S}(u)$ can destroy benign properties (like PL) that hold individually at each level, and this distortion can be circumvented by working in the higher-dimensional space of the penalty reformulation $L_\gamma(u,v)$. The fact that adding dimensions (moving from $F(u)$ to $L_\gamma(u,v)$) can smooth out the landscape is a counterintuitive and potentially general principle—it suggests that reformulating constrained bilevel problems through penalty methods is not just algorithmically convenient but landscape-theoretically advantageous.

## Suggestions

- Scale back the title and abstract framing to better match the scope: e.g., "Towards Global Optimality in Bilevel Optimization: A Pilot Study via Penalty Reformulation" would set more accurate expectations while preserving the "pilot study" framing.
- For the data hyper-cleaning result, add an explicit discussion of the diagonal Gram matrix assumption's restrictiveness and whether near-diagonal (or approximately orthogonal) data could yield approximate results.
- Consider adding one experiment on a slightly richer model (e.g., two-layer network with a fixed nonlinear feature map) to show whether the penalty reformulation's landscape advantage persists beyond pure linearity, even if convergence proofs don't yet cover it.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Local PL for overparameterized linear models | O0FOVYV4yo | 5.0 | Closest match: similar trajectory-based local PL verification for linear models. Our paper is stronger due to the bilevel novelty (Example 1, penalty reformulation insight) but weaker due to the very restrictive diagonal Gram matrix assumption in Theorem 3. |
| PL condition for residual networks | AbXGwqb5Ht | 7.0 | That paper proves PL for a more practical model class (residual networks), scoring higher. Our paper covers only linear models. |
| Global optimality of ICL for Markov chains | HuBFimORiz | 4.33 | That paper's theory was considered too narrow and specific with overly restrictive assumptions. Our paper has similar narrowness concerns but has a cleaner conceptual contribution. |
| Loss landscape via convex duality | 4xWQS2z77v | 8.0 | Far more comprehensive landscape analysis with broader implications—our paper is not at this level. |
| Exact linear-rate GD (vacuous theory) | 1NYhrZynvC | 2.5 | Our paper's theory is not vacuous or circular—it provides real convergence guarantees under verifiable (if restrictive) conditions. Much stronger than this anchor. |
| Adversarial training on non-separable data | SQLDXQ3IG8 | 5.25 | Similar pattern: interesting conceptual step beyond prior work but restrictive model assumptions limiting practical relevance. |

Our paper sits in the 5–5.5 range, comparable to the O0FOVYV4yo and SQLDXQ3IG8 anchors. It has genuine conceptual contributions (PL non-composition insight, penalty reformulation advantage) that elevate it above the 3–4 range where theory is purely incremental or vacuous. However, the very restrictive assumptions (especially the diagonal Gram matrix in Theorem 3) and the gap between broad framing and narrow delivery keep it below the acceptance threshold.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>