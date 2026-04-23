Now I have all the evidence I need. Let me write the final consolidated review.

## Summary

The paper introduces RTZ-VI-LCB, a model-based offline algorithm for robust two-player zero-sum Markov games (RTZMGs) with transition kernel uncertainty. The algorithm integrates robust value iteration with the principle of pessimism via lower confidence bounds, and achieves sample complexity $\tilde{O}(C_r^* H^4 S(A+B) f(\sigma^+, \sigma^-, H)/\varepsilon^2)$ under a robust unilateral clipped concentrability assumption. The paper also establishes information-theoretic lower bounds and extends the approach to multi-player general-sum Markov games.

## Strengths

- **Significant improvement in action-space dependence**: The algorithm achieves sample complexity with $S(A+B)$ dependence, improving over P²M²PO's $S^2 AB$ (Table 1, Theorem 1). Combined with the lower bound of $\Omega(C_r^* S H^3(A+B)/\varepsilon^2)$ from Theorem 2, this confirms optimality in $S$ and $\{A, B\}$, which is a genuine and meaningful advance for offline RTZMGs.

- **Robust unilateral clipped concentrability (Assumption 1, eq. 22)**: This is a meaningful conceptual contribution — it extends single-policy clipped concentrability to the robust multi-agent setting while only requiring unilateral (per-player) coverage, which is strictly weaker than requiring joint product-policy coverage. The clipping at $1/S(A+B)$ enables learning under partial coverage.

- **Dual reformulation for computational tractability (eq. 18)**: The strong duality transformation converts the $S$-dimensional optimization over the probability simplex into a one-dimensional maximization over $\alpha$, making each robust Bellman update computationally efficient.

- **Two-stage subsampling technique (Algorithm 1, Lemma 1)**: The adaptation of the two-fold subsampling method from single-agent RL to TZMGs is sound, with Lemma 1 providing the distributional equivalence guarantee (probability $\geq 1 - 8\delta$) and the useful count lower bound in eq. (16).

- **Lower bound construction (Theorem 2)**: The construction showing that robust TZMGs are at least as hard as standard TZMGs for small uncertainty is a clean information-theoretic result, independent of the specific distance metric.

## Weaknesses

### Fatal

None.

### Major

- **Inconsistency between Theorem 1's gap bound (eq. 23) and the claimed sample complexity — $f$ vs. $f^2$**: Equation (23) states $\text{Gap}(\hat{\mu}, \hat{\nu}) \leq c_1 \sqrt{\frac{C_r^* H^3 S(A+B) \log(KH/\delta)}{K}} \cdot f(\sigma^+, \sigma^-, H)$, with $f$ **outside** the square root. Setting $\text{Gap} \leq \varepsilon$ and solving for $K$ yields $K \geq c_1^2 \frac{C_r^* H^3 S(A+B) \log(KH/\delta) f^2}{\varepsilon^2}$, giving $T = KH \geq \tilde{O}\left(\frac{C_r^* H^4 S(A+B) f^2}{\varepsilon^2}\right)$. However, the paper repeatedly claims the sample complexity is $\tilde{O}\left(\frac{C_r^* H^4 S(A+B) f}{\varepsilon^2}\right)$ — with $f$, not $f^2$ (Section 1.1, Table 1, discussion after Theorem 1). If the theorem as written is correct ($f$ outside the square root), then for small uncertainty where $f = H$, the sample complexity becomes $\tilde{O}(C_r^* H^6 S(A+B)/\varepsilon^2)$ rather than the claimed $\tilde{O}(C_r^* H^5 S(A+B)/\varepsilon^2)$. If $f$ should be inside the square root, then the gap bound is presented incorrectly. Either way, this inconsistency undermines the paper's central optimality claims and must be resolved.

- **Inconsistency between Table 1 and Theorem 2 on the lower bound for small uncertainty**: Table 1 lists the lower bound for $\min\{\sigma^+,\sigma^-\} \lesssim 1/H$ as $\frac{C_r^* S H^3(A+B)}{\varepsilon^2}$ (with $H^3$). However, Theorem 2 (eq. 27) states the lower bound condition $T \leq c_2 \frac{C_r^* H^3 S(A+B) \min\{1/\min\{\sigma^+,\sigma^-\}, H\}}{\varepsilon^2}$. When $\min\{\sigma^+,\sigma^-\} \lesssim 1/H$, we have $\min\{1/\min\{\sigma^+,\sigma^-\}, H\} = H$, giving a lower bound of $\Omega(C_r^* H^4 S(A+B)/\varepsilon^2)$ — with $H^4$, not $H^3$. The paper's own discussion after Theorem 2 (line 308) confirms $H^4$: "no algorithm can find an $\varepsilon$-optimal robust policy with fewer than $\Omega(C_r^* S H^4(A+B)/\varepsilon^2)$ samples." Table 1 is inconsistent with both Theorem 2 and its textual interpretation. If the table is wrong, it misrepresents the lower bound; if the theorem/text is wrong, the optimality claims need recalibration.

### Minor

- **$f = H$ for most practical parameter regimes, limiting the impact of uncertainty-dependent analysis**: The function $f(\sigma^+, \sigma^-, H) = \min\{(H\sigma^+ - 1 + (1-\sigma^+)^H)/(\sigma^+)^2, (H\sigma^- - 1 + (1-\sigma^-)^H)/(\sigma^-)^2, H\}$ equals $H$ for most parameter values. Specifically, for $\sigma \leq 0.5$ and $H \geq 4$, the term $(H\sigma - 1 + (1-\sigma)^H)/\sigma^2$ substantially exceeds $H$, so $f = H$. Only when $\sigma$ is close to 1 does $f < H$ (e.g., $g(1) = H-1$). This means the robust and non-robust sample complexities coincide for most practical uncertainty levels, and the paper's emphasis on the $f$-dependent refinement is less impactful than presented. The paper should explicitly characterize the regime where $f < H$ and discuss its practical relevance.

- **Undefined gap function for the multi-player extension (Theorem 3)**: Theorem 3 (eq. 28) uses $\text{Gap}(\hat{\pi})$ without defining it. Equations (10–11) only define the gap for the two-player zero-sum case. The multi-player gap should be defined, since its meaning (and the corresponding equilibrium concept) determines whether "breaking the curse of multiagency" is a meaningful claim.

- **$C_r^*$ potentially much larger than the non-robust counterpart**: Assumption 1 (eq. 22) requires a supremum over all perturbed transition kernels $P \in \mathcal{U}(P^0)$. This means the data distribution must cover the optimal policies' visitation distributions under *every* kernel in the uncertainty set, not just the nominal one. For large uncertainty sets, $C_r^*$ could be dramatically larger than its non-robust counterpart, making the robust problem fundamentally harder from a data perspective — a point the paper does not discuss.

### Trivial

- The algorithm's per-step computation of robust matrix games is acknowledged as "generally PPAD-hard" (line 214) but the limitation is not discussed beyond a single mention. For a theory paper, a brief discussion of when this computation is tractable (e.g., two-player zero-sum case reduces to LP) would be helpful.

## Nice-to-Haves

- A plot of $f(\sigma, H)$ as a function of $\sigma$ for several values of $H$ would immediately reveal that $f = H$ for most $\sigma$ values, helping readers understand the practical influence of the uncertainty level.
- Empirical validation (even a small-scale tabular experiment) would help verify the $f$ vs. $f^2$ discrepancy and demonstrate whether the $S^2 AB \to S(A+B)$ improvement materializes in practice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that the multi-player extension is unsupported due to PPAD-hardness of computing Nash in general-sum games**: The "curse of multiagency" in this paper's context refers to *sample complexity* (exponential in number of players via $\prod_i A_i$), not computational complexity. The claim that sample complexity scales as $\sum_i A_i$ is a legitimate contribution in the sample complexity literature, regardless of computational hardness. However, the undefined gap function remains a valid weakness.

- **Demand for experiments as a "missing part"**: Experiments would strengthen the paper, but this is a pure theory paper and the absence of experiments is not a critical flaw in that context. Moved to Nice-to-Haves.

- **Burn-in cost criticism that the $\varepsilon$-independent burn-in is "practically misleading"**: The burn-in cost (eq. 24) is technically independent of $\varepsilon$ as claimed. While the $1/d_m^n$ factor can be large, this is a standard feature of offline RL analyses and not a unique weakness of this paper.

- **Criticism that the asymmetric cross-selection (max-player uses $\hat{Q}^-$, min-player uses $\hat{Q}^+$) is unjustified**: The algorithm's design follows naturally from the pessimism principle — each player uses the conservative estimate of their own value. While more explicit justification would be helpful, the rationale is clear from context.

- **Strength claim that the multi-player extension "breaks the curse of multiagency" is a supporting strength**: While the linear-in-$\sum_i A_i$ scaling is noteworthy, the undefined gap function and lack of proof in the main text weaken this claim. The multi-player result is best viewed as a promising extension rather than a fully substantiated contribution.

## Novel Insights

The most insightful observation from the review process is that the two mathematical inconsistencies ($f$ vs. $f^2$ and Table 1 vs. Theorem 2) are likely related: if $f$ should be inside the square root in eq. (23), then the upper bound sample complexity is correctly stated as $\tilde{O}(C_r^* H^4 S(A+B) f/\varepsilon^2)$, but the gap bound as written would need correction. Conversely, if the gap bound is correct, both the claimed sample complexity and Table 1 need updating. The resolution of one inconsistency constrains the other, and the authors should ensure all stated results (gap bound, sample complexity, Table 1, lower bound text) are mutually consistent.

## Suggestions

- Resolve the $f$ vs. $f^2$ inconsistency by either: (a) correcting eq. (23) to place $f$ inside the square root, or (b) updating all sample complexity claims to use $f^2$. Ensure all instances (abstract, Section 1.1, Table 1, discussion after Theorem 1) are updated consistently.
- Correct Table 1's lower bound for small uncertainty from $H^3$ to $H^4$ (or justify why $H^3$ is correct and Theorem 2 needs revision).
- Define $\text{Gap}(\hat{\pi})$ explicitly for the multi-player case in Theorem 3.
- Add a brief characterization of when $f < H$ and discuss whether this regime is practically relevant.

## Score and Decision

**Calibration anchors:**

- **High (avg 8.0)**: stUKwWBuBm — MARL with novel solution concept, clean theory, computational tractability proofs. This paper under review has genuine contributions but cannot match the theoretical cleanliness of a paper with verified, internally consistent results.
- **Medium (avg 7.0)**: AOlm45AUVS — Offline MARL with interaction rank, solid sample complexity bounds, consistent theory. This paper under review has similar scope but with mathematical inconsistencies that the AOlm45AUVS paper avoids.
- **Medium (avg 5.5)**: x36mCqVHnk — Zero-sum MG model-free algorithm with unclear technical contribution; rejected. This paper under review has stronger contributions but also more serious mathematical issues.
- **Medium-low (avg 4.6)**: r0kY4SS7ts — Markov games with mathematical errors changing theorem meanings; withdrawn/rejected. This paper under review has a similar pattern of mathematical inconsistencies in core theorems.
- **Low (avg 4.0)**: qybJSeG2VH — DRO-based offline RL with mathematical rigor issues in the proof of the claimed sample complexity; withdrawn/rejected.

The paper under review has real and significant contributions — the $S^2 AB \to S(A+B)$ improvement, the robust unilateral clipped concentrability concept, and the lower bound construction. However, the two mathematical inconsistencies ($f$ vs. $f^2$ in the core gap bound vs. sample complexity, and $H^3$ vs. $H^4$ in Table 1 vs. Theorem 2) directly affect whether the headline optimality claims hold. This places the paper in the same tier as r0kY4SS7ts (avg 4.6) and qybJSeG2VH (avg 4.0), both of which had mathematical inconsistencies in core results. The contributions here are somewhat stronger than those papers (the $S(A+B)$ result, if correct, is a genuine breakthrough for the subfield), so I place it slightly above but still in the reject range. If the inconsistencies are merely presentational (e.g., $f$ belongs inside the square root), they could be fixed in rebuttal, but the burden of proof is on the authors.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>