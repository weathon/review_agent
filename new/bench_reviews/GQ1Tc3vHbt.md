Now I have sufficient information for calibration. Let me now write the final review.

**Summary of calibration:**
- The Gorbunov et al. paper (0wmfzWPAFu) is the closest comparator—a concurrent work on convex (L₀,L₁)-smooth optimization that got accepted as a poster with scores 6,6,6,8,5. The paper under review is arguably stronger than Gorbunov et al.: it covers nonconvex + convex + acceleration, provides better acceleration complexity (polynomial vs exponential in L₁R), and derives principled stepsizes. 
- The lower bounds paper (ZjOXuAfS6l) scored 8,6,6,6,5 (Spotlight) — that paper proved lower bounds which is arguably harder than upper bounds. The paper under review doesn't prove lower bounds.
- The reject papers (UmMKbG2Ubr, O2GBkHujdP) have significantly weaker contributions.

The paper under review is stronger than Gorbunov et al. in terms of results and comprehensiveness, but the nonconvex complexity is misstated in the abstract/contributions (a real error), and it lacks numerical experiments. Overall, it aligns well with a 6-7 range at ICLR. Given the stronger results but presentation issue, I'll score around 7.

---

## Summary
The paper studies gradient methods for $(L_0, L_1)$-smooth optimization—a generalization of Lipschitz-smooth functions motivated by neural network training. The authors derive tighter first-order characterizations of this function class (Lemma 2.2), use these to derive principled stepsize rules (including connections to clipped steps), and establish improved complexity bounds: $\mathcal{O}(\frac{L_0 R^2}{\epsilon} + L_1 R \ln\frac{F_0}{\epsilon})$ for convex functions (Theorem 3.2), $\mathcal{O}(\frac{L_0 R^2}{\epsilon} + [L_1 R]^2)$ for parameter-free NGM and PS-GM, and $\nu\mathcal{O}(\sqrt{L_0 R^2/\epsilon} + (L_1 R)^{2/3} \ln\frac{F_0}{\epsilon})$ for an accelerated method (AGMsDR).

## Strengths

- **Principled derivation of stepsizes from the tight upper model.** The paper explicitly minimizes the surrogate upper bound from Lemma 2.2 to obtain stepsizes (9), then shows simplified (12) and clipped (13) variants satisfy the same per-iteration decrease (Eq. 11). This gives a clean structural reason for *why* clipping works—it approximates the surrogate-optimal stepsize—which is genuinely new insight not established in prior work.

- **Tighter first-order characterizations enabling the entire analysis.** Lemma 2.2's inequalities (3) and (4) are provably sharper than prior bounds (Zhang et al. 2020, Hübler et al. 2024), and the convex lower bound Lemma 2.4/Corollary 2.5 generalizes the classical quadratic lower bound of Nesterov (2018, Theorem 2.1.5) to the $(L_0,L_1)$ setting. The paper exploits these tighter bounds throughout.

- **Strong convex GM result (Theorem 3.2) with logarithmic $L_1$-dependence.** The $\mathcal{O}(\frac{L_0 R^2}{\epsilon} + L_1 R \ln\frac{F_0}{\epsilon})$ bound does not require $L$-smoothness, improving over Koloskova et al. (2023)'s $\mathcal{O}(\frac{L_0 R^2}{\epsilon} + \sqrt{L/\epsilon} L_1 R^2)$ which required additional $L$-smooth assumption, and avoids the initial-gradient dependence of Li et al. (2023). The paper correctly acknowledges the pessimistic $F_0$ case.

- **Modular AGMsDR theorem (Theorem 6.1).** The generic acceleration framework—which only requires any update rule with strictly positive decrease—cleanly separates the acceleration scaffold from the per-step progress mechanism. This modularity may be independently useful. The resulting $\mathcal{O}(\sqrt{L_0 R^2/\epsilon} + (L_1 R)^{2/3}\ln(F_0/\epsilon))$ complexity replaces the exponential $\exp(\mathcal{O}(1)L_1 R)\sqrt{L_0 R^2/\epsilon}$ bound in Gorbunov et al. (2024) with a polynomial in $L_1 R$.

- **Bounds free of exponential dependence on $L_0, L_1$ and free of initial gradient norm.** All complexity results avoid $\|\nabla f(x_0)\|$ and exponential dependence on $L_0, L_1$ (unlike Li et al. 2023), with the important caveat that $F_0$ can be exponentially large in the worst case (which the paper acknowledges).

## Weaknesses

### Fatal
None.

### Major

- **Nonconvex complexity is misstated in the abstract, contribution bullets (Section 1), and conclusion.** The abstract and Section 1 claim the nonconvex complexity is $\mathcal{O}(\frac{L_0 F_0}{\epsilon} + \frac{L_1 F_0}{\epsilon})$, but Theorem 3.1 states the bound requires $K+1 \geq \frac{2L_0 F_0}{a\epsilon^2} + \frac{3L_1 F_0}{a\epsilon}$—i.e., the $L_0$ term has $\epsilon^2$ in the denominator, not $\epsilon$. This is materially different in complexity order. The conclusion repeats the incorrect expression. A paper should not misstate its own headline result in multiple prominent locations. (This is a presentation error that does not appear to affect the actual theorem, which is correct, but it corrupts the contribution claims as written.)

- **The headline convex improvement is conditional on $F_0$ being reasonably bounded.** As the paper itself acknowledges in the paragraph after Theorem 3.2, $F_0$ can be as large as $\exp(L_1 R)\frac{L_0 R^2}{2}$, in which case $L_1 R \ln(F_0/\epsilon)$ becomes $\mathcal{O}([L_1 R]^2)$ — the same as the parameter-free methods. This means the claimed "significant improvement" over prior work depends on an extra favorable condition ($F_0$ bounded), which is not an assumption but a property of specific problem instances. The abstract and contribution bullets present the log dependence as the general result without adequate foregrounding of this conditionality.

### Minor

- **The $\nu$ factor in AGMsDR renders the acceleration comparison imprecise.** The comparison in Section 6 against Gorbunov et al. (2024) and Li et al. (2023) is done at the iteration level, but AGMsDR requires solving a 1D subproblem per step (costing $\nu$ oracle queries), while comparison methods do not. Since $\nu$ is left abstract, the oracle complexity comparison is not apples-to-apples. The paper acknowledges the line-search elimination as an open problem, but the comparison language should be qualified by the oracle model difference.

- **No experimental validation.** The paper is entirely theoretical. Given the ML motivation and the importance of practical method selection (optimal vs. simplified vs. clipped stepsizes), even a few numerical experiments on canonical $(L_0,L_1)$-smooth functions (logistic regression, $p$-norm objectives from Example A.1) would clarify whether the theoretical improvements materialize in practice and whether the $\nu$ overhead of AGMsDR is negligible.

- **NGM and PS-GM "adaptivity" claim is overstated.** The paper claims these methods "automatically adapt to the best possible $(L_0, L_1)$" (end of Section 4, end of Section 5, Conclusion), but this is a post-hoc analytic statement: the *bound* can be minimized over valid $(L_0, L_1)$ pairs, but the algorithm itself does not estimate or exploit this pair online. This is standard in the literature, but the phrasing overpromises algorithmic adaptivity.

### Trivial

- The paper calls stepsize (9) "optimal" without qualification; it is optimal for the surrogate upper model induced by (4), not globally optimal. This is a minor writing issue.

## Nice-to-Haves

- Provide lower bounds (even partial) for the $L_1 R \ln(F_0/\epsilon)$ and $(L_1 R)^{2/3} \ln(F_0/\epsilon)$ terms to establish whether these rates are optimal. The conclusion acknowledges this as an open question; even brief progress would strengthen significance claims.

- Extend NGM and PS-GM convergence analysis to the nonconvex case, or explain why this is fundamentally harder. The nonconvex setting is a primary ML motivation, and these parameter-free methods are practically important.

- Discuss or explore adaptive estimation of $\hat{R}$ for NGM (e.g., via doubling tricks), since the sensitivity to $\hat{R}$ is a practical limitation acknowledged in the paper.

- Add a complexity summary table comparing all methods (GM, NGM, PS-GM, AGMsDR) against prior work under unified assumptions and oracle models.

- Consider an extension to stochastic gradients, given the ML motivation (deep network training under $(L_0,L_1)$-smoothness is fundamentally stochastic).

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Removed — misleading about "exponential dependency"**: One reviewer flags that the "no exponential dependency" claim is undermined because $F_0$ can be exponentially large. However, the paper *explicitly acknowledges* this in the discussion after Theorem 3.2: "the second estimate ... comes from a very pessimistic bound on $F_0$ with the exponentially large quantity $\exp(L_1 R)\frac{L_0 R^2}{2}$." The complaint is that the abstract doesn't foreground this (subsumed under the major weakness above), but characterizing the result as having "exponential dependency on $L_0$ or $L_1$" misreads the paper.

**Removed — "twice differentiability is too restrictive"**: The paper states that its Definition 2.1 is equivalent to the $\alpha$-symmetric class with $\alpha=1$ for $C^2$ functions (Section 2), and that all $\alpha$-symmetric $C^2$ functions are also $(L_0,L_1)$-smooth. The paper's scope is explicitly twice-differentiable functions, and this is consistent with the original Zhang et al. (2019) definition. Whether non-$C^2$ examples exist is a separate question not central to the paper's contributions.

**Removed — "no stochastic extension limits ML relevance"**: The paper's scope is deterministic optimization, which is stated upfront. Requesting stochastic extensions is scope creep for this submission, though acknowledged above as a nice-to-have.

**Removed — "significance of NGM/PS-GM improvement over prior work"**: The reviewer from the competing paper (Gorbunov et al.) criticized that "only the non-dominant $O(\sqrt{1/\varepsilon})$ term" is improved. This critique applies to Gorbunov et al.; for the present paper, the improvement of NGM/PS-GM is against Koloskova et al. (2023) and Takezawa et al. (2024), which required extra $L$-smoothness assumptions. Removing a separate assumption is a genuine contribution.

## Novel Insights

The most genuinely novel conceptual contribution is the formal identification of clipped stepsizes as controlled approximations to the surrogate-optimal step derived from the tight upper model (4). This unifies several previously disparate stepsize heuristics (clipping, normalization, Polyak) under a single derivation principle, and the modular AGMsDR framework (Theorem 6.1) — which reduces accelerated convergence analysis to bounding $M_k = \|\nabla f(y_k)\|^2 / (2[f(y_k)-f(x_{k+1})])$ — is an independently interesting template that could extend beyond $(L_0, L_1)$-smoothness. The convex lower bound (Lemma 2.4) as a generalization of Nesterov's classical Theorem 2.1.5 is also a clean structural result that fills a gap in the theory.

## Suggestions

1. **Fix the nonconvex complexity in the abstract, Section 1 contribution bullet, and Section 7 conclusion**: replace $\mathcal{O}(\frac{L_0 F_0}{\epsilon} + \frac{L_1 F_0}{\epsilon})$ with $\mathcal{O}(\frac{L_0 F_0}{\epsilon^2} + \frac{L_1 F_0}{\epsilon})$ throughout.

2. **Qualify the convex bound more clearly in the abstract and contributions**: state that the improvement to $O(L_1 R \ln(F_0/\epsilon))$ over $O([L_1 R]^2)$ holds when $F_0$ is reasonably bounded, and give a representative condition (e.g., $F_0 \leq \text{poly}(L_0, L_1, R)$).

3. **Add oracle-model qualification to the AGMsDR comparison**: the stated complexity is $\nu$-oracle queries vs. 1-oracle per iteration for baseline methods; comparisons should make this explicit.

4. **Add at least one or two numerical experiments** on canonical $(L_0,L_1)$-smooth functions comparing the three stepsize variants and the four methods; this directly addresses whether the theoretical gains matter in practice.

## Score and Decision

**Calibration:**
- Gorbunov et al. (0wmfzWPAFu) — concurrent work on convex $(L_0,L_1)$-smooth optimization, accepted as poster, scores 6,6,6,8,5 (avg ≈ 6.2). The paper under review covers more ground (nonconvex + convex + acceleration) and achieves stronger results on acceleration and the convex GM bound, but has the nonconvex complexity error and lacks experiments. 
- Lower-bounds paper (ZjOXuAfS6l) — accepted as poster/spotlight, scores 8,6,6,6,5 (avg ≈ 6.2). That paper proved lower bounds (arguably harder), was also theory-only.
- Reject papers: AdaGrad convergence (UmMKbG2Ubr, scores 5,5,6,6) and I-NSGD (O2GBkHujdP, scores 3,3,5,6) are clearly weaker in contribution.

The paper under review is stronger than Gorbunov et al. in breadth and the acceleration result, but shares the weakness of being purely theoretical and having a presentation error in the headline nonconvex rate. The error is correctable but is real: the abstract states a complexity strictly stronger than what Theorem 3.1 proves. The convex bound is genuinely improved. The AGMsDR result is notably better than prior acceleration results (polynomial vs. exponential in $L_1 R$). Considering these factors and the calibration anchors, a score of **7** is appropriate — above the comparable accepted papers due to stronger technical content, but held back by the presentation error and absence of experiments.

**Novelty:** High — principled stepsize derivation, improved convex rates, superior acceleration  
**Technical soundness:** High for the theorems proved; moderate for the presentation accuracy  
**Empirical support:** None — purely theoretical  
**Significance:** High in the (L₀,L₁)-smooth theory literature  
**Clarity:** Good overall, with the notable exception of the nonconvex complexity mismatch

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>