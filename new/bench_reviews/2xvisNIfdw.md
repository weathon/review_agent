Now let me search for calibration papers to help score this paper.Now I have enough information. Let me write the consolidated review.

---

## Summary

This paper investigates global convergence in bilevel optimization — a longstanding open problem — using the lens of the penalty reformulation. The authors define two generalizations of the PL condition (joint PL and blockwise PL) on the penalized objective $L_\gamma(u,v)$, prove that PBGD converges to the global bilevel optimum under either condition (Theorem 1), and then verify these conditions rigorously along the PBGD trajectory for two specific applications: two-layer linear representation learning (Sec. 4) and data hyper-cleaning with a linear model (Sec. 5), establishing global convergence for these problems for the first time. The title includes "A Pilot Study" and the paper is explicit that it covers "certain (not all)" applications.

---

## Strengths

- **First global-convergence guarantee for a first-order bilevel gradient method.** Prior work only guarantees stationarity or local optimality. Theorems 2 and 3 establish an $\mathcal{O}(\log^2(\epsilon^{-1}))$ complexity to an $(\epsilon, \mathcal{O}(\epsilon))$ bilevel-optimal solution, a qualitatively stronger guarantee than anything previously known for this class of algorithms.

- **Compelling motivation through concrete landscape analysis.** Examples 1–5, Figures 1–2, and the contrast between the nested objective $F(u)$ and the penalized objective $L_\gamma(u,v)$ are genuinely illuminating. Example 1 demonstrates that PL on both levels does not guarantee PL on $F(u)$, which is a crisp, non-obvious result.

- **Technically non-trivial inductive trajectory analysis (T2).** The proof of Theorems 2 and 3 does not just invoke global PL conditions — because only *local, non-uniform* constants hold, the authors use induction combined with acute matrix perturbation theory to show $\sigma_{\min}(W_1^k), \sigma_{\min}(W_2^k)$ remain bounded away from zero, yielding $k$-independent lower/upper bounds on $\mu_k, L_k$. This is a substantive technical contribution.

- **Observation 2 is a clean, reusable tool.** The additivity lemma for strongly convex functions composed with linear maps (Observation 2) provides a transferable technical device for establishing PL in structured objectives, with potential utility beyond this paper.

- **Honest scope.** The paper is transparent about being a pilot study focused on linear models, draws an explicit parallel to how single-level global convergence theory (matrix completion, linear neural networks) also started with linear settings, and does not claim generality it does not prove.

---

## Weaknesses

### Fatal
*None.* The paper does not have a flaw severe enough to invalidate its core claim. The results are correctly stated within their scope.

### Major

- **The diagonal Gram matrix assumption in data hyper-cleaning (Lemma 2, Theorem 3) is highly restrictive.** Lemma 2 requires $X_{\text{trn}} X_{\text{trn}}^\top$ to be diagonal; Theorem 3 escalates this to requiring $[X_{\text{trn}}; X_{\text{val}}][X_{\text{trn}}; X_{\text{val}}]^\top$ to be diagonal — i.e., *all* training and validation samples must be mutually orthogonal. This is essentially the only setting where hyper-cleaning with a linear model is analytically tractable via the current approach, and the paper offers no discussion of whether or how the assumption might be relaxed. The lack of any relaxation or discussion is a genuine gap. Unlike the diagonal assumption in Section 4 (which is justified by the full-row-rank structure maintained along the trajectory), the diagonal Gram requirement here structurally decouples samples in a way that is hard to connect to realistic scenarios. The fact that $\mathcal{S}(u)$ becomes independent of $u$ under overparameterization is intrinsic to the problem (and the coupling is still present through the sigmoid-weighted penalty term), but the Gram diagonality assumption for Theorem 3 is a further, external restriction. The paper should at minimum discuss the prevalence and verifiability of this condition.

- **Experiments are limited to synthetic, theory-matched instances and do not validate global convergence directly.** The experiments (Figures 3–4) show convergence curves on small constructed problems that satisfy all theoretical assumptions by design. There is no comparison of the achieved objective value against an analytically computed or exhaustively searched global optimum — essential for a paper whose central claim is global (not just fast) convergence. For the tiny linear problems used, computing the true global bilevel optimum is feasible, and this comparison is absent. Additionally, no experiments probe behavior when key assumptions (orthogonality, full-rankness) are mildly violated, so the robustness of the findings is entirely unknown.

- **The claim "adaptable to multi-layer neural networks" (Sec. 4) is asserted without proof.** The paper states: "our analysis is adaptable to multi-layer neural networks as well." However, the proof relies critically on the bilinear/quadratic structure of the two-layer linear case — specifically, the ability to apply Observation 2 and track the singular values of exactly two weight matrices. For deeper networks, this structure breaks down. The paper should either provide a sketch or remove this claim.

### Minor

- **The blockwise PL case in Theorem 1 requires that $\arg\min_v L_\gamma(u,v)$ is independent of $u$.** This condition is stated as a formal requirement but receives little discussion. It is a meaningful restriction that removes a key source of bilevel difficulty (the lower-level solution's dependence on the upper-level variable). The paper should discuss which problem classes naturally satisfy this condition beyond the hyper-cleaning setting (where it follows from overparameterization).

- **Approximate equivalence between the penalized and original bilevel problem relies on a cited external result.** The paper delegates the optimality equivalence ($(ε, O(ε))$ bilevel solution from an $\epsilon$-solution of the penalized problem) to Shen et al. (2023). T3 in Sec. 1.3 mentions establishing application-specific equivalences, but this is deferred heavily to the appendix with little explanation in the main text. Providing more intuition for how this equivalence holds under local (rather than global) PL would improve accessibility.

- **PL constant for hyper-cleaning over $u$ depends on $c(W)$, the "minimum positive mismatch."** The lower bound on this constant is asserted based on perturbation theory in the appendix, but no intuition is given in the main text for why $c(W)$ stays bounded away from zero along the trajectory. A short remark explaining this would strengthen reader confidence in the result.

### Trivial

- **Figure 3's caption mentions $L_{\text{trn}}(W_1, W_2) - L_{\text{trn}}^*(W_1)$** as a measured quantity. It is not immediately obvious from the experimental setup how $L_{\text{trn}}^*(W_1)$ is computed at each step; this should be clarified in the experimental appendix.

---

## Nice-to-Haves

- **Visualization of singular values $\sigma_{\min}(W_1^k), \sigma_{\min}(W_2^k)$ over iterations** would directly verify the core induction hypothesis of Theorem 2, giving readers empirical confidence in the trajectory-based argument.

- **At least one experiment on real data (even if without formal guarantees),** e.g., Fashion-MNIST for hyper-cleaning with a linear model, would substantially boost the practical relevance of the framework.

- **A short discussion of the practical implications of $\gamma = \mathcal{O}(\epsilon^{-0.5})$ scaling** — specifically, whether this causes numerical instability for small $\epsilon$ and how large $\gamma$ must be in practice to achieve useful accuracy levels.

- **An example showing when $L_\gamma$ *fails* to satisfy joint or blockwise PL** (a small bilevel problem outside the studied cases) would clarify the boundary of the approach.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **[Harsh Critic — structural overstatement]** The critic argues the title/abstract "materially overstates the scope." This criticism is weakened by the fact that: (a) the title itself says "A Pilot Study"; (b) the abstract explicitly says "two specific bilevel learning scenarios"; (c) Sec. 1.1 says "for certain (not all) machine learning applications." The framing is standard for a foundational theory paper introducing new methodology on tractable instances. The claim is fair, but the degree of "overstatement" is mild and appropriately hedged throughout the paper. *Not included as a major weakness.*

- **[Harsh Critic] — $\mathcal{S}(u)$ independence "collapses the bilevel coupling."** The critic claims this makes the hyper-cleaning result vacuous. This overstates the issue: $\mathcal{S}(u)$ being independent of $u$ follows naturally from overparameterization (as in the linear representation learning setting), but the bilevel coupling through $u$ persists in the penalized objective via the sigmoid-weighted training loss $\ell_\gamma(u, W) = \ell_{\text{val}}(W) + \frac{\gamma}{2} \sum_i \psi(u_i)[\ldots]$ (eq. 12). The optimization over $u$ is nontrivial and the bilevel interaction is preserved in the penalty term. The diagonal Gram assumption (retained above as a Major weakness) is the more substantive concern. *Overstatement removed; core diagonal assumption concern retained.*

- **[Spark] — Comparison with branch-and-bound / SDP relaxation.** Requesting comparison against branch-and-bound or SDP methods is outside the paper's scope; these are not first-order gradient methods and are computationally intractable at the scales targeted. This is scope-creep criticism. *Removed.*

- **[Spark] — Total gradient complexity for $\gamma = \mathcal{O}(\epsilon^{-0.5})$ scaling.** While the concern about per-iteration cost scaling with $\gamma$ is real, both Theorems 2 and 3 provide explicit iteration complexity $\mathcal{O}(\log^2(\epsilon^{-1}))$ with step size $\alpha = \mathcal{O}(\gamma^{-1})$, which partially addresses this. Moved to Nice-to-Have for practical discussion. *Not a major weakness.*

- **[Harsh Critic] — Assumption 2 bakes in "alignment between train and validation objectives."** Assumption 2 merely ensures the existence of a near-full-rank bilevel solution — a non-degeneracy condition. Full-rank initialization is standard in two-layer linear network analysis (cf. Xu et al., 2023). This is not a problematic assumption. *Removed.*

---

## Novel Insights

The core novel insight — that the penalized bilevel objective $L_\gamma(u,v)$ can enjoy a PL geometry even when the nested objective $F(u)$ does not, and that this geometry can be *maintained along a specific algorithmic trajectory* via induction on matrix singular values — is a meaningful conceptual advance. It reframes the question "when does bilevel optimization have a benign landscape?" from a static, global property to a trajectory-dependent, algorithm-specific property, opening a path toward global-convergence analysis in other structured bilevel settings. Observation 2 (PL additivity under linear composition of strongly convex functions) is a clean, independently useful technical lemma that is likely to appear in future work on global optimization of linear neural architectures.

---

## Suggestions

1. **For Theorem 3, add a discussion of the diagonal Gram assumption:** explain what classes of data approximately satisfy it, whether it can be relaxed to approximate diagonality, and what breaks without it. A single remark in Sec. 5 would suffice.
2. **Add an experiment comparing achieved objective values against the analytically-computed global minimum** for the small synthetic problems used, to directly validate the "global convergence" claim.
3. **Remove or substantially caveat the claim about multi-layer network adaptability** until supported.
4. **In the main text, add a brief explanation of how the optimality equivalence (T3) is established** for each application to make the bridge between the penalized and original bilevel objectives accessible without requiring the appendix.

---

## Score and Decision

**Calibration against comparable papers:**

| Paper | Description | Human Scores | Decision |
|---|---|---|---|
| CvYBvgEUK9 | Penalty methods for nonconvex bilevel, stationary-point guarantees, broader scope | 6,6,6,5,8 | Accept (Spotlight) |
| 2fSyBPBfBs | Bilevel without LL strong convexity, stationary-point focus | 5,6,3,3,5,3 | Reject |
| cyPMEXdqQ2 | Constrained bilevel, gap functions, nonasymptotic convergence | 6,6,8,6 | Accept (Poster) |

**Positioning:** The paper under review targets a strictly harder goal than CvYBvgEUK9 (global rather than stationary convergence) with a more focused scope. The technical contribution — trajectory-dependent PL analysis with induction and matrix perturbation — is more intricate than what was needed in CvYBvgEUK9. The restrictive diagonal assumption in hyper-cleaning and the limited experiments are genuine downsides. Compared to CvYBvgEUK9 (avg ~6.2, accepted spotlight), the paper under review is somewhat narrower in scope but attacks a more ambitious goal with real technical novelty. Compared to cyPMEXdqQ2 (avg 6.5, poster), it is similar in style and ambition.

**Assessment on key axes:**
- *Originality:* High — first global-convergence guarantee for first-order bilevel gradient methods.
- *Importance of research question:* High — global vs. local convergence is a fundamental distinction.
- *Support for claims:* Moderate — theorems are well-stated and proofs are non-trivial, but the hyper-cleaning diagonal assumption and limited experiments reduce confidence in the generality of the contribution.
- *Soundness of experiments:* Below average for a 2025 ML paper — purely synthetic, no direct global-optimum comparison.
- *Clarity:* Good — well-organized, examples are instructive, scope is clearly stated.
- *Value to community:* Moderate-to-high as a pilot establishing methodology for future global-convergence work.

**Final Score: 6.0** — Marginally above acceptance threshold. The paper makes a genuine first-of-its-kind theoretical contribution on an important open problem, the technical work is non-trivial, and the scope is honestly stated. The diagonal Gram assumption in hyper-cleaning and the weak empirical validation prevent a higher score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>