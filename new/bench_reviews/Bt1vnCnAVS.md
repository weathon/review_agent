Now let me run calibration searches in parallel to anchor the score.Now I have sufficient information to write the final consolidated review.

---

## Summary

LOO-StabCP introduces a *leave-one-out* stability framework for conformal prediction that fits only a single model (trained on the observed training data $\mathcal{D}$) regardless of how many test predictions $m$ are needed, in contrast to the existing RO-StabCP that requires $m$ refits. The authors derive LOO stability bounds for RLM, SGD, kernel methods, neural networks, and derandomized bagging, and demonstrate that the LOO bound for SGD is exactly half the RO bound. They also extend the framework to a conformalized screening application (LOO-cFBH) for FDR-controlled multiple testing.

---

## Strengths

- **O(1) vs O(m) model fits with formal coverage guarantee (Table 1, Theorem 1):** The central computational contribution is clean and well-supported. Table 1 formally establishes that LOO-StabCP requires 1 model fit versus $m$ for RO-StabCP and $|\mathcal{Y}| \cdot m$ for FullCP, and Theorem 1 proves the finite-sample coverage guarantee under LOO stability — all without data splitting.

- **Factor-of-2 tighter SGD stability bound with mechanistic explanation (Theorem 3, Eq. 5):** $\tau^{\text{LOO}} = Rn\eta\gamma\nu_i\rho_{n+j}$ versus $\tau^{\text{RO}} = 2Rn\eta\gamma\nu_i\rho_{n+j}$. The proof is well-motivated: leaving out one point removes one gradient update, while replacing one point can reverse an update direction in the worst case, doubling the bound. This directly yields tighter intervals in SGD-based experiments (Figures 1–3).

- **Empirical speed advantage confirmed (Figures 1–2):** LOO-StabCP consistently matches SplitCP in computation time while producing shorter prediction intervals, and is dramatically faster than RO-StabCP at $m = 100$. The real-data experiments (Boston Housing, Diabetes) corroborate the simulation results.

- **Breadth of theoretical coverage (Theorems 2–5):** Stability bounds are derived for RLM, SGD (convex), SGD (non-convex/neural networks), kernel methods, and derandomized bagging, providing practitioners with concrete computable bounds for a wide range of standard algorithms.

- **Algorithm simplicity and transparency (Algorithm 1):** The method is straightforward to implement — one model fit followed by $O(mn)$ stability bound evaluations — making adoption practical without specialized infrastructure.

---

## Weaknesses

### Fatal
None.

### Major

- **Figure 4 contradicts the paper's stated conclusions about screening power.** The paper text (p. 9) asserts: *"Compared to cFBH, our method is more powerful, due to improved exploitation of available data for prediction."* Yet the extracted Figure 4 caption reads: *"cFBH (green) consistently shows lower FDP and higher power compared to RO-cFBH (orange) and LOO-cFBH (blue)."* These two statements are mutually exclusive — either LOO-cFBH or cFBH achieves higher power. This contradiction appears in the paper's most novel empirical contribution (the screening application) and cannot be resolved without clarifying which description is accurate. If the figure caption reflects the actual figure and cFBH wins on power, the primary claimed benefit of LOO-cFBH in Section 6 collapses. The authors must either fix the caption or fix the claim, and if the results do not support improved power, the screening section's contribution reduces to the computational speed advantage of LOO-cFBH over RO-cFBH.

- **FullCP is run with one-third the SGD epochs, artificially weakening its accuracy.** Section 4 states: *"Throughout, we ran SGD for R = 15 epochs for all methods, except R = 5 for the very slow FullCP."* Since SGD under FullCP uses a less trained model, interval-length comparisons between FullCP and LOO-StabCP in the SGD columns of Figures 1–2 partly reflect model quality differences rather than conformal method quality. The paper's claim that LOO-StabCP achieves accuracy "comparable to FullCP" under SGD is therefore overstated — it is comparable to a deliberately undertrained FullCP. A fair comparison requires equal training budgets or an explicit disclaimer that accuracy claims for FullCP under SGD are not directly comparable.

### Minor

- **No formal coverage proof for LOO conformal p-values (Eq. 7) in the screening application.** Theorem 1 establishes coverage for prediction intervals. Section 6 uses a different one-sided score ($S(y,z) = y - z$) and claims FDR control via the BH procedure. For FDR control to be valid, $p_j^{\text{LOO}}$ must be a valid conformal p-value (i.e., $\mathbb{P}(p_j^{\text{LOO}} \leq u) \leq u$ under $H_{0j}$). This requires its own formal argument; the paper asserts FDR validity without proving it. While the extension from intervals to p-values is conceptually natural, the omission of a formal statement is a gap in the theoretical foundation of Section 6.

- **Bagging result (Theorem 5) has no empirical validation.** Theorem 5 is the only stability result with no accompanying experiment — not even a small example using a random forest on the Boston or Diabetes datasets. The claim that bagging satisfies LOO stability is interesting but unverified empirically.

- **Neural network coverage relies on an unjustified heuristic.** Theorem 4 provides vacuous bounds in practice (as the paper correctly acknowledges, $\kappa = \prod_{i=1}^n(1 + \eta\varphi_i) \gg 1$). The actual neural network experiments use $\tau_{i,j}^{\text{LOO}} \approx R\eta \cdot \gamma\|X_i\|\|X_{n+j}\|$, which is derived from linear model analysis. The paper labels this "practical guidance" and honestly calls it a limitation, but the NN experiments are presented as evidence of robustness without a valid coverage guarantee. This distinction between empirical observation and formal guarantee should be stated more prominently.

- **Regime analysis missing for RLM bounds (Theorem 2).** For RLM, whether $\tau^{\text{LOO}} < \tau^{\text{RO}}$ depends on whether $\rho_{n+j} \geq \bar{\rho}$ (compare Eq. 4). The paper discusses only the SGD case where the LOO bound is always tighter by a factor of 2. For RLM, no discussion is provided of when the LOO bound may actually be looser than the RO bound, nor whether the experimental results reflect a setting where $\rho_{n+j} \approx \bar{\rho}$ throughout.

### Trivial

- The claim in the abstract and Section 6 that LOO-cFBH has "improved test power compared to state-of-the-art method based on split conformal" should be stated conditionally until the Figure 4 contradiction is resolved.

---

## Nice-to-Haves

- A systematic sweep over $m \in \{1, 10, 100, 500\}$ would more clearly characterize when LOO-StabCP's speed advantage materializes over SplitCP and when it becomes negligible.
- A random forest experiment (even in the appendix) would significantly strengthen the case for Theorem 5.
- Connecting the non-convex SGD stability result (Theorem 4) to recent uniform stability work on smooth non-convex objectives could narrow the theory–practice gap for neural networks.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Point: "Critical Issue #2 — Unfair accuracy comparison against FullCP"** as a *fatal* flaw. Demoted to Major: The paper does not claim to beat FullCP; it claims to be competitive. Under RLM, FullCP is run at the same budget. The SGD epochs issue is a real concern (retained as Major) but does not invalidate the paper's core contribution about computational efficiency.

- **Harsh Critic's characterization of Figure 4 as "unresolvable without the actual figure"** — The contradiction between text and caption is real and kept as a Major weakness, but the paper's Section 6 text and the Strength Finder's consistent reading suggest the figure likely matches the paper text. However, the caption directly contradicts this, so it must be resolved.

- **Harsh Critic Point: Neural network coverage guarantee as a "structural" flaw** — The paper explicitly acknowledges that the practical bounds are heuristic, calls this a future work direction, and cites prior work (Hardt et al., 2016; Ndiaye, 2022) for the heuristic. It is a limitation, not an undisclosed methodological error. Retained as Minor.

- **Strength Finder Strength: "LOO-cFBH achieves higher test power than cFBH"** — Conflicted with Figure 4 caption. Retained conditionally as a potential strength pending resolution of the contradiction, but not listed as a confirmed strength above.

---

## Novel Insights

The factor-of-2 tightness gap between LOO and RO stability for SGD — stemming from the observation that leaving out a training point *removes* one gradient step (bounded perturbation) while replacing a point *reverses* a gradient step direction in the worst case (doubling the perturbation) — is a genuinely insightful observation that illuminates why the type of stability used in conformal inference has concrete implications beyond computational convenience. This result is not merely an incremental refinement; it reveals that the definition of stability determines the interval width in a predictable, mechanistically-justified way. The extension to conformalized screening is also a natural but valuable application that had not been studied with stability-based conformal methods.

---

## Suggestions

1. **Resolve the Figure 4 contradiction immediately.** Either fix the figure caption to match the text or rerun experiments and update the claim. This is the most urgent revision needed.
2. **Either equalize SGD epochs for FullCP or explicitly disclaim the comparison.** Adding a parenthetical like "FullCP uses R=5 due to computational constraints; accuracy comparisons under SGD are approximate" would substantially increase transparency.
3. **Add Proposition/Corollary stating that $p_j^{\text{LOO}}$ is a valid super-uniform p-value** under $H_{0j}$ in the screening setting. The argument likely follows from Theorem 1 with minor adaptation.
4. **Include at least one random forest experiment** to empirically validate Theorem 5.
5. **Clarify when the LOO RLM bound is tighter than the RO bound** (i.e., when $\rho_{n+j} \gtrless \bar{\rho}$) in a remark following Theorem 2.

---

## Score and Decision

**Calibration anchors:**

- `vcX0k4rGTt` (*Approx Full CP for NNs via Gauss-Newton influence*, accepted poster, scores 6/8/5/6, avg 6.25): Most topically similar — also avoids refitting for full CP while maintaining coverage. That paper was narrower in scope (only NNs, single approach) but arguably more elegant in its linearization. No comparable figure/text contradiction issues.
- `33XGfHLtZg` (*Conformal Risk Control*, accepted spotlight, avg 7.3): Broader and more impactful generalization, cleaner experimental narrative, no contradictions.
- `W6hzM9DMMU` (*Bayesian Online CP*, rejected, avg 6.0, despite some 8s): Shows that theoretical novelty alone doesn't guarantee acceptance when reviewers disagree on significance.
- `0V311Uh8q1` (*Algorithmic Stability Unleashed*, rejected, avg 4.75): Stability-focused, narrower contribution, weaker experiments.

**Positioning:** The paper under review sits slightly above `vcX0k4rGTt` in theoretical breadth (five algorithm types vs. one) and has a real computational contribution, but has the Figure 4 contradiction (a genuine unresolved empirical claim in the most novel application), the FullCP epoch asymmetry, and missing formal validity for the p-values. Anchored by the cluster around 5.5–6.25 from similar conformal prediction papers, with the Figure 4 issue pulling slightly below the accepted poster at 6.25.

**Final score: 5.5 — Borderline Reject.** The theoretical contributions are real and the computational improvement is valuable, but the Figure 4 contradiction is a significant unresolved issue in the paper's key application, and the SGD epoch asymmetry undermines the most-cited accuracy evidence. With a corrected experimental narrative these issues could support acceptance, but in the current state the paper needs revision.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>