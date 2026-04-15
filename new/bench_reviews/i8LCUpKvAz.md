Now let me read the calibration papers to calibrate the score.Now I have enough information to write the consolidated review. Let me synthesize all the evidence.## Summary
EQO (Exploration via Quasi-Optimism) is a tabular RL algorithm for the finite-horizon episodic setting that achieves minimax-optimal regret using a simple exploration bonus proportional to $c/N(s,a)$, deliberately avoiding empirical-variance-based bonuses used by all prior minimax-optimal methods. The central theoretical innovation is the concept of "quasi-optimism": the estimated value function need not be fully optimistic, but its underestimation is controlled through a bounded additive correction, which is shown to suffice for optimal regret via a novel variance-decomposition argument. The paper also establishes PAC bounds and best-policy identification guarantees matching known lower bounds.

---

## Strengths

- **Algorithmically simple and principled.** The $c/N(s,a)$ bonus is arguably the most natural count-based bonus possible, and the fact that it achieves minimax optimality without empirical variances is a non-trivial and surprising result. Prior understanding held that Bernstein-type empirical-variance terms were necessary.

- **Tightest regret bound in the literature.** Theorem 1 achieves $\tilde{O}(H\sqrt{SAK} + HS^2A)$ with logarithmic factors $\mathcal{O}(\sqrt{\log(HSA/\delta)\log(KH)})$ that are strictly tighter than the state-of-the-art Zhang et al. (2021a) in the time-homogeneous setting. The non-leading term $O(HS^2A)$ also matches Zhang et al. at this level.

- **Genuinely weaker assumptions.** Assumption 1 requires bounded optimal *value functions* ($V_h^*(s) \in [0,H]$) plus per-step rewards bounded by $H$, rather than bounded realized *returns* ($\sum_h R_h \le H$). Since the bounded return assumption constrains all realized trajectories while bounded value only constrains the expected return, the inclusion is strict: bounded return implies bounded value but not vice versa. The paper's claim on this point is correct and supported by the discussion in Section 4.1.

- **Novel analytical technique.** The quasi-optimism framework (Lemma 2) and the form of Freedman's inequality used in Lemma 1—which isolates the variance and $1/N$ terms rather than mixing them in a $\sqrt{\text{Var}/N}$ term—are conceptually novel. The proof strategy of bounding $\sum_j \text{Var}(V_{j+1}^*)$ through a difference-type variance inequality (Lemma 27) without requiring bounded returns is a technical contribution of independent interest.

- **Complete theoretical package.** In addition to regret bounds, the paper provides mistake-style PAC bounds and best-policy identification bounds (Theorems 3–4) matching known lower bounds for small $\varepsilon$, demonstrating robustness of the approach.

- **Favorable empirical results.** On RiverSwim, EQO consistently outperforms UCRL2, UCBVI-BF, EULER, ORLC, and MVP in cumulative regret, and achieves lower execution time (Appendix G), consistent with the theoretical computational savings from avoiding variance computation.

---

## Weaknesses

### Fatal
None.

### Major

- **Experimental evaluation is too thin for the practical-superiority claims.** The paper asserts in the abstract and introduction that EQO "consistently outperforms existing algorithms in both regret performance and computational efficiency" and achieves "the best of both theoretical soundness and practical effectiveness." This conclusion rests solely on two RiverSwim configurations ($S=30,H=120$ and $S=40,H=160$). RiverSwim is a single hard-exploration benchmark with a specific chain structure that naturally favors simple count-based exploration. No error bars, no confidence intervals, and no statement of how many random seeds were used appear anywhere. No experiment on alternative MDP structures (random MDPs, sparse reward, chain MDPs with different topology) is provided. The gap between the experimental evidence and the strength of the practical claims is large enough to be misleading.

- **Notation errors undermine confidence in the presentation.** Two errors are visible without consulting the appendix:
  1. The regret of a policy $\pi$ is defined as $V_1^\pi(s_1) - V_1^*(s_1)$ (Section 2.1), which is $\le 0$ for any sub-optimal policy—the opposite sign of standard convention. The theorems bound Regret$(K)$ by positive quantities, so the analysis internally uses the correct direction, but the stated definition is wrong and may confuse readers.
  2. Proposition 1 writes $\sum_{k=1}^K \text{Regret}(K) \le \ldots$, but Regret$(K)$ is itself already the cumulative sum $\sum_{k=1}^K(\cdots)$. This double-sum notation is either a typo or a serious notational inconsistency in a central proposition.

### Minor

- **$c_k$ requires knowledge of $K$ in Theorem 1.** The constant $c = \max\{7H\ell_1, 1.4H\sqrt{K\ell_1/(SA\ell_{2,K})}\}$ depends on the total episode budget. The paper addresses this with an anytime result (Theorem 2) via a doubling-trick-style $c_k$ schedule, which reintroduces parameter complexity and slightly worse logarithmic factors. The practical "single parameter" simplicity claim thus applies to the known-$K$ setting; it should be stated more carefully.

- **No comparison with Tiapkin et al. (2022).** The related work acknowledges that Tiapkin et al. achieve the minimax bound without empirical variances via posterior sampling. This is the most directly comparable alternative approach, yet no theoretical table entry and no empirical comparison with it are provided. Even a theoretical comparison table row would be informative.

- **Large explicit constants.** The regret bound carries explicit constants of 38 and 256. For moderate $K$ (say $K \sim S^2A$ to $S^3A$), the non-leading term may dominate in practice. The paper claims "sharpest" bounds based on logarithmic factors but does not discuss where these constants sit relative to prior work in the same regime.

### Trivial

- Minor notational sloppiness: $Q_h(s,a)$ in Section 2.1 omits the policy superscript while being described as the action-value function of $\pi$.

---

## Nice-to-Haves

- **Additional environments.** Running on 2–3 other MDP types (random tabular MDPs, Deep Sea, SixArms, structured sparse-reward MDPs) would substantially substantiate the practical-superiority claim.

- **Sensitivity analysis for $c$.** A plot showing how cumulative regret changes when $c$ is multiplied/divided by a constant factor would clarify the robustness of the single-parameter advantage.

- **Ablation isolating bonus form.** Comparing $c/N$ against $\sqrt{\text{Var}/N}$ (Bernstein) and $1/\sqrt{N}$ (Hoeffding) while holding all other algorithm components fixed would isolate whether the empirical gain comes from the bonus form specifically.

- **Visualization of quasi-optimism dynamics.** Plotting $V_h^k(s) - V_h^*(s)$ over episodes would directly illustrate when underestimation occurs and how quickly it is corrected.

- **Statistical rigor in experiments.** Reporting means ± standard deviations over multiple seeds, and specifying the number of seeds used.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

**Harsh Critic: "Weaker boundedness assumption claim is unsupported and internally inconsistent" (Structural flaw #1)**
*Removed because the paper's claim is correct.* The paper's Assumption 1 bounds the optimal *value function* $V_h^*(s) \in [0,H]$ plus per-step rewards $R_h^k \in [0,H]$. Prior "bounded return" work (Zhang et al. 2021a, Zanette & Brunskill 2019) requires $\sum_h R_h \le H$ for every realized trajectory—a strictly stronger constraint on the realized randomness. Bounded return implies bounded value ($\mathbb{E}[\sum_h R_h] \le H$ trivially), but bounded value does not imply bounded return (the realized sum can exceed $H$ under the paper's assumption). The paper's inclusion ordering in Section 4.1 is therefore correct, and the harsh critic's objection that "per-step rewards may be as large as $H$" applies equally to both the paper's assumption and the bounded-return assumption (since if $\sum_h R_h \le H$ with $H$ terms, individual terms can still reach $H$). This criticism reflects a misreading of Section 4.1.

**Harsh Critic: "Empirical comparison may be unfair due to unspecified tuning protocols" (Methodological gap #3)**
*Removed as speculative.* The criticism assumes that baselines were not tuned comparably without evidence. The paper uses the same RiverSwim environment with published baselines, and tuning details for commonly studied tabular algorithms are standard in the literature. This does not rise to a substantive fairness concern.

**Human Finder: "Comparisons with more recent baselines (Ishfaq et al. 2024a/b)"**
*Removed per hard rule against citing missing related works.* Existence of specific recent papers cannot be verified; no external sources are available.

---

## Novel Insights

The paper's most genuinely novel contribution is methodological: it shows that the empirical-variance-based bonus—long considered essential to both the algorithmic design and the proof strategy of minimax-optimal tabular RL—can be replaced by a pure count-based bonus, provided the optimism requirement is relaxed to "quasi-optimism." The key enabling insight is the factored form of Freedman's inequality (Lemma 1), which separates the variance and $1/N$ contributions rather than coupling them in a $\sqrt{\text{Var}/N}$ term. This separation allows the $1/N$ piece to be absorbed by the bonus while the variance piece is bounded via a telescoping argument over the optimal value function's squared terms (Lemma 27). The resulting proof technique—bounding $\mathbb{E}[\sum_j \text{Var}(V_{j+1}^*)]$ without any bounded-return condition—may be of independent use in other regret analyses where assumption relaxation is desired.

---

## Suggestions

1. Fix the sign of the regret definition in Section 2.1 (should be $V_1^*(s_1) - V_1^\pi(s_1)$) and resolve the double-sum notation in Proposition 1.
2. Add at least 2 additional MDP environments to the experiments; report mean ± std over ≥5 seeds; move computational timing from appendix to main text.
3. Add a theoretical table row for Tiapkin et al. (2022), which is the only other prior work to achieve minimax bounds without empirical variances.
4. Discuss the regime (in terms of $K$ relative to $S,A,H$) where the non-leading term dominates, and compare the explicit constants 38 and 256 against those in Zhang et al. (2021a).
5. Qualify the abstract and introduction claims ("consistently outperforms," "best of both") to reflect the evidence available (RiverSwim results), or expand experiments to support the stronger claim.

---

## Score and Decision

**Calibration:**
- *txD9llAYn9* (Model-based RL, horizon-free + second-order bounds, Accept): Scores 6,8,8,6. A strong theory paper with broad results but limited empirical evaluation. The paper under review has comparable or slightly narrower scope but sharper algorithmic novelty (the simple bonus) and a cleaner proof concept.
- *en3NwykrHW* (Minimax RL with trajectory feedback, Reject): Scores 6,3,5,5,8,6. Rejected partly because of notation/polish issues and concerns about term dominance. The paper under review shares some of these issues but has a stronger algorithmic contribution.
- *h6ktwCPYxE* (Second-order bounds for contextual bandits, Accept): Scores 5,8,6,5. A theory paper valued for novelty of technique. Similar profile.

The paper under review sits between these anchors. The theoretical contribution (quasi-optimism, tightest known logarithmic factors, weakest assumptions) is solid and well-motivated; the quasi-optimism technique is genuinely new. The main deductions are the thin experimental section relative to the practical claims and two notation errors in a central proposition. Relative to *txD9llAYn9* (which was accepted with 6s and 8s), this paper is comparable in quality but makes more practical claims that its experiments cannot fully support, and has visible presentational errors. A score of **7.0** reflects a paper that makes a real theoretical contribution, is above the acceptance threshold, but requires revision to fix notation and strengthen the experimental section before camera-ready.

**Originality:** Strong — $c/N$ bonus for minimax-optimal RL is new, quasi-optimism is a novel analytical framework.
**Importance:** Good — addresses a real tension between theory and practice in tabular RL; the simplification is non-trivial.
**Claim support:** Mostly good for theory; overclaims on the empirical side.
**Experimental soundness:** Weak — single environment, no error bars, insufficient for broad practical claims.
**Clarity:** Mixed — proof sketch is unusually clear, but regret sign error and Proposition 1 notation are concerning.
**Community value:** Positive — the quasi-optimism technique may benefit future algorithm design beyond tabular settings.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>