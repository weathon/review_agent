Now let me do calibration searches to properly score this paper.Now I have enough information to write the final review. Let me synthesize everything I've found from reading the paper and the reviewer inputs.

---

## Summary

This paper extends Conservative Contextual Bandits (CCBs) beyond the linear setting by proposing two IGW-based algorithms—**C-SquareCB** and **C-FastCB**—that reduce the CCB problem to online regression oracles (squared-loss and KL-loss, respectively) while guaranteeing the safety constraint with high probability. C-SquareCB achieves $\tilde{O}(\sqrt{KT})$ regret and C-FastCB a first-order $\tilde{O}(\sqrt{KL^*})$ bound. Both are instantiated with neural networks and OGD to give end-to-end bounds, and evaluated on six OpenML classification datasets.

---

## Strengths

- **Novel oracle-reduction framework for CCBs** (Theorem 3.1, Section 3): Prior conservative bandit work (Kazerouni et al., 2017; Wu et al., 2016) relied on UCB-style confidence ellipsoids, which are intractable for general function classes. Replacing these with a squared-loss-based safety condition (Eq. 4) and bounding $n_T$ via regression regret (Lemmas 3.2–3.3) is the paper's central technical innovation and is cleanly executed.

- **Non-trivial bounding of $n_T$ via squared loss** (Lemmas 3.2–3.3): In the linear setting, confidence bounds around parameter estimates control baseline play. For general functions, the paper relates $n_T$ directly to the regression oracle's squared-loss regret, a novel argument noted in Remark 3.3.

- **Time-varying $\gamma_t$ analysis extending Foster & Rakhlin (2020)** (Lemma 3.4, Remark 3.4): The need to simultaneously bound the IGW regret and $n_T$ requires a carefully chosen adaptive $\gamma_t \propto \sqrt{|\mathcal{S}_t|}$, which extends the fixed-$\gamma$ analysis of prior IGW work and may be of independent interest.

- **First-order regret for C-FastCB** (Theorem 4.1): Achieving a bound that scales with $\sqrt{L^*}$ rather than $\sqrt{T}$ in the conservative setting requires an episodic $\gamma_t$-schedule (Remark 4.2), adding a $\sqrt{\log L^*}$ factor over the unconstrained case—a reasonable price.

- **Figure 2 directly validates the safety mechanism**: Comparing C-SquareCB/C-FastCB against their vanilla (non-conservative) counterparts on constraint violation rates (below 2% vs. up to 25%) provides a clean, informative test of the paper's core safety claim.

---

## Weaknesses

### Fatal
None.

### Major

- **Multiple errors in the stated theorems and algorithm specifications**: Three distinct issues in Sections 4–5 compromise the paper's published technical statements:
  1. *Algorithm 2, line 9 (safety condition)*: The inner sum $\sum_{i \in \mathcal{S}_{t-1}} \sum_{a \in [K]} p_{t,a} \hat{y}_{t,a}$ uses the *current-round* subscript $t$ on $p_{t,a}$ and $\hat{y}_{t,a}$, whereas the analogous condition in Algorithm 1 / Eq. (4) correctly uses $p_{i,a}\hat{y}_{i,a}$. As written, the term reduces to $|\mathcal{S}_{t-1}| \cdot \sum_a p_{t,a}\hat{y}_{t,a}$ — a scalar multiplied by set cardinality — which is semantically wrong. This is almost certainly a typo ($t \to i$), but it renders Algorithm 2 unexecutable as stated.
  2. *Theorem 5.1 self-reference*: The statement specifies "$\gamma_t$ **as in Theorem 5.1**" — a circular reference. The reader cannot determine the $\gamma_t$ schedule from the theorem. (The schedule from Theorem 3.1 can be inferred by cross-referencing, but the theorem as stated is incomplete.)
  3. *Theorem 5.2 wrong oracle*: The theorem says "We instantiate **Sq-Alg**…" but C-FastCB (Algorithm 2) uses **KL-Alg**, not Sq-Alg. The correct predictor (Eq. 18, the sigmoid ensemble) is described just above the theorem, but the theorem's instantiation statement names the wrong oracle.
  
  Together these mean that two of the four main results (Theorems 5.1 and 5.2) and the full specification of Algorithm 2 contain technical errors that must be corrected before the paper can be read without ambiguity.

- **$\Delta_t$ vs. $\Delta_l$ inconsistency in Theorems 5.1 and 5.2**: Both theorem statements write $\alpha y_l(\Delta_t + \alpha y_l)$ in the denominator of the conservative-play term, whereas Theorems 3.1 and 4.1 correctly use the constant $\Delta_l$ (from Assumption 2). The time-indexed $\Delta_t$ makes the bound ambiguous and inconsistent with the rest of the paper. While likely a typo, it appears in both neural-instantiation theorems.

### Minor

- **Experimental comparison does not quantify the price of safety**: Figure 1 compares only against C-LinUCB, which fails on nonlinear data due to model misspecification—not because of the conservative mechanism. The regret gap is driven by function-class mismatch. Comparing against unconstrained SquareCB/FastCB in Figure 1 would directly show how much regret the safety constraint costs, which is a central claim of the paper. Figure 2 partially addresses this for constraint violations but not for regret.

- **No ablation over $\alpha$**: The conservatism parameter $\alpha$ appears in every regret bound (Term II scales as $K / (\alpha y_l (\Delta_l + \alpha y_l))$) and is the paper's defining design parameter. No experiment varies $\alpha$ to confirm the theoretical dependence or guide practitioners.

- **C-FastCB's $\gamma_t$-schedule is not implemented in the experiments**: The paper explicitly uses a constant $\gamma$ in experiments (Section 6), while the first-order guarantee of Theorem 4.1 requires the episodic schedule from Appendix C. The claim that C-FastCB's empirical advantage is due to its first-order guarantee is therefore not directly supported.

### Trivial

- Minor inconsistency in $\gamma_t$-schedule reference: Theorem 4.1 says $\gamma_t$ is "chosen in ($\gamma_t$-Schedule)" without spelling out the schedule in the main text; Remark 4.2 gives the conceptual idea but no formula. For a result that is a key contribution, deferring the schedule entirely to the appendix weakens exposition.

---

## Nice-to-Haves

- An ablation varying $\alpha \in \{0.01, 0.05, 0.1, 0.5\}$ on at least one dataset, to validate the $O(K/\alpha^2)$ scaling in Term II and provide practitioners with guidance.
- A trajectory plot of the safety budget $\sum_{i=1}^t h(\mathbf{x}_{i,a_i}) / ((1+\alpha)\sum_{i=1}^t h(\mathbf{x}_{i,b_i}))$ over time to show how often the constraint is nearly tight.
- Discussion of the rate penalty incurred by the unbiased-estimate extension in Appendix E, since the known-baseline-cost assumption is nontrivial.
- A brief comparison of Term II's $O(K \log T / \alpha^2)$ scaling (for neural instantiation) against the $O(1/\alpha^2)$ term in Kazerouni et al. (2017) to make explicit the cost of generality.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "C-FastCB's outperformance falsely attributed to first-order optimality"** — Removed. The paper is transparent that $\gamma$ is tuned as a hyperparameter and does not claim the theoretical schedule is in use. The empirical C-FastCB benefit may arise from the KL-based weighting, not the first-order schedule per se; we can note this as a minor point rather than an invalidating criticism.
- **Harsh Critic: "Misspecification of $h(\mathbf{x}_{t,b_t})$ being known is too strong"** — The paper directly acknowledges this assumption and provides an Appendix E extension. This is standard in the conservative bandit literature (same assumption in Kazerouni et al., 2017). Removing as a weakness since the paper addresses it.
- **Harsh Critic: "Comparison against C-LinUCB is not a valid test at all"** — Weakened rather than fully removed. C-LinUCB is the only existing conservative bandit baseline for the general problem, making it a necessary comparison. The criticism is valid only insofar as a *complementary* comparison against unconstrained SquareCB/FastCB is missing; that is captured under Minor weaknesses.
- **Strength Finder: "Modular oracle-based design allows plugging in any regression algorithm"** — Removed as too generic; it restates the oracle-reduction framing without evidence beyond what Theorems 3.1 and 4.1 already provide.

---

## Novel Insights

The most technically interesting contribution is the use of *squared-loss regression regret* to upper-bound $n_T$, the number of conservative baseline plays (Lemmas 3.2–3.3). In prior linear CCB work, this quantity is controlled via confidence ellipsoids; here, the paper shows that the regression oracle's prediction quality (measured by squared loss) alone suffices to bound how often the safety condition forces the algorithm to revert to the baseline. This decouples the conservative mechanism from any confidence-set structure, enabling the extension to arbitrary function classes. The combination with a time-varying $\gamma_t$ (growing with $|\mathcal{S}_t|$) to simultaneously keep the IGW regret sublinear is an elegant technical contribution.

---

## Suggestions

1. **Fix Algorithm 2, line 9**: Replace $p_{t,a}$ and $\hat{y}_{t,a}$ with $p_{i,a}$ and $\hat{y}_{i,a}$ inside the sum over $i \in \mathcal{S}_{t-1}$ to match Algorithm 1's safety condition.
2. **Fix Theorem 5.1**: Remove the self-reference and explicitly state the $\gamma_t$ schedule (it follows from Theorem 3.1 as $\gamma_t = \sqrt{K|\mathcal{S}_t|/(\log T + \log(16\delta^{-1}))}$).
3. **Fix Theorem 5.2**: Replace "Sq-Alg" with "KL-Alg" and reference predictor (18) with the sigmoid ensemble.
4. **Fix $\Delta_t \to \Delta_l$** in Theorems 5.1 and 5.2.
5. **Add Figure 1 comparison against vanilla SquareCB/FastCB** to quantify the regret cost of the safety constraint.
6. **Add $\alpha$-ablation experiment** on at least one dataset.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Avg. Score | Decision |
|---|---|---|---|
| *Almost Optimal Batch-Regret Tradeoff for Batch Linear CBs* (`rakhNY32vw`) | Linear CB theory, optimal bounds | 7.0 | Accept (Poster) |
| *Second Order Bounds for CBs with Function Approximation* (`h6ktwCPYxE`) | CB theory, variance-aware bounds, some presentation issues | 6.0 | Accept (Poster) |
| *High Probability CBs for Optimal Dosage* (`z0B7A6Dh1H`) | Safe linear CBs, safety constraints | 6.0 | Reject |
| *Continuous Time MAB Regret* (`4jzjexvjI7`) | MAB theory, notation errors, unclear theorems | 2.3 | Reject |

This paper sits between the "accepted poster" cluster (6–7) and the "serious notation issues" cluster (2–3). Its theoretical contributions (oracle reduction, $n_T$ via squared loss, time-varying IGW analysis) are genuine and non-trivial — comparable in scope to `h6ktwCPYxE` (avg 6, accepted). However, unlike those accepted papers, this paper has *four* distinct errors in its stated theorems and algorithm specifications (Algorithm 2 wrong subscripts; Theorem 5.1 self-referential $\gamma_t$; Theorem 5.2 wrong oracle; $\Delta_t$ vs $\Delta_l$ in both neural theorems), plus a missing regret comparison for the safety-cost tradeoff and no $\alpha$ ablation. The errors are concentrated in Section 5 (the neural instantiation), which is the end-to-end theoretical showcase of the paper, making them harder to dismiss as cosmetic.

The underlying analysis (Sections 3–4) is sound as stated, and the paper's experimental Figure 2 provides a genuine safety validation. Relative to the anchor papers: it is clearly above the rejected low-quality papers (2–3 range) but below the clean high-quality theoretical papers (7–8 range). I place it at **5.5** — the theoretical core merits serious consideration but the presentation errors in the main theorems need correction before this work can be accepted.

**Originality:** Good — first oracle reduction for CCBs beyond linear functions.  
**Importance:** Moderate-high — CCBs are practically relevant and the general-function extension is meaningful.  
**Claim support:** Partially — Sections 3–4 are sound but Section 5 has multiple stated errors.  
**Experimental soundness:** Adequate for safety validation (Fig. 2); insufficient for quantifying the cost of conservatism.  
**Clarity:** Below standard for the main theorems and Algorithm 2 as published.  
**Value to community:** Real, once presentation issues are corrected.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>