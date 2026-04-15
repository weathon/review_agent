## Summary

This paper presents C-SquareCB and C-FastCB, two algorithms for Conservative Contextual Bandits (CCBs) with general nonlinear cost functions. The core idea is a reduction to online regression (square-loss and KL-loss respectively) combined with Inverse Gap Weighting (IGW) exploration and a safety gate that checks a predicted-cost budget condition before committing to an IGW action. The key novelty in the analysis is bounding the number of baseline plays $n_T$ without UCB-style confidence sets, instead relating it to online regression loss. The paper then instantiates the framework with neural networks and OGD, achieving $\tilde{O}(\sqrt{KT}+K/\alpha)$ and $\tilde{O}(\sqrt{KL^*}+K(1+1/\alpha))$ end-to-end regret bounds.

---

## Claims and Support

**Claim 1: First conservative CB algorithms for general nonlinear functions via reduction to online regression.**
*Supported.* Algorithms 1 and 2 are clearly specified, Theorems 3.1 and 4.1 assert high-probability safety + regret guarantees. The proof sketch is coherent and the four-step decomposition is well-presented. One genuine gap: Algorithm 1's $\gamma_t = \sqrt{K|\mathcal{S}_t|/(\text{Reg}_{\text{sq}}(m_T)+\log(8/\delta))}$ depends on the (unknown a priori) quantity $m_T$, and Algorithm 2's $\gamma_t$-Schedule is deferred entirely to Appendix C. This is acknowledged in Remark 4.2 but not operationally resolved.

**Claim 2: C-SquareCB achieves sublinear regret reducing to oracle square-loss regret.**
*Supported with caveat.* Theorem 3.1 gives $\tilde{O}(\sqrt{KT\,\text{Reg}_{\text{sq}}(T)}+K\,\text{Reg}_{\text{sq}}(T)/\alpha)$. Sublinearity holds whenever the oracle has sublinear regret, which is the framework's explicit assumption. The abstract's phrasing is marginally unconditional but acceptable for a reduction paper.

**Claim 3: C-FastCB achieves a first-order regret bound scaling with $L^*$.**
*Partially supported.* Theorem 4.1 states the expected regret bound as $\tilde{O}(\sqrt{KL^*\,\text{Reg}_{\text{KL}}(T)}+\ldots)$. However, the statement mixes probability quantifiers: safety is with probability $1-\delta$, but the regret bound is in expectation over arm distributions. The episodic $\gamma_t$-schedule, which is the mechanism preventing $\sqrt{T}$-dependence, is entirely absent from the main paper. The paper explicitly says in Remark 4.2 "has been pushed to Appendix C, for clarity."

**Claim 4: Neural instantiation yields $\tilde{O}(\sqrt{KT}+K/\alpha)$ and $\tilde{O}(\sqrt{KL^*}+K(1+1/\alpha))$ bounds.**
*Supported at the theorem level*, contingent on strong NTK assumptions (Assumptions 5–6). The paper correctly notes in Section 5 that these assumptions are standard in prior neural bandit work (Zhou et al., 2020; Zhang et al., 2021). The end-to-end bounds follow from combining oracle regret $O(\log T)$ from Deb et al. (2024a) with Theorems 3.1 and 4.1.

**Claim 5: Algorithms significantly outperform existing conservative baselines while maintaining the guarantee.**
*Weakly supported.* Figure 1 shows lower regret than C-LinUCB. Figure 2 shows lower constraint-violation percentage than vanilla (non-conservative) counterparts. The comparison set is very thin (one linear baseline), and the safety evaluation does not directly plot the pathwise budget process from Definition 2.2.

---

## Strengths

- **Novel technical contribution: bounding $n_T$ without confidence sets.** Remark 3.3 explains clearly that prior conservative CB analyses (Kazerouni et al., 2017) rely on UCB-style confidence bounds that are unavailable for general function classes. The paper relates $n_T$ to the regression oracle's loss via the predicted-cost safety gate, which is genuinely novel. Lemmas 3.2–3.4 formalize this with a clean decomposition.

- **First-order conservative extension.** The adaptation from SquareCB to FastCB in the conservative setting requires an episodic $\gamma_t$ schedule that avoids $\sqrt{T}$-dependence while still controlling $n_T$. This is a meaningful technical extension even if the schedule details are in the appendix.

- **Modular and broadly applicable reduction framework.** By expressing regret in terms of oracle regret (Assumptions 3 and 4), the results benefit automatically from any future improvement in online regression. The neural instantiation is a concrete and end-to-end example.

- **Practical safety mechanism design.** The key choice to use predicted cumulative costs (rather than observed costs) in the safety gate (Equation 4) is well motivated: the paper explains in Section 3 that using observed costs does not allow controlling $n_T$, which is necessary for sublinear regret. This trade-off is explained clearly.

- **Reasonable experimental scope.** Evaluation on six real OpenML datasets with 10–100 runs, width ablations, and an alternative heuristic $\gamma_t$ based on observed losses (Figure 3, Appendix F) shows adequate empirical care.

---

## Weaknesses

### Fatal
*None. The core claims are supported and the flaws are fixable.*

### Major

- **Safety evaluation does not match the formal guarantee (Critical).** The paper's defining contribution is the performance constraint in Definition 2.2: the pathwise cumulative cost inequality $\sum_{i=1}^t h(\mathbf{x}_{i,a_i}) \leq (1+\alpha)\sum_{i=1}^t h(\mathbf{x}_{i,b_i})$ for every $t$. Figure 2 instead reports "percentage of constraints violated" when the problem is perturbed by a noise parameter $\epsilon$, comparing conservative vs. non-conservative variants. This does not demonstrate the actual safety mechanism working—it shows relative comparison to unconstrained baselines, not that the proposed methods satisfy the cumulative pathwise constraint. A plot of the cumulative cost ratio over time for a typical run would directly validate the theorem's guarantee.

- **$\gamma_t$ depends on quantities unavailable online.** In Algorithm 1 (Theorem 3.1), $\gamma_t = \sqrt{K|\mathcal{S}_t|/(\text{Reg}_{\text{sq}}(m_T)+\log(8/\delta))}$ requires $m_T$ (total IGW plays at end of horizon) and an upper bound on $\text{Reg}_{\text{sq}}(m_T)$. In practice (Section 6), the authors substitute $\gamma_i = c\sqrt{t/\log(\delta^{-1})}$ and tune $c$, so the formal guarantee of Theorem 3.1 does not apply to the implementation. For C-FastCB, $L^*$ is unknown and $\gamma$ is treated as a tunable hyperparameter entirely outside the theory. This gap between theory and implementation is real and unaddressed; the paper should explicitly discuss whether a doubling-trick or adaptive schedule recovers the same guarantees.

- **Experimental baseline too weak for the paper's empirical claims.** The only conservative comparator is C-LinUCB, a linear method on deliberately nonlinear tasks. Beating a misspecified linear model does not isolate the benefit of the conservative mechanism, the IGW approach, or nonlinear function approximation. A heuristic conservative NeuralUCB or a non-conservative SquareCB/FastCB in the regret plots would substantially sharpen the empirical story.

### Minor

- **Algorithm 2 apparent notation error in safety condition.** Line 9 of Algorithm 2 reads $\sum_{i\in\mathcal{S}_{t-1}}\sum_{a\in[K]} p_{t,a}\hat{y}_{t,a}$, where the outer sum is over past rounds $i$ but the inner quantities ($p_{t,a}$, $\hat{y}_{t,a}$) use the current-round index $t$. Comparing with Equation (4) for Algorithm 1, which correctly uses $p_{i,a}\hat{y}_{i,a}$, this is a typo: it should be $p_{i,a}\hat{y}_{i,a}$. The neutral reviewer independently noted this.

- **C-FastCB regret is in expectation while safety is high-probability.** Theorem 4.1 states "with probability $1-\delta$... has the following bound on the **expected** regret." This mixing of probability quantifiers is not wrong, but it is somewhat confusing: readers may expect the regret bound to also be high-probability, as in Theorem 3.1. The abstract and introduction do not flag this asymmetry.

- **Assumption 2 (baseline gap lower bound $\Delta_l>0$) is restrictive.** The bound in Term II of Theorem 3.1 blows up as $\Delta_l\to 0$, covering cases where the baseline is near-optimal. While the paper correctly notes this is standard in the conservative bandit literature (matching Kazerouni et al., 2017), it deserves acknowledgment that the result may be weak precisely when conservative exploration is most needed—i.e., when the baseline is already very good.

- **No ablation on $\alpha$.** The conservatism parameter $\alpha$ appears directly in the regret bound's $K/\alpha$ term and in the safety condition. No experiment varies $\alpha$ to validate this dependence or illustrate the practical safety–regret tradeoff.

- **NTK assumptions limit generality of neural instantiation.** Assumptions 5 and 6 (Gaussian initialization, uniform positive-definite NTK) are acknowledged as standard, but the framing "general nonlinear functions" in the abstract overstates the scope. The neural results hold in a specific overparameterized NTK regime under realizability—not for arbitrary nonlinearity.

### Trivial

- **$\Delta_t$ vs. $\Delta_l$ in Theorem 5.1/5.2.** The regret expression uses $\Delta_t$ (time-indexed) where $\Delta_l$ (lower bound, constant) is likely intended. Minor notation inconsistency.

---

## Nice-to-Haves

- Provide a doubling-trick or adaptive version of $\gamma_t$ that does not require knowing $\text{Reg}_{\text{sq}}(T)$ or $L^*$ a priori, and re-prove guarantees under it. This would close the most significant theory–practice gap.
- Plot the cumulative budget ratio $\sum_{i\le t}h(\mathbf{x}_{i,a_i})\big/\big((1+\alpha)\sum_{i\le t}h(\mathbf{x}_{i,b_i})\big)$ over time for a single run to directly illustrate the safety mechanism in action.
- Plot fraction of rounds where the baseline action is played over time (connects $n_T$ theory to practice).
- Discuss what happens when $h\in\mathcal{H}$ is not realizable by the chosen neural network width.
- Discussion of large-/infinite-action settings beyond Remark 3.2's brief pointer.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh critic: "algorithms are partly nonconstructive."** The critic claims that needing $\text{Reg}_{\text{sq}}(T)$ makes the algorithms non-implementable. However, (a) treating oracle regret as a known upper bound is a standard convention in reduction-based bandit papers, (b) the paper provides an explicit practical substitute in Section 6 ($\gamma_i = c\sqrt{t/\log\delta^{-1}}$), and (c) Assumption 3 explicitly assumes the oracle regret is bounded by $\text{Reg}_{\text{sq}}(T)$. The concern is legitimate (kept as Major weakness) but the harsh critic overstates it as making the theory "false" or "nonconstructive." REMOVED the fatal framing; retained as a major weakness.

- **Harsh critic: "the high-probability safety guarantee is not fully checkable from the main text because key schedules are missing."** The paper explicitly states in Remark 4.2 that the $\gamma_t$-schedule has been "pushed to Appendix C, for clarity." For C-SquareCB, the schedule ($\gamma_t=\sqrt{K|\mathcal{S}_t|/(\text{Reg}_{\text{sq}}(m_T)+\log(8/\delta))}$) is stated in the theorem. This is a stylistic deferral, not a missing argument. REMOVED the "not verifiable" framing; noted as a weakness in presentation.

- **Harsh critic: "the experimental baseline is 'too weak to support the broad empirical claim.'"** The harsh critic characterizes this as "decision-grade evidence" failure. Kept as a major weakness but REMOVED the catastrophizing; the comparison is legitimately limited but does demonstrate the basic efficacy of the approach.

- **Harsh critic: "Claim 2 overstates sublinearity in T."** The abstract says "sub-linear in $T$" for C-SquareCB, which is conditional on oracle regret being sublinear (Assumption 3). This is entirely standard framing in oracle-based work and not an overstatement. REMOVED.

- **Human finder: "missing related works" and citing reviewer comparisons to other papers.** Per hard rules, removed as potential fabrications without independent verification.

- **Human finder: "$\tilde{O}$ notation hides important dependencies."** The paper's use of $\tilde{O}$ is standard, and the actual $\log$ factors are clear from context (e.g., $\text{Reg}_{\text{sq}}(T) = O(\log T)$ for the neural case). REMOVED as a formatting nitpick.

---

## Novel Insights

The most genuinely novel technical insight in this paper is the use of **predicted cumulative costs rather than observed costs** in the safety gate (Equation 4), and the subsequent analysis showing that this enables bounding $n_T$ through the oracle's squared regression loss rather than through UCB-style confidence ellipsoids. This is a non-trivial departure from prior conservative bandit analysis: it avoids the need for confidence sets around function estimates (which are intractable for general nonlinear classes) and instead leverages the structural properties of online regression. The complementary insight—that an episodic $\gamma_t$ schedule with $O(\log L^*)$ episodes can preserve first-order scaling in the conservative setting—extends the Foster–Krishnamurthy (2021) analysis in a meaningful way by simultaneously controlling both IGW regret and conservative overhead without resorting to $\sqrt{T}$-type bounds.

---

## Suggestions

1. **Fix the theory–practice gap for $\gamma_t$:** Add a theorem (even in the appendix) showing that a doubling-based or estimate-based adaptive $\gamma_t$ achieves the same regret order. Explicitly state in Section 6 that the practical $\gamma$ does not satisfy the conditions of Theorems 3.1/4.1, and discuss the implications.
2. **Correct the Algorithm 2 safety condition:** Change $p_{t,a}\hat{y}_{t,a}$ to $p_{i,a}\hat{y}_{i,a}$ in the inner sum to match the intended semantics and align with Equation (4).
3. **Add a direct safety validation plot:** Show the cumulative constraint slack $\sum_{i\le t}h(\mathbf{x}_{i,b_i})(1+\alpha)-\sum_{i\le t}h(\mathbf{x}_{i,a_i})$ over time, at minimum for one representative dataset. This is the experiment that most directly validates the paper's central safety claim.
4. **Strengthen experimental baselines:** Include non-conservative SquareCB/FastCB in the regret plots (to show the cost of conservatism) and a conservative heuristic based on NeuralUCB (to show the benefit of the IGW approach over the obvious alternative).
5. **Vary $\alpha$:** A regret-vs-$\alpha$ curve would validate the theoretical $K/\alpha$ dependence and clearly illustrate the practical safety–regret frontier.
6. **Clarify the quantifier mismatch in Theorem 4.1:** State explicitly that safety holds with probability $1-\delta$ and regret is bounded in expectation, and explain why this asymmetry arises.

---

## Score and Decision

**Originality:** Good. The paper is the first to extend conservative contextual bandits to general nonlinear function classes via a clean oracle reduction. The key technical ideas (predicted-cost safety gate, time-dependent IGW for conservatism) are novel extensions of prior work.

**Importance:** Moderate-to-good. Conservative bandits with nonlinear cost functions is a practically relevant problem. The reduction framework is modular and extensible.

**Claims vs. support:** Mostly well-supported at the theorem level, with a real gap between the theoretical $\gamma_t$ and the implemented version. The formal safety guarantee is not directly empirically validated.

**Soundness of experiments:** Weak. A single conservative baseline (C-LinUCB), indirect safety validation, no ablation on $\alpha$.

**Clarity:** Good. The proof structure is clear and well-organized. The deferral of the $\gamma_t$-schedule to the appendix for C-FastCB is a mild clarity issue.

**Value to research community:** Solid. Provides a complete framework—theory, neural instantiation, and experiments—that fills a genuine gap in the conservative bandit literature.

The paper makes a genuine theoretical contribution and should be accepted, but the experimental evaluation needs strengthening and the theory–practice gap for $\gamma_t$ should be explicitly acknowledged.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>