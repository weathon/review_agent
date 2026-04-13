=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
The paper studies zeroth-order nonsmooth convex optimization and stochastic multi-armed bandits under **symmetric heavy-tailed noise** with bounded \(\kappa\)-th moment for **any \(\kappa>0\)**, including regimes where the noise may have unbounded expectation. Its main idea is to use a component-wise median over repeated two-point gradient-difference samples to build an estimator that remains unbiased and has bounded second moment, enabling high-probability convergence rates that, in the key Lipschitz-oracle setting, match bounded-variance zeroth-order rates and yield \(\tilde O(\sqrt{dT})\) regret for MAB.

This is a meaningful theoretical contribution: the paper identifies symmetry as the structural property that breaks the usual \(\kappa>1\) barrier and extends median-based robustness to zeroth-order optimization and bandits. However, the empirical side is materially weaker than the theoretical side, and several practical claims in the abstract/introduction are overstated relative to the evidence shown.

## Strengths
- **A nontrivial theoretical extension beyond the \(\kappa>1\) regime.** The paper explicitly targets the gap left by prior clipped zeroth-order methods whose rates deteriorate as \(\kappa\to 1\), and proves results for any \(\kappa>0\). This is central and substantive, not incremental.
- **The oracle/noise modeling is genuinely tailored to exploit symmetry.** Assumption 3 is not just a rephrasing of standard heavy-tail assumptions: it is formulated on the distribution of the two-point noise difference \(\phi(\xi\mid x,y)\), which is exactly what enables median-based analysis. The paper also clearly contrasts this with prior oracle assumptions in §3.1.1.
- **The key technical step is substantial:** Lemma 1 states that the medianized two-point estimator is unbiased and has bounded second moment despite potentially infinite first moment of the raw noise. If correct, this is the paper’s core conceptual advance and underlies the downstream optimization and MAB guarantees.
- **The main rates are theoretically significant in the intended regime.** In particular, Theorem 1 gives bounded-variance-style rates for the Lipschitz oracle, and Theorem 3 gives \(\tilde O(\sqrt{dT})\) regret for stochastic MAB under symmetric heavy-tailed noise, matching the bounded-variance lower-bound scaling.
- **The paper does a useful job delineating scope and limitations.** Section 6 explicitly acknowledges the symmetry assumption and the need to know \(\kappa\) to choose \(m\), which are real restrictions.

## Weaknesses

### Major:
- **The empirical evidence for the MAB contribution is not convincing, and Figure 1 appears inconsistent with the paper’s textual claim.**  
  In §5.1, the only bandit experiment uses just **2 arms**, which is far too narrow for a contribution whose theorem is dimension-dependent and claims \(\tilde O(\sqrt{dT})\) regret. More importantly, the text says: “HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does,” but the displayed figure/caption indicates that **HTINF attains a much higher probability of selecting the best arm** than the proposed method. Even allowing for some ambiguity between “expected regret” and “best-arm identification,” the figure as presented does not support the stated conclusion. Since MAB is one of the paper’s three headline contributions, this mismatch matters.
- **The real-world portfolio experiment does not validate the analyzed MAB setting.**  
  §5.2 explicitly states that this task has **full feedback** (“we observe the whole income vector for each asset”) and that the authors **adjust** their method accordingly. That is not the same problem analyzed in §4, which is a stochastic MAB / bandit-feedback setting. The baselines there (“hold ETH” and Efficient Frontier) are also not the relevant heavy-tailed bandit baselines. This section is acceptable as an application vignette, but it is not meaningful evidence for the paper’s MAB theorem.
- **The practical claims are broader than the experiments justify.**  
  The abstract says the methods “do not lose to SOTA approaches and dramatically outperform them for \(\kappa \le 1\).” The evidence shown in the main paper does not support such a broad claim. For bandits, there is only a 2-arm synthetic study and an inapposite full-feedback portfolio example. For zeroth-order optimization, the main text reports only **3 launches** in §5.3, which is not enough to characterize behavior under heavy-tailed stochasticity, especially when the theory emphasizes high-probability guarantees. The empirical section is suggestive, but not strong enough to support the paper’s strongest practical messaging.
- **The “matching bounded-variance rates” message is incomplete unless total oracle/sample cost is foregrounded.**  
  The paper’s high-level presentation emphasizes iteration complexity, but each iteration costs \((2m+1)b\) oracle calls in Theorem 1 and \((2m+1)\) in Theorem 2, with \(m=2/\kappa+1\). Moreover, Lemma 1’s variance proxy contains factors like \((4/\kappa)^{2/\kappa}\), which become enormous as \(\kappa\to 0\). So the paper’s improvement is real at the level of asymptotic iteration dependence on \(\varepsilon\), but the practical cost for very small \(\kappa\) can still be severe. This does not invalidate the theory, but the current framing overstates the sense in which the method “matches bounded-variance rates.”

### Minor
- **The symmetry assumption is powerful but restrictive, and the paper’s defense of it is not fully persuasive.**  
  The paper does acknowledge this in §6.1, but the argument that one can simply run several algorithms in practice does not really mitigate the theoretical narrowness of the assumption. The improvement here comes from changing the admissible noise class, not from improving guarantees under standard generic heavy-tail assumptions.
- **MAB experiments are under-scaled relative to the theorem.**  
  Using only \(d=2\) arms leaves the paper without empirical evidence for the claimed \(\sqrt d\) scaling advantage. A few experiments at larger \(d\) would substantially improve the paper.
- **The zeroth-order experiments do not validate the high-probability nature of the theory.**  
  Since the paper emphasizes high-probability results, empirical tail behavior across many runs would be much more appropriate than a small number of trajectories.
- **There is some ambiguity/inconsistency in the experimental presentation around the median size \(m\).**  
  In §5.3 the text says “For ZO-clipped-med-SSTM, the best median size is \(m=2\),” while later the tuning paragraph says the grid search range is \([3,5,7]\). This could be a notation/parser issue, but as extracted it is inconsistent and should be clarified.
- **The method depends on knowing \(\kappa\) to set \(m = 2/\kappa + 1\).**  
  The paper acknowledges this limitation, but it remains a real practical gap since \(\kappa\) is generally unknown.

### Trivial
- The theorem-level distinction between the strongest result being in the **Lipschitz-oracle** setting versus the **independent-oracle** setting could be communicated more prominently in the introduction/summary claims. In Theorem 1, the independent-oracle case still carries a stronger noise-dependent term, so the bounded-variance-style message is most compelling for the Lipschitz-oracle regime.

## Nice-to-Haves
- Add experiments varying **dimension \(d\)** for both ZO and MAB, since the claimed rates are dimension-dependent.
- Report **total oracle/sample complexity**, not only iteration counts, especially as a function of \(\kappa\).
- Include at least one **asymmetric-noise experiment in the main paper**, since practical robustness outside exact symmetry is important and is currently deferred to the appendix/discussion.
- Add an ablation on sensitivity to **median size \(m\)** and possibly clipping levels, since the paper claims \(m\) is not very sensitive in practice.
- For MAB, include a more direct heavy-tailed clipping-based baseline if available in the cited setup, and evaluate at \(d \gg 2\).

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper lacks standalone constrained ZO experiments, so Theorem 2 is unsupported.”**  
  This is too strong. It is fair to say the constrained setting is not directly validated, but the paper does provide theory and some bandit-related constrained optimization evidence. This is a scope/coverage limitation, not a substantive flaw.
- **Generic complaints about missing hyperparameters or reproducibility details.**  
  The main issue is not missing low-level details; it is that the experiments shown are too narrow to support the strongest claims.
- **Claims about unavailable/nonstandard baselines or cited methods.**  
  Per instruction, criticisms questioning existence/release/availability of cited tools/methods are removed.
- **Pure formatting/writing complaints.**  
  The extraction has parser artifacts, and style issues are not central here.

## Novel Insights
The clearest synthesis across the reviews is that this paper is best understood as a **theory-first contribution whose main novelty is not merely “median is more robust than clipping,” but that symmetry in the *two-point oracle noise difference* is the precise structural lever that restores bounded-second-moment behavior in zeroth-order estimation even when raw moments may not exist**. That is the real conceptual spark. At the same time, the paper somewhat obscures its own strongest message by overselling practical superiority: the core result is a sharp separation between generic heavy-tail assumptions and symmetric heavy-tail assumptions, whereas the experiments mainly show promise rather than decisive practical dominance.

## Suggestions
- **Tighten the empirical claims** in the abstract/introduction to match what is actually demonstrated. The theory appears stronger than the experiments; the paper should say so rather than oversell practical superiority.
- **Fix the interpretation of Figure 1** and explicitly explain what metric is being shown and why the conclusion follows. As written, the text and figure seem at odds.
- **Reframe §5.2 as an application demo**, not as validation of the MAB theorem, unless the authors add a true bandit-feedback real-world experiment.
- **Add larger-scale MAB experiments** with more arms and stronger baselines aligned with the theory.
- **Report total oracle complexity** including the \((2m+1)\) factor and discuss the practical effect of the \((4/\kappa)^{2/\kappa}\) constants.
- **Increase the number of random runs** in ZO experiments and present empirical distributions / failure probabilities if the paper wants to emphasize high-probability guarantees.
- **Clarify the median-size tuning inconsistency** around \(m\) in §5.3.
- **Emphasize in the high-level summary that the strongest bounded-variance-style claim is for the Lipschitz-oracle setting under symmetry.**

Overall, the paper looks **theoretically interesting and potentially significant**, with a real novelty in how symmetry is exploited in zeroth-order and bandit settings. The main obstacle is that the **empirical section does not yet carry the same level of credibility or support the breadth of the paper’s practical claims**.

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 5.0, 6.0]
Average score: 5.5
Binary outcome: Reject
