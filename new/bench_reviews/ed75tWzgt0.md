Now I have enough context from the calibration papers. Let me synthesize the review.

Key calibration anchors:
- INPO (Accept Oral, scores 6,6,6,6): Similar game-theoretic RLHF paper with theory-practice gap, overclaimed connections
- XPO (Accept Poster, scores 6,6,8,6): Provably efficient RLHF with exploration bonus, but no empirical experiments
- COPO (Accept Spotlight, scores 6,8,8): Exploration bonus in online RLHF with theory and experiments
- Global Convergence RLHF (Reject/Withdrawn, scores 5,3,5,3): Theory-practice disconnect for neural networks with unrealistic assumptions
- Sample Efficient RLHF via Active Exploration (Reject, scores 5,6,3,5,5): Theory-practice gap, heuristic algorithm differs from theory
- MPO (Accept Poster, scores 6,6,8,6): Unclear theory-empirical connection

The paper under review has a significant theory-practice gap (theoretical framework ≠ practical algorithm), overclaims theoretical guarantees for practical algorithms, has a questionable Nash equilibrium interpretation, and limited experiments. This pattern is similar to rejected papers (Global Convergence RLHF) rather than accepted ones. However, it does have real empirical improvements unlike pure theory papers.

Let me now write the final review.</think>## Summary

This paper proposes TANPO (Two-Agent Nash Policy Optimization) and SADPO (Single-Agent Diversity-driven Optimization), self-play frameworks for online RLHF. The theoretical framework (Section 3.2) optimizes over a finite reward class with a value-plus-loss objective, claiming sublinear regret under TGEC conditions. The practical TANPO algorithm uses DPO-style updates on two agents (a max-player and a min-player with an additional log-likelihood bonus), and SADPO approximates this with a single agent via extremal rejection sampling. Empirically, TANPO and SADPO outperform baselines on AlpacaEval 2.0, MT-Bench, and academic benchmarks using Zephyr-7B-SFT.

## Strengths

1. **The two-agent data diversity mechanism is a practical and interesting idea.** The insight that generating training pairs from two policies with different objectives (one exploiting, one exploring) increases data diversity and improves the max-player's performance—even though the max-player's own loss is standard DPO—is empirically supported (Table 1, Figures 1 and 3). The TANPO max-player vs. Online DPO comparison directly isolates the effect of data diversity, as they share the same objective but differ in training data.

2. **The derivation from the theoretical framework to TANPO's practical objectives (Eqs. 11, 13) is technically competent.** The reparameterization argument and minimax interchange to go from Eqs. (4)-(7) to closed-form DPO-style losses follows established techniques in the literature (Rafailov et al. 2024) and yields implementable algorithms.

3. **Consistent empirical gains across multiple evaluation metrics.** TANPO achieves 27.66% LC win rate on AlpacaEval 2.0 (vs. 24.36% for Online DPO), top MT-Bench average scores, and improvements on academic benchmarks (Figure 2). The improvement over SELM and Hybrid GSHF is also noteworthy.

4. **SADPO provides a practical single-agent alternative** that avoids the cost of maintaining two models while capturing some of the diversity benefit.

## Weaknesses

### Major:

1. **The core claim of "provable efficiency" for the practical algorithms is unsupported.** Theorem 1 provides a sublinear regret bound for the theoretical algorithm framework (Section 3.2), which operates over a finite, realizable reward class \(\mathcal{R}\) (Assumption 1). TANPO (Algorithm 1) operates on neural network policies with DPO-style updates and no explicit reward class—PairRM provides feedback, not a learned reward in \(\mathcal{R}\). SADPO further replaces the min-player with rejection sampling. The paper states that "the theoretical analysis in this section also applies to TANPO... provided the reward function class \(\mathcal{R}\) meets Assumption 4" (Section 5.1), but Assumption 4 is only in the appendix and its satisfaction by neural policies is never discussed, nor is any formal approximation bound between the theoretical algorithm and TANPO established. The abstract's claim of a "provably efficient and practical self-play RLHF algorithm" conflates two distinct objects. This is not a minor notational gap—it is the paper's central positioning claim.

2. **The "Nash equilibrium" interpretation is not meaningfully justified.** The constructed value function \(V(\pi,\mu) = \mathbb{E}[r^*(a^1) - r^*(a^2) - \alpha KL(\pi||\pi_{\text{ref}}) + \alpha KL(\mu||\pi_{\text{ref}})]\) (Eq. 3) defines a game where the min-player is incentivized to have high KL from \(\pi_{\text{ref}}\). In a real RLHF setup, there is no strategic opponent; the "min-player" is simply another policy used to generate diverse comparison data. The paper does not establish that the Nash equilibrium of this constructed game corresponds to any desirable alignment objective, nor that the practical TANPO min-player's learning objective (Eq. 13) actually computes a best response. The claim that low regret implies "the resulting policy of TANPO is approximately a Nash equilibrium policy" (Section 5.1) therefore does not carry the alignment significance the paper implies.

3. **The exploration story is weakly supported.** In TANPO, the max-player has no exploration term (its objective is standard DPO). The min-player's bonus \(\mathbb{E}_{x,a\sim\pi^t}[\log \mu(a|x)]\) encourages \(\mu\) to assign higher probability to actions sampled from \(\pi^t\)—this is not clearly an exploration incentive in the standard sense of optimism under uncertainty or information gain. It increases the diversity of training pairs, but "diversity" is distinct from "exploration" in the bandit-theoretic sense. The only evidence for improved diversity (Figure 1) shows a small difference in \(|\log\pi_{\text{ref}}(a^1) - \log\pi_{\text{ref}}(a^2)|\) (0.08 vs. 0.11) without variance estimates, and without an ablation removing the bonus term from the min-player while keeping the two-agent structure.

4. **No ablation study isolating the proposed mechanism.** The paper does not present: (a) TANPO without the min-player exploration bonus, (b) SADPO with different K values, (c) SADPO with random pair selection instead of extremal selection, or (d) Online DPO trained for the same number of iterations as TANPO for the overfitting comparison. Without these, it is impossible to determine whether the gains come from the proposed game-theoretic formulation, from the exploration bonus, from simple data augmentation via two-policy sampling, or from longer training. This is a significant gap for a method paper.

5. **SADPO's approximation to TANPO has an unexplained discrepancy.** The TANPO min-player objective (Eq. 13) includes \(\mathbb{E}_{a\sim\pi^{t+1}}[\log\mu(a|x)]\), while SADPO (Eq. 16) uses \(\mathbb{E}_{a\sim\pi_{\text{ref}}}[\log\pi(a|x)]\). The sampling distribution changes from the current max-player to the static reference policy, and the optimized policy changes from \(\mu\) to \(\pi\). No theoretical or intuitive justification is provided for this substitution, undermining the claim that SADPO is "supported by both theoretical analysis and empirical evidence" as stated in the abstract.

### Minor:

6. **The overfitting claim in Figure 4 lacks a proper baseline.** The paper claims TANPO "mitigates overfitting" because AlpacaEval win rates continue to improve across 6 iterations (2 epochs). However, no baseline (e.g., Online DPO or SELM) is trained for the same number of iterations/epochs. It is plausible that any online method running additional iterations on fresh data would continue improving.

7. **Limited experimental scope.** All experiments use a single 7B model and a single dataset (UltraFeedback). The claims about generality and "provably efficient" alignment are not tested across scales or data regimes.

8. **Missing comparisons with closely related self-play baselines.** The paper cites SPPO, INPO, and Nash-MD as related work, but none appear in the experimental comparison. Comparing against other self-play/Nash-style methods would be essential to establish the specific advantages of the proposed formulation.

## Nice-to-Haves

- Experiments on a larger model (e.g., 13B or 70B) and/or a different dataset to test generalizability.
- A safety/harmlessness evaluation (given the paper's primary area is alignment/safety), as all reported metrics are capability-based.
- Reporting of computational cost (GPU hours, inference cost of PairRM per iteration) to help practitioners assess tradeoffs.
- Confidence intervals or multiple seeds for benchmark results, especially for the academic benchmarks in Figure 2.

## Removed Points

- **"SADPO has no theoretical analysis at all":** The abstract claims it is "supported by both theoretical analysis and empirical evidence." While there is no separate theorem for SADPO, Section 4.2 provides an informal argument for why it approximates TANPO. The real issue (captured in Weakness 5) is the unexplained distribution change, not a complete absence of discussion. *(Treated as a weaker version of a valid weakness, not removed entirely.)*

- **"Missing related work on XPO/COPO/SPIN/etc.":** Per the rules, I do not flag missing related works as a weakness since I cannot confirm their existence and relevance. The absence of comparison with concurrently published self-play baselines (Weakness 8) is based on papers the authors themselves cite.

- **Formatting nitpicks and minor notation issues:** Removed per rules on formatting nitpicks.

- **"Reproducibility concerns about PairRM/implementation details":** Removed per rules; if the paper cites PairRM and UltraFeedback, these are treated as existing and available.

- **"No safety evaluation":** Moved to Nice-to-Haves. The paper is in the alignment area and this is a valid suggestion, but all reported metrics are standard for RLHF papers and evaluating capabilities is the direct test of the method's core claim.

## Novel Insights

The paper's most interesting finding is indirect: the max-player in TANPO improves over Online DPO despite having the identical objective function (DPO loss), with the only difference being the training data generated by the min-player. This suggests that even without explicit exploration bonuses, a policy can benefit substantially from training on more diverse preference pairs—an insight that aligns with recent work on on-policy versus off-policy DPO but adds the dimension of *adversarial diversity* through the min-player. However, this compelling empirical story is obscured by the Nash equilibrium framing, which does not cleanly explain this phenomenon.

## Suggestions

1. **Refine the theoretical claims:** Restrict the "provably efficient" claim to the theoretical algorithm framework in Section 3.2, and clearly state that TANPO and SADPO are heuristic approximations not covered by Theorem 1. This would actually strengthen the paper by making it more honest.

2. **Add ablations:** At minimum, run TANPO without the min-player bonus, and run Online DPO for 6 iterations on the same data schedule, to isolate the mechanism.

3. **Justify or modify SADPO's exploration term:** Either explain why changing \(\pi^{t+1}\) to \(\pi_{\text{ref}}\) in the bonus term is valid, or use \(\pi^t\) in SADPO to maintain theoretical consistency.

4. **Soften the Nash equilibrium narrative:** Be explicit that this is a *constructed* zero-sum game whose equilibria need not correspond to standard RLHF optima, and that the benefit of the two-agent scheme is better diversity rather than converging to a game-theoretic equilibrium.

## Score and Decision

**Calibration:**
- INPO (Accept Oral, 6,6,6,6): Similar game-theoretic RLHF framing; theory-practice gap noted; but cleaner theory-empirical mapping and stronger experiments.
- XPO (Accept Poster, 6,6,8,6): Provably efficient RLHF with exploration bonus; strong theory but no experiments; accepted for theoretical contribution.
- COPO (Accept Spotlight, 6,8,8): Exploration bonus in online RLHF with theory and experiments; simpler theory but tight theory-practice connection.
- Global Convergence RLHF (Reject, 5,3,5,3): Theory for neural RLHF with unrealistic assumptions; no experiments; theory-practice disconnect was a key weakness.
- Sample Efficient Active Exploration RLHF (Reject, 5,6,3,5,5): Theory-practice gap where theoretical algorithm differs from practical implementation.

This paper sits in a similar space to the rejected papers in terms of its theory-practice disconnect (the theoretical framework uses a finite reward class and explicit reward optimization; the practical algorithm uses neural DPO-style updates with no explicit reward class). However, unlike purely theoretical papers that were rejected, this paper does show real empirical improvements. The gap between the claimed "provably efficient" guarantee and the actual algorithms is substantial—the theorem covers a stylized algorithm that is not run, and the Nash equilibrium narrative is not justified. The lack of ablations further makes it impossible to attribute the gains to the proposed mechanism specifically.

This is a paper with a genuinely interesting empirical method (two-agent/diversity-driven preference optimization) that is substantially oversold by a theoretical framework that does not actually cover it. The empirical contribution alone—consistent gains on standard benchmarks—is real but not exceptional by current standards in the RLHF literature.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>