## Summary

This paper proposes RTZ-VI-LCB, a model-based algorithm for offline robust two-player zero-sum Markov games (RTZMGs) under partial data coverage. The authors introduce a novel "robust unilateral clipped concentrability coefficient" to characterize data quality without requiring full coverage, derive finite-sample complexity bounds that are near-optimal in state and action space dimensions, establish information-theoretic lower bounds for various uncertainty regimes, and extend the algorithm to multi-player general-sum games.

## Strengths

- **Novel algorithmic contribution for a challenging setting:** The paper addresses the intersection of offline RL, robustness, and multi-agent learning—a setting with limited prior theoretical treatment. The RTZ-VI-LCB algorithm properly integrates pessimism principles with robust value iteration and TV distance uncertainty sets (Section 3, Algorithm 2).

- **Weaker data coverage assumption:** The robust unilateral clipped concentrability coefficient $C_r^*$ is a genuine improvement over the maximum density ratio $C_r$ used in prior work like P²M²PO. As stated in Section 1.1, $C_r^* \in [\frac{1}{S(A+B)}, \infty)$ captures distribution shift without requiring proportional scaling when occupancy distributions exceed $\frac{1}{S(A+B)}$, enabling learning under partial coverage.

- **Near-optimal sample complexity for key parameters:** The derived sample complexity $\tilde{O}\left(\frac{C_r^* H^4 S(A+B)}{\varepsilon^2} f(\sigma^+, \sigma^-, H)\right)$ matches information-theoretic lower bounds (Theorem 2) with respect to state space $S$ and action spaces $\{A, B\}$—a first for offline RTZMGs (Table 1 and Theorem 1 discussion).

- **Meaningful lower bound analysis:** Theorem 2 establishes that learning RTZMGs is at least as hard as learning standard TZMGs when uncertainty is small ($\min\{\sigma^+,\sigma^-\} \lesssim 1/H$), and provides tighter bounds for larger uncertainty levels. This contextualizes the algorithm's difficulty relative to well-studied settings.

## Weaknesses

- **Inconsistency between Table 1 and main text regarding horizon exponent:** Table 1 states the lower bound for small uncertainty as $\Omega\left(\frac{C_r^* SH^3(A+B)}{\varepsilon^2}\right)$, while the discussion following Theorem 2 in Section 4 states $\Omega\left(\frac{C_r^* SH^4(A+B)}{\varepsilon^2}\right)$. This discrepancy directly affects the paper's optimality claims and must be resolved. The upper bound is $H^4$ in both places, so either the lower bound exponent is wrong in one location, or the claimed "optimality except for $H$" is inaccurate.

- **Confusing notation in Assumption 1 (Equation 22):** The definition uses $\sup_{(s, a, b, h, P) \in \Delta(A) \times \mathcal{S} \times \mathcal{A} \times \mathcal{B} \times [H] \times \mathcal{U}^{\sigma^-}(P^0)}$, which writes $\Delta(A)$ (a probability simplex) as an element in a tuple subscript—this is not standard mathematical notation. The intent appears to be taking a supremum over policies $\mu^-$, but this should be written explicitly for clarity.

- **Computational tractability gap for general-sum extension:** Theorem 3 claims the extension to multi-player general-sum games "break[s] the curse of multiagency," achieving sample complexity depending on $\sum_i A_i$ rather than $\prod_i A_i$. However, Section 3.2 briefly acknowledges that "solving these robust matrix games is generally PPAD-hard." The paper does not clarify whether this extension assumes a computational oracle, relies on a weaker equilibrium concept (e.g., CCE), or is purely information-theoretic. Without this clarification, the practical relevance of the general-sum result is unclear.

- **No empirical validation:** The paper is purely theoretical with no experiments demonstrating the algorithm's practical viability, convergence behavior, or sensitivity to uncertainty parameters. While not strictly required for theory papers at ICLR, empirical validation would strengthen the contribution—particularly given the novel concentrability coefficient and penalty term construction.

## Nice-to-Haves

- **Tighten the horizon dependency:** Investigate whether the $H^4$ upper bound can be reduced toward $H^3$ to fully close the gap with the lower bound. Even an explanation of why the current proof techniques yield $H^4$ would help readers understand the bottleneck.

- **Provide guidance on uncertainty set calibration:** The sample complexity depends on $f(\sigma^+, \sigma^-, H)$, but no discussion is offered on how practitioners should select $\sigma^+$ and $\sigma^-$ in practice or how sensitive performance is to these choices.

- **Add discussion on verifying concentrability:** The assumption on $C_r^*$ is theoretically useful but practically unverifiable from data alone. Discussing how one might estimate or bound this coefficient would enhance applicability.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Criticism of (s,a,b)-rectangularity as "strong assumption":** This structural condition is standard in robust RL literature (cited Iyengar, 2005) and enables the Bellman recursion. Requesting comparison with weaker conditions like s-rectangularity is scope creep beyond the paper's stated contribution.

- **Request for discussion of different divergence functions per player:** The paper already notes both players can use different divergence functions but restricts to the same one for the main results. This is a modeling choice within scope.

- **Minor notational inconsistencies (e.g., $C_n$ vs $C_h$):** These are proofreading issues that do not affect technical correctness.

- **Burn-in cost dependence on $d_m^n$ being "prohibitive":** This is standard in offline RL; the paper provides the standard $d_m^n$ condition alongside the concentrability assumption. Flagging this as a flaw misrepresents the offline RL literature.

## Novel Insights

The robust unilateral clipped concentrability coefficient $C_r^*$ represents a meaningful conceptual advance: by clipping the occupancy distribution at $\frac{1}{S(A+B)}$ before taking the ratio, the assumption becomes significantly weaker than requiring full coverage or proportional scaling. This insight—that partial coverage suffices when combined with pessimism—transfers the single-agent offline RL insight of Li et al. (2024a) to the multi-agent robust setting. The key innovation is adapting the clipped concentrability concept to account for adversarial perturbations in both transition dynamics and opponent policy, requiring consideration of worst-case occupancies under model perturbations. The derivation of the Bernstein-style penalty that properly accounts for the uncertainty set's non-linear transformation of the transition kernel is also noteworthy, though the proof intuition is deferred to the appendix.

## Suggestions

- **Resolve the H^3 vs H^4 discrepancy immediately:** Verify whether Table 1 or the main text is correct for the lower bound exponent, and update consistently. If both are correct under different conditions, explicitly state the condition boundaries.

- **Clarify Assumption 1 notation:** Rewrite Equation 22 to explicitly show the supremum over policies $\mu^-$ and $\nu^+$, matching how $d_h^{\mu^-, \nu^+, P}$ is used.

- **Add an explicit statement about computational assumptions in Theorem 3:** Either state that the general-sum result assumes an oracle for solving robust matrix games, or specify that CCE/CE is targeted instead of NE. Currently, readers cannot assess the practical meaning of the "curse of multiagency" claim.

- **Consider adding synthetic experiments:** Even simple matrix game experiments would demonstrate that the penalty term correctly handles uncertainty and that convergence follows the theoretical rate.

## Evaluation

**Novelty:** High. This is the first work to achieve near-optimal sample complexity for offline robust Markov games under partial coverage, with a novel concentrability coefficient adapted to the robust multi-agent setting.

**Technical soundness:** The core theoretical framework is sound, but the inconsistency between Table 1 and the main text regarding the horizon exponent is a significant oversight that affects optimality claims. The notation in Assumption 1 needs repair.

**Empirical support:** None provided. The paper is entirely theoretical.

**Significance:** Significant for the offline robust MARL literature. The theoretical tools (concentrability coefficient, robust Bellman analysis with TV distance, lower bound construction) are valuable for future work.

**Clarity:** Generally well-written, but the Assumption 1 notation and the H-exponent inconsistency impede full understanding of the key technical claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>