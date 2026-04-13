=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary

This paper proposes **SELECT**, a general algorithmic template for satisficing regret minimization in bandit optimization. SELECT uses any learning oracle with sub-linear standard regret to iteratively identify candidate satisficing arms, performs forced sampling, and applies lower confidence bound testing to verify satisficing status. The key contribution is achieving constant satisficing regret (independent of the satisficing gap Δ_S, instead depending on the exceeding gap Δ*_S) for settings including concave and Lipschitz bandits where Δ_S = 0 makes prior bounds vacuous.

## Strengths

- **Conceptual contribution of exceeding gap:** The insight that Δ_S can be zero in continuous arm spaces (Remark 4 correctly notes this for concave and Lipschitz bandits), and that Δ*_S = r(X*) − S is the right quantity for non-vacuous bounds, is significant. This directly addresses a fundamental limitation of prior work (Michel et al., 2023; Garivier et al., 2019).

- **General algorithmic template:** SELECT works with any oracle satisfying Condition 1 (sub-linear regret), enabling instantiations across diverse problem classes. The three-component design (oracle trajectory sampling, forced sampling, LCB testing) is well-motivated, with each component serving a clear analytical purpose explained in Remark 2.

- **Dual guarantees:** Theorem 1 and Theorem 2 together establish that SELECT achieves constant satisficing regret in realizable cases while preserving oracle-level standard regret in non-realizable cases—a principled approach to handling realizability uncertainty.

## Weaknesses

- **All proofs are deferred to a separate document.** The paper repeatedly cites "Appendix A/B of the full version (Feng et al., 2025)" for proofs of Propositions 1, 2, and all theorems. This makes the submission non-self-contained and prevents verification of the theoretical claims. For a venue like ICLR, this is a significant issue.

- **Algorithm requires explicit knowledge of oracle parameters.** SELECT requires α, β, and C₁ from the oracle's regret bound to set γ_i = 2^{−i(1−α)/α} and T_i. These parameters may not be known in practice. The paper never discusses sensitivity to parameter misspecification or whether adaptive estimation is possible.

- **Pseudo-code lacks explicit stopping condition.** Algorithm 1 inputs time horizon T but the outer loop "for round i = 1, 2, ... do" has no termination condition. The inner while loop exits when LCB < S, but the mechanism for stopping when T is reached is unspecified.

- **Exponential-in-dimension constants for Lipschitz bandits.** Corollary 3 shows the satisficing regret bound scales as L^d / (Δ*_S/2)^{d+1}. For moderate d (e.g., d=5) and small Δ*_S (e.g., 0.1), this constant exceeds 64 million—making the "constant regret" claim practically meaningless for realistic time horizons. The paper presents this as a strength without acknowledging the practical limitation.

- **Figure 4b appears inconsistent with Theorem 2.** Theorem 2 states SELECT's standard regret is bounded by C₁T^α · polylog(T)—the same order as the oracle. Yet Figure 4(b) shows SELECT achieving ~1000 standard regret while Uniform UCB (the oracle) achieves ~4000. A 4× improvement over the oracle would require explanation, as the theorem only guarantees parity.

- **Experiments test only favorable parameter regimes.** The realizable instances use large Δ*_S values (0.07, 0.5, 0.7 across settings). No sensitivity analysis shows how performance degrades as Δ*_S → 0, which is critical given the 1/Δ*_S dependence in the bounds.

- **No error bars or confidence intervals in experiments.** All experiments report averages over 1000 runs without variance information, making it difficult to assess statistical significance of differences between algorithms.

- **Gap between upper and lower bounds remains undiscussed.** Corollary 1 gives O(K/Δ*_S · polylog) while Theorem 3 gives Ω(1/Δ). The K-factor gap is not addressed—whether it's necessary or an artifact of the analysis.

## Nice-to-Haves

- Analysis of SELECT's robustness when oracle parameters (α, β, C₁) are misspecified or estimated online.

- Discussion of whether warm-starting the oracle across rounds (rather than cold-starting each round) could improve practical performance while preserving guarantees.

- A clear statement of when SELECT is preferred over prior methods (e.g., when Δ_S << Δ*_S, prior methods may be tighter; this trade-off should be explicit).

- Extended experiments with smaller Δ*_S values and time horizons beyond T=5000 to verify asymptotic behavior.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **"Missing Limitations section"**: This is a formatting requirement rather than a substantive weakness. ICLR encourages but does not mandate dedicated Limitations sections.

- **"LCB formula uses factor of 4 instead of standard 2"**: This is a technical detail of the analysis; different constants are common across bandit literature and the factor of 4 is presumably needed for the concentration argument in Proposition 2.

- **"No lower bound for Lipschitz bandits"**: This would strengthen the paper but its absence is not a weakness. The paper provides lower bounds for the two settings (finite-armed, concave) where tight characterization matters most.

- **"SAT-UCB+ has no theoretical guarantees"**: This is noted by the authors (they call it "a heuristic"). Beating a heuristic baseline is still valid empirical evidence.

## Novel Insights

The key insight is that continuous arm spaces naturally induce Δ_S = 0 (since arms arbitrarily close to but below threshold S exist), making prior satisficing regret bounds that depend on 1/Δ_S vacuous. The paper correctly identifies that the exceeding gap Δ*_S = r(X*) − S remains positive in realizable cases and provides the right dependence for non-trivial guarantees. The algorithmic insight—that oracle trajectory sampling avoids expensive uniform exploration over large arm spaces, while forced sampling followed by LCB testing correctly handles the exploration-exploitation trade-off—elegantly addresses why prior approaches fail in continuous settings.

## Suggestions

- Include all proofs in the main submission or appendix, not a separate document.

- Add a termination condition in Algorithm 1 that explicitly tracks time horizon T.

- Provide a sensitivity analysis varying Δ*_S to empirically validate the 1/Δ*_S scaling.

- Add error bands to experimental figures.

- Explain or investigate why SELECT appears to outperform its oracle in Figure 4(b), as this is not predicted by theory.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
