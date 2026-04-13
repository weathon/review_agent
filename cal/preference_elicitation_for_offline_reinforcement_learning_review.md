=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary

The paper addresses preference elicitation for offline reinforcement learning, proposing Sim-OPRL—an algorithm that queries preferences on simulated rollouts from a learned environment model rather than on trajectories from the offline dataset. The key insight is combining pessimism for handling out-of-distribution states (from the offline RL literature) with optimism for acquiring informative preferences (from the preference-based RL literature). Theoretical analysis shows that Sim-OPRL eliminates the reward concentrability coefficient present in prior offline preference methods, and experiments across five environments demonstrate improved sample efficiency.

## Strengths

- **Clear theoretical contribution**: The paper provides formal guarantees distinguishing OPRL (sampling from offline data) from Sim-OPRL (simulated rollouts). Theorems 5.1 and 6.1 clearly show that Sim-OPRL eliminates the reward concentrability term $C_R$ that appears in OPRL's bounds, offering a principled motivation for the proposed approach.

- **Novel problem formulation**: The paper formalizes the problem of preference elicitation for offline RL (Definition 3.1), properly framing the challenge of combining conservatism (for offline data) with exploration (for active preference learning). This is a meaningful contribution at the intersection of two important RL subfields.

- **Practical implementation details**: Section 6.3 provides concrete implementation strategies using neural network ensembles for uncertainty estimation and Lagrangian penalties for pessimism, making the theoretical framework implementable and reproducible.

- **Empirical validation across domains**: Experiments include D4RL (HalfCheetah), Gridworld environments, and a healthcare-relevant sepsis simulation, demonstrating applicability across different domains with varying state/action spaces and dynamics complexity.

## Weaknesses

- **Missing direct comparison to FREEHAND**: Table 1 identifies FREEHAND (Zhan et al., 2023a) as the closest related offline method with robustness guarantees, yet it is absent from all experiments. The authors state in Appendix C that their implementation framework can run FREEHAND with minor modifications, making its omission from the evaluation a notable gap. Including FREEHAND would strengthen the empirical claims about Sim-OPRL's superiority.

- **Notation inconsistencies**: (i) The symbol $\sigma$ is used both as the link function in Eq. (1) and as the batch size in Section 6.3 ("we sample preferences in batches of $\sigma$"), creating confusion. (ii) The quantity $\kappa$ is defined differently in Theorems 5.1 ($\kappa = \sup_r 1/\sigma^2(r)$) and 6.1 ($\kappa = \sup_r 1/\sigma(r)$)—these are different quantities and the inconsistency should be corrected.

- **Undefined concentrability coefficient $C_R$**: The reward concentrability coefficient $C_R$ appears centrally in Theorem 5.1 but is never formally defined in the paper. While $C_T$ receives Definition 3.3, $C_R$ is referenced only implicitly, making the theorem's interpretation incomplete without consulting Zhan et al. (2023a).

- **Dependency on transition model quality**: While Section 6.2 discusses the trade-off between transition and preference model quality, the method's performance fundamentally relies on the learned model generating meaningful trajectories. In environments where dynamics are hard to learn (complex continuous control, partially observable settings), simulated rollouts may query preferences on unrealistic states, potentially harming reward learning—this risk is acknowledged but not empirically quantified.

- **Limited benchmark breadth**: Only one D4RL environment (HalfCheetah-Random) is tested. The "random" dataset has specific characteristics (diverse but suboptimal behavior). Testing on "medium" or "medium-expert" datasets would better assess whether Sim-OPRL's advantages hold across common offline RL data regimes.

- **Experiments use synthetic preferences**: All preference labels are derived from the ground-truth reward function. Real human feedback includes noise, inconsistency, and potential non-transitivity. The active elicitation strategy may behave differently with imperfect human labels.

## Nice-to-Haves

- **Computational cost comparison**: Sim-OPRL requires ensemble training plus inner-loop policy optimization to find exploratory policies. Reporting wall-clock time or FLOPs alongside OPRL would clarify whether sample efficiency gains come at computational cost.

- **Ablation with oracle dynamics**: An experiment providing the true transition model would isolate whether gains come from the elicitation strategy versus better dynamics modeling.

- **Sensitivity to pessimism hyperparameters $\lambda_T, \lambda_R$**: These control the conservatism-exploitation trade-off; understanding robustness to misspecification would strengthen practical applicability claims.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that Definition 3.1 has a "typo"**: The notation $V_{\pi,R}^\pi$ is indeed confusing (the paper should write $V_{T,R}^{\hat{\pi}}$), but the text clarifies that $\hat{\pi}$ is the estimated policy. The intent is recoverable even if notation could be clearer.

- **Demand for 10+ seeds**: 6 seeds with 95% confidence intervals is within typical RL experimental practice. While more seeds would strengthen claims, this does not invalidate the presented results.

- **Demand for human user study**: While valuable for real-world validation, human-subject experiments are beyond the scope of an algorithmic contribution. Synthetic preferences from reward functions are standard practice in preference-based RL research.

- **Claim that $\alpha < 1$ for uncertainty sampling is unproven**: The paper correctly states "Uncertainty sampling learns accurate reward models with fewer preference queries when $\alpha < 1$, but can perform like uniform sampling in harder problems ($\alpha = 1$)." This appropriately hedges rather than over-claiming.

## Novel Insights

The key insight is that offline preference elicitation faces a fundamental tension: OPRL-style sampling from offline data is safe but may waste queries on trajectories far from optimal, while simulated rollouts can target informative regions but risk querying unrealistic states. The paper's solution—pessimistic rollouts that stay near the data distribution while optimizing for preference uncertainty—elegantly bridges this tension. The theoretical result that this eliminates $C_R$ while relying on $C_T$ (transition concentrability) is meaningful because $C_T$ is often better behaved in practice: offline datasets for offline RL are typically collected with some exploration, making transition coverage reasonable even when reward coverage is poor.

## Suggestions

1. Include FREEHAND in the experimental comparison, since the implementation framework already supports it.

2. Define $C_R$ explicitly alongside $C_T$ in the preliminaries section.

3. Correct the $\sigma$ symbol collision (use a different symbol for batch size) and reconcile the $\kappa$ definitions across theorems.

4. Add at least one D4RL "medium" dataset to validate performance across common offline RL data quality regimes.

5. Add a brief experiment with noisy preference labels (e.g., 5-10% random flips) to demonstrate robustness to imperfect human feedback.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0, 6.0]
Average score: 6.8
Binary outcome: Accept
