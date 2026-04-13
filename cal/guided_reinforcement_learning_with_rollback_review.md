=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary

This paper introduces Guided Reinforcement Learning (GRL) and GRL with Roll-Back (GRL-RB), methods for warm-starting RL agents using a prior guide policy while maintaining performance above a user-defined threshold. The authors derive theoretical sampling rates α for the guide policy under specific MDP assumptions, and introduce a roll-back mechanism to recover from threshold violations when convergence assumptions fail. Experiments on Combination Lock and AntMaze environments demonstrate that GRL-RB maintains performance better than static or linear-decay sampling baselines.

## Strengths

- **Novel theoretical contribution:** The derivation of guide sampling rates α that provably maintain evaluation return above a degradation threshold (Equations 3–7) addresses a real gap in guided RL literature, where sampling rates are typically chosen empirically. The idea of grounding the sampling rate in a formal guarantee is conceptually clean and meaningful for safety-critical applications.

- **Practical roll-back mechanism:** GRL-RB provides a simple but effective corrective mechanism when the core convergence assumption is violated. Figure 4 demonstrates that while standard GRL fails under poor hyperparameter choices, GRL-RB recovers by reverting to previous sampling rates—a practical solution to the theoretical fragility.

- **Clear motivation and honest limitations discussion:** The paper clearly articulates the problem (performance degradation during guide-to-learner transfer) and is transparent about the convergence assumption and the reactive nature of roll-back. The limitations section acknowledges that roll-back triggers *after* violations and that the theoretical scope is limited.

- **Guide policy flexibility:** The method is agnostic to guide policy format (heuristic rules, decision trees, or pre-trained networks), making it applicable to systems where prior knowledge exists in non-neural formats. This is a practical strength for real-world deployment.

## Weaknesses

- **Theory-practice gap:** The theoretical derivations (Equations 3–7) are derived specifically for the Combination Lock MDP structure—binary optimal/non-optimal actions with immediate termination on error. The extension to AntMaze (continuous state/action, dense rewards) relies on heuristic estimation of β_l and β_g without theoretical justification. The paper acknowledges that β_l "might be difficult to determine" and defaults to conservative estimates, but this introduces a hyperparameter that undermines the guarantee's practical utility. The theoretical guarantee effectively becomes heuristic guidance when applied to environments like AntMaze.

- **Missing closed-form solution for positive dense reward:** Equation 7 provides only an implicit inequality for positive dense rewards, with no algorithm or procedure to solve for α. Practitioners cannot readily compute the required sampling rate for this reward type without additional work not provided in the paper.

- **Incomplete baseline comparison:** JSRL (Uchendu et al., 2023)—the most closely related prior work—is only compared in AntMaze, not in Combination Lock. Additionally, the IQL baseline is tested with a cleared replay buffer, which is non-standard for offline-to-online RL and disadvantages the baseline. A standard IQL online fine-tuning setup (retaining the replay buffer) should be included.

- **Limited empirical scope:** Experiments cover only one toy environment (Combination Lock) and three variants of AntMaze navigation. No validation on standard continuous control benchmarks (e.g., MuJoCo locomotion), robotic manipulation, or domains with different reward structures is provided. This limits confidence in generalizability.

- **No ablation of key hyperparameter μ:** The degradation threshold parameter μ directly controls the safety-speed trade-off, yet no sensitivity analysis is provided. Understanding how different μ values affect convergence speed and threshold adherence is critical for practitioners.

- **Reactive rather than preventive safety:** For safety-critical applications highlighted in the introduction (robotics, autonomous vehicles), a single threshold violation could have real consequences. The paper correctly acknowledges this limitation, but the framing of "maintaining" performance and "guarantees" should be tempered accordingly—the method *recovers* from violations rather than preventing them.

## Nice-to-Haves

- **Online estimation of β_l:** A mechanism to estimate learner suboptimality online rather than requiring manual specification would improve practical deployability.

- **Predictive roll-back:** A look-ahead mechanism to anticipate threshold violations before they occur would strengthen safety claims for critical applications.

- **Analysis of convergence rate:** The curriculum strategy reduces α by (1−α) each time the return surpasses R̂_πg, but no analysis of how many steps this requires or how it scales with environment complexity is provided.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **Algorithm placement in appendices:** The criticism that Algorithms 1 and 2 should be in main text is a formatting preference, not a substantive weakness.
- **Equation 1 discrete-space framing applied to continuous:** While the return definition states it's for discrete spaces, the extension to continuous is standard practice and not a fundamental issue.
- **"5 seeds" criticism:** The paper uses 50 seeds for Combination Lock. AntMaze seed count isn't explicitly stated, but this is a minor reproducibility concern rather than a core flaw.
- **Static baselines as weak strawmen:** While static sampling baselines are simple, they remain relevant for isolating the effect of adaptive sampling and are used in prior guided RL work.

## Novel Insights

Beyond the paper's contributions, an interesting tension emerges: the theoretical guarantee becomes most fragile exactly where it would be most valuable—in complex, continuous domains where convergence between curriculum steps cannot be assured. The paper's honest acknowledgment that GRL alone requires convergence while GRL-RB is reactive suggests a deeper opportunity: rather than reactive rollback, predictive models of policy improvement (e.g., value function extrapolation) could anticipate violations before they occur. Additionally, the β_l parameter represents a fundamental information gap—the learner's initial suboptimality is unknown in practice, making the guarantee's practical instantiation dependent on conservative guesses that may slow transfer unnecessarily.

## Suggestions

- **Add ablation for μ:** Include experiments varying μ (e.g., μ ∈ {0.5, 0.75, 0.9}) to demonstrate the trade-off between safety and transfer speed.
- **Add standard IQL baseline:** Compare against IQL fine-tuning with retained replay buffer to ensure fair baseline comparison.
- **Clarify β estimation for continuous domains:** Provide a concrete procedure or empirical validation for estimating β_l and β_g in continuous state/action spaces, beyond defaulting to conservative values.
- **Add at least one MuJoCo locomotion domain:** Even a single additional domain beyond navigation would strengthen claims of broad applicability.
- **Report AntMaze seed count explicitly:** Clarify the number of random seeds used for statistical significance.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
