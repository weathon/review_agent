## Summary
This paper introduces two novel Monte-Carlo Tree Search (MCTS) algorithms, CATSO and PATSO, which combine distributional return estimates at Q‑nodes with Thompson Sampling and an explicit polynomial optimism bonus. The authors provide non‑asymptotic regret bounds (O(n⁻¹ᐟ²) simple regret at the root) and a novel connection to Wasserstein Distributionally Robust MDPs. Empirical evaluation on synthetic stochastic trees and 12 Atari games shows competitive performance against strong baselines, with comprehensive ablations analyzing each component.

## Strengths
- **Novel algorithmic synthesis:** The paper cleanly unifies distributional value representations (categorical or particle-based), Thompson sampling for action selection, and count-based optimism within a single MCTS framework. This integration is clearly specified and well-motivated for stochastic settings.
- **Strong theoretical grounding:** Non‑asymptotic regret guarantees are provided for both algorithms, matching the state‑of‑the‑art rate for fixed‑depth MCTS. The analysis elegantly models nodes as non‑stationary bandits and lifts results to the full tree. The connection to Wasserstein distributionally robust MDPs offers an insightful robustness interpretation and a sample‑complexity bound.
- **Thorough empirical validation:** Experiments cover a range of synthetic tree configurations and 12 Atari games, with careful comparisons against multiple strong baselines. The ablation studies systematically isolate the effects of distributional Q‑nodes, Thompson sampling, optimism, and backup rules, providing clear evidence for each design choice.
- **Practical engineering contributions:** The “merge‑on‑insert” mechanism for PATSO caps memory usage while preserving theoretical guarantees. Hyperparameter sensitivity analyses show robustness, and runtime/memory measurements are reported, making the algorithms practical.

## Weaknesses
### Major:
- **Theoretical assumptions may not fully capture MCTS non‑stationarity:** The analysis relies on an asymptotic stationarity assumption (Assumption 1) that each node’s reward process stabilizes to i.i.d. samples from a limiting distribution. While justified intuitively, this assumption simplifies the interdependence of nodes in a growing tree, and the resulting guarantees might not hold if the assumption is violated in practice.
- **Empirical improvements are modest and not consistently superior:** The core claim that distributional Q‑nodes provide a key advantage in stochastic environments is not strongly supported. Ablations show scalar Thompson sampling with the same optimism bonus (ScalarTSOpt) performs similarly to CATSO with mean backup (Table 2). In deterministic/low‑noise settings, Power‑UCT often outperforms the proposed methods (Table 4). Atari results show CATSO/PATSO are competitive but not dominant, with UCT and Power‑UCT winning or tying in several games (Table 1). This undermines the claim that distributional representations are crucial for robust performance.

### Minor:
- **Connection to Wasserstein robustness is underdeveloped:** The link to Wasserstein Distributionally Robust MDPs is presented as an after‑the‑fact interpretation rather than a design principle. The sample‑complexity bound (Theorem 5) has an exponential dependence on horizon, limiting its practical relevance, and no experiment validates the robustness interpretation.
- **Scalability demonstration is limited:** Experiments are confined to synthetic trees of moderate depth/branching and deterministic Atari games. The paper does not show performance in extremely large, continuous, or genuinely stochastic/partially observable domains, leaving open questions about broader applicability.
- **Presentation of results could be more statistically rigorous:** Table 1 reports wins/ties counts that may inflate perceived success without statistical significance testing. Standard deviations are provided, but confidence intervals or significance tests would strengthen the comparisons.
- **Omission of a key baseline in Atari comparisons:** The theoretically relevant Fixed‑Depth‑MCTS baseline (Shah et al., 2022) is included in synthetic tree experiments (Figure 1) but not in the main Atari table (Table 1), making it harder to assess relative performance against this state‑of‑the‑art method.

### Trivial:
- *None*

## Nice-to-Haves
- Ablation of the optimism bonus (pure Thompson sampling without bonus) to isolate its contribution.
- Empirical validation of the WDRMDP connection via robustness tests under dynamics/reward perturbations.
- Visualization of Q‑node distributions over time to illustrate how the algorithm differentiates actions.
- Guidance on setting hyperparameters (e.g., optimism constant C) for new domains.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticism that V‑nodes remaining scalar severely limits the distributional nature:** The paper explicitly discusses this design choice in Section 3.3, justifying it for tractability, and ablations isolate the backup rule’s effect. It is a reasoned trade‑off, not a flaw.
- **Claim that the extension from bandits to the tree is “hand‑wavy”:** The paper provides Theorems 3 and 4 with proofs in the appendix; the extension is formally stated.
- **Suggestion that hyperparameter insensitivity implies the distributional parameterization is not crucial:** Flat sensitivity (Tables 5‑6) indicates robustness, not a lack of importance, and the ablation studies directly test the distributional component.
- **Request for comparisons to modern distributional RL planning methods (e.g., IQN, QR‑DQN):** This would require adding new baselines beyond the paper’s scope; the paper already compares to several strong MCTS variants.

## Suggestions
- Provide statistical significance tests (e.g., confidence intervals or p‑values) for the Atari results to clarify whether performance differences are meaningful.
- Include Fixed‑Depth‑MCTS in the Atari comparison table if feasible, or explain why it was omitted.
- Discuss the limitations of the asymptotic stationarity assumption more thoroughly, possibly with empirical evidence of node reward stabilization in the tested environments.