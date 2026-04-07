## Summary

This paper validates the Goal-Oriented Environment Inference (GOEI) algorithm—previously proposed for abstract environments—in a competitive two-player card game, "Hol's der Geier." The authors demonstrate that GOEI reduces the state representation to approximately 2.9% of reachable observations (452 states from 15,542) while achieving near-Nash-equilibrium performance against a strong opponent. The work provides empirical evidence that minimal core state representations can support effective strategies in sequential decision-making settings.

## Strengths

- **Impressive state compression with preserved performance:** GOEI achieves near-optimal reward rates (~-0.010 vs. NE's 0.000) while reducing states to 2.9% of the observation space reachable under the training distribution (Table 1). At rounds t=2 and t=3, GOEI uses even fewer states than the NE strategy itself (Figure 2B), suggesting efficient information encoding.

- **Rigorous experimental design for isolating inference capability:** The authors cleanly separate environment inference training (on Rand vs. NE games) from strategy evaluation (against NE), preventing confounding between inference quality and strategy adaptation (Section 3.3). This design choice allows clear attribution of performance to the learned state representation.

- **Informative information-theoretic analysis:** Section 4.2 and Figure 3 provide mutual information analysis between learned states and individual observation features, revealing that information about score difference (SD) is preserved at round t=4 while agent/opponent hand information (AH, OH) is largely compressed. This diagnostic offers insight into what the algorithm identifies as "core."

- **Honest acknowledgment of limitations:** The authors transparently discuss the simplified five-card constraint, the offline learning setup, and the gap between state reduction and genuine explainability. This transparency is commendable.

## Weaknesses

- **Incremental novelty:** The core algorithm (GOEI) is imported wholesale from Takahashi et al. (2024). The paper contributes empirical validation on a small card game but introduces no new algorithmic, theoretical, or architectural innovations. For ICLR, an application paper must demonstrate significance beyond applying existing methods—even the state reduction analysis relies on the prior work's framework.

- **Training-test distribution overlap:** GOEI is trained on games between Rand and NE strategies and tested against NE. Since NE appears in both training and evaluation, the learned state representation may be specialized to this opponent distribution. The paper provides no experiments against alternative opponents (e.g., π₀, π₁, or adaptive strategies) to assess generalization of the learned core states.

- **Explainability claim not substantiated:** The introduction motivates GOEI as a solution to the "lack of explainability" in DNN-based agents, yet Section 5 admits "we could not give a verbal explanation of the reduced state representation more concretely than Figure 3." State reduction is necessary but not sufficient for explainability; this paper demonstrates compression without demonstrating interpretability.

- **Weak baseline comparison:** The only baseline is tabular Q-learning. The paper does not compare against modern model-based RL methods (e.g., Dreamer, MuZero) or principled state abstraction approaches (bisimulation, φ-abstraction). Without such comparisons, it remains unclear whether GOEI's compression is superior to standard latent-state methods or simply appropriate for this specific small-scale setting.

- **Statistical reporting gaps:** While 21 training runs with median/quartile reporting are provided, no formal statistical tests compare GOEI's performance to NE equivalence or to Q-learning. The claim of "nearly optimal strategy equivalent to the Nash equilibrium" lacks confidence intervals or significance testing.

- **Limited scale undermines broader claims:** The paper is constrained to a five-card version due to GPU memory (12GB). The observation space grows combinatorially with card count, and the lack of results on standard game sizes limits the scalability conclusions. The authors' suggestion that GOEI "may apply to versions with more than five cards" is speculative without empirical support.

## Nice-to-Haves

- **Online interactive learning experiments:** Testing GOEI in a setting where environment inference and strategy optimization occur simultaneously would strengthen real-world applicability claims. The authors note this as future work.

- **Semantic interpretability of learned states:** A post-hoc analysis mapping reduced states to human-understandable concepts (e.g., "winning position," "must-win round") would substantiate the explainability motivation.

- **Comparison with modern state-abstraction baselines:** Benchmarking against DeepMDP, bisimulation methods, or world-model approaches would clarify GOEI's relative merits.

- **Computational cost metrics:** Wall-clock time and memory usage during training would help readers assess whether the state reduction yields practical efficiency gains beyond theoretical compression.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unknown dynamics" is misleading**: The paper uses "unknown game dynamics" appropriately—the opponent's policy is unknown to the agent. While "unknown opponent strategy" might be more precise, this criticism is overly pedantic and does not harm the paper's technical correctness.

- **Demand for standard 15-card game validation**: The paper explicitly acknowledges the memory constraint (12GB GPU) limiting validation to 5 cards. Requiring results on larger games is beyond the current scope and would require algorithmic extensions for memory efficiency—this is appropriately noted as future work.

- **Demand for human subject studies for explainability**: User studies are not standard expectations for algorithmic contributions at ICLR. The paper's admission that reduced states lack verbal explanation is sufficient transparency about this limitation.

- **Markov assumption as fatal flaw**: The critic suggests history-dependent strategies are needed for optimal play. However, the Nash equilibrium is computed under the same Markov assumption, making this a fair comparison. The assumption is transparently stated and consistent throughout.

- **Demand for online learning validation**: The paper explicitly scopes this to future work and provides the offline setup as a clean isolation of inference capability. While interactive learning is important for real deployment, criticizing its absence is scope creep—the paper's stated contribution is validating GOEI's state reduction in a controlled setting.

## Novel Insights

The observation that GOEI achieves *fewer* states than the Nash equilibrium representation at early rounds (t=2, t=3) while maintaining comparable performance is genuinely interesting. This suggests that the NE strategy, while optimal, may encode redundant information for early-game decisions—potentially because early-round play conditions on less relevant features. This finding hints at an asymmetry in information importance across game stages that merits deeper theoretical investigation: perhaps the Markov "core" needed for optimal play genuinely shrinks in early rounds and expands only as the game approaches its terminal stage (where score difference becomes critical). This could inform adaptive state abstraction strategies that allocate representational capacity dynamically across episode horizons.

## Suggestions

- **Add confidence intervals** around the reward rate comparisons in Table 1 (e.g., via bootstrap) to substantiate claims of "near equivalence" to NE.

- **Include at least one alternative opponent** in the evaluation set (e.g., π₀ or a simple heuristic strategy) to demonstrate generalization beyond the training distribution.

- **Clarify the contribution statement** in the introduction—the paper should explicitly position itself as empirical validation of prior theoretical work rather than claiming novelty in algorithm or theory.