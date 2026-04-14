## Summary

This paper formalizes the novel problem of *preference elicitation for offline reinforcement learning*, where a policy must be learned from an offline dataset without environment interaction and without a reward function—only human preference queries are available. The authors provide a theoretical analysis of sampling-from-buffer approaches (OPRL), then propose Sim-OPRL, which generates candidate trajectories via rollouts in a learned pessimistic world model, optimistically selecting pairs to maximize reward uncertainty. The key theoretical result is that Sim-OPRL eliminates the reward concentrability coefficient C_R present in OPRL's bound (Theorem 6.1), theoretically justifying its empirically observed sample efficiency across five diverse environments.

---

## Strengths

- **Elimination of C_R in Theorem 6.1 is a genuine and non-trivial theoretical improvement.** By designing the elicitation strategy to always cover the near-optimal policy set Π_offline, the authors remove the reward concentrability term that Zhan et al. (2023a) showed can be arbitrarily large for buffer-sampled preferences. This is the paper's central theoretical insight and is well-supported by the proof structure.

- **The dual pessimism/optimism decomposition is cleanly motivated and empirically validated.** The ablation in Figure 2 confirms that removing pessimism w.r.t. transition uncertainty in either the output policy or the rollout strategy leads to significant performance degradation—this is not a trivial result and directly supports the theoretical framing.

- **The problem formalization in Section 3, particularly Definition 3.1 and the ε_T-floor optimality criterion, is precise and practically grounded.** Acknowledging that preference elicitation cannot beat the irreducible transition-model error is an honest scoping of the problem that strengthens the theoretical analysis.

- **Theorem 5.1 delivers a theoretical analysis of OPRL that was previously absent**, recovering Zhan et al. (2023a) as a special case and formally capturing the α ≤ 1 coefficient that explains the empirical advantage of uncertainty sampling over uniform sampling (Section 7, Figure 1 and Table 2).

- **The Sepsis simulation result is particularly meaningful**: OPRL-uniform fails entirely (marked ✗ in Table 2), while Sim-OPRL succeeds with 225 ± 46 queries. This extreme gap, in a medically-motivated environment with sparse rewards, is strong evidence for the method's advantage in exactly the settings the paper targets.

---

## Weaknesses

- **FREEHAND (Zhan et al., 2023a) is listed in Table 1 as offline + practical implementation + robustness guarantees, but is absent from the experimental comparisons.** Since FREEHAND is the closest prior offline preference-based RL algorithm with both theory and code, its omission is a substantive gap. OPRL (Shin et al., 2022) lacks robustness guarantees, making it a weaker baseline than FREEHAND on the method's own comparison table. The authors should explain why OPRL is sufficient as the primary offline baseline, or include FREEHAND.

- **D4RL evaluation is limited to HalfCheetah-Random—the easiest split in terms of offline RL difficulty**, where even a weak model can capture the transition dynamics well. Medium or medium-expert splits, where the behavioral policy is more structured and concentrability is more challenging, would provide far more convincing evidence of Sim-OPRL's scalability to realistic continuous-control settings. Restricting to HalfCheetah-Random also means the hardest case for the world model (OOD extrapolation) is not tested.

- **There is a potential notation inconsistency between Theorems 5.1 and 6.1 on the link function coefficient κ.** Theorem 5.1 defines κ = sup 1/σ²(r), while Theorem 6.1 defines κ = sup 1/σ(r)—a different functional form without explanation. This is either a typo affecting the stated bound or an unexplained difference in proof technique that must be clarified.

- **All ablations (Figure 2) are conducted exclusively on StarMDP**, the simplest of the five environments. It is unclear whether the conclusions about pessimism hold in HalfCheetah-Random or Sepsis, where the state space is large and the model may be significantly less accurate. At a minimum, the paper should acknowledge this limitation explicitly.

- **The theory-to-practice gap is substantial but incompletely addressed.** The theoretical guarantees rely on well-calibrated confidence sets 𝒯 and ℛ; the practical implementation uses neural network ensemble disagreement, which does not satisfy this property for OOD inputs—exactly the regime offline RL must handle. The paper cites Appendix A.2 for a lower-bound result under "well-calibrated uncertainty estimates," but the condition under which ensemble disagreement qualifies is never formally justified.

- **Computational cost of Sim-OPRL is entirely unaddressed.** Sim-OPRL requires training N_T transition model ensembles, N_R reward model ensembles, N_R separate policy models (one per reward ensemble member), and performing rollouts for all policy pairs per preference query. This is substantially more expensive than OPRL, which simply samples from a buffer. For practitioner guidance—and especially given the healthcare framing—the compute overhead should be quantified or at least discussed.

- **The anomalous behavior in Figure 3b (density ratio = 1 requiring ~22 preferences vs. ~5 for ratio = 2–5) is explained only qualitatively.** While the authors attribute it to low trajectory diversity in a near-deterministic dataset, this interesting failure mode (where "more optimal data" is actually harder for Sim-OPRL) has no theoretical treatment and no parallel in the OPRL curve.

---

## Nice-to-Haves

- **Robustness to preference noise**: All experiments use noiseless Bradley-Terry feedback drawn from the true reward. Adding even simple label-flip noise (5–15%) would strengthen claims about applicability to real human annotators.
- **Hyperparameter sensitivity for λ_T, λ_R**: Offline RL methods are known to be sensitive to pessimism coefficients. A sweep or sensitivity plot would address whether the reported gains require careful tuning.
- **Computational cost comparison**: A wall-clock time or FLOP comparison between Sim-OPRL and OPRL per preference query would let practitioners assess whether the sample efficiency gain justifies the computational overhead.
- **Visualization of simulated vs. real trajectories**: Side-by-side plots in simple environments would transparently expose whether the model generates plausible rollouts or hallucinates OOD states.
- **Discussion of safety risks of synthetic queries in healthcare**: The paper argues that querying on synthetic trajectories is preferable in healthcare settings (lines 251–252), but if the learned model is inaccurate, clinicians could evaluate clinically implausible or harmful trajectories, biasing the reward in dangerous ways. This failure mode warrants explicit discussion.

---

## Removed Points

*These points are flagged to be removed—treat them with caution.*

- **"Claim of overstatement of novelty"**: The harsh critic argues that FREEHAND and Zhu et al. (2023) make the paper's framing of "bridging the gap" overstated. In fact, as the paper correctly notes, neither FREEHAND nor Zhu et al. consider *active* preference elicitation in the offline setting—they study passive offline preference RL. The actual novelty claim (active elicitation + offline constraint) is accurate.

- **"Realizability Assumption 3.1 is not satisfied in practice with neural networks"**: This is standard in all theoretical RL work with general function approximation and is not a specific weakness of this paper. The paper explicitly acknowledges the gap between theory and practice.

- **"6 seeds is below ICLR norms"**: 6 seeds with 95% CIs reported is within acceptable norms for this type of paper, especially for tabular/small-scale environments. This is a generic nitpick.

- **"The density ratio can be infinite"**: The paper correctly cites and defines the concentrability coefficient, notes it is upper-bounded by the density ratio, and works within the standard offline RL assumption that it is finite. Flagging infinite-density-ratio edge cases is standard theoretical concern for all offline RL papers, not specific to this work.

- **"Π_offline computation is intractable"**: The paper explicitly addresses this in Section 6.3 (one policy per ensemble reward member), following Lindner et al. (2021). The approximation is clearly stated; demanding a formal bound on approximation error relative to the theoretical Π_offline is beyond the scope of an empirical systems paper.

- **"Claim that α < 1 condition is not precisely defined"**: The theorem correctly states α ≤ 1 for uncertainty sampling and α = 1 for uniform, with the coefficient arising from the proof structure; the verbal description is consistent with the mathematical statement.

- **"Fixed ε=20 threshold is unjustified"**: With returns normalized to [0,100], ε=20 represents a 20% suboptimality gap—a reasonable and interpretable threshold for comparing sample complexity.

---

## Novel Insights

The reviews collectively surface one genuinely sharp observation: the paper provides evidence that the fundamental advantage of active policy-based elicitation (minimizing candidate optimal policy uncertainty) over information-gain-based elicitation (maximizing reward entropy) holds equally in the offline model-based setting as it does in the online setting—despite the fundamental structural difference that the "queries" are now filtered through a pessimistic world model rather than the true environment. The Sepsis result, where OPRL-uniform fails catastrophically while Sim-OPRL converges at 225 queries, is more than an empirical win: it suggests that in sparse-reward, long-horizon tasks with low offline coverage, reward-function-agnostic exploration at the policy level is the only viable elicitation strategy. The Figure 3b anomaly (near-optimal data being harder for Sim-OPRL than moderately suboptimal data) also hints at an underexplored identifiability problem: when all plausible policies agree on trajectories, preference queries carry no discriminative signal—a phenomenon distinct from either coverage or model error, and potentially worth formalizing.

---

## Suggestions

1. **Add FREEHAND as an experimental baseline**, or provide an explicit argument (e.g., equivalence to one of the OPRL variants under specific parameter settings) for why OPRL is a sufficient proxy.
2. **Evaluate on at least one additional D4RL split** (e.g., HalfCheetah-Medium or Hopper-Random) to demonstrate that the world model quality advantage of Sim-OPRL persists beyond the easiest offline split.
3. **Clarify or correct the κ definition discrepancy** between Theorems 5.1 and 6.1—either add a sentence explaining why the proof techniques differ, or fix the typo.
4. **Extend Figure 2 ablations to at least one continuous environment** (HalfCheetah or Sepsis) to validate that the pessimism conclusions hold beyond StarMDP.
5. **Add a compute cost discussion or table** comparing Sim-OPRL and OPRL in terms of wall-clock time or total model training steps per preference query.

---

**Novelty**: High — the problem formalization and active elicitation strategy for the offline setting are genuinely new, and the theoretical result eliminating C_R is a clean contribution.

**Technical soundness**: Moderate-to-high — the core theorems are correct and well-structured; the κ inconsistency and theory-to-practice gap (ensembles vs. confidence sets) are unresolved issues of moderate severity.

**Empirical support**: Moderate — consistent gains across 5 environments with 6 seeds, but limited to HalfCheetah-Random from D4RL, no continuous-control ablations, and missing the most natural offline baseline (FREEHAND).

**Significance**: High for the safety-critical offline RL community; the Sepsis result in particular demonstrates practical stakes.

**Clarity**: Good — the algorithm, theory, and experiments are well-organized and clearly described; the practical gap is partly glossed over.

MY FINAL SCORE: <pineapple>6.2</pineapple>