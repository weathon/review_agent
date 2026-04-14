---

## Summary

This paper introduces CORMAB, a new class of restless multi-armed bandit (RMAB) problems where combinatorial action constraints couple the arms such that standard Lagrangian/Whittle-index decoupling becomes inapplicable. To solve CORMAB, the authors propose SEQUOIA, which embeds a trained DQN Q-network into a mixed-integer linear program (MILP) to select combinatorial actions at each timestep during both training and inference. Four concrete CORMAB instantiations are introduced (multiple interventions, path constraints, bipartite matching, capacity constraints), and SEQUOIA is empirically shown to outperform myopic and iterative baselines by an average of 24.8% across these settings.

---

## Strengths

- **Genuinely novel problem class and formulations.** The CORMAB framing cleanly identifies a gap in the existing RMAB literature—settings where arms are *strongly coupled* so Lagrangian relaxation breaks down—and provides four well-motivated instantiations tied to real-world resource allocation. To the authors' knowledge (and consistent with the related work survey), no prior work addresses sequential stochastic planning under per-timestep combinatorial action constraints of this kind.

- **Integration of MILP-based action selection into the RL training inner loop.** The key technical contribution—using the MILP not only at inference but also during on-policy training to compute Bellman targets—is non-trivial and distinguishes SEQUOIA from the prior work of Delarue et al. (2020), which addresses a single-shot, deterministic setting. The encoding of ReLU networks into MILPs (Fischetti & Jo, 2018) is well-adapted to this training loop.

- **Infeasible-action warm-starting is a specific and clever insight.** The observation that per-arm state transitions are defined independently, so Q-network training can exploit *infeasible* actions to improve diversity without corrupting the simulation, is a non-obvious and practically useful contribution rather than a generic "we used data augmentation" claim.

- **Robustness across settings without per-domain tuning.** Using the same two-layer architecture and hyperparameters across all four CORMAB instantiations and still achieving strong performance demonstrates that the method is not over-fitted to any single domain, which is an important practical property for a general-purpose approach.

- **Evaluation scale.** Averaging over 30 random seeds across 50 evaluation episodes of length 20 is more thorough than typical RL papers. The ablation against ITERATIVE DQN—which uses the same Q-network but only a greedy construction heuristic—is a clean isolation of the MILP's contribution to action selection.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing model-based planning baselines.** The paper explicitly states in Section 2 that "transition dynamics and reward are known a priori." With known dynamics, methods such as rollout-based planning, approximate dynamic programming, or Monte Carlo Tree Search (MCTS) are natural and often highly sample-efficient competitors. The comparison is limited entirely to model-free baselines (DQN variants and myopic policies). The conclusion mentions connections to RDDL planning and acknowledges a Gurobi-based planner that cannot handle stochasticity, but no systematic comparison against any model-based method is provided. This omission matters because the choice of model-free RL when the model is available is itself a design decision that needs justification, not just assertion.

- **No error bars or statistical significance analysis.** Despite reporting 30-seed averages, Figure 3 presents only bar heights with no visible confidence intervals. For improvements like the 12.1% gap between SEQUOIA and ITERATIVE DQN in the capacity-constrained setting, it is impossible to assess whether the difference is statistically significant. This is a critical gap for an empirically grounded paper.

- **The abstract's 24.8% claim is not precisely attributed.** The abstract leads with "outperforms existing methods by an average of 24.8%." The main text never unpacks this number: which baseline defines the denominator, which settings are included in the average, and how the average is computed across varying J and N. A headline quantitative result this prominent should be precisely defined in the main text.

- **The IPOPT continuous relaxation result is buried and not evaluated as a baseline.** Section 7 (Conclusion) reports that on the capacity-constrained CORMAB, solving a continuous nonlinear relaxation with IPOPT "finds significantly better actions than Gurobi, if we allot the same amount of time to each solver." This is a substantive empirical finding that directly bears on whether exact MILP is necessary—yet it is presented only as a future research direction, without any comparison in the main experiments. If a heuristic relaxation is faster *and* better, the motivation for exact MILP needs stronger support.

### Minor

- **Memoization effectiveness is not quantified.** Section 4.2 describes memoization of MILP solutions as a key computational mitigation, but the actual cache hit rate, the fraction of solves avoided, and the resulting wall-clock speedup are never reported. The MILP computational burden (640,000 solves for a modest run) is prominently flagged as the central bottleneck; the reader has no evidence that the mitigations are actually effective.

- **No runtime-vs.-reward tradeoff analysis.** The MILP is the acknowledged bottleneck, and there is no plot showing training or inference wall-clock time versus achieved reward, making it difficult to assess the practical viability of SEQUOIA relative to baselines that do not incur MILP costs.

- **Path-constrained experiments compare against only weak baselines.** The paper correctly explains (Section 5) that ITERATIVE MYOPIC and ITERATIVE DQN have no natural extension to path constraints (no simple iterative construction for a valid cycle). However, this leaves SEQUOIA compared only against MYOPIC and RANDOM in Figure 3d—its two weakest competitors. The result is less informative than for the other three settings.

### Tiny

- **Big-M bound selection is unaddressed.** The MILP encoding (Eqs. 4–6) uses big-M constraints whose tightness is critical for solve time. This is a well-known issue in the NN-to-MILP literature; even a brief comment on how bounds are derived (e.g., via interval arithmetic) would strengthen the technical presentation.

- **The claim that Lagrangian relaxation cannot be applied to CORMAB is stated without a formal argument.** While the intuition is clear and Ou et al. (2022) are cited, a one- or two-sentence formal justification would sharpen what is the central modeling claim of the paper.

---

## Nice-to-Haves

- An ablation of the three warm-starting strategies (myopic initialization, perturbed myopic, infeasible actions) would clarify which components drive the computational gains and whether all three are necessary.
- A wall-clock runtime plot (training time vs. reward for SEQUOIA vs. baselines) would provide the context needed to assess practical viability.
- Scalability experiments beyond J=100 arms, or at least a characterization of the MILP solve-time scaling curve, would help establish where the method becomes infeasible.
- A comparison against a PPO or actor-critic variant with a constrained projection or feasibility-filtering layer would be informative, even if such methods are expected to be weaker—the negative result would further motivate the MILP-based approach.
- Consider promoting the IPOPT continuous relaxation to a proper baseline in the experiments, given the finding that it outperforms Gurobi under the same time budget for capacity constraints.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"The notation $P^\times \to [0,1]^J$ conflates marginal with joint probabilities"** — The paper immediately clarifies in context that per-arm marginals are intended, and the arm-independence relaxation is precisely what CORMAB models. This is a pure notational nitpick with no impact on correctness.

- **"The claim to be the first to consider sequential combinatorial settings is overstated compared to Delarue et al. (2020)"** — The paper explicitly addresses Delarue et al. in Section 6, distinguishing it as a single-shot, deterministic setting vs. SEQUOIA's stochastic, multi-step sequential planning. The distinction is well-made; the claim is defensible.

- **"Infeasible action training introduces systematic Q-value bias near the feasibility boundary"** — The paper justifies this by the fact that per-arm state transitions are defined independently, so the simulator can compute valid transitions even for infeasible action vectors. The concern about boundary bias is speculative and not empirically demonstrated.

- **"The restriction to known dynamics deserves more prominent acknowledgment"** — Section 2 opens with: "We consider offline, stochastic planning for restless bandits with combinatorial action constraints, where the transition dynamics and reward are known a priori." The scope is clearly stated from the first sentence. The limitation is acknowledged in the conclusion as well. There is nothing hidden here; the criticism of insufficient acknowledgment is factually incorrect.

- **"The path-constrained experiments are less meaningful and weaken the paper"** — The absence of iterative baselines is explained by a genuine structural reason (no natural iterative construction for a valid cycle). The absence is disclosed transparently. The criticism would apply equally to any paper that faithfully reports which baselines cannot be adapted.

- **"No comparison to actor-critic methods with Lagrangian-relaxed policy gradient"** — Adapting PPO or similar methods with a constrained projection to combinatorial action spaces requires non-trivial and potentially paper-length engineering. This is scope creep beyond what the paper claims to address. Moved to Nice-to-Haves as a suggestion.

---

## Novel Insights

The spark-finder correctly identifies one genuinely underappreciated point: the paper's own conclusion reveals that a continuous nonlinear relaxation (IPOPT) finds better actions than the exact MILP Gurobi solve on capacity-constrained CORMAB under equal time budgets. This is not a fatal flaw but it is a substantive finding that the paper sidesteps—it raises the possibility that the exact MILP may be unnecessarily slow in some settings, and a hybrid solver strategy (switching between MILP and continuous relaxation based on time) could improve both performance and speed simultaneously. Elevating this from a paragraph in the conclusion to a first-class experimental finding would meaningfully strengthen the paper's contribution and provide actionable guidance for practitioners.

---

## Suggestions

1. **Add confidence intervals to all bar charts in Figure 3** — Even simple ±1 standard deviation bands over 30 seeds would immediately clarify which improvements are robust.
2. **Precisely define the 24.8% headline number** in the main text (which baseline, which settings, which J values).
3. **Include IPOPT as a baseline** in at least the capacity-constrained experiment to enable a direct comparison, and consider extending it to other settings.
4. **Add a wall-clock runtime analysis** (even a simple table of median MILP solve times per setting and J) to substantiate the computational claims and the effectiveness of memoization.
5. **Include or discuss model-based baselines** (e.g., a rollout-based planner that uses the known $P^\times$), even if only to demonstrate that they fail to scale or are dominated by SEQUOIA.
6. **Run an ablation** toggling off each of the three warm-starting components (myopic init, perturbed myopic, infeasible actions) to isolate their individual contributions to convergence speed and final performance.

---

**Evaluation Summary:**
- *Novelty*: High — CORMAB and the SEQUOIA training-loop integration of MILP action selection are genuinely new contributions not found in prior RMAB or RL-for-CO literature.
- *Technical soundness*: Moderate-high — The DQN+MILP framework is technically correct, but computational analysis is incomplete (memoization, big-M, solver failure rates).
- *Empirical support*: Moderate — 30 seeds is commendable, but missing error bars, missing model-based baselines, the unexplored IPOPT finding, and the thin runtime analysis collectively weaken the empirical story.
- *Significance*: Moderate-high — Opens a useful new problem class with real-world relevance; the general method is applicable beyond restless bandits.
- *Clarity*: Good — Well-organized with clear problem statements; some key computational details are deferred entirely to the appendix.

MY FINAL SCORE: <pineapple>6.4</pineapple>