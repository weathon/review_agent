## Summary

SEQUOIA addresses restless multi-armed bandits with strongly coupled combinatorial actions (CORMAB), a class of problems where standard Lagrangian/Whittle-index decoupling fails. The authors introduce four novel CORMAB problem formulations (multiple interventions, bipartite scheduling, capacity constraints, and path planning) and propose an algorithm that embeds a trained ReLU Q-network into a mixed-integer linear program (MILP) to tractably select combinatorial actions at each timestep. Empirical evaluation across all four settings shows SEQUOIA outperforms myopic and iterative baselines by an average of 24.8% over a RANDOM baseline.

---

## Strengths

- **Novel problem class with four well-motivated instantiations.** The CORMAB formulations—multiple interventions with sigmoid-link compounding effects, scheduling as bipartite matching, capacity-constrained generalized assignment, and path-constrained routing—are all genuinely new for the restless bandit literature and cover a broad, practically relevant design space. Most prior RMAB work is confined to simple budget constraints.

- **Clean and principled integration of DQN and NN-in-MILP.** Taking the state-action pair as Q-network input (rather than the standard output-per-action design) is a necessary and non-obvious architectural choice that enables the MILP embedding. The resulting pipeline is well-specified in Algorithm 1 and Figure 2, and correctly distinguishes training- and inference-time MILP usage.

- **Careful warm-starting strategy that makes the approach tractable.** The three-phase initialization (myopic pre-training, on-policy seeding via MILP, action space diversification with perturbed/infeasible samples) directly addresses the otherwise prohibitive MILP solve count. That this is needed at all, and how it is designed, is a substantive engineering contribution.

- **Reproducibility.** The authors provide a public GitHub repository, detailed appendices for each domain (transition dynamics, MILP formulations, hyperparameters), and evaluate across 30 random seeds per setting, which is commendable.

- **Honest and well-scoped related work.** The paper clearly differentiates SEQUOIA from Delarue et al. (2020) (deterministic, single-stage) and from prior RMAB work (weak coupling), and acknowledges the RDDL connection as an open direction with identified technical obstacles.

---

## Weaknesses

### Fatal
None.

### Major

- **No performance upper bound, making it impossible to assess solution quality.** All results are normalized so that RANDOM = 1, but RANDOM achieves very different fractions of optimal across problem settings. The 24.8% headline number is uninterpretable without knowing how far any method is from optimal. Even a small-scale exact MDP solution or LP-relaxation upper bound for J=20 would allow readers to assess whether SEQUOIA recovers 60% of available improvement or 99%. As stated, the experiments establish *relative ordering* among baselines but not *absolute quality*.

- **No ablation of the infeasible-action training trick.** Section 4.2 introduces including infeasible actions as a key diversity technique: *"Incorporating perturbed and even infeasible actions greatly increases the diversity of potential samples."* This design choice could distort Q-values near the feasibility boundary, since the network is trained on a mixture of feasible and infeasible state-action pairs while the MILP at inference only queries the feasible region. Whether the performance difference between SEQUOIA and ITERATIVE DQN is driven by MILP-based global optimization versus this infeasibility training trick is never disentangled. An ablation comparing SEQUOIA with and without infeasible samples is necessary to validate the core claim.

- **Scalability characterization is insufficient.** The largest evaluation is J=100 arms with N=20 workers and H=20. Many real-world RMAB deployments cited in the paper involve hundreds or thousands of patients. Since MILP is NP-hard and computational cost is explicitly acknowledged as a bottleneck (640,000 solves for a modest setting), the paper should characterize where the method breaks down—either empirically (J=200, 500) or by analyzing MILP solve-time growth versus J and N. Without this, the practical reach of SEQUOIA is unclear.

### Minor

- **Runtime analysis is relegated entirely to Appendix G.** Given that computational cost is described as the primary bottleneck and is central to the method's practical value, at minimum a summary table comparing wall-clock training time and per-step inference time against baselines should appear in the main text. The Spark Finder's suggestion of reward-vs.-wall-clock-time curves would further clarify whether the performance gains are worth the computational investment.

- **No error bars or confidence intervals in Figure 3.** Results are averaged over 30 seeds, but Figure 3 shows only bar heights with no indication of variance. For comparisons where the gap between SEQUOIA and ITERATIVE DQN is visually moderate (e.g., capacity constraints at J=100), statistical significance cannot be assessed.

- **Big-M constraint values are not discussed.** The MILP encoding of ReLU activations uses big-M constraints (Equations 4–6), and the paper cites Huchette et al. (2023) which extensively documents the numerical and computational difficulties of loose M values. No discussion of how M values are selected or whether bound-tightening is employed is provided. This is a practically important omission given that MILP tractability is the method's key bottleneck.

- **Transition notation $P^\times : \mathcal{S}^J \times \mathcal{A}^N \times \mathcal{S}^J \rightarrow [0, 1]^J$ is non-standard.** The codomain $[0, 1]^J$ suggests J independent probability scalars rather than a distribution over $\mathcal{S}^J$. Since the paper alternates between per-arm independence (Section 2.1) and joint transitions (Section 2.2), this notation ambiguity should be resolved.

- **The bipartite matching formulation (Section 3) is significantly less detailed than the other three.** Transition dynamics for patients between scheduling windows, the relationship between the scheduling horizon K and the MDP horizon H, and the MILP constraints are not described in the main text. This makes the setting difficult to reproduce without access to the appendix.

- **Memoization efficiency is not characterized.** The memoization strategy (Section 4.2) is presented as a key computational optimization, but the cache hit rate is never measured or reported. In large or continuous state spaces, cache effectiveness could be negligible.

### Tiny

- The $\gamma \in (0, 1]$ specification allows $\gamma = 1$, but the undiscounted infinite-horizon value function is only well-defined under ergodicity conditions that are not stated. This should either be restricted to $\gamma < 1$ or accompanied by a brief remark.

- The conclusion (Section 7) discusses IPOPT as a promising heuristic that "finds significantly better actions than Gurobi" on capacity-constrained CORMAB given equal time. If this holds generally, it significantly changes the method's practical recommendation and warrants a more prominent empirical comparison rather than a brief remark.

---

## Nice-to-Haves

- **Whittle-index-with-ignored-coupling as a baseline.** For settings where Whittle index can be applied by ignoring coupling (e.g., capacity-constrained CORMAB reduced to a simple budget constraint), including this baseline would directly quantify the *cost of ignoring coupling*, strengthening the paper's central claim.

- **Ablation of warm-start phases individually.** Separating the contribution of (1) myopic pre-training, (2) on-policy seeding, and (3) action space diversification would give readers insight into which components drive computational savings versus reward gains.

- **Sensitivity analysis to constraint tightness.** As constraints tighten (e.g., smaller budgets in capacity-constrained CORMAB), the MILP feasible region shrinks and solve times may increase. Characterizing how performance degrades with tightening constraints would be informative.

- **IPOPT relaxation as a full alternative method in experiments.** Given the conclusion's positive comment about IPOPT on capacity-constrained CORMAB, a systematic comparison of Gurobi vs. IPOPT across all four settings would strengthen the paper's practical impact.

- **Graph neural network architectures.** Replacing the generic MLP with a GNN that reflects the arm-coupling structure (e.g., the graph in path-constrained CORMAB) is a natural and potentially impactful extension briefly mentioned in Section 7.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Claim of being first should be supported in the abstract"** — Stylistic/formatting nitpick; the related work section adequately supports this claim.

- **"Policy gradient methods (PPO/REINFORCE with combinatorial decoder) should be discussed"** — The paper states explicitly that it is unaware of applicable RL algorithms from the literature due to the combinatorial action structure, and cites RLlib as an example. Demanding coverage of every potential alternative methodology goes beyond standard expectations for an empirical systems paper; appropriate as a suggestion, not a weakness.

- **"Delarue et al. distinction needs more careful treatment"** — The paper's distinction (stochastic + sequential vs. deterministic + single-stage) is correctly stated. The critic's argument that VRP sub-tours constitute a sequential problem conflates the *construction process* of a single solution with a *multi-step MDP with stochastic transitions*—these are genuinely different.

- **"Reward shaping / fairness objectives are missing"** — Fairness constraints on the reward function are explicitly outside the paper's stated scope, which focuses on action-space coupling. This is scope creep.

- **"Q-function approximation error bounds"** — Demanding theoretical guarantees from an empirical systems paper is non-standard for this community.

- **"Actor-critic methods for large discrete action spaces (Branching DQN, etc.) are missing from related work"** — Since the paper focuses on *combinatorial* (not merely large discrete) action spaces, and the cited branching DQN methods are not designed for combinatorial feasibility constraints, this is a questionable missing-reference criticism. Per instructions, no claims about missing related work are made.

- **"Unfair baseline comparison"** — No such issues were identified; all comparisons appear to be symmetric or favor baselines.

---

## Novel Insights

The paper surfaces an underappreciated distinction between *weakly coupled* RMABs (where Lagrangian relaxation enables per-arm decoupling) and *strongly coupled* CORMABs (where the constraint structure of the action set prohibits any such decomposition). The formalization of this distinction via four concrete instantiations—each with a different combinatorial constraint topology—is the paper's most intellectually original contribution. A secondary insight is that training on *infeasible* actions can be beneficial when the per-step feasible set is small or hard to explore: because per-arm transitions are defined independently, the Q-network can learn from infeasible state-action pairs while the MILP restricts inference to the feasible set. This idea of deliberately *expanding the training distribution beyond the inference feasible set* in combinatorial RL is novel and deserves more rigorous investigation than the paper provides.

---

## Suggestions

1. **Add a small-scale upper bound** (exact value iteration or LP relaxation) for J=20 in each of the four settings. This single addition would transform the evaluation from a purely relative ranking into an assessable measure of solution quality.

2. **Provide an ablation table** separating SEQUOIA (a) with infeasible samples vs. without, and (b) with full warm-start vs. cold-start DQN. These are the two most consequential undocumented design choices.

3. **Move a runtime summary into the main text.** A table showing median per-episode wall-clock time (training and inference) for each method and each problem size (J=20/40/100) would be sufficient and is essential given the paper's computational emphasis.

4. **Report error bars on Figure 3.** With 30 seeds this is straightforward; even small shaded regions or standard-deviation bars would allow significance to be visually assessed.

5. **Clarify and tighten the big-M discussion.** At minimum, state the M-setting strategy (e.g., interval arithmetic pre-processing) used and cite relevant bound-tightening techniques from the NN verification literature.

6. **Formalize the bipartite matching setting** in the main text (at least one paragraph on transition dynamics and the relationship between scheduling horizon K and MDP horizon H) to ensure the problem is fully reproducible from the main paper.

---

**Assessment across axes:**

- **Novelty**: Moderate-to-high. The CORMAB problem class and four formulations are original; the algorithmic approach is an integration of known components (DQN + NN-in-MILP), but the integration itself and the warm-starting strategy are non-trivial.
- **Technical soundness**: Moderate. The algorithm is mechanically correct, but key design choices (infeasible-action training, M-value selection) are unablated and underjustified.
- **Empirical support**: Moderate. Broad coverage (4 domains, 30 seeds) but lacks upper bounds, error bars, runtime comparisons in the main text, and ablations of critical components.
- **Significance**: Meaningful for AI for Social Good and operations research; the problem class addresses a real gap that existing RMAB and combinatorial RL literatures do not. Scalability limitations constrain near-term deployment impact.
- **Clarity**: Good overall; notation in Section 2.2 and the bipartite matching description are the main gaps.