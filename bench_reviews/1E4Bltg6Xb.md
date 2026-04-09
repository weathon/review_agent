## Summary

This paper proposes a Dynamics Feature Representation (DFR) framework for RL-based Dynamic Path Planning (DPP) in urban road networks. DFR hierarchically refines high-dimensional global traffic dynamics into compact, decision-relevant features through two stages: (1) a policy attention mechanism that uses a pre-trained static shortest-path policy to extract a task-relevant subgraph, and (2) an n-hop neighborhood method that further decouples this subgraph into agent-centric local features. Experiments on three urban road networks demonstrate improved RL performance and faster planning times compared to full-dynamics baselines.

## Strengths

- **Principled hierarchical approach to the completeness-efficiency trade-off**: The two-stage refinement (global task filtering via policy attention → agent-centric local encoding via n-hop neighborhoods) is a well-motivated architectural solution to a genuine problem. Rather than heuristically choosing between local and global state, DFR provides a structured middle path, which is a meaningful design contribution.
- **Substantial empirical efficiency gains with maintained performance**: DFR reduces average planning time by 85.59% (DQN), 46.08% (GCN+DQN), and 79.32% (PPO) compared to all-dynamics baselines, while simultaneously improving or matching success rate and mean GAP. These are non-trivial efficiency gains for real-time planning.
- **Systematic ablation study**: The (k, n) ablation in Figure 6 provides useful practical insights—showing that n exhibits diminishing returns while k has more complex behavior—and the authors honestly report these findings rather than cherry-picking.

## Weaknesses

### Major:

- **Static prior filtering risks excluding the dynamic optimum**: The policy attention mechanism filters the state space to edges along the top-k *static* shortest paths (πd*). In dynamic environments where the objective is travel time (not distance), the true optimal dynamic path may deviate substantially from static shortest paths—for example, taking a longer detour to avoid severe congestion. If the dynamic optimal path lies outside the static top-k subgraph, the RL agent is structurally prevented from learning it, making the policy suboptimal by design. The paper acknowledges that "distance naturally serves as one of the most fundamental constraints" (Section 4.3) but does not analyze the risk or magnitude of this filtering error. An empirical analysis quantifying what fraction of dynamic-optimal edges are retained by the static subgraph (a "recall" metric) would directly address this concern.

- **PSR theoretical claims are overstated**: Section 4.2 claims that "Grounding DFR in PSR principles thus guarantees that the resulting representations are compact, temporally predictive, and theoretically sufficient." However, PSR requires that the state representation enable prediction of *all* future observation sequences given action sequences. The paper provides no formal proof that an n-hop neighborhood of a static subgraph satisfies this sufficiency criterion, particularly when traffic dynamics exhibit long-range spatial correlations (e.g., congestion propagating from beyond the n-hop radius). The invocation of PSR is currently a loose analogy rather than a rigorous justification, and the word "guarantees" should be replaced with a more measured claim or supported by formal analysis.

### Minor:

- **Unnecessary use of RL for static subgraph generation**: The policy πd* is obtained by training an RL agent on static distance-based rewards (Section 4.3). For static shortest paths on a known graph, exact algorithms like Dijkstra or Yen's algorithm (for top-k paths) are both faster and exact. Using an approximate RL policy introduces unnecessary approximation error and training overhead. The paper does not justify this design choice.

- **Synthetic dynamics without specified temporal correlation**: The congestion factor β ∈ [0.1, 1.5] is applied to real OSM topologies, but the paper does not specify how β evolves over time—whether it is i.i.d., Markovian, or has longer-range temporal dependencies. This matters because the n-hop state design implicitly assumes that local spatial context captures the relevant temporal dynamics. Without temporal correlation structure, the claim of "realistic" evaluation (Section 5.1) is partially unsupported.

- **Unclear AD baseline implementation for MLP-based methods**: For DQN (MLP-based), the "All Dynamics" baseline must handle the full graph's edge weights as input. Since MLPs require fixed-size input, it is unclear how the variable-sized graph is encoded (flattened? padded?). This ambiguity makes it hard to assess whether the AD baseline is a fair comparison or a strawman.

- **Limited baseline comparison with recent RL-based DPP methods**: The paper compares only DQN, PPO, and GCN+DQN against DFR-enhanced versions. While the paper's focus is on the impact of state representation within the RL paradigm, comparisons with more recent RL-based DPP or state representation methods would better contextualize the contribution.

### Trivial:

- **Imprecise language about Markov property**: The Introduction states "insufficient state representation may undermine the Markov property." Technically, insufficient representation creates partial observability (POMDP), not a violation of the environment's Markov property. This is a language issue rather than a conceptual error—the authors' intended meaning is clear.

## Nice-to-Haves

- Empirically verify the Markov property claim by testing whether adding history buffers to the DFR state improves policy performance. If history helps significantly, the compressed state is not informationally sufficient as claimed.
- Develop an adaptive mechanism for selecting k and n (e.g., based on traffic volatility or graph density metrics), as the authors themselves identify this as a limitation.
- Validate on real-world traffic traces (e.g., historical speed data from PeMS or similar) rather than synthetic congestion factors, to test robustness under realistic temporal correlations.
- Visualize the policy attention subgraph overlaid on the ground-truth optimal dynamic path for specific episodes, to reveal whether gains come from noise reduction or information loss.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Road closures / topological changes invalidate pre-computed subgraphs.** The paper explicitly assumes V remains constant in Section 3.1 ("It is assumed that V remains constant"). Criticizing the absence of topological changes is scope creep—the paper's stated problem is weight dynamics, not structural dynamics.
- **Weakness: Statistical significance testing missing.** Single-run or few-seed evaluation without formal significance tests is the norm for this type of RL experiment. Demanding t-tests or confidence intervals is a nice-to-have, not a core flaw.
- **Weakness: Parameter sensitivity of k and n.** The authors already acknowledge this limitation in the Conclusion ("the two parameters of k and n in DFR are manually selected in this study, which may limit its practical applicability") and propose adaptive selection as future work. Criticizing what the authors already reasonably address is double-counting.
- **Weakness: Pre-computing top-k paths for all source-destination pairs is O(N²).** The paper states that both policy attention and n-hop neighborhoods "depend only on the fixed road network topology, allowing offline computation and reuse." The concern about storage is reasonable but speculative without evidence that it is actually a bottleneck; the paper demonstrates feasible computation on the tested networks.
- **Weakness: Missing related works.** Per hard rules, we cannot confirm the existence of suggested missing references.
- **Weakness: Formatting/style issues.** Per hard rules, these are removed.

## Novel Insights

The most insightful observation across the reviews is the fundamental tension at the heart of DFR: the method's primary strength (drastic dimensionality reduction via a static structural prior) is also its primary vulnerability (the static prior may systematically exclude the dynamic optimum). This is not merely a theoretical concern—it creates a testable prediction. In low-volatility regimes where dynamic optima align with static shortest paths, DFR should excel; in high-volatility regimes where congestion forces large detours, DFR's performance should degrade relative to full-dynamics baselines. The paper's current evaluation does not test this prediction, and doing so would either substantiate the framework's robustness or reveal its operational boundaries. This analysis would also inform the design of the adaptive k mechanism the authors envision.

## Suggestions

- **Quantify the "subgraph recall" of the policy attention mechanism**: Compute what fraction of edges on the dynamic optimal path (found by dynamic Dijkstra) are retained in the static top-k subgraph under varying congestion levels. This directly measures the information loss from static filtering and would either validate the design or reveal its failure modes.
- **Replace RL-based πd* with Yen's algorithm**: Since the static subgraph is defined by top-k shortest paths, using an exact algorithm would eliminate approximation error and simplify the method without changing its essence.
- **Add a high-volatility experimental condition**: Create scenarios where β has high variance and strong spatial correlation (e.g., a major corridor experiencing congestion that forces routes far from the static shortest paths). This tests the robustness of the static prior assumption.
- **Clarify the AD baseline implementation**: Explicitly state how the full graph dynamics are encoded as input for MLP-based DQN, so readers can assess the fairness of the comparison.

---

**Assessment by axis:**

- **Novelty**: Moderate. The combination of static-policy-based hard attention with n-hop local features for DPP state representation is a distinct architectural contribution, though each component individually is well-established.
- **Technical soundness**: The core mechanism works empirically, but the theoretical claims (PSR sufficiency, Markov guarantee) are overstated relative to what is proven, and the static-dynamic tension in the policy attention design is a meaningful conceptual concern that is not analyzed.
- **Empirical support**: Adequate for the basic efficacy claim (DFR improves efficiency and performance), but limited by synthetic dynamics, lack of analysis on when the method fails, and basic baselines. The ablation study is a strength.
- **Significance**: Moderate-to-good. The efficiency gains are practically meaningful for real-time planning, and the state representation perspective on DPP is valuable. Significance is bounded by the unresolved question of whether the static prior becomes a liability in volatile scenarios.
- **Clarity**: The paper is generally well-organized and readable, with some imprecise theoretical language (Markov property, PSR guarantees) that could mislead.