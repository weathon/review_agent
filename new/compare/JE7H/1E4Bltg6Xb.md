---
job_id: 7dedb012-5cc6-4d05-a810-bc61d90a6e43
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 1E4Bltg6Xb.pdf
paper: Learning Dynamics Feature Representation via Policy Attention for Dynamic Path Planning in Urban Road Networks
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about reinforcement learning, representation learning on graphs, and dynamic path planning in urban road networks, which fits squarely within ICLR topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present and in English. The method is technically coherent and experiments are nontrivial, though there are notable weaknesses in positioning, analysis, and empirical depth, none of which are “fatal” in the sense of desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No signs of prompt injection, hidden instructions, or manipulation aimed at automated reviewers are present in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper studies dynamic path planning (DPP) on urban road networks using reinforcement learning, focusing on how to represent time-varying traffic dynamics in the state. It proposes a Dynamics Feature Representation (DFR) framework that first uses a pre-trained distance-based “policy attention” to extract a task-specific subgraph (top‑k shortest paths from source to goal), then further restricts dynamics to an $n$‑hop neighborhood around the agent to form a compact local feature $W_t''$. Experiments on three real urban subgraphs (Nanjing, Beijing Chaoyang, Shanghai Pudong) with DQN, PPO, and GCN+DQN show that adding DFR can improve success rate and reduce cost gap while substantially decreasing feature dimensionality and planning time.

## Strengths

1. **Clear problem focus on state representation in RL-based DPP.**  
   The paper squarely targets the long-standing tension between global versus local state representations for RL in large graphs. Section 4.1 articulates this trade-off crisply in terms of fully observable vs partially observable MDPs and the need for “sufficient yet compact” dynamics features.

2. **Conceptually simple, practically plausible framework.**  
   The DFR pipeline in Equation (5) and Figure 3 is straightforward: precompute a static distance-based policy $\pi_d^*$, extract a subgraph from top‑$k$ shortest paths (policy attention, $\Psi$), then specialize it per step using $n$‑hop neighborhoods ($\Phi$). This builds on well-understood primitives (shortest paths, neighborhoods) and is easy to integrate into many RL algorithms, which makes the idea attractive from a practical deployment perspective.

3. **Empirical evidence that feature compression helps training and runtime.**  
   Section 5.2 shows that DFR not only improves planning quality metrics (1 − GAP, SR) but also significantly reduces planning time. The reported reductions in planning time (e.g., 85.59% for DQN, 46.08% for GCN+DQN, 79.32% for PPO) provide concrete evidence that the representation change has real computational impact, beyond just accuracy.

4. **Systematic ablation on the two key hyperparameters ($k$ and $n$).**  
   The ablation in Section 5.3 and Figure 6 disentangles the effects of the policy attention strength ($k$) and neighborhood radius ($n$). The heatmaps for Mean GAP, Success Rate, and Compactness Rate concretely show regimes where DFR brings clear gains over the baseline $(k = -1, n = -1)$ and the non-monotonic behavior of $k$, which is insightful and actionable for practitioners.

5. **Use of realistic urban networks and a clear visual depiction.**  
   Figure 4 gives an intuitive visualization of the three OSM-derived subgraphs and how regions are selected, which enhances credibility and clarity about the experimental setting. The networks look reasonably large and complex, so the claimed efficiency gains are meaningful.

6. **Good integration with standard RL formulations.**  
   The MDP formulation in Section 3.2, with Equation (3) and Equation (4), is standard and correct, and the DQN loss in Equation (10) is consistent with prior work. This makes it easier for other researchers to reproduce and extend the method, and the paper stays close to common RL practice.

7. **Figures support the conceptual narrative.**  
   Figures 1 and 2 effectively illustrate the dynamics sequence and how the policy interacts with it over time. Figure 2, in particular, visually connects the evolving $W_t$ dynamics to the state sequence $s_t$ and the policy outputs $a_t$, which helps clarify what is actually being compressed in DFR.

## Weaknesses

1. **Limited conceptual novelty and relatively ad‑hoc design of “policy attention.”**  
   The core technical idea is to precompute a distance-based shortest-path policy and then use its top‑$k$ paths as a hard attention mask (Section 4.3 and Figure 3). While this is reasonable, it is essentially a static top‑$k$ shortest paths heuristic combined with $n$‑hop neighborhoods. Existing routing and graph RL literature often already uses shortest-path heuristics, subgraph extraction, and neighborhood truncation. The paper does not convincingly distinguish DFR from such prior abstractions, nor does it provide any theoretical advantage beyond qualitative PSR arguments. As a result, the contribution feels more like a specific engineering recipe than a fundamentally new representation concept.

2. **Theoretical claims around PSR and “Markov sufficiency” are informal and unsubstantiated.**  
   Section 4.2 cites Predictive State Representations and argues that $W_t''$ “serves as a predictive representation of the state” and “guarantees” sufficiency via Equations (6), (7), and (8). However:
   - Equations (6) and (7) only state $\pi^*(v^t,v_g;W_t') \approx \pi^*(v^t,v_g;W_t)$ and similarly for $W_t''$, but no formal measure of approximation, assumptions, or proof is given.  
   - The PSR argument is qualitative: there is no construction showing that the particular choice of shortest-path-based subgraph and $n$‑hop neighborhood indeed preserves the predictive statistics required by PSR theory.  
   - The text claims that DFR “aligns with the Markov assumption” and “guarantees” compactness and sufficiency, but strictly speaking, DFR simply drops information; it may or may not be sufficient depending on traffic patterns.  
   This overstates the theoretical guarantees and could mislead readers about the strength of the result.

3. **Experimental baselines and comparisons are relatively weak and narrow.**  
   The evaluation (Section 5.2) compares three RL backbones (DQN, PPO, GCN+DQN) with and without DFR. There are several issues:
   - There is no comparison with non-RL path planners (e.g., dynamic Dijkstra, A*, D* Lite) in terms of planning time and cost gap, even though these methods are explicitly discussed in Section 2 and dynamic Dijkstra is used as the “ground truth” generator. This omission makes it hard to judge whether RL+DFR is competitive with standard dynamic routing.  
   - There is no comparison to more sophisticated graph RL or path-planning architectures that explicitly handle dynamics, such as temporal GNN-based RL or multi-step lookahead policies.  
   - The “AD” baselines use “All Dynamics” as input, but there is no comparison to simpler heuristics like “local neighborhood only” or “static shortest path plus myopic dynamic corrections,” which would directly test whether the policy-attention-driven subset is better than generic locality-based truncation.  
   As a result, the effectiveness of DFR is mostly demonstrated relative to an arguably unrealistic baseline that feeds the entire graph dynamics into a relatively small network.

4. **Lack of quantitative detail on feature dimensionality and graph scale.**  
   The paper repeatedly claims substantial input compression and reduced computational overhead, but it gives almost no concrete numbers:
   - We never see the original dimension $|W_t|$, the size of the policy-attention subgraph $|W_t'|$, or the final local feature $|W_t''|$ in absolute terms.  
   - Compactness Rate (CR) is used, and Figure 5 plots $1 - \mathrm{CR}$, but no table or explicit values per configuration are given; only a few scattered numbers (e.g., “CR remains below 5.7%” in Section 5.3).  
   - Graph statistics like number of nodes and edges per subgraph are only vaguely mentioned in Figure 4 and not tabulated.  
   Without a detailed table of dimensions and graph sizes, it is hard to assess whether DFR is compressing from, say, 10k to 500 dimensions or from 100 to 50 dimensions. This weakens the empirical evidence for scalability.

5. **No explicit evaluation of planning quality vs. dynamic Dijkstra beyond GAP.**  
   Mean GAP is defined as the relative difference in path cost vs a dynamic Dijkstra oracle, but only aggregated GAP is reported, and there is no explicit comparison of planning time vs Dijkstra. For a dynamic routing problem, one expects a table or figure where DFR-based RL is benchmarked directly against dynamic Dijkstra (and maybe static shortest-path heuristics) in both cost and computation. Currently:
   - Dynamic Dijkstra is only used to define ground-truth paths.  
   - Planning Time (PT) is reported only relative among RL baselines, not against Dijkstra.  
   This makes it unclear whether RL+DFR is truly competitive or is simply an internal RL improvement.

6. **Dataset and dynamics generation lack realism and detail.**  
   Traffic dynamics are generated via a congestion factor $\beta(v_i, v_j; t) \in [0.1, 1.5]$ (Equation (9)), but the paper does not specify:
   - How $\beta$ is sampled or correlated over time and space (i.i.d., Markov, periodic, event-driven?).  
   - Whether dynamics are derived from any real traffic data or purely synthetic.  
   - Whether episodes share common patterns or are random per episode.  
   Since the central claim is robust performance under dynamic traffic, the lack of a clear and realistic dynamics model significantly limits external validity. It is hard to know whether DFR will behave similarly under realistic rush-hour patterns, incidents, or correlated congestion.

7. **Limited analysis of failure modes and where DFR can hurt.**  
   The ablation heatmaps in Figure 6 show some regimes where performance degrades (e.g., low $n$ or high $k$), but there is no deeper analysis of *why* and *when* DFR might systematically remove necessary information. For instance:
   - If congestion shifts to alternative routes outside the precomputed top‑$k$ static paths, DFR might systematically ignore better options; this failure mode is not explored.  
   - There is no analysis of robustness when dynamics strongly deviate from static distances, which is arguably the whole point of dynamic planning.  
   Without such analysis, the robustness claim remains qualitative.

8. **No tables of quantitative results or detailed statistics.**  
   All main quantitative results are presented via radar plots (Figure 5) and heatmaps and learning curves (Figure 6). There is no standard table summarizing per‑scenario Mean GAP, SR, CR, PT with means and standard deviations. This omission makes it difficult to precisely compare methods and to understand variance across runs or seeds.

9. **Some mathematical and notational issues and oversights.**  
   While the main RL equations are fine, there are several issues that hurt clarity and rigor:
   - In Equation (1), the summation index is written as $\sum_{k=1} w(p_k, p_{k+1}; t_k)$ without explicit bounds; presumably this should be $\sum_{k=0}^{n-1} w(p_k, p_{k+1}; t_k)$. The current form is ambiguous and slightly inconsistent with the earlier path definition.  
   - Equation (2) writes the reward as $-c(v^t, v^{t+1}; W_t) + b \cdot \mathbb{I}(v^{t+1}==v_g)$, but $c(\cdot,\cdot)$ is defined in Equation (1) in terms of $w$ at $t_k$, and Equation (9) uses $W_t$ explicitly. The notation could be clarified, especially how $t_k$ relates to decision step $t$.  
   - The equation for the TD loss (Equation (10)) uses $r(s,a,s')$ while the rest of the paper uses $R(s,a,s')$. This is minor but symptomatic of inconsistent notation.  
   These issues are not fatal but indicate the math could be tightened.

10. **DFR overhead and pretraining cost are not properly quantified.**  
    Section 4.3 claims that policy attention and $n$‑hop neighborhoods impose “negligible additional computational overhead” because everything is precomputable offline. However, the paper does not detail:
    - How many source–goal pairs are needed to train $\pi_d^*$ and how expensive that training is.  
    - How many top‑$k$ paths per pair are stored, and whether this scales quadratically in nodes or linearly in some sampling scheme.  
    - How the overhead compares to simply running Dijkstra per query on a static graph.  
    Without these details, the claim of negligible overhead is not fully convincing.

## Potentially Missing Related Work

All of the following appear directly relevant and are not cited in the paper. They should be discussed in Section 2 and, where applicable, in the experimental comparison or discussion of limitations:

1. **Zhou, Z., Zhou, B., Liu, H., “DynamicRouteGPT: A Real-Time Multi-Vehicle Dynamic Navigation Framework Based on Large Language Models”, 2024.**  
   Addresses real-time dynamic navigation and explicitly balances global and local optimality in dynamic traffic environments. It should be cited in the related work on dynamic path planning and contrasted with the RL-based approach here, especially regarding how each method handles dynamic information and representation.

2. **Lai, Y., Kanoh, H., “Connected vehicles’ dynamic route planning based on reinforcement learning”, 2024.**  
   Proposes RL for dynamic route planning in connected vehicles with dynamic weighting of traffic conditions. Highly relevant to the RL-based DPP framing and should be compared in Section 2 and possibly in the discussion of state design.

3. **Liu, Z., Chen, B., Zhou, H., “MAPPER: Multi-Agent Path Planning with Evolutionary Reinforcement Learning in Mixed Dynamic Environments”, 2020.**  
   Uses RL for path planning in dynamic environments, including multi-agent interactions. It is pertinent to the discussion of RL for dynamic routing and possible state abstraction strategies; should be referenced in the RL-based path planning section.

4. **Xu, G., Chen, L., Zhao, X., “Dual-Layer Path Planning Model for Autonomous Vehicles in Urban Road Networks Using an Improved Deep Q-Network Algorithm with PID Control”, 2025.**  
   Combines A* with an improved DQN for urban path planning. This is structurally similar to using classical search plus RL and thus directly comparable to DFR’s use of a shortest-path prior. It should be discussed in Section 2 and contrasted with DFR’s policy attention mechanism.

5. **Alsuwaiket, M. A., “Optimizing Autonomous Vehicle Navigation Through Reinforcement Learning in Dynamic Urban Environments”, 2025.**  
   Integrates PPO and GNNs for dynamic urban navigation, overlapping both in setting and method (PPO, GNNs). It should be cited and compared to the PPO and GCN+DQN baselines, particularly on how dynamics are represented.

6. **Nallamala, S. K., Putha, S., Kasaraneni, B. P., “AI-Based Autonomous Vehicle Control Systems for Urban Environments: Leveraging Reinforcement Learning for Decision-Making, Path Planning, and Collision Avoidance under Dynamic Traffic Conditions”, 2022.**  
   Surveys or develops RL-based control and path planning in dynamic urban contexts; relevant background for positioning DFR and justifying the focus on representation.

7. **Zhang, X., Li, C., Zhao, M., “In-station UAV path planning based on multi-agent reinforcement learning and dynamic environment modeling”, 2026.**  
   Deals with dynamic environment modeling and RL-based path planning, even if in a UAV context. It could be cited to emphasize that dynamic representations and RL path planning are studied in other domains.

8. **Zhang, X., Li, C., Zhao, M., “Automated Path Planning Method for Urban Street-Corner Parks Based on Multi-Agent Deep Reinforcement Learning”, 2026.**  
   Another path planning work in a complex urban context with multi-agent RL; relevant to broader related work and differences in state representation.

9. **Anvil Labs, “Dynamic Path Planning with Reinforcement Learning”, 2025.**  
   Describes dynamic path planning via RL for drones in complex environments. While perhaps more applied, it underscores the general RL DPP line of work and should be mentioned in the context of dynamic environments and representation choices.

## Questions

1. **Dynamics generation and realism.**  
   How exactly is the congestion factor $\beta(v_i, v_j; t)$ generated over time and space? Is it i.i.d. per edge and time, or does it follow a temporal process (e.g., AR, Markov) or a spatial diffusion model? Are there any real traffic datasets informing $\beta$, or is it entirely synthetic?

2. **Pretraining of $\pi_d^*$ and storage cost.**  
   Could you provide quantitative details on the cost of training the distance-based policy $\pi_d^*$ (number of tasks, steps, training time) and the memory footprint of storing top‑$k$ paths for all relevant source–goal pairs? How does this compare to, say, storing all‑pairs shortest paths or running Dijkstra per query?

3. **Absolute dimensionalities of $W_t$, $W_t'$, and $W_t''$.**  
   For each of the three subgraphs, can you report the absolute dimensionality of the original dynamics feature $W_t$, the policy-attention-filtered $W_t'$, and the local feature $W_t''$ (for typical $(k, n)$)? This would substantiate the claimed compression more concretely than CR alone.

4. **Comparison with dynamic Dijkstra and A*.**  
   Can you provide a direct comparison of path cost and planning time between your best RL+DFR variant and dynamic Dijkstra (and possibly A* with predicted or uniform costs) on the same tasks? This would greatly clarify whether RL+DFR is practically competitive.

5. **Robustness when static distances misalign with dynamic costs.**  
   Have you evaluated settings where the fastest dynamic paths differ substantially from the shortest static paths (e.g., systematic congestion on static shortest paths)? In such cases, does policy attention still help, or does it hurt by pruning dynamically good but statically longer routes?

6. **Alternative local representations.**  
   Have you compared DFR against a baseline that only uses $n$‑hop neighborhoods around the agent (without policy attention) but with the same feature dimensionality? This would help isolate the contribution of the global, task-aware pruning versus local truncation.

7. **Variance and statistical significance.**  
   How many random seeds were used for training each configuration, and what is the variance in Mean GAP and SR across runs? Some of the trends in Figure 6 (especially for large $n$) appear close; error bars or standard deviations would help assess robustness.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is conceptually coherent and the RL components are standard and correctly implemented, but theoretical claims about sufficiency and Markovity are overstated, and the empirical validation, while nontrivial, could be strengthened with more realistic dynamics and stronger baselines.

## Presentation Rating

3: good.  
The paper is generally clear and well organized, with helpful figures (e.g., Figures 2, 3, 4, 5, 6). However, some notation is inconsistent, important quantitative details (dimensions, graph sizes) are missing, and the theoretical discussion around PSR is too informal relative to the strength of the claims.

## Contribution Rating

3: good.  
The contribution is a practically useful and conceptually clear framework for compressing dynamics features in RL-based DPP, and the improvements over the AD baselines are meaningful. At the same time, the novelty is moderate, and the lack of comparison with stronger or more diverse baselines somewhat limits the assessed impact.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper offers a clean and practically relevant idea for hierarchical dynamics feature representation in RL-based dynamic path planning, with supportive though not exhaustive empirical results. Conceptual novelty is moderate and several aspects of theory and experimental design could be improved, but the work is sound enough and relevant enough to the ICLR community to merit a positive but cautious recommendation.

## Reviewer Confidence

4: confident.  
I am comfortable with RL, graph-based planning, and representation learning, and have carefully checked the methodological details and equations; some uncertainty remains mainly around the exact realism of the simulated dynamics and potential missing related work in specialized traffic RL communities.