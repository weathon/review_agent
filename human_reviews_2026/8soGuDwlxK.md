# SymLight: Exploring Interpretable and Deployable Symbolic Policies for Traffic Signal Control

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Deep Reinforcement Learning have achieved significant success in automatically devising effective traffic signal control (TSC) policies. Neural policies, however, tend to be over-parameterized and non-transparent, hindering their interpretability and deployability on resource-limited edge devices. This work presents SymLight, a priority function search framework based on Monte Carlo Tree Search (MCTS) for discovering inherently interpretable and deployable symbolic priority functions to serve as the TSC policies. The priority function, in particular, accepts traffic features as input and then outputs a priority for each traffic signal phase, which subsequently directs the phase transition. For effective search, we propose a concise yet expressive priority function representation. This helps mitigate the combinatorial explosion of the action space in MCTS. Additionally, a probabilistic structural rollout strategy is introduced to leverage structural patterns from previously discovered high-quality priority functions, guiding the rollout process. Our experiments on real-world datasets demonstrate SymLight's superior performance across a range of baselines. A key advantage is SymLight's ability to produce interpretable and deployable TSC policies while maintaining excellent performance. Our codes will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed a Monte Carlo Tree Search method to improve the interpretability of traffic signal control. Experiments show that the performance is achieved a new SOTA.

### Strengths
Using Monte Carlo Tree Search to improve the interpretability of traffic signal control sounds interesting. Experiments show that the performance is achieved a new SOTA. This method is very simple and effective.

### Weaknesses
1. Why use repeat WI in [+, −, ×, WO, WI, WI]? This example makes reader confused.

2. More newly baseline methods should be compared. More scalability datasets should be experimented.

3. How is the robustness of your method? In real-world deployment, there may be some noise in perception, could the method still keep high performance[1,2]?

4. If we add running vehicles as input, can the performance be improved? 

5. This performance could be getting stuck in local optima due to greedy selection of the best reward in current step, it ignores long-term reward payoffs. The authors should discuss this issue and address how to resolve it as future work.


[1]Robustlight: improving robustness via diffusion reinforcement learning for traffic signal control. ICML 
[2]Fuzzylight: a robust two-stage fuzzy approach for traffic signal control works in real cities

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper develops a method for identifying a symbolic priority function for traffic intersection management. The approach uses MCTS to design interpretable, symbolic decision rules that score lanes for selection. Empirical results show that the method reduces travel time and increases vehicle throughput across 6 intersection scenarios. Through its use of a symbolic function, the resulting decision rule is automatically interpretable.

### Strengths
- The use of a symbolic priority function is a well-motivated approach for providing interpretability in an application domain where interpretability is a main bottleneck to realizing gains from machine learning.
- The paper presents its main ideas clearly.
- The empirical analysis supports the central claims of the paper: improved intersection metrics and interpretability of the learned priority function.

### Weaknesses
- Clarity and relevance to learning: it took me a couple of read-throughs of the method to understand how the priority function was implemented and what the role of MCTS was. I still give the paper a good rating for clarity because -- once I understood the method -- I think the organization makes sense. So this weakness is more about, "could the paper have been more clear?" My confusion was that MCTS is usually not a learning method itself but is a decision-time search method. So at first I expected the method to use MCTS to select the next phase. There then appeared to be no learning in the paper which would make it out of ICLR's scope. Ultimately, I see that MCTS is used **offline** to learn a fixed priority function. This is not a major weakness but I believe more clarity could be added in this regard.
- Table 1's font is difficult to read without zooming in a lot.
- It wasn't fully clear how datasets are being used. I am assuming that the data provides the demand profiles for each scenario so that you know how many vehicles to simulate?

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SymLight, a priority function search framework for traffic signal control (TSC) that balances interpretability, efficiency, and deployability. Unlike deep reinforcement learning (DRL) models, which are opaque and resource-intensive, SymLight discovers explicit symbolic priority functions through Monte Carlo Tree Search (MCTS) to decide traffic light phases based on real-time traffic features. It proposes a concise symbolic representation, an adaptive reward shaping mechanism, and a probabilistic structural rollout strategy that leverages structural patterns from high-quality expressions to guide efficient exploration. SymLight directly optimizes system-level objectives (e.g., average travel time) and produces lightweight, human-understandable control policies suitable for deployment on low-cost edge devices. Extensive experiments on six real-world traffic networks (Hangzhou, Los Angeles, Atlanta, Jinan, and Manhattan) show that SymLight outperforms state-of-the-art DRL and symbolic baselines in both travel time reduction and throughput improvement.

### Strengths
1. The paper proposes a symbolic-policy formulation for traffic signal control, representing control logic as explicit priority functions. This approach is promising to bridge the gap between high-performance learning methods and human-interpretable rule-based systems, addressing a long-standing limitation in DRL-based TSC.
2. By integrating MCTS with a concise symbolic representation and the proposed probabilistic structural rollout strategy, SymLight enables efficient exploration of large discrete spaces.
3. Experiments on six real-world traffic networks demonstrate consistent improvements in both travel time reduction and throughput over strong DRL and symbolic baselines.

### Weaknesses
1.	While the central idea of this work lies in leveraging a priority function search for traffic signal control, the paper provides limited background or theoretical motivation for this concept. As a result, the rationale for adopting priority functions and their novelty relative to existing symbolic or TSC works remains unclear.
2.	The Monte Carlo Tree Search has been explored in traffic signal optimization [1]. The authors are expected to clarify the difference and novelty compared to the existing literature. 
3.	Can you intuitively explain how your model achieves better interpretability and deployability than existing methods? 
4.	The code and implementation details are not publicly available, which hinders the reviewers to reproduce and verify the reported results.

[1] Monte Carlo Tree Search-based intersection signal optimization model with channelized section spillover. Transportation Research Part C: Emerging Technologies. 2019.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces SymLight, a traffic-signal control method that searches for symbolic priority functions—compact algebraic expressions mapping lane-level features to phase priorities—via MCTS. The priority function is encoded as a token list over basic operators (addition, negation, multiplication, min/max, protected division) and lane features. The search adds a Probabilistic Structural Rollout (PSR) that biases rollouts using parent-child token statistics from the top-k expressions.  The reward is adaptively normalized by the current best observed travel time to stabilize UCT.  Experiments on six CityFlow road networks (Hangzhou, Los Angeles, Atlanta, Jinan, Manhattan) report lower average travel time and higher throughput than conventional.

### Strengths
1. Interpretable policy class with a easy understand over meaningful lane features; protected division and min/max are practical choices.  

2. Empirical improvements on six CityFlow networks with significance testing. 

3. Search framework is simple to implement; PSR is a plausible way to avoid uninformative rollouts.

### Weaknesses
1. The core claim is that SymLight yields strong, deployable policies; however, the offline search costs (simulation calls, expansions, rollouts, wall-clock per network/intersection) are not quantified. Without this, it’s unclear whether the approach scales to larger grids or frequent retuning.

2. The reward normalizes inverse travel time, but multi-objective considerations (emissions, per-approach fairness, pedestrian delay) are absent, and it is unclear whether the method over-optimizes one metric at others’ expense. 

3.  Regarding the interpretability, more analysis case should added. The robustness of model interpretability is need to be considered.

4. More symbolic RL works need to be compared, which can help hightlight the contribution of this work.

Discovering symbolic policies with deep reinforcement learning, ICML, 2021.

Learning Neurosymbolic Generative Models via Program Synthesis, 2019, ICML.

Neurosymbolic Reinforcement Learning with Formally Verified Exploration, 2020, ICML.

5.  What are the per-network wall-clock hours, simulator steps, node expansions, and rollout counts for MCTS+PSR to reach the reported policies? Please tabulate alongside DRL training costs.

6. How does performance vary with k and \alpha? Any evidence of PSR biasing search toward substructures that generalize poorly across intersections?

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2
