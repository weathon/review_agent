# Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Long-horizon goal-conditioned tasks pose fundamental challenges for reinforcement learning (RL), particularly when goals are distant and rewards are sparse. While hierarchical and graph-based methods offer partial solutions, their reliance on conventional hindsight relabeling often fails to correct subgoal infeasibility, leading to inefficient high-level planning. To address this, we propose Strict Subgoal Execution (SSE), a graph-based hierarchical RL framework that integrates Frontier Experience Replay (FER) to separate unreachable from admissible subgoals and streamline high-level decision making. FER delineates the reachability frontier using failure and partial-success transitions, which identifies unreliable subgoals, increases subgoal reliability, and reduces unnecessary high-level decisions. Additionally, SSE employs a decoupled exploration policy to cover underexplored regions of the goal space and a path refinement that adjusts edge costs using observed low-level failures. Experimental results across diverse long-horizon benchmarks show that SSE consistently outperforms existing goal-conditioned and hierarchical RL methods in both efficiency and success rate.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a novel method for hierarchical reinforcement learning, which leverages a frontier-based and failure-aware meta-policy replay buffer to ensure that goals that are reachable and at the frontier of experience are sampled. The method keeps track of this information with a graph-based encoding of the nodes in the environment. To ensure proper exploration, they leverage an exploration policy that is aware of all grid cells in 2D/3D space, and samples goals (at least partially) from these cells. 

The main contribution are the novel replay buffer, and the method for having the graph be aware of what paths will fail, and having exploration be decoupled for goal space coverage.

### Strengths
- A key strength of their method is that it learns transitions between goals, enabling more efficient planning (as implied by Figure 1). It appears that they are effectively learning a high-level policy.  
- The method achieves strong training performance compared to baseline approaches.  
- The evaluations demonstrate that each component of the algorithm plays an essential role in achieving good performance in the environment.  
- The experiments clearly illustrate the benefits of the proposed components (e.g., Figures 2 and 3).  
- The overall presentation is clear and well-explained.

### Weaknesses
- "We assume the existence of a mapping φ such that φ(s) ∈ G, allowing the agent to infer goal progress from the current state."
  This seems like a strong assumption. Is this mapping learned, or predefined? Clarifying this would help understand how generalizable the approach is.

- The algorithm appears to involve substantial hand-crafting, which may limit its applicability to more complex or continuous environments. For instance, sampling from a grid-based estimator may not scale beyond grid-world settings.  
  Similarly, assuming that subgoal reachability can be determined by $||\phi\left(s_{t^{\prime}}\right)-\tilde{g}_t||<\lambda$ is a strong and potentially unrealistic assumption.  Using this grid discretization for failure awareness also seems restrictive.

### Questions
- how is $\lambda$ set? I'm surprised there's no ablation on this.
- the Strict Subgoal Execution (SSE) framework updates the high-level policy with positive returns only when the low level successfully reaches the assigned subgoal. This seems like the same high-level ideas as [1]. Could you contrast to this work? This seems like a graph-based version of that idea?
- when comparing to methods like HIRO, what steps do you take to ensure that HIRO can use similar assumptions as this method, e.g. access to grid discretization scheme?
- isn't having exploration be decoupled from goal space coverage a common strategy in this graph-based planning setting, e.g. [2], which you did not cite.

[1] Self-Imitation Learning
[2] Successor feature landmarks for long-horizon goal-conditioned reinforcement learning

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Strict Subgoal Execution (SSE), a hierarchical reinforcement learning framework designed for long-horizon, goal-conditioned tasks with sparse rewards. SSE incorporates Frontier Experience Replay (FER) to identify and filter out unreachable subgoals, improving the reliability and efficiency of high-level planning. It also uses a decoupled exploration policy to better explore the goal space and a path refinement mechanism to adjust edge costs based on low-level failures. Experiments on multiple benchmarks demonstrate that SSE achieves higher efficiency and success rates than existing goal-conditioned and hierarchical RL methods.

### Strengths
1. The manuscript is clearly written, with figures that effectively elucidate the proposed methodology and equations that comprehensively convey its technical details.

2. The experimental design is well structured, incorporating appropriate selections of baselines. The ablation study provides a thorough examination of the contribution of each component and offers a detailed analysis of the method’s sensitivity to hyperparameters.

### Weaknesses
1. To make a stronger case for the method's generality, the paper should include results from a broader set of environments. Specifically, in line with the existing HRL works, the authors may want to evaluate the method on Pusher, AntFall, AntGather and Ant4Rooms, in addition to the relatively simpler maze-based tasks. This would provide a better understanding of how well SSE adapts to various state spaces and task complexities. Including additional tasks would also help demonstrate whether the method can avoid unstable regions and enforce subgoal completion.

2. The use of only five random seeds in the comparative study may be insufficient to establish statistical significance. Increasing the number of seeds would lead to more reliable and robust experimental conclusions.

### Questions
Would it be possible to include comparative studies on a broader range of environments, using a larger number of random seeds as suggested?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper has proposed a novel goal-conditioned hierarchical reinforcement learning approach, called SSE, to achieve reliable long-horizon planning. The proposed approach is evaluated in a set of simulated navigation tasks.

### Strengths
•  The paper addresses an important problem in hierarchical reinforcement learning — unreliable subgoal execution — which is crucial for long-horizon tasks.

•  The introduction of Frontier Experience Replay (FER) is conceptually clear and provides a principled way to delineate reachable and unreachable subgoals, improving training stability.

### Weaknesses
•  The proposed framework assumes that the goal space is known and low-dimensional, which may not hold for complex real-world manipulation or visual tasks where the goal representation itself is high-dimensional and uncertain.

•  The technical novelty is moderate — SSE combines known components (graph-based HRL, experience replay, path cost reweighting) rather than introducing fundamentally new learning principles.

•  All experiments are conducted in simulators; there is no validation in real-world robotic systems, limiting the practical credibility of the claimed reliability.

### Questions
1.	Does the proposed method assume that the goal space is low-dimensional? This may not be true for complex manipulation tasks.

2.	Is the proposed method applicable to real-world tasks? Only conducting experiments in simulators is not convincing enough.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper mentioned that in long-horizon, sparse-reward tasks, high-level policies often choose subgoals that the low-level controller can’t reliably reach. When HER is applied at the high level, failures get relabeled. The authors propose Frontier Experience Replay (FER), which stores three kinds of high-level transitions—success, stop-on-failure (zero-return, early termination), and partial success to the last reliably reached waypoint. They introduce a decoupled exploration policy that prioritizes under-explored goal-space regions (simple grid-density estimator) alongside an ε-greedy high-level policy to improve coverage. The authors add failure-aware path refinement that inflates edge costs in high-failure regions of the goal graph, nudging Dijkstra planning away from unstable corridors. SSE substantially outperforms HRL (HIRO, HRAC) and graph-based methods (HIGL, DHRL, NGTE, PIG, BEAG) on success rate and often on learning speed

### Strengths
Strong empirical results on a diverse suite, including tasks that require implicit sequencing
Ablation coverage is thoughtful: removing FER or replacing with HER largely breaks performance on harder tasks

### Weaknesses
The density estimator and failure statistics hinge on a grid. This is fine for 2D/3D but will be problematic in higher dimensions and for goals that include orientation or other factors.
No guarantees or formal properties regarding convergence or bias introduced by early termination/FER.

### Questions
This is a strong empirical paper with a simple, well-motivated idea that addresses a real failure mode in hierarchical goal-conditioned RL. Have you thought about how to extend to analyze theoretically with HER?
How often does path refinement genuinely alter the planned path (e.g., fraction of episodes where the refined path differs from shortest path)?
have you tried replacing grid-based novelty with learned density?if so, do the gains persist?

### Soundness
3

### Presentation
3

### Contribution
3
