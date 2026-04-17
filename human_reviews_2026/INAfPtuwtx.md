# Scalable Multi-Agent Autonomous Learning in Complex Unpredictable Environments

- Decision: Reject
- Scores: 6, 4, 6, 8

## Abstract
This research introduces a novel multi-agent self-learning solution for large and complex tasks in dynamic and unpredictable environments where large groups of homogeneous agents coordinate to achieve collective goals. Using a novel iterative two-phase multi-agent reinforcement learning approach, agents continuously learn and evolve in performing the task. In phase one, agents collaboratively determine an effective global task distribution based on the current state of the task and assign the most suitable agent to each activity. In phase two, the selected agent refines activity execution using a shared policy from a policy bank, built from collective past experiences. Merging agent trajectories across similar agents using a novel shared experience learning mechanism enables continuous adaptation, while iterating through these two phases significantly reduces coordination overhead. This novel approach was tested with an exemplary test system comprising drones, with results including real-world scenarios in domains like forest firefighting. This approach performed well by evolving autonomously in new environments with a large number of agents. In adapting quickly to new and changing environments, this versatile approach provides a highly scalable foundation for many other applications tackling dynamic and hard-to-optimize domains that are not possible today.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel two-phase iterative multi-agent reinforcement learning (MARL) framework for scalable autonomous learning in large, dynamic, and unpredictable environments. The approach targets cooperative tasks involving groups of homogeneous agents, such as drones in forest firefighting. In Phase 1 (Refocus), a task distributor decomposes the overall task into activities based on the current environment state (e.g., using heuristics like A* planning, combined with domain-specific pipelines for fire edge progression and hotspot prediction via transformers) and assigns them to suitable agents, reducing the state space and coordination overhead. In Phase 2 (Refine), agents execute their assigned activities using policies from a shared policy bank, collect trajectories, and refine policies through merged shared experiences (e.g., selecting and merging best/worst trajectories from homogeneous agents). The framework iterates between phases for continuous adaptation, leveraging algorithms like PPO, Actor-Critic, or DQN for policy updates.

### Strengths
1. The framework addresses a critical gap in MARL: handling thousands of agents in non-stationary, partial-observable environments. By decoupling task distribution (Phase 1) from policy refinement (Phase 2), it avoids the exponential state-action space explosion in joint MARL. The use of homogeneous agent groups and shared policy banks enables efficient experience sharing, leading to faster learning 
2. Experiments cover ablation studies (e.g., removing transformers reduces returns by ~50%), comparisons to MARL baselines (two-phase PPO outperforms MAPPO by 2-3x in returns), scalability tests (up to 1,000 agents vs. MARL's limit of ~30), and statistical validation (ANOVA p<0.05 for trajectory strategies). Results on real fires (e.g., 40%+ time savings for large fires with 30 hotspots) are compelling, with visualizations (e.g., heatmaps in Figures 3-4) aiding interpretability. The custom simulator, grounded in established models (WRF-Fire, Hansen's extinguisher efficacy), adds credibility.
3. The approach is versatile for dynamic domains like urban firefighting, medical rescue, or flood control, where pure RL fails due to variance. It emphasizes result-oriented learning without over-relying on end-to-end RL, potentially inspiring hybrid systems.

### Weaknesses
1. The paper justifies avoiding SMAC or other MARL benchmarks due to their focus on small-scale competitive games, but this limits comparability. Custom forest fire scenarios, while realistic, may be tailored to favor the method (e.g., homogeneous drones suit shared policies). Comparisons to human firefighters feel mismatched—drones have advantages like aerial mobility but ignore real-world constraints (e.g., battery life, regulatory issues). Traditional MARL baselines are tested with only 25 agents, potentially understating their performance in scaled-down settings.
2. Propositions and proofs are concise but overly assumptive.

### Questions
1. The custom simulator and drones are detailed in the appendix, but code isn't mentioned (though reproducibility statement claims full details). Hyperparameters (e.g., PPO clip ϵ=0.1, α=2500/β=3500 in rewards) are specified, but sensitivity analyses are missing. Real-drone tests are limited to 3 units coexisting with simulations, raising questions about full-scale physical deployment (e.g., communication delays, sensor noise).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address large-scale, dynamic, and unpredictable real-world tasks, this paper proposes a two-stage iterative multi-agent reinforcement learning framework that alternates between a Refocus and Refine process. In each iteration, the Refocus phase determines the current optimal task allocation scheme, while the Refine phase leverages collective intelligence for optimizing execution and policy learning at the agent level. The proposed framework is validated through realistic firefighting simulations with drone swarms, demonstrating strong empirical performance.

### Strengths
1. The paper features real-world experiments with substantial engineering achievements — a type of contribution that is relatively rare in the ICLR community.

2. It represents a valuable attempt to apply MARL methods in real-world scenarios, bridging the gap between theoretical frameworks and practical deployment.

### Weaknesses
1. Outdated Literature Review

The literature discussed in the Related Work section is largely outdated, with many works from four or five years ago. I understand that the authors may have chosen representative or characteristic studies to highlight the novelty of their own work, but the selected references do not reflect the current state of research, especially in Hierarchical Reinforcement Learning and Task Partitioning & Role Assignment.
Recent works relevant to dynamic and non-stationary task environments include (but are not limited to):

Hierarchical Reinforcement Learning (HRL):

[1] End-to-end hierarchical reinforcement learning with integrated subgoal discovery. IEEE TNNLS, 2021.

[2] Hierarchical reinforcement learning for non-stationary environments. SSCI, 2023.

[3] Exploring the limits of hierarchical world models in reinforcement learning. Scientific Reports, 2024.

[4] Hierarchical reinforcement learning with uncertainty-guided diffusional subgoals. ICML, 2025.

Task Partitioning & Role Assignment:

[5] Ldsa: Learning dynamic subtask assignment in cooperative multi-agent reinforcement learning. NeurIPS, 2022.

[6] Learning to transfer role assignment across team sizes. AAMAS, 2022.

[7] Dynamic role discovery and assignment in multi-agent task decomposition. Complex & Intelligent Systems, 2023.

[8] Automatic grouping for efficient cooperative multi-agent reinforcement learning. NeurIPS, 2023.

[9] Dynamic subtask representation and assignment in cooperative multi-agent tasks. Neurocomputing, 2025.

2. Clarity of Notation and Contextualization

Section 3.1 provides a clear and rigorous formal setup with well-defined notation. However, I strongly recommend that the authors illustrate the meaning of each symbol using the forest firefighting example, to help readers understand the motivation and reasonableness behind each assumption. The current version, while clearly written, still leaves several conceptual ambiguities (see “Questions” section below).

3. Figure 1 is Confusing

As the core figure of the paper, Figure 1 is confusing and fails to clearly depict the two phases of the framework. The caption actually conveys more clarity than the figure itself. A redesign that visually differentiates the Refocus and Refine processes is highly recommended.

4. Limited Methodological Novelty

The paper reads more like a technical report than a methodological contribution. Essentially, it presents an intuitive framework rather than a novel algorithmic method. While proposing a framework is valuable, most components in the experiments are based on or quite similar to existing methods, and the analysis largely explores their behavior within the proposed setup. Overall, this work is stronger in engineering and application than in methodological innovation.

### Questions
1. In $A_g$, there is no explicit dependence on $t$. Does this mean $A_g$ does not vary over time?

2. Can two groups $A_{g_i}$ and $A_{g_j}$ simultaneously handle overlapping subsets of activities?

3. Could the simultaneous partitioning of both tasks and agents potentially reduce overall efficiency? For instance, might some agents need to wait for others to finish certain sub-tasks before proceeding?

4. The constraint that “each group’s number of activities should be no less than the number of agents in that group” seems overly strong. With a large number of agents, can this always be satisfied? Does it imply that each agent performs multiple sub-tasks at every time step $t$? If so, how is this feasible, given that an agent’s action $u_i$ might benefit some sub-tasks but hinder others?

5. Are $r(w^t_j)$ and $\kappa(a_i)$ prior knowledge or expert-defined heuristics?

6. How are real drones and simulated drones coexisting in the same environment (line 300)? The description seems inconsistent: at one point (line 320) you mention 3,000 simulated drones, while later (line 339) you refer to 25 agents in a MARL comparison. Could you clarify this setup?

7. Is drone capability evaluated solely based on fire-extinguishing capacity?

8. You mentioned testing the framework on several MARL benchmarks. Could you provide detailed results on standard benchmarks such as SMAC? This is crucial to assess the framework’s potential impact on future MARL research. I genuinely appreciate the engineering effort and real-world implementation, but practical success alone does not establish superiority. Demonstrating that existing MARL algorithms perform better under your proposed framework on well-known benchmarks like SMAC or GRF would greatly strengthen your contribution. Without such evidence, I am currently leaning toward a negative evaluation. However, if the authors can show competitive or superior benchmark performance, I would be delighted to see this work presented to the ICLR community.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a two-phase framework to improve the scalability of multi-agent reinforcement learning in large, dynamic environments. The key idea is to decouple global coordination and local learning: the system first decomposes and redistributes tasks among homogeneous agent groups, then each group learns and refines activity-specific policies through shared experience. By iteratively alternating between these phases, the framework enables adaptive, efficient cooperation across many agents.

### Strengths
- The paper presents a clear 2-stage framework that separates global task allocation from local policy learning, effectively improving scalability in large multi-agent systems.

- The iterative design enables continuous adaptation to dynamic environments, the performance seems to be better than the baseline.

### Weaknesses
- The task decomposition process relies heavily on heuristic and domain-specific components, limiting generality and reproducibility.

- Theoretical guarantees are based on strong assumptions (e.g., agent homogeneity and sufficient decomposition) that are generally not met in practice.

### Questions
1. The theoretical results rely heavily on the assumption that agents are perfectly homogeneous, making their trajectories fully interchangeable. In realistic systems, agents often differ in sensing, actuation, or local context—how would the proposed merging and shared policy updates behave when this assumption is relaxed?

2. The framework assumes that shared experience merging among agents improves learning efficiency, but for inhomogeneous agents this could introduce bias or destabilize training. Has the method been tested or theoretically analyzed under partially heterogeneous conditions, and what mechanisms (e.g., importance weighting or representation alignment) might mitigate these effects?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a two phase approach to solving multi-agent learning in complex unpredictable and dynamically changing environments.
The method assumes sets of homogeneous agents, with each set having a library of pre-trained policies.
Phase one involves using agents to gather information about the environment to use in a task distributor. The task distributor then decomposes the environment into sub-tasks, with the at least as many sub-tasks as there are agents to avoid wasted agent assignment. The tasks are then assigned to the agents.
Phase two involves each agent having been assigned a sub-task and chooses a policy from their library that is best for solving that task. The agents then perform their tasks and merge the best and worst experiences. Depending on the variation of the setups described, this might only use the best experiences or a weighted distribution over the experience. These are then used to refine the policies and store the updated in the policy bank. The policies are trained using PPO, DQN and/or AC methods.

They demonstrate this in a fire fighting scenario in which they have different classes of drones, small, medium, large. The simulations are derived from previous works which simulate the fire scenarios. This was simulated, and done with actual drones, with fire depicted by coloured fabric. (I have concerns for experiment with the actual drones that this might not follow a repeatable rule in updating the environment. Whereas the simulated ones are backed by data. The focus of the research appears to be on the algorithm and not the simulation however, so this is less of a concern for the purpose of the algorithm.)
They compare their method this to a base line of actual firefighters over various task sizes, varying the number of fire units and hotspots. They show that the algorithm outperforms the baseline by about 15-40%, which larger improvements on large task sizes.
They do an ablation on the task distribution with various setups, showing using a transformer and EdgeProgression performs the best.
They use three two-phase models: PPO, AC, DQN. All three beat traditional MARL methods: MAPPO, A2C and DQN. The two-phase PPO is shown to perform the best.
The weighted shared experience learning is shown to have the best performance.
They also show that the scalability of the method outperforms traditional MARL methods by a large margin.

Problem addresses: A scalable method to solving dynamic complex environments with multi-agent systems.

The proposed method/idea: A two phase approach for task assignment for different sets of homogeneous agents, and updated learning during the task.

The main claim is that they contribute to perform complex tasks with a large number of agents, and this is backed by empirical results as well as some theoretical results.

This work tackles an important problem in MARL scalability for complex and dynamic tasks.

Overall I believe this is good work. There are a few details about the experiment setup, particularly the real drones one, that could be expanded upon to make it clearer.

### Strengths
1) The paper clearly describes the methods.
2) The paper tests over a number of different methods and ablations.
3) The simulations are backed up by previous works.
4) The results suggest this method outperforms the baseline as well as different traditional MARL methods.
5) It appears to be a novel combination of task assignment and merged learning techniques to solve an important problem.

### Weaknesses
1) I may have missed something, but it is not entirely clear how the reward works, or how that is measured in the real life simulations. Perhaps expanding on this might help.
2) I'm not 100% certain about the baseline of the firefighters. I would love an elaboration on that.

3) Minor typos in the paper such as line 109 "the maximum numbers of activities than agent..."
I didn't track all of them, but a good read through should be done, as I did pick up a few others.

### Questions
1) I am assuming that agents cannot be reassigned tasks until it's current task has been completed? At least that's the understanding I got from the way you describe the tasks having time lengths. I ask because it seems like there could be a strange edge case where the task assignment phase could update a task to an agent and then reassign it again getting stuck in a loop if that is not the case.
2) With the policy library being updated, does this not introduce a non-stationarity issue, it seems the results don't suggest this to be a problem. It's a question out of curiosity.

### Soundness
3

### Presentation
3

### Contribution
3
