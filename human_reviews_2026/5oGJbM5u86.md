# RewardFlow:  Propagating Reward in the State Graphs of Agentic Learning with LLMs

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 0, 4

## Abstract
Large Language Models (LLMs) can operate as agents that interleave reasoning, action, and observation. Training them with reinforcement learning (RL) in multi-turn scenarios remains challenging due to long horizons and sparse terminal rewards, where this setting provides limited guidance for intermediate states, leading to unreliable credit assignment and diluted token-level updates. We address these limitations by proposing *RewardFlow*, a graph-based framework for reward modeling that represents agentic contexts as graphs, with states as nodes and actions as edges. RewardFlow constructs a state graph from multiple rollouts and propagates terminal rewards from successful states to all visited states using graph propagation methods such as Breadth-First Search and Personalized PageRank. This produces dense, state-wise, task-centric reward signals that indicate whether actions move the agent closer to or farther from success. Across text and visual domains on four challenging agent environments and three model sizes, RewardFlow consistently improves task success and training efficiency over strong group-based RL baselines. These results show that RewardFlow is a simple, scalable, and effective framework for mitigating sparse-reward credit assignment in agentic RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates reward shaping in RL training, particularly in multi-turn scenarios where sparse terminal rewards fail to provide fine-grained, reliable feedback for intermediate states. To address this challenge, the authors propose RewardFlow, which abstracts the relationships between intermediate states as a graph constructed from multiple rollouts and refined using carefully designed pruning strategies. The final trajectory-level reward is then back-propagated through the graph to assign task-relevant, fine-grained reward signals to each intermediate state (node). Extensive experiments across various benchmarks demonstrate that RewardFlow consistently improves task success.

### Strengths
1. Reward shaping is crucial for overcoming challenges associated with sparse outcome-based rewards in multi-turn RL tasks. The proposed RewardFlow is **an efficient solution** that avoids the complexity of process reward models, while still achieving fine-grained and reliable reward assignment. By representing relationships between states and actions as a graph, RewardFlow exploits logical connections to back-propagate trajectory-level rewards, assigning intermediate rewards to each state. The design choices, such as graph pruning and state-level advantage estimation, are well-motivated and grounded in rational principles.

2. The experiments demonstrate the effectiveness of RewardFlow, showing significant performance improvements compared to existing RL baselines (e.g., GRPO, GiGPO). RewardFlow achieves higher task success rates across benchmarks and for diverse L(V)LM backbones, highlighting its generalizability. Additionally, further analysis of training dynamics (e.g., training rewards and validation success rates) reveals a stable training process facilitated by RewardFlow.

3. The paper is clearly written, with well-explained concepts, notations, and design principles. The motivation and rationale behind RewardFlow are easy to understand and logically presented. The consistent notations and vivid figures further facilitate comprehension of the technical details, making the methodology accessible to readers.

### Weaknesses
1. A key concern is that RewardFlow may have limited applicability in scenarios where internal states are not explicit, distinct, or frequently recurring enough to be modeled as nodes. In structured domains like Sokoban, the grid layout provides clear, distinguishable states within a fixed state space, making it suitable for graph-based modeling. However, in tasks involving reasoning represented in natural language, the state space (e.g., text intents or representations) is vast, difficult to enumerate, and less differentiable. This could make it challenging to apply RewardFlow effectively in such domains.

2. Minor concerns of experiments
   * Training convergence: In Figure 3, for Qwen2.5-VL-3B-Instruct, both training and validation curves show signs of under-convergence within 100 steps. This raises concerns about whether the step limit of 100 is sufficient. Extending the training steps could provide insights into the model’s performance under more stable and converged settings.
   * Backbone models: While RewardFlow shows significant improvements, it is unclear whether these gains are partly due to the relatively limited capabilities of the L(V)LM backbones used. Would RewardFlow still be effective for larger-scale L(V)LMs with stronger reasoning and generalization capabilities?
   * Latency: While the paper implies that building the graph and computing state-level rewards does not significantly impact efficiency, an explicit efficiency analysis would strengthen the claim. This could include measurements of latency or computational overhead introduced by RewardFlow.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper "RewardFlow: Propagating Reward in the State Graphs of Agentic Learning with LLMs" proposes an approach to shape step-wise rewards for multi-turn RL tasks, using LMMs as agents that interact with the environment. The approach consists in sampling a group of rollouts for a given same task using the current policy (as in GRPO), grou

### Strengths
- Shaping state-wise rewards is important for complex RL tasks
- Experiments look validate the approach on simple tasks
- Easy to follow paper

### Weaknesses
- Presentation : The paper is generally very easy to follow, but it spends considerable space discussing rather elementary concepts, often with noticeable redundancy. For instance, the two-page explanation of the representation of an MDP as a graph seems unnecessarily long  and could be significantly condensed. In the meantime, the paper lacks analysis depth. 

- Applicability of the approach : The proposed approach appears to be applicable mainly to small-scale MDPs — in terms of the number of states reachable through LLM-based agents, both for computational tractability and for ensuring a representative graph structure. It also assumes a discrete setting (or at least one in which states can be meaningfully clustered according to some metric). Finally, the method relies on a largely deterministic environment, as the notion of distance in terms of the number of hops to the goal is not well defined in stochastic settings.

- Analysis: No analysis is provided regarding the relationship between the entropy of the state distribution induced by the policy used for rollouts and the resulting graph coverage for the target task, either from a theoretical or empirical perspective. At a minimum, the paper should include an experiment investigating how the number of performed rollouts affects the number of states represented in the constructed graph or the overall density of that graph.

- Usefulness : From my point of view, the usefulness of such an approach that performs an exhaustive visitation of all possible ways mixing performed is questionnable. First from a scalability perspective (ok breadth first search is linear in term of states that can be reached at each depth, but it requires strong memory requirements). Second, if such search is possible many classical search algorithms in graphs  might be more effective (e.g., A*, Dijkstra, etc.), or at least should be discussed and experimented as alternatives.

### Questions
- I do not understand why authors talk about inconsistent states that could be generated by the LLM. As we are interacting with an environment, the state is always valid (it is a state of the environment),  the only component that can be invalid is the action, which corresponds to text that much match valid actions from the environment. But in that case authors explain than these invalid actions lead to self-loops (no move), which are eventually removed from the graph. Please clarify. 

- Authors propose to transform the succesor graph as a undirected graph. This may corresponds to a strong limitation for many environments, where actions cannot be reversed.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of sparse rewards and credit assignment in multi-turn agentic LLMs. It proposes RewardFlow, a graph-based reward propagation framework that represents agent trajectories as state–action graphs and propagates terminal rewards back to intermediate states via graph algorithms (BFS and Personalized PageRank). This yields dense, state-wise reward signals which are then used in a PPO-style update with state-level advantage estimation. Experiments on text-based (ALFWorld) and visual (Sokoban) agents show improvements in success rate and sample efficiency over GRPO and RLOO.

### Strengths
1. Novelty: The use of a state graph with propagation algorithms for shaping dense rewards is a creative solution to the sparse-reward credit assignment problem in agentic RL.
2. Clarity: The paper is well-organized and clearly written, with helpful diagrams and examples (e.g., Sokoban and ALFWorld).
3. Strong Results: RewardFlow achieves substantial gains over existing baselines—up to 28% improvement in success rate on Sokoban and 12.5% on ALFWorld.
Fine-grained Credit Assignment: By moving from trajectory-level to state-wise advantage estimation, the method improves granularity and stability.
4. Scalability: RewardFlow is tested on multiple LLM scales (1.5B, 3B, 7B), demonstrating generalizability across model sizes and tasks.

### Weaknesses
1. Dependence on Graph Quality: The success of RewardFlow depends on the accurate construction of the state graph, which is derived from sampled rollouts. This can be problematic in highly stochastic or ambiguous environments.
2. Ambiguity in Textual States: The paper briefly mentions challenges in identifying identical states in text-based environments (e.g., ALFWorld), but doesn’t provide a detailed strategy to resolve this.
3. Evaluation Scope: While results on Sokoban and ALFWorld are strong, more environments (e.g., real-world GUI agents or web agents) would improve claims of generality.
4. Limited Comparison to Reward Learning: The work does not compare with recent reward model learning techniques, such as step-wise feedback or preference-based learning, which are also used to deal with sparse supervision.

### Questions
1. How sensitive is the method to rollout quality and diversity? Could poor exploration affect the construction of the state graph and, therefore, misguide reward propagation?
2. Have you considered extending the graph to partially observable or stochastic environments where state aliasing is common?
3. Could you comment on the potential for using learned reward propagation functions (e.g., GNNs) instead of fixed BFS/PageRank?
How do you resolve contradictions when different rollouts assign conflicting transitions or values to similar states?

### Soundness
3

### Presentation
3

### Contribution
3
