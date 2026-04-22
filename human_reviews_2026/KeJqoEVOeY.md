# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large Language Models (LLMs) have recently emerged as promising agents for Traffic Signal Control (TSC) due to their strengths in reasoning and generalization. However, current LLM-based approaches treat intersections as independent agents without inter-intersection cooperation, limiting their effectiveness in network-wide optimization. To address this gap, we propose CoLLMLight, the first cooperative LLM agent framework for network-wide traffic signal control. CoLLMLight enables agents to perform in-depth spatiotemporal reasoning for cooperation, while ensuring real-time responsiveness through an asynchronous cooperative decision architecture. The reasoning process runs asynchronously, deriving cooperative control guidance from dynamic interactions among intersections. This guidance is cached and incorporated as contextual input for real-time signal decisions. To enhance cooperation quality while ensuring reasoning efficiency, we propose cost-aware cooperation optimization. It first applies adaptive reasoning chain optimization to enable the LLM to adjust its reasoning depth according to traffic complexity. The model is then refined with reinforcement learning using reward signals that promote network-wide performance while penalizing excessive reasoning. Extensive experiments on four real-world traffic networks demonstrate that CoLLMLight consistently outperforms existing methods, achieving more effective and generalizable cooperation while maintaining real-time responsiveness and efficient token usage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper, CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control, proposes a framework where multiple LLM-based agents collaboratively manage a network of intersections. The key ideas are an asynchronous cooperative decision architecture that separates reasoning from real-time control, and a cost-aware cooperation optimization mechanism that balances reasoning depth and computational efficiency. Experiments on several real-world traffic datasets show promising improvements over both reinforcement learning (RL) and LLM-based baselines.

### Strengths
Novel problem framing: The paper extends prior LLM-based traffic control works (e.g., LLMLight) to a cooperative multi-agent setting, which is a meaningful and realistic step toward network-level optimization.

Asynchronous architecture: Separating reasoning and decision modules to maintain real-time control is a practical and well-motivated design.

Cost-aware optimization idea: Introducing adaptive reasoning and reinforcement learning to regulate reasoning depth shows awareness of the token and latency constraints in LLM systems.

### Weaknesses
- The proposed method appears as an incremental extension of LLMLight with cooperation and token-efficiency modules. Both the asynchronous design and adaptive reasoning could be considered engineering optimizations rather than fundamentally new learning mechanisms.

- The cooperative reasoning process relies on textual prompting, but there is little analysis of why or how the LLM reasoning benefits coordination beyond the cached contextual hints. No ablation or visualization clarifies what the LLM is learning about spatial-temporal dependencies.

- Although the asynchronous module mitigates latency, the experiments rely on simulation rather than real-time control. It’s unclear whether the system can truly operate under realistic time constraints with communication and inference overhead among dozens of intersections.

- The zero-shot evaluation uses synthetic training and real-world testing but does not report variance or statistical significance.

- No human or expert evaluation validates the interpretability or safety of the LLM’s decisions.

- Comparison with strong graph-based MARL methods (e.g., multi-agent transformers) is missing.

- The paper devotes extensive space to performance tables and prompts but offers limited insights into why the model works, missing a deeper understanding of LLM-agent cooperation dynamics.

### Questions
Besides weakness, I also have the following questions:

How are LLM reasoning traces (SR) concretely represented and reused during decision making? Are they natural-language summaries or latent embeddings?

How is the asynchronous communication among intersections implemented? Do agents share text tokens, numeric signals, or both?

How robust is CoLLMLight to communication failures or stale reasoning caches in dynamic conditions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses a relevant and emerging topic in large-scale urban traffic signal control by extending LLM-based single-intersection agents to a cooperative multi-agent framework.

The proposed CoLLMLight architecture integrates asynchronous reasoning–decision decoupling with cost-aware reinforcement optimization, aiming to balance coordination performance and reasoning efficiency.

The idea is interesting and experimentally validated across multiple city networks.

However, several aspects—including the theoretical motivation, reward design rigor, and consistency of coordination—require further clarification and analysis to fully establish the framework’s soundness and generality.

### Strengths
1.Explores a novel and timely research direction by bridging large language models (LLMs) with multi-agent coordination for network-wide traffic signal control.

2.The proposed Spatiotemporal Reasoning–Real-Time Decision (SR–RD) decoupling effectively mitigates LLM latency issues in real-time control.

3.Experiments on multiple city-scale networks demonstrate the scalability of the approach beyond toy environments.

4.The paper is generally well-written and clearly organized, making it accessible to both MARL and LLM audiences.

### Weaknesses
1.The paper does not clearly explain why large language models (LLMs) are more suitable for multi-interaction cooperation than conventional multi-agent reinforcement learning (MARL) frameworks. 

2.There are fewer recent baselines in the comparative experiments.

3.The SR (Spatiotemporal Reasoning) and RD (Real-Time Decision) modules operate asynchronously, and the reasoning latency of SR is not fixed. This could lead to desynchronized decisions among intersections. 

4.Cooperation among intersections is primarily achieved through prompt aggregation and message sharing at the text level. The framework would benefit from a more explicit or interpretable mechanism for cooperative policy modeling to reveal how coordination behaviors emerge across agents.

### Questions
1.The paper does not clearly explain why large language models (LLMs) are more suitable for multi-interaction cooperation than conventional multi-agent reinforcement learning (MARL) frameworks. A theoretical or conceptual discussion on the limitations of existing RL-based coordination strategies is necessary to justify the shift toward LLM-based control.


2. Comparing more recent baselines will enhance the persuasiveness of the experimental results.


3. The paper should analyze how asynchronous reasoning affects network-wide coordination and feedback consistency.

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
4

### Summary
This paper introduces CoLLMLight, the first framework that uses cooperative large language model agents for managing traffic signals across entire road networks. Unlike existing LLM-based controllers that operate intersections independently, CoLLMLight enables intersections to share spatiotemporal information and coordinate decisions asynchronously, improving network-wide traffic flow. The model includes a cost-aware cooperation optimization mechanism that balances reasoning depth and computation efficiency through adaptive reasoning and reinforcement learning. Experiments on four real-world traffic networks show that CoLLMLight outperforms traditional, reinforcement learning, and prior LLM-based methods in all evaluation metrics, while maintaining real-time responsiveness.

### Strengths
1. The problem discussed in the paper is a very important and interesting problem.

2. The ablation study of the work is comprehensive enough to understand how different parts of the framework contribute, especially in the inference time part.

### Weaknesses
1. Some important details are missing in the paper. For example, what is the base model that the authors try to optimize.  And the introduction to the framework is somehow vague, lacking a case study to show how the system really works.

2. The experimental setting is not very standard for RL-based baselines. Normally, in the original paper, RL-based methods are trained on the same map as the evaluation setup, with a different traffic scenario. However, this paper uses a transfer learning style evaluation. Though I understand LLM-based methods have a better generalization on this transfer style evaluation, which is also one of the advantages of the LLM-based method,  I hope to see a comparison for a normal setting for the RL-based method to better understand the performance.

3. This paper lacks solid proof of the effectiveness of the proposed framework. In Table3, the results show that the proposed method will perform worse than LLMLight-8B (From Table 1) without any policy refinement. Therefore, I doubt that the performance gain is coming from the training process in policy refinement instead of the multi-agent framework.

### Questions
Please see the weakness part

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CoLLMLight, a cooperative large language model framework for network-wide traffic signal control. Unlike existing LLM agents that manage intersections independently, CoLLMLight enables inter-intersection cooperation through an asynchronous decision architecture. It performs spatiotemporal reasoning to guide real-time signal decisions, ensuring both cooperation and responsiveness. A cost-aware optimization strategy further balances reasoning depth and efficiency using adaptive reasoning and reinforcement learning. Experiments on four real-world traffic networks show that CoLLMLight significantly outperforms existing rule-based, RL-based, and LLM-based methods, achieving better traffic flow and faster decision-making.

### Strengths
1. Novel Contribution: This paper proposes the first cooperative LLM-based framework for network-wide traffic signal control. It shows a new perspective for the TSC community.
2. Innovative Architecture: The asynchronous cooperative decision architecture cleverly decouples reasoning from real-time control, ensuring both deep cooperation reasoning and real-time responsiveness.
3. Adaptive and Efficient Reasoning: The introduction of cost-aware cooperation optimization and adaptive reasoning chain optimization enables the model to balance reasoning depth with computational cost, improving both efficiency and scalability.

### Weaknesses
1. Clarification issue: The paper should clearly explain how Cooperative Reasoning and Reflection are implemented. How the implementation fits the need of Cooperation and Reflection function, is not described.

2. The neighborhood radius (e.g., one-hop) and the precise set of lanes included are not fully specified; bandwidth, latency and the practical deployment cost, especially across large city-scale networks with many intersections, are not quantified or discussed.

### Questions
1. Can you provide clearer methodological details or examples/intuitions on how Cooperative Reasoning and Reflection are implemented within the pipeline?

2. Can you discuss and, if possible, test larger networks?

### Soundness
3

### Presentation
3

### Contribution
3
