# M$^2$-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Graphical User Interface (GUI) agent is pivotal to advancing intelligent human-computer interaction paradigms. Constructing powerful GUI agents necessitates the large-scale annotation of high-quality user-behavior trajectory data (i.e., intent–trajectory pairs) for training. However, manual annotation methods and current GUI agent data mining approaches typically face three critical challenges: high construction cost, poor data quality, and low data richness. To address these issues, we propose M$^2$-Miner, the first low-cost and automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search (MCTS). For better data mining efficiency and quality, we present a collaborative multi-agent framework, comprising InferAgent, OrchestraAgent, and JudgeAgent for guidance, acceleration, and evaluation. To further enhance the efficiency of mining and enrich intent diversity, we design an intent recycling strategy to extract extra valuable interaction trajectories. Additionally, a progressive model-in-the-loop training strategy is introduced to improve the success rate of data mining. Extensive experiments have demonstrated that the GUI agent fine-tuned using our mined data achieves state-of-the-art performance on several commonly used mobile GUI benchmarks. Our work will be released to facilitate the community research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces M-Miner, a low-cost and automated framework for mining high-quality intent–trajectory pairs to train mobile GUI agents. It builds on Monte Carlo Tree Search and uses a collaborative multi-agent setup with InferAgent for exploration guidance, OrchestraAgent for coordination and speedup, and JudgeAgent for trajectory evaluation. An intent recycling strategy increases intent diversity, and a progressive model-in-the-loop scheme improves mining success over time. Experiments indicate that agents fine-tuned on the mined data achieve state-of-the-art on several mobile GUI benchmarks.

### Strengths
1. The M-Miner multi-agent framework automatically collects data based on intents and appears to be the first work to do so; it reduces the time and cost of manual annotation.
2. Models trained on this dataset achieve strong performance, indicating the dataset’s potential value.
3. The framework demonstrates sufficient novelty, and ablation experiments show that each component plays a sufficient role.
4. The paper is clearly structured and easy to understand.

### Weaknesses
1. The chapter organization needs adjustment. The theoretical preliminaries of MCTS are not difficult for researchers in RL-based agent domains. The authors devote about a page to this, which seems unnecessary. It could be moved to the appendix, freeing space to detail the three-stage intent component and the training procedure. A table-style exposition would suffice rather than a full appendix-level example.
2. Although the results are SOTA, the gains are quite limited relative to some Qwen2-VL–based methods; see the Questions for specifics.
3. The approach assumes expansion from an existing dataset rather than starting entirely from scratch. It is unclear why data were not collected from scratch on the same apps. As presented, the method resembles data augmentation.
4. The quality validation reveals shortcomings in the method, and the paper does not discuss the cost of human quality review.

### Questions
1. I think the comparison in Table 1 is unfair and lacks practical significance. Were the time costs considered? Table 2 reveals limitations of the method. How exactly is the human quality review conducted? And its cost?
2. In Figure 3, the content measured by the acc metric is not defined.
3. Line 373, the explanation of CAGUI may be problematic. Since the training data are not released, how do you ensure there is no test-data leakage during training? Generalization experiments are generally conducted entirely on unseen apps. 
4. Baselines are missing, e.g., [1] Falcon-UI: Understanding GUI Before Following User Instructions and [2] MobileIPL: Enhancing Mobile Agents’ Thinking Process via Iterative Preference Learning. To my knowledge, these are not concurrent works.
5. Table 1 mentions AMEX and GUI-Odyssey as standard GUI evaluation datasets. I recommend adding experiments on these datasets (not strictly required during the rebuttal phase, given time constraints).
6. In Section 4.4, the intent evolution procedure appears not very different from the instruction filtering in [3] DiGIRL: Training in-the-wild Device-control Agents with Autonomous Reinforcement Learning. Is this understanding correct? 
7. M2-Miner-Agent training dataset + other datasets like AITZ do demonstrate the validity of the data, but how can we ensure that there is no training data leakage, because the newly generated instructions are very similar to the original ones, but we all know that the original train and test instructions are also very similar and share the same data distribution.
8. If the original data is removed from the ablation experiment and only the mined data is used, what will the model results be?
9. What is the difference between mining data from scratch using brand new instruction templates in your framework, compared to replacing existing instructions with slots? I think a comparison might be necessary, as the paper doesn't explain why it's necessary to mine data on existing data.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper propose M2-Miner, the first low-cost and automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search(MCTS). For better data mining efficiency and quality, the paper present a collaborative multi-agent frame work, comprising InferAgent, OrchestraAgent, and JudgeAgent for guidance, acceleration, and evaluation. To further enhance the efficiency of mining and enrich intent diversity, this paper design an intent recycling strategy to extract extra valuable interaction trajectories. Additionally, a progressive model-in-the-loop training strategy is introduced to improve the success rate of data mining.

### Strengths
1.This paper propose a fully automated framework for mobile GUI agent data mining. By introducing MCTS and designing a collaborative multi-agent framework, the method improve data mining efficiency while enhancing data quality. 
2.The intent recycling strategy further enhances both mining efficiency and intent richness, while the progressive model-in-the-loop training paradigm boosts success rates in both familiar and novel environments.
3.Extensive experiments show that GUI agents trained on the mined data achieve SOTA performance.

### Weaknesses
1. The paper propose an automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search(MCTS). Monte Carlo tree search is a classic algorithm, is its innovation insufficient?
2.The background knowledge of MOBILE GUI AGENT DATA MING was not sufficiently introduced in the paper writing, making it difficult to understand.

### Questions
see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-agent framework for mobile GUI agents. By introducing a tree-based structure, it organizes and stores learned operations in a structured and reusable manner. The framework incorporates three specialized agents to optimize the MCTS search process and reward estimation, effectively avoiding inefficient random exploration. Furthermore, it introduces a model-in-the-loop training paradigm that enables continual learning and self-improvement during deployment, expanding the system’s learning capability from a relatively small initial dataset. Experimental results show significant performance improvements, and comprehensive ablation studies validate the effectiveness of each component.

### Strengths
S1. Strong Experiments. It compares with 13 methods and analyzes the effect of agent numbers and online learning strategies, showing solid and comprehensive evaluation.

S2. Practical Significance. The framework is scalable and adaptable, demonstrating potential for real-world GUI automation and broader mobile applications.

S3. Trajectory Recycling is an interesting and computation-efficient design.

### Weaknesses
W1. Writing and Presentation Issues. The paper contains several typos and minor writing problems that affect readability:

1. Line 52: “we presents” ->“we present”
2. Line 274: “where i denotes the i-th visit to the node” appears twice.
3. Line 353” “This is crucial when targeting new application scenarios.” is unclear — please specify what scenarios are referred to.
4. Line 480 “significantly improve” -> “improves”.
5. Line 484 “an solid foundation” -> “a solid foundation”
6. Line 485 “Statement of Using LLM” lacks a period at the end.

W2. Outdated Baselines. The comparison does not include the latest agent-based methods, such as AppAgent[1]. The authors should either explain the exclusion or add these newer methods to the experiments.

W3. Clarity of Agent Interaction

The description of how the three agents (Infer, Orchestra, and Judge) coordinate during the MCTS process remains vague. A clearer explanation of their information flow and role boundaries would improve reproducibility and reader understanding.

[1]. Zhang, Chi, et al. "Appagent: Multimodal agents as smartphone users." *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. 2025.

W4. The novelty of applying MCTS and multi-agent in the GUI data mining is unclear.

### Questions
Q1. Consider comparing with more recent agent-based methods, such as AppAgent [1], or provide a clear justification for why these methods were excluded.

Q2. Ablation Study. The ablation study lacks clarity on how different components are decoupled. Specifically, how does the InferAgent function when the JudgeAgent is removed? Please elaborate on how the agents’ dependencies are handled during ablation.

Q3. Experimental Clarity. It is unclear how large language models (LLMs) are wrapped or packaged as agents for GUI interaction in your experiments. Please explain how the LLMs receive GUI state inputs and produce executable actions, and whether environment feedback is included in this loop.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes M²-Miner, an automated framework for mining mobile GUI agent training data using Monte Carlo Tree Search (MCTS) enhanced with a collaborative multi-agent system. The framework introduces InferAgent, OrchestraAgent, and JudgeAgent to enhance expansion and simulation efficiency, an intent recycling strategy that extracts multiple intent-trajectory pairs from a single search tree, and a progressive model-in-the-loop training approach. Experiments show that GUI agents trained on M²-Miner data achieve state-of-the-art performance on several mobile GUI benchmarks while significantly reducing annotation costs.

### Strengths
The intent recycling strategy re-evaluates sibling paths to extract multiple intent-trajectory pairs from a single search tree, significantly improving data diversity and mining efficiency without additional exploration costs.

The progressive model-in-the-loop training implements a three-stage training strategy, allowing agent capabilities to improve progressively in tandem with data complexity, which enhances the mining success rate in unseen scenarios.

### Weaknesses
- The ablation study should be expanded: include a baseline using the stronger 72B model for InferAgent and JudgeAgent, but without the model-in-the-loop (MITL) strategy. This is necessary to validate the true effectiveness of MITL.

- The paper mentions using 8 A100-80G GPUs for training and "retraining for 2 epochs on the full mined dataset at each stage". These significant computational costs, as well as the API costs for Qwen2.5-VL-72B, seem to be omitted from the 196 total cost claimed in Table 1. It is better to clarify whether the 196 figure covers this computational overhead.

- The dataset partitioning is unclear. Your training set (during warm-up ) and test set both use AC and AITZ. You need to provide more detailed information, such as statistics on the exact splits, to clarify how data overlap is prevented, especially since ICLR appendices have no page limits.

### Questions
Did you build a custom framework to run the model-in-the-loop training strategy? If so, please provide more details. If not, please specify the base framework used.

### Soundness
3

### Presentation
3

### Contribution
3
