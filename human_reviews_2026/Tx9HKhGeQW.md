# Reinforcing Query-Level Meta-Agents

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework, FLOWREASONER, aimed at automating the design of query-level multi-agent systems, a promising and significant research direction. The authors formulate the problem as training a reasoning meta-agent via reinforcement learning to dynamically generate customized workflows for each specific user query, a clear distinction from traditional task-level approaches. The methodology, combining data distillation, supervised fine-tuning, and reinforcement learning, is logically sound. The experimental results also demonstrate its superiority on several code benchmarks.

### Strengths
The paper's primary strength is its novel query-level meta-agent, FLOWREASONER, which enhances system adaptability by dynamically generating a custom multi-agent system for each unique user query. It innovatively uses Reinforcement Learning guided by real-world external execution feedback and a multi-purpose reward function to learn effective planning without relying on complex search algorithms. This approach is empirically validated, with experiments showing that FLOWREASONER significantly outperforms existing methods.

### Weaknesses
1. The core contribution of this paper appears to be demonstrating that a smaller model can be taught how to combine a series of predefined roles (Operators) in a query-specific manner. However, these foundational roles (e.g., Code Generator, Review Operator) are explicitly borrowed from prior work (e.g., Aflow). Therefore, the novelty does not seem to lie in designing new agentic behaviors, but rather in the process of learning to combine them.

2. The training data originates from three specific code benchmarks (BigCodeBench, HumanEval, and MBPP). To verify that the model has learned a general capability for code planning rather than patterns specific to these datasets, it is crucial to test it on other code datasets that are outside of this training distribution. The paper currently lacks this validation.

3. The paper's central claim is about training a general "query-level meta-agent." However, all experiments remain within the code domain. This makes it impossible to determine if the method can generalize to tasks outside of code that also require complex planning (e.g., mathematical word problems, multi-hop QA).

### Questions
1. The experiments are confined to the code generation domain. Has the meta-agent learned a general, transferable planning logic, or has it merely learned specific patterns highly optimized for the code generation scenario?

2. The paper states the reward function is a multi-purpose one considering performance, complexity, and efficiency. Could the authors elaborate on how complexity and efficiency are quantified and integrated into the final scalar reward?

3. The multi-purpose reward function (performance, complexity, efficiency) is key to the RL stage. What kind of workflows would the model generate if only performance were used as the reward? How significant are the roles of the complexity and efficiency regularizers?

4. The initial data distillation stage is foundational to the entire approach. What would the performance degradation be if this step were skipped and RL training were conducted directly on a general-purpose base model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FlowReasoner, a meta-agent that automates the design of multi-agent systems. Unlike existing methods, FlowReasoner can generate a customized system for each user query. The approach employs RL to optimize system generation, and the proposed FlowReasoner-14B model demonstrates superior performance compared to existing meta-agent baselines on coding tasks.

### Strengths
1. The paper introduces a two-stage training pipeline, including SFT and RL, to optimize the meta-agent. Experimental results show that even a relatively small model (14B) can effectively design multi-agent sytems and enhance overall performance.

2. FlowReasoner supports query-level system generation, improving adaptability to diverse queries.

### Weaknesses
1. The evaluation is limited to coding tasks, which restricts the generalizability of the method. Prior works, such as AFlow and MaAS, include broader experiments on QA, math, and tool-use tasks.

2. In Table 2, RL fine-tuning yields less than a 1% improvement, suggesting only marginal gains. Furthermore, the absence of results for a meta-agent directly using DeepSeek-R1-Distill-Qwen without SFT makes it unclear how much benefit SFT contributes.

3. Figure 4 lacks explanation for the differences indicated by color lightness.

4. (minor) A space is missing before the references in Lines 314-315.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The author proposes the query-level meta-agent FLOWREASONER, achieving "one system per user query".
It eliminates the need for manual set searching, uses external execution feedback as rewards, and trains an o1-like inference model through RL (GRPO), balancing performance, complexity, and efficiency.

### Strengths
1.FLOWREASONER improves state-of-the-art performance by 10.5% on average across three code benchmarks, surpassing o1-mini, AFLOW, MaAS, and other open-source code and models.

2.The case study shows that the same model generates more complex workflows for complex engineering tasks compared to simple algorithm problems, verifying that FLOWREASONER adapts to query difficulty.

### Weaknesses
1.Although the author emphasizes "one system per user query", FLOWREASONER is only tested on code generation (which can be executed automatically and rewarded with 0/1) task to prove its effectiveness. This limited task type may indicate that the FLOWREASONER has poor scalability. The author may consider conducting additional experiments on other types of tasks (web search Q&A, mathematics, etc.) or even real-world tasks where external feedback is difficult to obtain.

2.The design motivation of multi-purpose rewards has not been elaborated, and the impact of complexity and diversity rewards on the final efficiency and performance of the system should be further explored.

Presentation issue：
1. Section 5.1 Ablation of Meta-agents and Workers：Figure 1 -> Figure 4

### Questions
1.What are the costs of data synthesis and training 7B and 14B models respectively?

2.In section 5.2, you only prove the generalization ability of the method on different worker models. Does it generalize on unseen tasks? For example, what is the final performance after removing MBPP from the training set?

3.For subsection "Ablation of Meta-agents and Workers" and Figure 4: What methods are used in these experiments? What do the dark and light scores for each model in the diagram of Figure 4 represent?

### Soundness
3

### Presentation
2

### Contribution
3
