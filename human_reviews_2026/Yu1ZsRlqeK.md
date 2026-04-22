# MARFT: Multi-Agent Reinforcement Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
The rapid rise of LLM-based agents has led to the emergence of LLM-based Multi-Agent Systems (LaMAS), which show strong potential in complex, collaborative tasks such as presentation generation and even scientific research. While Reinforcement Learning is well-established in enhancing LLM-based agent performance, its success has largely focused on single-agent settings. In contrast, applying Multi-Agent Reinforcement Learning to LaMAS remains limited. This is due to fundamental mismatches between traditional MARL assumptions and the unique dynamics of LaMAS, including action asynchronicity, dynamic organization, characteristic profiles, etc., which present significant new challenges. To address these challenges, we first formalize LaMAS optimization as a Flex-MG, capturing agent heterogeneity and interdependence, and then propose a novel paradigm termed Multi-Agent Reinforcement Fine-Tuning (MARFT), introducing a new optimization framework for LaMAS. Two naive instantiations of MARFT are implemented on the action-level and token-level. Comparative experiments demonstrate MARFT's superior stability and performance over representative methods, while extensive ablation studies and analysis on math problem-solving and coding benchmarks further validate its effectiveness and efficiency, establishing it as a principled and generalizable approach for tuning LaMAS. As this work establishes a new paradigm, we conclude by highlighting the limitations of current research and pinpointing promising directions for future work.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **MARFT (Multi-Agent Reinforcement Fine-Tuning)**, a framework that aims to extend reinforcement learning (RL) principles to **LLM-based multi-agent systems (LaMAS)**. The authors first formalize a new game model, the **Flexible Markov Game (Flex-MG)**, which introduces a dynamic dependency function ( D(a_i, a_j) ) to capture conditional relations between agents. Building on this formulation, they propose two concrete implementations:

* **MARFT-A** (action-level fine-tuning), where each agent’s policy is optimized with a PPO-like objective under sequential dependencies;
* **MARFT-T** (token-level fine-tuning), which treats every token as an action and defines a token-level Bellman backup.

The authors evaluate MARFT on **mathematical problem-solving** (MATH, CMATH, GSM8K) and **coding** (CodeForces) environments using Qwen2.5 models in single-, dual-, and triple-agent configurations. Experimental results show modest accuracy improvements (e.g., +3 p.p. on MATH500 and +4 points on CodeForces) compared with vanilla multi-agent or single-agent baselines.

The paper positions MARFT as a new paradigm that unifies large-language-model fine-tuning and multi-agent reinforcement learning by leveraging heterogeneous, dynamically organized agents interacting through language.

### Strengths
The paper addresses a timely problem, optimizing large language model-based multi-agent systems (LaMAS) through reinforcement learning, and presents a reasonably clear exposition of the proposed framework. The idea of introducing a Flexible Markov Game (Flex-MG) to capture dynamic dependencies among agents is conceptually interesting, and the implementation of two variants (MARFT-A and MARFT-T) demonstrates some engineering effort and reproducibility. The manuscript is well written and organized, with clear figures and experimental details that make the pipeline understandable. Overall, the paper is sound at a conceptual level and fair in its presentation quality, showing adequate awareness of related work and providing a structured attempt to formalize multi-agent fine-tuning for LLMs, even though the originality and practical contribution remain limited.

### Weaknesses
The main weakness of this paper lies in its **lack of conceptual clarity and empirical justification**. Although the Flex-MG formulation introduces a dependency function $D(a_i, a_j)$, the paper does not clearly demonstrate how these dependencies are represented or learned in practice. No concrete examples or visualizations in experiment part are provided to help readers understand how one agent’s action influences another, leaving the proposed mechanism largely theoretical and disconnected from real implementation. The **experimental design is weak**, as all evaluations are conducted on mathematical and coding tasks that do not inherently require multi-agent coordination. The “Reasoner–Actor” and “Coder–Reviewer” setups appear artificial and fail to convincingly illustrate any genuine inter-agent interaction or dependency. Moreover, the **performance improvements are minor** and could easily stem from additional fine-tuning rather than from the multi-agent reinforcement framework itself. The paper also **fails to position MARFT relative to RLHF**: it uses standard PPO objectives without explaining the conceptual or methodological distinction from human-feedback-based reinforcement learning, which is currently the dominant paradigm for post-training LLMs. Overall, the work feels **premature and speculative**, with limited novelty beyond reinterpreting existing MARL ideas under LLM settings and insufficient empirical evidence to support its claimed contributions.

### Questions
1. **Clarification on action dependency modeling:** Could the authors provide a concrete example or visualization showing how the dependency function ( D(a_i, a_j) ) is computed or updated during training? For instance, how does one agent’s output affect another’s input in the MARFT framework, and how is this handled across asynchronous timesteps?

2. **Justification for multi-agent design in math and coding tasks:** Why are tasks like MATH and CodeForces appropriate for evaluating multi-agent reinforcement learning? Can the authors demonstrate any scenario where coordination between agents (e.g., Reasoner–Actor or Coder–Reviewer) is essential to solving the task, rather than just increasing model complexity?

3. **Connection and difference with RLHF:** The paper repeatedly refers to “reinforcement fine-tuning,” but appears to use standard PPO without human feedback. Could the authors clearly articulate how MARFT differs from RLHF in terms of training signal, objective function, or supervision source?

4. **Empirical evidence of coordination benefits:** Beyond modest accuracy gains, is there any qualitative or behavioral analysis (e.g., communication traces, dependency activations) showing that MARFT leads to more coordinated or efficient agent behaviors?

5. **Scalability and practical feasibility:** The proposed setup requires separate GPUs for each agent and sequential rollouts. How would this approach scale to realistic multi-agent LLM systems with more than three agents or longer tasks? Are there plans for efficient parallelization or off-policy training to address this limitation?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Multi-Agent Reinforcement Fine Tuning (MARFT), which is a framework for optimizing LLM-based multi-agent systems. The authors develop two algorithms: MARFT-A (action-level) and MARFT-T (token-level). The method is theoretically supported and empirically validated.

### Strengths
- The formulation is original, and the idea to apply multi-agent RL for LLM agents is novel, to the best of my knowledge. 
- The use of Theorem 1 to justify the sequential nature of the system is sound. It provides a better understanding of the approach.
- The empirical study is showing that the prorposed approach is effective.

### Weaknesses
- If I understand the setting correctly, independent RL can also be applied.  Namely that each agent is not aware of the existence of the others. This can be a valid baseline to compare with but it is not shown in the experiments. Actually there is no baseline in the experiments other than the vanilla performance.

- Following the previous point,  it is therefore hard to evaluate the proposed approach without baselines. So at this moment I cannot tell how useful it is to formulate it at a multi-agent level (rather than single agents)

- Figure 3 contains way more information than is presented in the main contents. Concepts including Central Critic Head, Buffer, "inst" are not explained.  So I would suggest to either simplify the figure a lot, or provide more explanation in text.

- Minor: certain notations are missing definition. The advantage function in Theorem 1 is not formally defined.

### Questions
- How sensitive is MARFT-A to the order of agent updates? In some cases certain agents can running in parallel rather than sequentially. For example, in figure 1, calendar agent and location agent can run at the same time. Although we can still model it as a sequential problem, does agent ordering affect stability or final performance?
- In section 5.5, do you mean the implementation is without vllm and sglang, and is it because they cannot be incorporated due to some technical issues?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MARFT (Multi-Agent Reinforcement Fine-Tuning), a new paradigm for optimizing LLM-based Multi-Agent Systems (LaMAS) via reinforcement learning. The authors first formalize the Flexible Markov Game (Flex-MG) to account for agent heterogeneity, asynchronous execution, and dynamic organizational dependencies. Building upon this, they propose MARFT-A (action-level) and MARFT-T (token-level) instantiations, extending PPO-like optimization to multi-agent language systems. Experiments on math problem solving and coding tasks demonstrate consistent performance improvements over vanilla LaMAS baselines.

### Strengths
1. The paper generalizes reinforcement fine-tuning from single-agent MARL to LaMAS systems, addressing an important theoretical gap.
2. The proposed Flex-MG formulation is reasonable and effectively models dynamic dependencies among agents. 
3. The experiments on coding and math problem-solving tasks verify the effectiveness of the proposed method, particularly MARFT-A, which shows stable and consistent improvement.

### Weaknesses
1. The method and experiments are limited to single-round tasks, which raises concerns about the framework’s scalability to more complex or multi-turn interactive environments. Could MARFT be extended to handle richer LaMAS settings (e.g., multi-turn reasoning or tool-use workflows)?
2. The paper lacks comparison with relevant RL-based LaMAS works, such as MAPoRL[1]. 
[1] Park, Chanwoo, et al. "Maporl: Multi-agent post-co-training for collaborative large language models with reinforcement learning." arXiv preprint arXiv:2502.18439 (2025).

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces MARFT (Multi-Agent Reinforcement Fine-Tuning), a novel framework that extends reinforcement fine-tuning from single-agent language models to multi-agent LLM systems.The authors formalize a new theoretical model called Flexible Markov Game (Flex-MG) to represent asynchronous, dependency-driven agent interactions, and prove the Multi-Agent Advantage Decomposition Theorem, which allows global rewards to be decomposed into per-agent advantages.Two concrete instantiations are implemented and evaluated on reasoning (MATH, CMATH, GSM8K) and coding (CodeForces) environments. Results show that MARFT significantly improves multi-agent collaboration performance over standard supervised fine-tuning baselines.

### Strengths
- The paper systematically extends RL-based fine-tuning into the multi-agent LLM regime, which has not been thoroughly studied before.
- The Advantage Decomposition Theorem provides a clean bridge between joint and sequential optimization, addressing credit assignment across agents.-
- Well-written and well-organized

### Weaknesses
- Limited experimental scope. Experiments are restricted to relatively small-scale environments and specific LLMs (mainly Qwen-based), which raises questions about generalization to larger or open-ended LaMAS systems.
- While MARFT is compared to SFT-based LaMAS baselines, it would be valuable to include multi-agent RL algorithms for a more comprehensive comparison, especially MARTI and MAPoRL, which were mentioned in the introduction.

### Questions
- How sensitive is MARFT-A to the order of agents (since the framework assumes sequential dependencies)?

### Soundness
2

### Presentation
3

### Contribution
3
