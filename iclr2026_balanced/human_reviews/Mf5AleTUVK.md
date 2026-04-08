## Human Reviewer 1

### Summary
The paper introduces AgentFlow, a trainable agentic framework designed to address the challenges of tool-augmented reasoning in large language models (LLMs). By decomposing tasks into specialized modules (planner, executor, verifier, generator) and optimizing the planner through in-the-flow, on-policy training, AgentFlow achieves significant improvements in long-horizon tasks. The proposed Flow-GRPO algorithm tackles the sparse-reward, multi-turn credit assignment problem by converting it into tractable single-turn updates, aligning local decisions with global outcomes. Empirical evaluations across ten benchmarks demonstrate the framework’s effectiveness, with substantial accuracy gains over existing baselines and even larger proprietary models like GPT-4o.

### Strengths
1. The introduction of AgentFlow as a trainable and modular agentic framework is innovative, particularly in how it addresses key limitations of monolithic policies and static, offline training approaches. The decomposition into specialized modules is a well-motivated design that aligns with the principles of scalability and adaptability to diverse tools and tasks.
2. The paper provides a clear and robust methodological foundation. The Flow-GRPO algorithm is a well-constructed solution to the long-horizon, sparse-reward problem, and the use of trajectory-level outcomes to guide local updates is both elegant and effective.
3. The experimental results are impressive, showcasing consistent gains across a wide range of benchmarks, including search, agentic, mathematical, and scientific tasks. The 14.9% average improvement in search tasks and 14.5% gain in mathematical tasks highlight the framework’s practical value.
4. The clarity of the code and modular design of AgentFlow suggest strong potential for adoption by the community. Open-sourcing this framework would likely facilitate further research and development in agentic LLM systems.

### Weaknesses
1. While the framework demonstrates strong performance across selected benchmarks, the paper could provide more discussion on how well AgentFlow generalizes to entirely unseen tools or tasks, especially in zero-shot or low-resource settings.
2. The on-policy training approach, while effective, may introduce additional computational complexity compared to training-free or offline methods. A discussion on the trade-offs between performance gains and computational costs would strengthen the narrative.

### Questions
1. How does the Flow-GRPO algorithm scale with increasing model size or when dealing with a significantly larger number of tools? Are there any bottlenecks or limitations observed during training?
2. How robust is AgentFlow in noisy or adversarial environments where tools may fail or provide incorrect outputs? Does the verifier module effectively mitigate such issues?

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This work introduces AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules through an evolving memory, optimizing the planner on-policy within the multi-turn reasoning loop. It further proposes Flow-GRPO, a novel reinforcement learning algorithm that converts long-horizon, sparse-reward multi-turn optimization into tractable single-turn updates by broadcasting a trajectory-level reward to all turns. Experiments across benchmarks demonstrate substantial gains over state-of-the-art models.

### Strengths
1. The idea of in-the-flow reinforcement learning for agentic systems is both useful and interesting.
2. The proposed Flow-GRPO provides a stable, elegant formulation for long-horizon credit assignment.
3. The authors conduct comprehensive evaluation across multiple domains, outperforming competitive baselines.

### Weaknesses
1. The authors conduct comprehensive experiments. It's better to add discussion on computational cost and training stability.
2. The proposed method performs good on text-based tasks. How about the results on dynamic environments or multi-modal settings?
3. Minor: It's better to add some task description presented in appendix in Figure 1 or 2 to make the solution more clear.

### Questions
It's a well-written and comprehensive paper. Some potential improvements please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces AGENTFLOW, a trainable agentic system that enables multi-turn reasoning and tool use by optimizing its planner in the flow of execution. Traditional tool-augmented LLMs train a single, monolithic policy that struggles with long-horizon reasoning. AGENTFLOW instead decomposes reasoning into four interacting modules that coordinate through a shared evolving memory.
The key technical contribution is Flow-based Group Refined Policy Optimization (Flow-GRPO), an on-policy RL algorithm that converts sparse, long-horizon reinforcement learning into tractable per-turn updates by broadcasting a single trajectory-level reward to all turns and using group-normalized advantages for stability. This design allows for end-to-end optimization of the planner within the live agentic loop.

### Strengths
- The paper addresses the sparse reward problem in LLM-based reasoning and enables the training of modular, multi-tool systems in dynamic environments

- strong empirical results

### Weaknesses
- representation should be improved; it is super hard to follow

- The reliance on LLM-as-a-Judge rewards risks evaluation leakage. I also find the direct distribution of the global reward across turns unconvincing.

- Although the results are mostly qualitative and good, the paper does not report the computational cost, which raises concerns about efficiency.

### Questions
First of all, I would like to thank the authors for their work.
I would like to ask how reward hacking is avoided when global rewards are distributed directly to each turn? In such cases, even undesirable behaviors within a successful trajectory could be reinforced, potentially leading to unnecessarily long or inefficient trajectories. Additionally, could the authors comment on the computational cost of the training process?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
6

### Confidence
2