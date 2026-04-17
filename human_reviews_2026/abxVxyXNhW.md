# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. 
Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pre-training data. 
To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. 
This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1’s fast, intuitive thinking with System 2’s deliberate reasoning within LLMs. 
MARS strategically integrates multiple external tools—such as Google Search, Google Scholar, and Python Interpreter—to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity.
Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency.
Extensive experiments demonstrate MARS achieves substantial improvements of 3.86\% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9\% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Inspired by the dual-process theory of human cognition, this paper proposes a collaborative two-agent framework in which one agent (System 2) conducts deliberate reasoning and invokes external tools when necessary, while the other agent (System 1) summarizes and distills the tool outputs into concise feedback for the reasoning agent. The authors extend Group Relative Policy Optimization (GRPO) into a multi-agent reinforcement learning (RL) setting, incorporating bin-packing optimization and balanced sampling to improve learning efficiency and stability.

### Strengths
The proposed framework enhances reasoning through multiple rounds of tool usage and effective cooperation between the two systems. The results demonstrate consistent improvements over several baselines and achieve an HLE score comparable to large proprietary models such as Claude 3.7 Sonnet.

### Weaknesses
1. Writing quality: The exposition can be improved for clarity and coherence. The intended connection to the dual-process theory (System 1 vs. System 2) is not immediately clear and is somewhat confusing in the abstract.
2. Novelty concerns: The idea of using tool calls (e.g., retrieval, computation) has been explored extensively in prior work (e.g., WebGPT). Likewise, the computational optimization via bin-packing is a known technique rather than a novel contribution.
3. Baseline limitations: Although the method is positioned as a multi-agent system, most comparisons are against single-agent RAG or R1-style baselines. Including comparisons with multi-agent frameworks (e.g., CAMEL, and MetaGPT) would strengthen the empirical evaluation.

### Questions
1. Are System 1 and System 2 implemented and trained using the same policy LLM but differentiated only by prompts or roles? If so, why is this setup referred to as “multi-agent”?
2. The reported HLE result for WebThinker appears inconsistent with the original paper [1]; could the authors clarify this discrepancy?
3. What causes the increase in the number of tools per question as training progresses? Why does the agent predominantly use Google Search after approximately 50 training steps? What would happen if we only use Google search as the tool? 
5. The mean response length increases sharply near the end of training, while the HLE score drops. How do the authors interpret this inverse correlation?

## [Reference]
[1] Li, Xiaoxi, et al. "Webthinker: Empowering large reasoning models with deep research capability." arXiv preprint arXiv:2504.21776 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The proposed MARS framework is a technically ambitious attempt to formalize the dual-process theory of cognition (System 1 for intuition/speed, System 2 for deliberation/reasoning) into a collaborative, multi-agent reinforcement learning  paradigm for LLMs. The paper is attempting to address the twin issues of LLM inefficiency (over-analysis on simple tasks) and static knowledge bases by delegating high-volume external data processing to a System 1 agent and strategic reasoning/planning to a System 2 agent. This system, trained using an extension of Group Relative Policy Optimization, demonstrates measurable gains on challenging benchmarks, notably HLE.

### Strengths
1. The dual-system approach is a principled, interpretable design choice that formalizes the intuitive trade-off between reasoning depth and efficiency, moving beyond monolithic LLM-based agents. The extension of GRPO to concurrently optimize two interconnected, interdependent agents (System 1 and System 2) with distinct functions is a non-trivial advancement in applying MARL to LLM-based systems.

2. Achieving substantial and rigorous performance gains (e.g., +3.86% on the challenging HLE benchmark) validates the architectural and training complexity, suggesting the system is learning truly superior decision-making policies.

### Weaknesses
1. A fundamental challenge in MARL is accurately attributing the final reward to individual agent actions. Since System 1 and System 2 policies share the same underlying LLM, the paper must provide a more rigorous breakdown of how the GRPO extension effectively disentangles the reward signal to assign credit distinctly to the System 1 (summarization) vs. System 2 (planning) policies.

2. The technical contribution of "bin-packing optimization" is a key claim, yet its utility compared to simpler alternatives (e.g., standard vector search filtering, or fixed-length truncation of results) is not sufficiently isolated and quantified. This complexity may not be justified if simpler methods yield similar gains.

3. Multi-agent RL training is notoriously resource-intensive. The paper lacks a necessary detailed comparison of the computational overhead (wall-clock time, total token consumption) of the MARS MARL training pipeline against standard supervised fine-tuning or single-agent RL baselines, making the real-world utility hard to gauge.

### Questions
How does the proposed GRPO extension and its advantage estimation method rigorously address the challenge of Temporal Credit Assignment and Policy Interference? Specifically, if a final answer is correct, how can the method assign distinct, non-interfering learning signals to the S1 agent (rewarding it for efficient, high-fidelity summarization) and the S2 agent (rewarding it for optimal, multi-turn tool-call planning)? Since S1's fast-generation policy and S2's deliberate-reasoning policy share the same neural parameters, what mechanisms are in place during the update step to ensure that optimizing S2 for deliberate reasoning does not degrade S1's learned efficiency/distillation capability, and vice versa?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
MARS presents a dual-system multi-agent RL framework that unifies intuitive (System 1) and deliberate (System 2) reasoning within an LLM, jointly optimized via GRPO to improve deep research and reasoning performance across complex tasks.

### Strengths
1. The paper is clearly written and well-structured

2. Proposes a dual-system multi-agent RL framework that explicitly models human-like System 1/System 2 reasoning, an interesting conceptual extension of existing multi-agent paradigms.

3. Demonstrates measurable gains on challenging reasoning benchmarks

### Weaknesses
1. The proposed dual-system framework essentially resembles a standard RL-based tool-use pipeline augmented with a learnable summarizer that condenses the environment’s returned content before feeding it back. While the integration is well-engineered, the conceptual difference from existing RL tool-use or summarization-based reasoning systems is limited.

2. Because the entire trajectory shares a single scalar reward, it is unclear how meaningful credit is assigned to System 1’s summarization behavior. Without step-level or component-wise feedback, System 1 receives only a weak and noisy learning signal, making it difficult to understand how it learns to produce more useful summaries. The paper could be strengthened by introducing more fine-grained supervision or ablation analyses that clarify how System 1’s updates contribute to overall improvement.

3. In Table 1, several entries marked as best (bold) and second-best (underlined) appear to be incorrect. This is misleading to readers. 

4. The ablation study mainly analyzes the impact of removing different external tools (Google Search, Scholar, Python), but this aspect is peripheral to the paper’s main contribution. Since the core claim of MARS lies in the joint optimization and coordination between System 1 and System 2, the paper would benefit much more from ablations that directly test this interaction—for example, mixing trained and untrained versions of System 1/2, or disabling their shared optimization to assess whether the two systems truly co-adapt. 

5. Even so, I find the results in the Ablation Study on Tools for HLE rather confusing. For example, in Chem, the setup with all three tools performs the worst, while both without Scholar and Scholar-only achieve the best results — which makes it unclear whether Scholar is actually helpful or not; Also both without Search and Search-only achieve the Second. Similarly, in CS/AI, the best setup is without Search, yet Search-only also performs noticeably higher than most others; and in Engineering, Python-only gives the highest score, but without Python ranks second. Overall, the patterns seem quite inconsistent or even random. Given how close these numbers are, I wonder whether you ran multiple inference trials and averaged the results. The apparent randomness in this table makes it hard to trust the conclusions on HLE.

6. Could you clarify whether the comparison between MARS and the baselines is fully fair in terms of tool usage? Specifically, do all methods have equal access to the same tools (Python, Search, and Scholar)? The results suggest that the presence or absence of certain tools has a large impact on performance, and in most subjects, removing a specific tool even makes MARS perform worse than most baselines. This raises concerns about whether the comparison setup is fully fair. It would be important to provide more details on the tool configurations for all baselines and ensure that all methods are evaluated under comparable conditions. Moreover, additional ablations are needed to justify that the reported gains on HLE truly come from the proposed dual-system RL framework, rather than differences in tool availability or usage.

### Questions
Please refer to the weaknesses section for main questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a dual-system framework named MARS. System 1 distills information gathered from various tools and feeds it to System 2, while System 2 decomposes the user’s question, selects the tool and provides parameters, and provides a “purpose” to guide System 1 in compressing the information. To support the proposed approach, the work further devises a bin-packing optimization technique to improve System 1’s rollout efficiency and employs an advantage-function-weighted sampling strategy to construct the training buffer, ensuring that Agent 1 and Agent 2 are optimized balancedly.

### Strengths
1) The paper identifies the imbalance issue in multi-agent optimization and proposes an advantage-function-weighted sampling strategy to construct the training buffer, ensuring that both agents are optimized in a balanced manner. This design is readily generalizable to broader multi-agent optimization scenarios.
2) The experimental section provides a thorough analysis of how metrics and tool-usage ratios evolve throughout the multi-agent RL training process.

### Weaknesses
1) The baseline selection for the experiments in Tables 1 and 2 appears overly simplistic: only MARS was trained on the training set listed in Table 6. This seems unreasonable; it would be advisable to compare with other RL-based methods that use the same training set.
2) The implementation that solving the rollout sequence-lengths is quite straightforward. Could you clarify whether any entirely novel design was introduced for this component?

### Questions
In terms of method design, why should System 1 and System 2 share a checkpoint? Has there been any discussion on what the effects would be if two different models were used? Is it because these two tasks have a mutually improving effect?

### Soundness
2

### Presentation
3

### Contribution
2
