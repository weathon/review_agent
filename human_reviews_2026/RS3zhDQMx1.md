# FlashResearch: Real-time Agent Orchestration for Efficient Deep Research

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Deep research agents, which synthesize information across diverse sources, are significantly constrained by their sequential reasoning processes. This architectural bottleneck results in high latency, poor runtime adaptability, and inefficient resource allocation, making them impractical for interactive applications. To overcome this, we introduce *FlashResearch*, a novel framework for efficient deep research that transforms sequential processing into parallel, runtime orchestration by dynamically decomposing complex queries into tree-structured sub-tasks. Our core contributions are threefold: **(1)** an **adaptive planner** that dynamically allocates computational resources by determining research breadth and depth based on query complexity; **(2)** a **real-time orchestration layer** that monitors research progress and prunes redundant paths to reallocate resources and optimize efficiency; and **(3)** a **multi-dimensional parallelization framework** that enables concurrency across both research breadth and depth. Experiments show that FlashResearch consistently improves final report quality within fixed time budgets, and can deliver up to a 5x speedup while maintaining comparable quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
FlashResearch converts deep research into a tree-structured, real-time orchestrated search process, parallelizes sub-queries and prunes low-value branches under a hard time budget. An adaptive planner chooses breadth and depth per node, while an asynchronous execution engine enables speculative deepening and early termination. Evaluated on DeepResearchGym and DeepResearch Bench (though a little bit narrow), the system processes up to 4× more nodes than the GPT-Researcher baseline and yields slightly higher report scores within the same wall-clock limit.

### Strengths
- Demonstrates measurable throughput-quality lift over a strong retrieval baseline under fixed time budgets.  
- Provides a clean, asynchronous tree abstraction that cleanly separates planning, orchestration, and execution concerns.

### Weaknesses
1. FlashResearch appears to focus primarily on an offline setting, where data are collected from a local corpus (for instance, the setting in Figure 1 seems to be strongly related to DeepResearchGym). However, I understand that one of the major challenges of deep research actually lies in tool selection and allocation within online web search scenarios—different task types may require different tools (e.g., some can be efficiently solved via a Wikipedia Search API, while others may benefit from Playwright-based visual browsing). Yet, I did not observe any adaptation or discussion regarding this aspect. I understand that DeepResearchGym may only require a single retriever tool, but what kinds of tools does DeepResearchBench actually provide?
2. The paper’s survey of current state-of-the-art deep research methods is somehow insufficient. The cited works (e.g., AFlow, Flow, and EvoFlow) remain confined to traditional math or coding tasks, whereas there now exist many multi-agent approaches specifically designed for deep research, including but not limited to [1, 2, 3, 4]. I believe it is necessary to mention, cite, and even compare these works.
3. In Line 135, the authors claim that most existing agentic workflows “lack support for real-time replanning.” This statement is inaccurate, as a number of studies have already demonstrated the ability to dynamically adjust plans in real time [5, 6]. The authors should take this into account; there is also a growing body of research on dynamically adjusting multi-agent workflows according to different task requirements/complexity, including but not limited to [7, 8].
4. The experimental presentation seems rather rushed. At least a few aspects could be analyzed in greater depth: for example, reporting economical cost (LLM API consumption); examining whether, as task complexity increases (e.g., wrt the number of ground-truth documents related to each query), FlashResearch actually learns to allocate different search resources (e.g., varying the number of search nodes); and conducting case studies to analyze the quality of replanning when failed leaf nodes occur (Appendix B seems to omit this discussion).
5. Regarding the utility function U(r) in Line 200, how is it concretely implemented?

---


[1] OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation  
[2] AgentOrchestra: Orchestrating Hierarchical Multi-Agent Intelligence with the Tool-Environment-Agent(TEA) Protocol   
[3] Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training 
[4] OAgents: An Empirical Study of Building Effective Agents    
[5] https://github.com/ZTE-AICloud/Co-Sight 
[6] https://github.com/huggingface/smolagents   
[7] FlowReasoner: Reinforcing Query-Level Meta-Agents   
[8] Weak-for-Strong: Training Weak Meta-Agent to Harness Strong Executors

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes FlashResearch, a framework for deep research. It introduces an adaptive planner for reasoning depth, a real-time orchestration layer that dynamically adjusts compute allocation, and a multi-dimensional parallelization scheme that explores both the breadth and depth of reasoning in parallel. On benchmark datasets, the framework achieves strong accuracy while improving inference efficiency.

### Strengths
1: The paper is clearly written and tackles an important efficiency problem in deep research; the method and experiments are well aligned with the motivation. 

2: By integrating adaptive planning, real-time orchestration, and multi-dimensional parallelization, the system simultaneously boosts throughput and reduces latency, making it readily deployable in production.

### Weaknesses
1: Both the adaptive planner and the real-time orchestration policies rely on LLM judgments to decide when to expand, deepen, or terminate paths. This makes the decision process hard to interpret. 
2: Evaluating the goal-satisfaction level and quality score of every node requires on-the-fly calls to an LLM, incurring measurable extra overhead.
3: The choice of 2-minute and 10-minute time budgets is motivated by prior human–computer interaction findings, but the paper lacks a broader sensitivity analysis across time budgets and does not discuss external/API-induced latency that could waste the budget. This limits understanding of robustness under different deployment conditions.

### Questions
See weakness

### Soundness
3

### Presentation
3

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
Given the limited runtime adaptability of existing methods for deep research tasks, this paper introduces a dynamic planner designed to adjust the breadth and depth of the research tree in real time. The planner leverages utility-guided strategies, where utility is determined by LLM-based judges. The proposed approach is evaluated across multiple deep research benchmarks, showing improved accuracy and reduced latency.

### Strengths
1. The examples provided in Section 3.2 highlight the necessity and potential benefits of improving the configuration of breadth and depth.
2. The explicit formulation of controlling the research tree's breadth and depth is both logically sound and empirically effective.
3. The paper is well-written and easy to understand.

### Weaknesses
1. Because the method relies on the current implementation of LLM-as-a-judge, its effectiveness depends heavily on the LLM’s ability to make accurate judgments.
2. The evaluation is limited to a single model family (Gemini-2.5); testing with additional models, such as smaller open-source ones, would strengthen the results.

### Questions
1. Can the proposed approach be generalized to utility models beyond LLM-as-a-judge?
2. Given the asynchronous operations and increased number of explored nodes, does this result in higher computational or operational costs?

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
3

### Summary
This paper presents the FlashResearch framework, designed to enable efficient and scalable deep research. The framework operates in parallel, dynamically decomposing complex queries into subtasks at runtime. An adaptive planning module determines the appropriate breadth and depth of exploration, while a real-time orchestration layer monitors progress and optimizes research pathways for maximum efficiency.

### Strengths
- The paper is clearly written and well-structured.
- Demonstrates improved research quality under fixed computational budgets, achieving up to a 5× speedup.

### Weaknesses
- The proposed framework is evaluated on only two benchmarks, with relatively small sample sizes (100 and 50 examples). The rationale for not using the full datasets is unclear. Evaluating on larger datasets or additional benchmarks would strengthen the empirical validity of the results.
- The use of LLMs for evaluation raises concerns regarding metric reliability and potential bias. This should be discussed in greater detail.
- Comparisons are limited to GPT-Researcher. Including experiments with additional models would improve the robustness and generalizability of the findings.
- The performance results are mixed, and further analysis is needed to clarify the conditions under which the proposed method is most effective.
- The paper lacks an ablation study, which would help isolate and assess the contribution of individual components within the framework.

### Questions
Was any error analysis performed to better understand the sources of the model’s successes and failures?

### Soundness
3

### Presentation
3

### Contribution
2
