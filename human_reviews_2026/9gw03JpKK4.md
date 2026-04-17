# Gaia2: Benchmarking LLM Agents on Dynamic and  Asynchronous Environments

- Decision: Accept (Oral)
- Scores: 10, 6, 8

## Abstract
We introduce **Gaia2**, a benchmark for evaluating large language model agents in realistic, asynchronous environments. Unlike prior static or synchronous evaluations, Gaia2 introduces scenarios where environments evolve independently of agent actions, requiring agents to operate under temporal constraints, adapt to noisy and dynamic events, resolve ambiguity, and collaborate with other agents. Each scenario is paired with a write-action verifier, enabling fine-grained, action-level evaluation and making Gaia2 directly usable for reinforcement learning from verifiable rewards. Our evaluation of state-of-the-art proprietary and open-source models shows that no model dominates across capabilities: GPT-5 (high) reaches the strongest overall score of 42% pass@1 but fails on time-sensitive tasks, Claude-4 Sonnet trades accuracy and speed for cost, Kimi-K2 leads among open-source models with 21% pass@1. These results highlight fundamental trade-offs between reasoning, efficiency, robustness, and expose challenges in closing the “sim2real” gap. Gaia2 is built on a consumer environment with the open-source **Agents Research Environments** platform and designed to be easy to extend. By releasing Gaia2 alongside the foundational ARE framework, we aim to provide the community with a flexible infrastructure for developing, benchmarking, and training the next generation of practical agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces Gaia2, a benchmark for evaluating large language model agents in realistic, asynchronous environments. The results from this paper highlight fundamental trade-offs between reasoning, efficiency, robustness, and expose challenges in closing the “sim2real” gap. Gaia2 is built on the open-source Agents Research Environments platform and designed to be easy to extend.

### Strengths
1, I think the experiments are comprehensive, including all the state of arts models
2, The contribution makes sense to me including releasing Agents Research Environments, a general-purpose platform for
building asynchronous, event-driven benchmarks that support scalable evaluation and data generation for RL; introducing Gaia2, the first benchmark unifying asynchronous execution, temporal reasoning, noise robustness, ambiguity resolution, and multi-agent collaboration under a verifiable evaluation framework directly usable for RLVR; evaluating leading proprietary and open-source models on Gaia2, exposing fundamental trade-offs between reasoning strength, efficiency, robustness, and cost.

### Weaknesses
To be honest, I don't see any weakness of this paper. I really think all the experiments make sense to me and are comprehensive.

### Questions
No question. Good paper.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Gaia2, a benchmark and platform for evaluating LLM agents in dynamic, asynchronous environments using action-level verification. Gaia2 simulates smartphone-like environments with realistic apps and event timelines, testing seven essential agent capabilities such as execution, temporal reasoning, robustness to noise, ambiguity resolution, and multi-agent collaboration. Scenarios are built as event and action DAGs and evaluated by a custom ARE Verifier. Experiments across many leading models reveal that no single agent excels in all areas, and that inference speed can be a bottleneck for time-sensitive tasks. Gaia2’s fine-grained verification and open-source framework support RLVR training and future extensible research in agent evaluation.

### Strengths
High-Quality, Extensible ARE Platform: The paper introduces the Agents Research Environments (ARE) platform, an open-source and extensible framework for building and testing agents. Simulating a smartphone environment with numerous apps and tools, it provides a robust foundation for community-driven benchmarking.

Asynchronous Design Enables Temporal Awareness: A key innovation is the shift from static to dynamic, asynchronous environments. By having events occur independently of the agent, the benchmark can evaluate critical, real-world capabilities like temporal awareness, responsiveness, and the ability to operate under time constraints, revealing how model latency directly impacts task success.

Reliable, Fine-Grained Action-Level Verification: The paper proposes a "write-action verifier" that evaluates each state-changing action against a human-annotated oracle graph. This verifier is shown to be highly reliable (0.98 agreement with human labels, 0.99 precision) and moves beyond simple final-answer checks. This fine-grained approach not only provides robust evaluation but also makes the benchmark directly usable for RLVR.

### Weaknesses
Dependence on a Single Orchestration Scaffold: The evaluation (Section 5) uses a single, "simple ReAct-like scaffold" for all models. It's unclear how much the reported performance is bottlenecked by this specific orchestration choice versus the model's intrinsic capabilities. The paper itself notes (Section 5.2) that its "single-threaded scaffold" cannot handle "concurrent actions."

Limited Scope (Mobile-Only): The paper claims to provide "a foundation for developing, benchmarking, and training the next generation of practical agent systems" (Abstract). However, the benchmark is currently limited to a simulated "Mobile" environment. While this is a complex and valuable domain, this scope is narrow for such a broad claim, and it's unclear how these findings would translate to other common agentic environments (e.g., desktop operation, web browsing).

Unclear Agent2Agent Mechanism: The Agent2Agent (A2A) split (Section 4.1, 5.3) introduces "app-agents" that replace native tools, forcing collaboration. However, the mechanism for their creation and operation is not fully detailed. The findings (collaboration helping weak models but not strong ones) are interesting but difficult to interpret without a clearer understanding of the sub-agent's lifecycle.

Verifier Fragility: The paper commendably includes a discussion of reward hacking (Appendix B.2.3), where an agent learned to exploit the LLM-based "soft check." While the authors mitigated this, it underscores the inherent fragility of any evaluation pipeline that still relies on an LLM as a judge, even in a hybrid system.

### Questions
How might the results change with more sophisticated orchestration instead of just ReAct? Is the benchmark senstive to this?

Where do these sub-agents come from?

The paper mentions mitigating the observed reward hacking by adding a "style" check. Could the authors elaborate on this? Do they anticipate other, more subtle forms of hacking? What is the long-term strategy for ensuring verifier robustness as agents become more adept at gaming these systems?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Gaia2, a novel benchmark for LLM agents, and the underlying platform Agents Research Environments (ARE) building it. This paper's contributions are threefold, aimed at advancing the evaluation of LLM agents in dynamic environments and realistic scenarios.:

- First, it open-sourced the ARE platform for building and running agentic evals. ARE is designed from the ground up for building asynchronous, event-driven benchmarks while also support scalable evaluation and data generation for Reinforcement Learning (RL).

- Second, the paper presents the Gaia2 benchmark. Gaia2 is the first benchmark to unify a set of critical, real-world agent capabilities under one evaluation, including asynchronous execution, temporal reasoning, robustness to environmental noise, ambiguity resolution, and multi-agent collaboration.

- Third, the authors conduct a comprehensive empirical study on Gaia2 using a suite of leading models. This study exposes fundamental trade-offs between agent reasoning strength, efficiency (latency), robustness, and operational cost.

### Strengths
- True Asynchronicity and Temporal Dynamics enabled by the async nature of the ARE platform and the event driven/DAG design for constructing more tass.

- A Scalable, Open-Source Platform (ARE) that allows the community to build new, more complex scenarios on top of this work, ensuring its long-term relevance.

- Well-Designed Verifier for RLVR: The new action-level verifier is proven to be effective, achieving much higher precision and recall than a LLM-only judge.

- The analysis of cost-performance-time trade-offs provides a practical, multi-dimensional view of agent performance that is highly relevant for real-world deployment.

- The benchmark is not saturated and successfully identifies clear weaknesses in current SOTA models. The low scores on Noise, Ambiguity, and Agent2Agent splits provide a clear roadmap for future research.

### Weaknesses
- The verifier is designed to check against a "minimal oracle sequence" of write actions, which might be brittle in some cases. It appears to evaluate path optimality rather than goal completion. An agent that makes a mistake, self-corrects (e.g., books the wrong cab, cancels, books the correct one), and ultimately reaches the correct state would be marked as a failure. 

- The "inverse scaling" result might be confounded by the agent's orchestration. Given that all models use a single threaded simple ReAct-like scaffold, this might conflate model inference latency with scaffold inefficiency. A more advanced orchestration (e.g., parallel tool calls, or a planner that coordinate) might allow a "slow" model to succeed. And overall the extremely low score on this split seem more for consideration under specific circumstance rather than general capability limitation.  

- The capability taxonomy is somewhat muddled, mixing action types (Execution, Search) with problem constraints (Time, Noise). While the design to stress test one capability in isolation is reasonable. Real-world tasks are almost never this clean; they require a composition of skills and the lack of a "Compositional" split is a weak point

### Questions
- Could the verifier be modified to support "goal completion" rather than "path optimality"? How would you propose handling valid, non-minimal trajectories that involve agent self-correction? Overall feel it might be helpful to better structure the categorization of the capability, and some break down/analysis on individual categories, e.g., does task complexity/length affect the performance on execution/search

- Given that the current scenarios test capabilities in isolation, why not adding compositional split or there are some tests showing it is not needed?

- Agent2Agent setting seem to be a bit different than the other settings, which can be used to evaluated both the main agents' planning and coordination skill, especially given partial observability of the action space, as well as the sub-agents ability on effectively communicate with the main agent and complete task, the experiments and design for this part could worth further exploration.

### Soundness
3

### Presentation
3

### Contribution
3
