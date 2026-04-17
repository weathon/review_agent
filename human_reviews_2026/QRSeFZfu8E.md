# Generalizable End-to-End Tool-Use RL with Synthetic CodeGym

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Tool-augmented large language models (LLMs), hereafter LLM agents, leverage external tools to solve diverse tasks and interface with the real world. However, current training practices largely rely on supervised fine-tuning (SFT) over static trajectories or reinforcement learning (RL) on narrow tasks, which generalize poorly beyond development settings and lead to brittleness with new tools and unseen workflows. Because code execution reflects many structural patterns of real-world workflows, we use coding problems as a structured substrate to build tool-use agent training environments with diverse task configurations. To this end, we introduce **CodeGym**, a scalable framework that synthesizes diverse, verifiable, and controllable multi-turn tool-use environments for agent RL, enabling LLM agents to explore and master various workflows actively. CodeGym converts static coding problems into interactive environments by extracting atomic functions or logic into callable tools, yielding verifiable tasks that span various tool-execution workflows. Models of varying sizes and chain-of-thought configurations trained in CodeGym exhibit consistent out-of-distribution generalizability; for example, Qwen2.5-32B-Instruct achieves an absolute accuracy gain of 8.7 points on the OOD benchmark $\tau$-Bench. These results highlight CodeGym as a step toward scalable general-purpose RL environments for training tool-use behaviors that align with real-world agent workflows. Our code is publicly available at https://github.com/StigLidu/CodeGym.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CodeGym, a scalable pipeline that rewrites coding problems into interactive, verifiable, multi-turn tool-use environments (extracting atomic functions as callable tools; rewards come from unit tests), then uses RL (GRPO) to train LLM agents that generalize to new tools and workflows; models fine-tuned on CodeGym show consistent OOD gains

### Strengths
1. Clear problem framing: SFT on static traces and narrow RL don’t transfer, and a concrete, verifiable-reward environment design grounded in real execution logic can help
 2. The paper proposes an automated synthesis-and-verification pipeline that scales to 13k environments with documented tools, complexity filters, and solvability checks; 
3. Thorough empirical evidence across model sizes and benchmarks showing broad OOD improvements and they also did behavior analyses.

### Weaknesses
1. Despite breadth, the environment family is still derived from coding tasks, so transfer to non-code, high-latency, or noisy real-world tools  may be limited; 
2. Binary, terminal rewards and fixed turn caps can under-shape credit assignment for long-horizon agentic tasks; 
3. Environment quality depends on LLM-driven synthesis, rewrite and unit tests

### Questions
How can large companies develop similar environments in real-world scenarios?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces CodeGym, a framework that converts static coding problems into interactive, verifiable reinforcement learning environments for training large language model (LLM) agents. Instead of directly generating code, models interact with environments by calling “tools”, i.e., atomic functions extracted from coding solutions, and receive verifiable feedback based on unit tests. The goal is to train models to develop generalizable tool-use capabilities rather than pure coding ability. CodeGym includes ~13k environments and 86k task configurations, each representing different tool-use workflows. Experiments with Qwen2.5 and QwQ models demonstrate improved out-of-distribution generalization on benchmarks such as τ-Bench and ALFWorld, suggesting that training in CodeGym enhances multi-turn reasoning and tool orchestration skills.

### Strengths
- The work proposes a scalable, controllable RL environment for LLMs. As scaling trial-and-error experiences becomes increasingly important in LLM training, it is valuable to see a framework that provides a controlled, verifiable setup for agent RL. Building RL environments from existing code problems makes CodeGym practical and reproducible.
- Leveraging code as the substrate for tool environments is both natural and effective. Code inherently defines structured logic and function calls, which map well to the notion of tools in agentic workflows. The decomposition of coding problems into callable primitives offers a principled and scalable way to generate diverse and verifiable tasks.
- The results are solid and consistent. Across multiple model scales and benchmarks, CodeGym-trained models show measurable gains, particularly in tool-use and multi-turn interaction benchmarks. The improvements over SFT baselines are convincing and empirically well-supported.

### Weaknesses
- The paper’s framing and terminology are confusing at first read. The title and abstract give the impression that CodeGym is a gym for improving coding ability, while the core contribution is about tool-use training using code-derived environments. Clarifying this early that the goal is not coding but interactive tool-use would significantly improve readability and accessibility.
- Wording such as “synthesize” in the abstract may mislead readers. The current phrasing suggests that environments are generated from scratch, while in reality they are transformed from existing coding problems. Making this grounding explicit (e.g., “transforming coding problems into interactive environments”) would make the contribution clearer.
- The experimental evidence for scalability is somewhat limited. Although the method is described as scalable, the RL training runs are short (around 100 steps) and already saturate in reward. This leaves open whether the framework can truly support long-horizon or large-scale training. A longer training curve or evidence of continued scaling benefits would strengthen the claim.

### Questions
see weakness.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CodeGym, a pipeline that converts coding problems into interactive, verifiable, multi-turn tool-use environments for training LLM agents with RL. The method rewrites solutions into atomic callable tools, wraps tasks as Gym-style POMDPs and verifies solvability via unit tests and a pass@K oracle-generation step before using them for RL. Experiments across Qwen2.5 (7B–72B) and a long-CoT model show consistent gains on OOD agent benchmarks (e.g., average +7.3 for Qwen2.5-32B; τ-bench improvements averaging +8.7 across domains), with analyses showing increased, more structured tool-call behavior after training. The work matters because it offers a scalable, verifiable-reward RL setting for tool-augmented agents that better reflects real-world workflows than static SFT trajectories.

### Strengths
- The paper’s pipeline for turning coding problems into verified, multi-step environments is technically well thought out. The idea of checking solvability through unit tests and pass@K is clean and reproducible, and the dataset scale (tens of thousands of examples) makes it a potentially useful benchmark for RL-style training.

### Weaknesses
- Uncertain real-world mapping of “program → workflow.” The central motivation—that program structure mirrors real-world workflows—is argued conceptually, not demonstrated empirically. The environments are synthetic and limited to code-style function calls, not real API interactions or systems with state, latency, or uncertainty. The authors don’t justify why converting code problems into multi-step RL tasks helps with general agent performance. If the goal is to improve planning or tool use, why not build tasks from more realistic domains (e.g., APIs, documents, web agents)? Right now, it feels like a proxy task without a demonstrated external benefit (showing the performance improvements on more out-of-domain benchmarks compared to the training domain would demonstrate this!)

- Almost all the tool-use results come from τ-bench and τ²-bench, which share design and simulator assumptions. Similarly, the CodeGym trajectory format seems very similar to Alfworld's -- i.e., Alfworld also shares design & simulator assumptions with CodeGym. These are not broad or diverse enough to show real generalization. I'd love to see more tool use evaluation beyond the scope/domain setup of tau-bench to further validate the proposed approach (e.g., on some benchmark that tests "MCP tool calling capability" of models, or even running agent on SWE-Bench, etc)

### Questions
- What’s the actual real-world analogy for these “converted coding problems”? Do you see this as a proxy for software agents, or just a sandbox for multi-step reasoning?
- I'd love to see more results from benchmarks, that's a lot different compared to tau-bench and alfworld

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CodeGym, a pipeline that converts coding problems into interactive, verifiable multi-turn tool-use environments for RL training of LLM agents. Environments are synthesized by extracting atomic functions as callable tools, validated with unit tests, and filtered by tool-use complexity/difficulty. Training (GRPO) on CodeGym reportedly improves out-of-distribution performance and scales to 13,116 environments with 86,165 task configurations.

### Strengths
The observation convincingly reflects real-world workflow structures. The code naturally encodes sequences of function calls, preconditions, and validations, highly consistent with how tool-enhanced agents operate. CodeGym leverages this to create synthetic, scalable reinforcement learning environments while maintaining their verifiability.

The environment generation process in Figure 2 comprises two phases: synthesis and validation. This process employs automated LLM-assisted tools for extraction and unit testing for validation. The use of pass@K checks ensures solvability and stability, crucial for generating large-scale synthetic data.

Binary rewards and deterministic unit testing eliminate noisy or subjective reward models. The CPU-intensive environment servers, deployed in a distributed system, are carefully designed for scalability.

CodeGym training delivers continuous improvement across various model scales – particularly with object-oriented design (OOD) tools and multi-round benchmarking (τ-Bench, ALFWorld). Figure 6 shows smooth, non-overfitting training curves, indicating diverse and stable environments.

Ablation experiments directly compared reinforcement learning (RL) with SFT/distillation methods, verifying the hypothesis that "RL has stronger generalization ability than SFT". The results strongly support this hypothesis, with RL's OOD score being significantly higher than SFT's.

### Weaknesses
While the paper argues that code structure reflects real-world workflows, CodeGym's environment remains purely text- and algorithmic, lacking the noisy, unstructured dynamics common in real-world tools (APIs, browsers, bots). This may limit its generalization ability in non-code scenarios.

All tasks originate from coding problems (mostly algorithmic or numerical). While they exhibit logical diversity, this does not represent semantically or context-dependent tool usage, such as webpage navigation or database queries. The term "generalizable" should be limited to "within the domain of code-like tools."

The paper presents reward curves and average tool calls but does not analyze the agent's qualitative behavior, such as the diversity of planning structures, recovery strategies, or tool choices. This makes it difficult to determine what kind of reasoning reinforcement learning (RL) actually induces.

Due to sparse rewards and long paths, the efficiency and integral allocation of reinforcement learning are crucial. The paper mentions GRPO training but neglects exploration strategies or entropy regularization settings, factors that affect sample efficiency and stability.

Some reported OOD benchmarks (such as τ-Bench and SWE-Gym) remain code-dependent. A clearer definition of the boundary between structural generalization and semantic generalization would help strengthen the assertion of its broad applicability.

### Questions
To what extent can the behavior trained by CodeGym be transferred beyond the coding environment? Have you tested its ability to generalize across domains to non-code-derived tool use cases (e.g., retrieval, Web API)?

Under sparse binary rewards and long trajectories, which stabilization techniques are most critical for convergence? Have you tried reward shaping, lesson learning, or entropy regularization to improve exploration?

Besides the average number of tool calls, have you performed qualitative or trajectory-level analysis of what the agent has learned (e.g., improved planning, error recovery, or adaptive exploration patterns)?

Since the environment generation process relies on LLM (Seed-1.6-Thinking) for code rewriting and tool extraction, how sensitive is the generated dataset to the quality of this generator? Will a weaker LLM affect the stability of validation?

### Soundness
3

### Presentation
3

### Contribution
3
