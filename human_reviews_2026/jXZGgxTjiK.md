# ALMC: Adaptive LLM-based Multi-Agent Collaboration Across Diverse Task Domains

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Large language model-based multi-agent systems (LLM-MAS) are effective at solving complex tasks by coordinating specialized agents. However, existing frameworks rely on a small set of predefined scenarios with static role configurations and rigid collaboration structures, limiting their adaptability across diverse task domains. We propose the Adaptive LLM-MAS Collaboration (ALMC) framework, which dynamically recruits agents and configures collaboration patterns according to task demands through three collaborative components: a Manager Agent that synthesizes task-specific role compositions and an executable workflow, a Judge Agent that evaluates execution quality, and a Solution Optimizer Agent that persists and reuses high-quality configurations via retrieval-augmented generation. The framework supports human-in-the-loop review and creates a learning loop where previous superior configurations improve future executions on similar tasks. By using ALMC, collaborations become adaptive, auditable, and reusable across domains. Code is available at: https://anonymous.4open.science/r/ALMC-2E0F.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new framework called ALMC (Adaptive LLM-based Multi-agent Collaboration), which is an adaptive multi-intelligent body collaboration framework based on the Large Language Model (LLM). The framework aims to address two major challenges that exist in current multi-intelligent body systems: (1) The contradiction between generality and specialization: existing systems are either too general, leading to poor performance on complex tasks, or too specialized, making it difficult to adapt to new task domains and (2) lack of accumulated experience: most systems are unable to learn and reuse solutions from past successes. The experiments were conducted using different base bigram models (e.g., GPT-3.5, GPT-4o-mini, Llama-3.1-8B, etc.). The results show that ALMC maintains strong performance on different base models, proving the robustness of its framework.

### Strengths
- The ALMC framework demonstrates robust adaptability by dynamically generating task-specific agent compositions and workflows. This is strongly supported by its superior performance not only across four distinct domains but also in the out-of-domain stress test (chemistry), where it surpassed both general-purpose and specialized baseline methods.
- The integration of the Judge Agent and Solution Optimizer Agent creates a novel learning loop. This allows the system to systematically assess, store, and reuse successful collaboration patterns, addressing a common limitation in multi-agent systems and contributing to more stable and improved performance over time.
- The paper provides comprehensive empirical evidence showing that ALMC achieves state-of-the-art or highly competitive performance against strong baselines. The results are consistent across multiple base LLMs (e.g., GPT-3.5-turbo, GPT-4o-mini), highlighting the robustness and effectiveness of the proposed framework itself.

### Weaknesses
- There has been a significant amount of existing works like AgentVerse [1], CaptainAgent [2], GPT-Swarm [3], etc, about dynamically building an agent team for task solving, and I cannot identify the difference between this work and those previous works (and also the author didn't describe the difference between these works and the proposed work).
- The core claim of the Solution Optimizer and Judge Agent is that the system learns from past successes. However, the experiments do not provide direct evidence of this learning process. A crucial missing experiment would be a longitudinal study: by processing a dataset in sequential chunks (as described in the `Section 4.3.2`), the authors should demonstrate a clear and consistent performance improvement from the first chunk to the last. Without this, the benefit of experience accumulation remains more of a theoretical assertion than an empirically proven advantage.
- The experiments are conducted on well-defined, single-turn tasks (like Q&A or generating a single function). It is unclear how ALMC would scale to more complex, multi-step, or long-horizon problems (e.g., developing a complete software module from scratch, requiring iterative refinement and dependency management). An experiment on a more complex benchmark would be necessary to test the limits of the Manager Agent's planning capabilities and to assess whether the dynamically generated workflows remain coherent and efficient as task complexity increases.

Refs:

[1] Chen, Weize, et al. "Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors." The Twelfth International Conference on Learning Representations. 2023.

[2] Song, Linxin, et al. "Adaptive in-conversation team building for language model agents." arXiv preprint arXiv:2405.19425 (2024).

[3] Zhuge, Mingchen, et al. "Language agents as optimizable graphs." arXiv preprint arXiv:2402.16823 (2024).

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
ALMC proposes a task-adaptive multi-agent framework where a Manager retrieves prior high-quality workflows and synthesizes a task-specific configuration of roles, phased steps, and collaboration rules, execution proceeds via pairwise dialogues that freeze intermediate artifacts for the next phase, and a Judge plus Solution Optimizer evaluate results and persist successful configurations into a RAG memory for reuse across similar tasks; experiments across coding, medical QA, quantitative reasoning, and finance suggest higher accuracy than common debate or voting baselines, with additional transfer to chemistry-style tasks.

### Strengths
- Proposes a task-adaptive orchestration that learns and reuses workflows via a Manager–Judge–Solution Optimizer loop with RAG-backed memory, and emphasizes pairwise, phase-scoped dialogues with frozen intermediates.
- Presents a clear end-to-end algorithm specifying configuration synthesis (R,P,G), execution, evaluation, and persistence, with ablations on agent count and Judge/Optimizer contributions, plus cost/latency reporting alongside accuracy across multiple domains.
- The paper delineates roles, phases, collaboration rules, and intermediate artifacts in a structured manner; figures and pseudocode make the dataflow and decision points easy to follow.
- Demonstrates consistent gains over common multi-agent baselines across coding, medical QA, quantitative reasoning, and finance, and showcases transfer via configuration reuse, timely given ongoing evidence that stronger orchestration can outperform naive debate or simple voting.

### Weaknesses
- Report results under the same backbone, temperature, stopping rules, tool permissions, and token budgets, and include variance over multiple random seeds with significance tests. This avoids hidden advantages from longer debate chains or richer tool access.
- The claimed determinism of pairwise collaboration should be tested against Multi-Agent Debate, Tree-of-Thoughts, and Graph-of-Thoughts under matched budgets, with accuracy-vs-tokens frontiers.
- Persisting and retrieving configurations without clear indexing fields, similarity thresholds, and domain isolation risks cold-start inefficiency, unstable cross-domain transfer, and evaluation contamination (including leakage and privacy exposure). Add ablations for “no-memory / in-domain / cross-domain” and disclose retrieval specifics and leakage mitigations.
- The paper lacks discussion and head-to-head comparison with closely related lines

[1] Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration

[2] Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks

[3] AutoAgents: A Framework for Automatic Agent Generation

[4] Voyager: An Open-Ended Embodied Agent with Large Language Models

### Questions
- Provide controlled ablations that swap ALMC’s intra‑phase pairwise controller with Multi‑Agent Debate (MAD) and search‑style controllers such as Tree‑of‑Thoughts or Graph‑of‑Thoughts under matched budgets. Plot accuracy‑vs‑tokens frontiers and comment on stability.
- When the memory is empty or sparse, what is the fallback (e.g., “no‑memory ALMC”) and performance delta?
- What index fields and similarity thresholds gate reuse across tasks/domains? Provide ablations for no‑memory / in‑domain / cross‑domain reuse.
- State exact decoding settings (temperature, top‑p, max turns), stop rules, tool permissions, and token budgets shared by all baselines; report mean±std over multiple seeds and significance tests. This helps avoid hidden advantages from longer debate chains or richer tool use.

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
2

### Summary
This work proposes an adaptive framework based on dynamic role synthesis, reusable experiential memory, and human-AI collaborative review. The system autonomously generates task-specific role combinations and execution workflows according to task requirements, while continuously optimizing performance through experience reuse. It successfully achieves both generality without requiring pre-defined domain libraries and specialization through customized task configurations, thereby enabling efficient and stable cross-domain task execution.

### Strengths
1. The paper conducts comprehensive experiments across five challenging domains (coding, medicine, mathematics, finance, and chemistry), using both in-domain and out-of-domain datasets to evaluate ALMC. Results show consistent improvements over both general-purpose and domain-specific baselines.
2. The proposed “dynamic configuration of roles and collaboration modes + RAG-based experience reuse” allows the system to automatically generate role configurations, phase divisions, and workflows tailored to each task. This effectively overcomes the rigidity of predefined roles and static workflows, enhancing adaptability and knowledge transfer efficiency in multi-agent systems.

### Weaknesses
1. The proposed method relies heavily on architectural intuition and empirical evidence rather than formal theoretical analysis. It lacks discussion of convergence guarantees, expressivity trade-offs, or provable limits of adaptive composition mechanisms.
2. Although the paper claims to reduce human engineering effort, each task still requires a Human-in-the-Loop Gate (HITL-Gate) review before execution. The actual utility and cost of this human involvement are not quantitatively analyzed.
3. Each task phase adopts a pairwise dialogue structure (two agents interacting over multiple turns), which is claimed to prevent deadlocks. However, for large-scale or parallel tasks requiring multiple agents, such a fixed structure could become a bottleneck. The paper does not discuss these limitations or demonstrate performance in more complex or asynchronous settings.

### Questions
1. Can the authors quantify the actual intervention rate or the overhead introduced by human-in-the-loop review? Is this process scalable for real-world deployment?
2. Is there any theoretical or empirical analysis of when and why the pairwise agent mechanism converges to high-quality solutions, and under what conditions suboptimal negotiation or agent conflicts may arise?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes *ALMC*, a multi-agent framework with three roles (Manager, Optimizer, Judge) aiming to achieve adaptive collaboration across different task domains (code, medical, math, finance). The authors claim that ALMC dynamically designs agent roles, generates stage-wise workflows, and reuses past experiences through RAG-based retrieval.

### Strengths
This paper resembles a synthesis of prior multi-agent ideas, implemented via prompt engineering. The baselines are outdated, the “adaptivity” is ill-defined, and the claims overreach the evidence.

### Weaknesses
“Manager–Judge–Optimizer” is widely used multi-agent patterns already exist. The method is not a novel solution.

Additionally, “adaptive” aspect is not learned or optimized, but *prompt engineering. 

The claim of “continuous self-improvement” is unsupported.
The “RAG memory” is simply text retrieval without validation or analysis of retrieval quality.

For experiments: 
The paper repeatedly asserts domain generalization, but the experimental setup only involves standard benchmarks (HumanEval, MedQA, MMLU subsets).
There is **no transfer or few-shot evaluation** proving actual *adaptation*.
The authors compare only against very early/old frameworks.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
