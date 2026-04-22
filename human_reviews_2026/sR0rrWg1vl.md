# ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real‑world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi‑hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development of LLM agents, we construct a benchmark from a controlled Azure tenant including a SQL environment covering 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated test questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning.
Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces ExCyTIn-Bench, a QA benchmark built upon Microsoft Azure security logs. Currently, there is no existing benchmark that can automatically evaluate the capability of LLM agents in cyber threat investigation. ExCyTIn-Bench leverages security log resources from Azure and simulated cyberattack scenarios designed by Microsoft security experts. The authors extracted the corresponding event logs and constructed provenance graphs, proposing a graph-based algorithm that automatically generates natural language questions and answers based on multiple alerts and their associated graph nodes and paths. In addition, the paper evaluates several models and LLM strategies on ExCyTIn-Bench to analyze their performance.

### Strengths
The paper addresses an important area at the intersection of LLMs and security. ExCyTIn-Bench is practically useful and will be valuable for other researchers to build upon.

### Weaknesses
The question generation is fully automated and lacks human expert validation. This may result in questions that lack authority, persuasiveness, and sufficient depth. Given that there are only 589 questions, manual expert verification and refinement seem feasible and would greatly improve the benchmark’s credibility.

Also, it is not clear whether the tables listed in the paper can always connect all the steps of an attack. The granularity of the logs and the coverage of the logs are not clearly explained. For example, existing techniques based on system auditing collect logs from kernels to record every system calls, and can reveal attack steps around processes accessing files and network channels but not on memory-based attacks. It is not clear about the coverage of the logs in this paper and what their limitations are. 

The paper proposes that the reward is suited for RL training, but no supporting experiments are provided. I recommend adding RL experiments to demonstrate that the reward signal indeed helps train an effective LLM agent.

### Questions
Q1. Have you considered incorporating human experts to validate generated questions to establish the benchmark’s authority and reliability?

Q2. Can you elaborate on the coverage and the limitations of the logs? Do you have empirical results on the types of attacks that can be revealed by the logs?

Q3. Can existing RL algorithms such as PPO or GRPO directly use your reward signal to train an agent? If so, how well does the trained agent perform compared with the baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ExCyTIn-Bench, a benchmark for assessing LLM agents on end-to-end cyber threat investigation over a realistic Azure/Sentinel–style SQL log environment. Questions are graph-anchored from bipartite alert. A broad model comparison shows the task is hard, while prompting/test-time strategies like ReAct/Reflect/Best-of-N improve performance. Ablations highlight the importance of alert tables and show reward patterns vs. turns and cost.

### Strengths
1.  The benchmark leverages logs with a live SQL environment, so agents must actually explore schemas, compose multi-hop queries,
and integrate evidence, which is much closer to analyst workflows than static QA. 

2. The partial-reward design evaluates process, not just final answers, and is well-suited for RL.

3. Anchoring Q/A to explicit alert avoids generic, ungrounded questions and ensures verifiable evaluation.

### Weaknesses
1. The benchmark is constructed from a single simulated Azure tenant containing only 8 attack scenarios, which is even predefined. Hence, the diversity of benchmarking is quite limited regarding log formats, and environment. However, the author has over-claimed that this is "the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation..." where the scope is much larger.

2. The QA generation relies heavily on bipartite alert-entity graphs from the "SecurityIncident" and "SecurityAlert" tables. This design assumes high-quality alerting, which may not hold in real deployments where alert noise and false negatives are frequent. In fact, the author has acknowledged that removing alert logs drastically decreases performance, which implies a strong dependency on curated signals rather than raw investigative capability.

3. The agent environment is limited to MySQL query-based interactions, without considering other sources of forensic evidence commonly found in security operations (e.g., packet captures, decompiled binaries, endpoint telemetry). This narrow scope underrepresents the full breadth of threat investigation tasks.

4. Although 589 questions were selected for testing, the benchmark initially generated over 7,500 questions. The selection and filtering criteria are not sufficiently detailed.

### Questions
Please see the comments above.

### Soundness
3

### Presentation
3

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
The paper presents ExCyTIn-Bench, a benchmark and interactive environment for evaluating LLM agents on cyber-threat investigation tasks. It organizes simulated enterprise security logs and alerts from an Azure tenant into a MySQL database, constructs bipartite alert–entity graphs, and uses these graphs to generate multi-hop question–answer pairs with corresponding reference solution paths.
Agents interact with the database by issuing SQL queries, interpreting schema outputs, and are rewarded via a discounted process-level reward that assigns partial credit for intermediate reasoning steps matching the gold solution path.
The authors evaluate both closed- and open-source LLMs (GPT-4, Claude, Gemini, Llama-4, etc.) across multiple agent frameworks (Base, ReAct, Strategy, Reflection, Best-of-N), and analyze the relationships between reward, query success rate, and computational cost.

### Strengths
1. The paper introduces what appears to be the first benchmark explicitly designed to evaluate LLM agents on cyber-threat investigation tasks, filling a notable gap in current agent-evaluation research.

2.  The use of alert–entity graphs and graph-path–based QA generation provides automatically grounded question–answer pairs with verifiable solution paths, ensuring reproducibility and objectivity.

3. The interactive MySQL-based environment effectively measures agent performance through tool usage and reasoning steps, going beyond static factual-recall benchmarks. This setup also lays the groundwork for reinforcement learning in cybersecurity with verifiable, process-level rewards.

4. The paper presents a comprehensive baseline study covering diverse model families, agent strategies, and cost–performance trade-offs, offering valuable insights into behavior and efficiency trends among LLM agents.

### Weaknesses
1. The paper uses numeric bracket citations instead of the required author–year natbib format. If reformatted according to the official ICLR template, the main text may exceed the 9-page submission cap, which could make the paper desk-rejectable under the ICLR 2026 Author Guide. This issue has been flagged for AC and PC clarification.

2. The benchmark relies on an LLM-based judge (GPT-4o) for grading open-ended answers, which introduces potential bias and lack of transparency. No quantitative inter-rater validation, such as human versus LLM agreement or kappa scores, is provided. For tasks with fixed or structured answers such as URLs, IDs, or entities, deterministic matching or simple NLP metrics could yield more reliable and reproducible scores.

3. The benchmark currently includes only eight attack scenarios, all from a single Azure tenant. This limited scope reduces diversity across organizations and threat types and may restrict generalization to broader real-world conditions.

4. The evaluation considers only the shortest reference path as the gold trajectory. This design may penalize alternative but valid reasoning paths, especially when multiple investigative routes lead to the same correct answer. A possible improvement would be to use path-set matching or edge-overlap credit to enable fairer evaluation of creative or divergent reasoning strategies.

**Minor**

1. Typo in abstract for mentioning "automatic thread investigation" that should be "threat".
2. What is "x" in ExCyaTIN-Bench?

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a benchmark to evaluate LLM agents in realistic cybersecurity investigation settings. Using a fictional Microsoft Azure environment, it constructs bipartite alert-entity graphs and generates Q&A tasks for evaluation. Each question is tied to explicit nodes and edges in the incident graph, enabling deterministic evaluation and partial credit for intermediate reasoning steps. The benchmark includes a MySQL-based interactive environment where agents query logs, reason over evidence, and receive rewards based on investigation accuracy. Experiments show the task difficulty and future potential.

### Strengths
- Q&A case generation via paths on the alert-entity graph is a viable method for producing realistic threat analysis tasks from real-world data
- The paper presents comprehensive evaluation of multiple families of LLMs, along with different prompting structures for extensive performance analysis

### Weaknesses
- Only a single fictional Microsoft Azure tenant is used to generate the benchmark, which limits its scope and realism to diverse environments and attack scenarios
- The use of discounted rewards to show partial progress is understandable, but it is not well motivated why this is the only evaluation criteria. It is natural to expect a Q&A benchmark to report accuracy (i.e. correctness of final solution) and provide partial progress as secondary information.
- Based on Algorithm 2, it seems that in the case when final solution is not correct, the reward can range from 0 to 1.66.. ( $\frac{1}{1-0.4}$ ). Taking even the extreme case that only the last step is correct, the reward is 1. This makes the partial progress reward much higher than the reward for correct solution (1), which introduces an imbalance in the evaluation
- The constraint where agents are not provided with the database schemas and need to explore themselves seems artificial and is not well motivated. Given access to the database, it is expected to have access to the schema as well because the schema is not a hidden or implicit structure of the environment but an explicitly defined quantity

### Questions
- Please answer the concerns listed in Weaknesses
- Given that answers are also LLM-generated, why is exact match performed on the final answer as opposed to an approximate semantic match for correctness
- What is the motivation behind storing raw text log data into SQL? Is this a standard industry practice?
- Introduction line 82 says "We use an LLM as an evaluator by default, but deterministic checking of answer is also available", however later in the paper, the checking is described as perfect-match based. Please clarify.

### Soundness
2

### Presentation
3

### Contribution
2
