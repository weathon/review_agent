# AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large language models (LLMs) based agents have demonstrated strong proficiency
in leveraging external tools to address complex problems. However, existing evalu-
ations largely overlook the temporal dimension of tool invocation, particularly the
practical impact of inherent tool response latency, and they are typically confined
to single-task scenarios. In realistic applications, tasks often need to be executed in
parallel, and overall efficiency critically depends on the ability to utilize idle time
during tool response delays. We denote this capability as asynchronous tool calling.
To address the lack of evaluation in this area, we propose ASYNCTOOL, which, to
the best of our knowledge, is the first benchmark specifically aimed at assessing
the asynchronous multitasking abilities of LLM-based agents within interactive
tool-use contexts. ASYNCTOOL consists of composite tasks with intra-task step
dependencies that must be executed concurrently while incorporating realistic tool
response delays. Through a hybrid data evolution strategy, we construct a diverse
and representative asynchronous multitasking dataset that covers multiple scenarios
and exhibits a wide range of tool use patterns. We further assess performance from
three levels, namely Step Level, Sub-Task Level, and Task Level, covering perspec-
tives from fine-grained to coarse-grained. Extensive experiments on ASYNCTOOL
show that even state of the art models experience notable performance degradation
when confronted with complex asynchronous workflows. Our analysis identifies
the main failure modes of current tool agents and provides practical guidelines
for designing future systems with stronger temporal reasoning and coordination
capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ASYNCTOOL introduces a benchmark for evaluating LLM’ ability to handle asynchronous, multi-task tool use, where tool responses are delayed. The authors construct the dataset by combining 2–3 independent tool-based tasks and simulating one-round response latencies, creating ground-truth trajectories that utilize idle time. Evaluation on top models shows that they struggle with this temporal coordination, exposing a gap in current agentic reasoning.

### Strengths
The paper addresses a timely and underexplored gap in agentic LLMs.

The motivation and novelty are clear: testing agents’ temporal reasoning and their ability to manage idle time.

The authors conduct a solid set of experiments across diverse models and uncover systematic failure modes.

### Weaknesses
1- The benchmark assumes a uniform one-turn delay for all tool calls. In practice, real-world tools have variable latencies. The paper lacks ablations or robustness analysis under different delay distributions, which limits generalizability.

2- Although ASYNCTOOL introduces three levels of evaluation (step, subtask, task-level), these are still based on correctness metrics. There is no explicit metric to quantify how well an agent utilizes idle time or performs efficient scheduling. As a result, true gains in scheduling behavior are not isolated.

3- Presentation issues:
Table 2 is difficult to interpret; column labels are not clearly defined. Also, the paper does not clearly explain how tool delays are simulated during evaluation (distinct from dataset construction). Qualitative examples showing how delayed or repetitive tool calls from the agent lead to the observed drop in metrics would help readers better connect the evaluation setup to the failure patterns.

4- The benchmark assumes a single reference trajectory that represents the optimal schedule under fixed latency. However, there can be multiple valid ways to use idle time effectively. The evaluation may penalize models that produce alternative but equally efficient schedules, making the metric sensitive to arbitrary ordering choices rather than genuine reasoning quality.

5- The paper does not explore potential solutions. Exploring mitigation strategies, like a basic fine-tuning experiment, would strengthen the contribution.

### Questions
See in the weaknesses above.

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
4

### Summary
This paper addresses a key gap in LLM agent evaluation: most benchmarks ignore tool response latency and focus only on single-task scenarios. The authors propose ASYNCTOOL, which is the first benchmark to evaluate an agent's asynchronous multitasking capabilities by simulating realistic tool response delays and requiring concurrent task execution. A high-quality dataset was also constructed using a four-step pipeline (collection, AI reconstruction, human annotation, multi-task composition). The experiments are conducted at the Step, Sub-Task, and Task levels, and show that even SOTA models struggle significantly with these complex asynchronous workflows, revealing key failure modes.

### Strengths
1. The authors introduce tool response latency into LLM evaluation, which is critical for real-world deployment but often overlooked in existing benchmarks.

2. The proposed three-level evaluation system (step, sub-task, task) provides a nuanced understanding of model behavior. Including 19 models across different scales strengthens its empirical contribution.

3. The pipeline for building the benchmark is transparent and well-documented, which enhances reproducibility and trustworthiness.

4. The analysis of failure modes (e.g., tool confusion, task neglect) is clear and provides specific guidance for future research.

### Weaknesses
1. The use of a fixed "one-round delay" is unrealistic. Real-world latency is variable and unpredictable, so the benchmark may only test a simple heuristic rather than true asynchronous management. 

2. The reliance on single, deterministic ground-truth paths turns the task into "plan following" rather than "plan generation". In addition, the benchmark does not take into account resource contention or mutual exclusion (e.g., a tool being locked by one task). All of these shortages limits the benchmark's realism.

2. The current evaluations only focus on correctness (e.g., accuracy, F1), not considering efficiency (such as total task completion time, token usage, or number of tool calls), which is important in real-world multitasking scenarios.

4. The evaluations do not penalize excessive or unnecessary tool calls, which could lead to gaming the system through brute-force switching strategies.

### Questions
1. Could you clarify how the latency of each tool call is determined? Is it uniformly set to one round for all tools? If so, what is the justification for this simplification, and do you plan to support heterogeneous or stochastic latency in future versions?

2. Have you conducted the experiments without tool-call latency (i.e., synchronous setting)? Such a comparison would help isolate the impact of asynchrony and better demonstrate the value of the benchmark.

3. If a model is strong at single-task tool use, it might achieve high scores by frequently switching tasks with a carefully crafted prompt. Have you observed such behavior? Are there any mechanisms or penalties in place to ensure that the evaluation rewards strategic scheduling rather than heuristic over-invocation?

4. Can you elaborate on the trade-off between evaluation simplicity (using deterministic paths) and realism? Is ASYNCTOOL evaluating planning or just the ability to follow a complex instruction?

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
3

### Summary
This paper introduces ASYNCTOOL, a new benchmark for evaluating LLM agents' capability to handle multiple tasks concurrently through asynchronous tool calls. The authors identify a critical gap in existing benchmarks, which typically assume instantaneous tool responses and focus on single-task scenarios. ASYNCTOOL addresses this by simulating tool response latency, thereby requiring agents to manage idle time by interleaving the execution of different tasks. The paper details the construction of a multi-task dataset via a "hybrid data-evolution strategy" and evaluates a wide range of LLMs using a three-level metric system (Step, Sub-Task, Task). The main finding is that even SOTA models struggle significantly with asynchronous workflows, with the paper providing an analysis of common failure modes.

### Strengths
1. Novel Problem Formulation: The paper is the first, to my knowledge, to systematically formalize and create a benchmark for asynchronous multi-task tool use, a critical and under-explored area for LLM agents.
2. Rigorous Data Curation: The multi-step data construction process, combining LLM-based generation with intensive human verification, is a strong point and likely results in a high-quality, internally consistent dataset.
3. Comprehensive Analysis: The paper provides a good qualitative analysis of agent failure modes, offering useful insights into why current models struggle with this task.

### Weaknesses
1. Unverifiable SOTA Results: The use of unreleased and hypothetical models (GPT-4.1, GPT-5) for key results is a major flaw. It makes the reported SOTA performance impossible to reproduce and sets an unstable target for the research community. This significantly reduces the benchmark's practical utility.
2. Unrealistic Task Environment: The benchmark's core mechanics—a fixed one-round latency and a single deterministic correct trajectory—do not reflect the variability and flexibility required in real-world scenarios. This limits the generalizability of the findings and risks over-fitting future models to this specific, simplified setup.
3. Lack of Quantitative Error Breakdown: The paper qualitatively describes failure modes but misses the opportunity to provide a quantitative breakdown in the main text. Quantifying the prevalence of "tool confusion" vs. "task neglect" across different models would provide much stronger and more actionable evidence.

### Questions
1. Justification for Unreleased Models: Could the authors justify the decision to benchmark and prominently feature hypothetical or private preview models? Given the negative impact on reproducibility and the benchmark's utility, would it not be more scientifically sound to re-frame the results around the best-performing publicly accessible models as the primary baseline?
2. Defense of Deterministic Trajectories: Please provide a stronger defense for enforcing single, deterministic solution paths. Have you analyzed how many tasks in your dataset could have alternative valid paths? How does this design choice not risk penalizing more advanced, flexible reasoning agents in favor of those better at sequence imitation?
3. Impact of the Latency Model: Can you provide any ablation or theoretical argument on how the results might change with a more realistic, variable latency model? Does the current fixed-delay model truly test "temporal reasoning," or does it primarily test context-switching and short-term memory?

### Soundness
3

### Presentation
2

### Contribution
3
