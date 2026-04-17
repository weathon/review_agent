# Gistify: Codebase-Level Understanding via Runtime Execution

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
As coding agents are increasingly deployed in large codebases, the need to automatically design challenging, codebase-level evaluation is central. We propose Gistify, a task where a coding LLM must create a single, minimal, self-contained file that can reproduce a specific functionality of a codebase. The coding LLM is given full access to a codebase along with a specific entrypoint (e.g., a python command), and the generated file must replicate the output of the same command ran under the full codebase, while containing only the essential components necessary to execute the provided command. Success on Gistify requires both structural understanding of the codebase, accurate modeling of its execution flow as well as the ability to produce potentially large code patches. Our findings show that current state-of-the-art models struggle to reliably solve Gistify tasks, especially ones with long executions traces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GISTIFY, a novel task designed to evaluate code generation models on codebase-level understanding through runtime execution. Given a codebase and an entry-point command, the goal is to generate a single, self-contained, and minimal gistified file that reproduces the original runtime behavior. The authors formalize three evaluation metrics, Execution Fidelity, Line Execution Rate, and Line Existence Rate, to measure model performance. Experiments are conducted across multiple frameworks and state-of-the-art models on popular Python repositories. Results show that even advanced models struggle with complex execution traces, while agentic frameworks and access to execution-aware tools yield consistent improvements. The paper argues that GISTIFY offers a lightweight, reproducible benchmark that reflects real-world developer workflows and provides valuable distilled code artifacts for downstream applications.

### Strengths
- The paper formalizes a runtime-based approach to codebase understanding, which is underexplored in prior benchmarks.
- The three metrics (Execution Fidelity Rate, Line Execution Rate, Line Existence Rate) together provide a comprehensive evaluation of the GISTIFY task, offering a well-rounded assessment of the model’s effectiveness.
- Experiments cover multiple models and frameworks, providing comparative insights.

### Weaknesses
- The Execution Fidelity can only be signed to 1 or 0, which lacks granularity in capturing partially correct implementations.
- While error types are categorized (Import Error, Missing Test, etc.), deeper causal insights (why import errors dominate, or which repository structures cause more failures) would make the analysis stronger.
- Insufficient documentation of evaluation details. Critical hyperparameters, such as temperature settings and the number of repeated runs per task, are not specified.
- The evaluation uses only 25 test instances per codebase, which is insufficient for drawing robust conclusions.
- The benchmark exclusively evaluates Python codebases, ignoring other programming languages (e.g., Java). This narrow focus limits insights into cross-language performance and fails to address challenges unique to diverse paradigms.

### Questions
- Could partial credit be integrated (e.g., per-test-case success rate) to provide smoother performance gradients?
- According to the definitions of Line Execution Rate, this metric is 100% when every line of the gistified file is executed. However, this may misalign with practical minimality when syntactically necessary code (e.g., unused branches in conditional statements) is retained for correctness but not executed under test inputs. For example, an if-else block where only one branch runs would penalize the metric score despite the code being functionally minimal.
- Could the line existence rate penalize valid simplifications?

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
This paper introduces **GISTIFY**, a new repository-level evaluation benchmark where LLM agents must compress a command's behavior (e.g., a specific pytest) into a single, self-contained file. The authors define clear task requirements, propose metrics for execution fidelity and minimality, and benchmark four leading LLMs across three agent frameworks on five SWE-Bench repositories plus debug-gym. The evaluation includes error analysis, ablations on prompting strategies and tools, and a difficulty analysis based on execution coverage.

### Strengths
- Unlike prior repository-level benchmarks (SWE-Bench, RepoBench), GISTIFY uniquely requires runtime-guided single-file reconstruction. This frames an interesting and practical challenge.
- The paper tests four models across three frameworks, includes thoughtful ablations on prompting strategies and tool usage, and extracts concrete insights about agent behavior and design trade-offs.
- The task is well-explained, metrics are intuitive, and the error categorization (import errors, missing functions, etc.) provides useful diagnostic insights.
- The finding that state-of-the-art agents still struggle on long execution traces is noteworthy and motivates future work on execution-aware reasoning.

### Weaknesses
- The main experiments cover only 25 tests per repository across 5 Python projects. It's unclear how well this generalizes to larger codebases, other languages, or more diverse domains. The repository selection criteria and expansion plans are not well explained.
- The line execution/existence rates are intuitive but not validated. Without human reference gists or user studies, it's hard to know if these metrics truly measure "minimal and faithful" distillations or just correlate with them.

### Questions
1. Do you have plans to expand the benchmark beyond 25 tests per repo? What's the timeline, and will the expanded sets be publicly available?
2. Have you considered validating the minimality metrics against human judgments? Even a small pilot study would strengthen confidence in your metrics.

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
This paper introduces GISTIFY, a benchmark for evaluating codebase-level understanding in large language models. Given a repository and a command, the model is required to generate a minimal, self-contained file that reproduces the same runtime output as the original codebase. The benchmark is automatically constructed from real GitHub projects and paired with execution traces that capture ground-truth runtime behavior.Evaluation metrics include execution fidelity, line existence rate, and line execution rate. Experiments with multiple LLMs and agent frameworks show that while models can handle simple single-module tasks, they struggle on multi-file or long-trace scenarios.

### Strengths
This work introduces a new approach to evaluate models' code repair ability from a more comprehensive and systematic perspective. The benchmark is built from real execution traces, collected via an automated pipeline, and the evaluation criteria are well thought out.

### Weaknesses
1. Since code execution inevitably produces different errors due to various factors, it is unlikely that data filtering alone can fully prevent them. How should such errors be handled during evaluating? Could there be missing special filtering or handling mechanisms?
2. The benchmark is quite novel, but the overall size might be somewhat limited.

### Questions
See weakness please

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
GISTIFY tests whether LLMs truly understand whole codebases—not just snippets—by requiring them to generate a single minimal Python file that exactly replicates a command’s runtime behavior (e.g., pytest output). Success demands four criteria: files must be self-contained (inline dependencies), execution-faithful (identical output), minimal (only essential code), and grounded (zero hallucinations, strictly sourced). Evaluating top models (GPT-5, Claude-4) via agent frameworks like SWE-Agent, the authors introduce three practical metrics: binary execution fidelity, Line Execution Rate (minimality), and Line Existence Rate (source fidelity). Results reveal even leading models struggle significantly with complex tasks, especially long dependency chains. Error analysis shows failures stem from incorrect inlining or over-pruning critical code. This benchmark exposes core weaknesses in current LLMs' cross-file reasoning.

### Strengths
1. This paper addresses a critical gap in current LLM code evaluation, which leans heavily on static analysis. Requiring full runtime behavior replication forces models to grasp entire codebase execution flows—crucial as AI coding assistants scale toward industrial systems handling multi-file complexity.
2. With just a repository and entrypoint command, GISTIFY generates evaluation cases automatically for any codebase without relying on manually labeled data such as GitHub issues. This sidesteps annotation biases and enables direct cross-project comparisons.

### Weaknesses
1. The main results in Table 1 are based on 25 tests from each of 5 repositories. While the analysis section uses a larger set of 50 tests from pylint, the primary claims about model performance would be more robust if supported by a larger and more diverse set of test instances in the main experiment.

### Questions
1. How does the normalization and block-matching algorithm handle semantically equivalent but syntactically different code changes, such as rewriting a for-loop as a list comprehension or rearranging independent class methods? Could you include the relevant pseudocode or a more detailed description of the matching logic?
2. For handling non-deterministic program outputs (e.g., timestamps, memory addresses, or unordered dictionary iteration in legacy Python versions), what normalization methodology is applied to stdout/stderr prior to comparison?
3. In this experiment, the static model receives a strong oracle. The observed conclusion indicates dynamic file selection achieves superior performance. It is hypothesized that the static model's suboptimal results may stem from context overload. Could implementing alternative prompting strategies—such as single-turn multi-step Chain-of-Thought (CoT) that directs initial identification of essential code segments before file synthesis—effectively mitigate this performance gap?

### Soundness
3

### Presentation
3

### Contribution
4
