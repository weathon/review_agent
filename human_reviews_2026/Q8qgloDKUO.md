# Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large language model-based agents show promise for software engineering, but environment configuration remains a bottleneck due to heavy manual effort and scarce large-scale, high-quality datasets.
Existing benchmarks assess only end-to-end build/test success, obscuring where and why agents succeed or fail.
We introduce the Environment Configuration Diagnosis Benchmark, EnConda-bench, which provides process-level trajectory assessment of fine-grained agent capabilities during environment setup-planning, perception-driven error diagnosis, feedback-driven repair, and action to execute the final environment configuration. 
Our task instances are automatically constructed by injecting realistic README errors and are validated in Docker for scalable, high-quality evaluation.
EnConda-bench combines process-level analysis with end-to-end executability to enable capability assessments beyond aggregate success rates.
Evaluations across state-of-the-art LLMs and agent frameworks show that while agents can localize errors, they struggle to translate feedback into effective corrections, limiting end-to-end performance.
To our knowledge, EnConda-bench is the first framework to provide process-level internal capability assessment for environment configuration, offering actionable insights for improving software engineering agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark, EnConda-Bench, for evaluating LLM-based agents on environment configuration tasks in software engineering. The key innovation is providing process-level trajectory assessment across four capabilities: environment setup-planning, perception-driven error diagnosis, feedback-driven repair, and action execution. This work constructs 4,201 tasks from 323 GitHub repositories. Their evaluation across multiple LLMs and agent frameworks reveals that. while agents can localize errors reasonably well (F1 ~60%), they struggle to translate feedback into effective corrections.

### Strengths
1. This paper propose a new benchmark with process-level evaluation for environment configuration.

2. From repository selection to filtering and validation, the multi-stage dataset construction pipeline demonstrates rigor.

3. The paper clearly articulates the problem (environment configuration bottleneck), motivation (limitations of end-to-end metrics), and solution (process-level evaluation with synthetic errors).

### Weaknesses
**1. Limitations In Language Coverage**: The benchmark focuses exclusively on Python repositories. Given that environment configuration challenges exist across all programming languages, this can limit the generalizability of the evaluation and conclusions.

**2. Limitations In Synthetic Error Validity**: While the authors validate that injected errors cause failures, there's insufficient evidence that these errors represent the *distribution* of real-world configuration problems. The difficulty comparison (Table 2) shows similar mean scores among different benchmarks, but doesn't validate whether the *types* of errors match real-world distributions.

**3. Limitations In Evaluation Metrics:** Pass@1 metric doesn't account for partial progress (e.g., fixing 1 of 2 errors).

**4. Limitations In Analysis:** The specialized environment agents (e.g., Repo2Run) are evaluated but not deeply analyzed for why they perform better. There are also no ablation studies examining which agent design choices matter most.

**5. Limitations In Data Construction Transparency**: The repository selection criteria (10+ stars, 1000+ commits, 10+ issues) seem arbitrary, while no justification are provided. Also, the process of "manual checking" is mentioned but not detailed (e.g., how many annotators? what was the failure rate?).

**6. Limitations In Paper Presentation:** Some figures are hard to read. For example, all the scatter points and X-axis labels and titles are very hard to read in Figure 7, and model names are also very hard to read in Figure 5.

### Questions
My questions are following several aspects mentioned in weakness:

- How did you validate that your synthetic error distribution matches real-world configuration problems?
- How did you take partial credit into consideration, as it cannot be effectively assessed by Pass@1?
- Can you provide more detailed analysis of *why* Repo2Run performs better? Is it the dual-environment architecture, the rollback mechanism, or something else?
- Through case study, what are the common failure modes or patterns? What are their distributions? What are the implications for future improvements?
- Can you give more detailed explanation for selection, filtering, and validation criteria in dataset construction?
- Can you revise the figures to make them clearer and more readable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose a framework for analyzing process-level trajectories of LLM agents in environment configuration. This moves evaluation beyond simple pass/fail build outcomes to diagnose where and why an agent fails. The dataset is created by injecting errors (six defined categories) into valid README files followed by automatic Docker-based validation. However, this methodological strength also possibly limits the benchmark's real-world applicability (discussed more in Weaknesses section).

### Strengths
1. This work addresses a critcal yet unexplored bottleneck, moving from end-to-end pass/fail metrics to process-level trajectory assessment. This is useful to extract actionalble feedback that is useful for agent designers.

2. The decomposition of evaluation into the perception, feedback, and action provides fine-grained diagnostics beyond aggregate success metrics.

### Weaknesses
1. The six error types chosen are said to be "guided by failure modes frequently encountered in practice", but no citation or prior empirical study is provided to substantiate this taxonomy. Without grounding in developer-observed data, it is unclear whether these six types cover real-world failure modes, or simply reflect intuitive assumptions.

2. Each erroneous README is created by injecting two errors per file. This raises two concerns: (i) since both synthesis and evaluation rely on LLM behavior, the resulting benchmark may reflect model-specific phrasing or error styles, rather than human-authored configuration errors; (ii) generating only two errors per category constrains diversity, a stronger design would sample multiple error candidates and retain a stratified subset verified by humans.

3. The results lack ablation on error difficulty levels, number of injected errors, and impact of Docker environment variations. These would strengthen the claim that process-level evaluation provides deeper insight than end-to-end metrics.

### Questions
1. Were the six error types derived from any minded empirical study or developer survey of configuration failures?

2. Did you experiment with multiple variants per error type to check whether the evaluation metrics remain stable?

3. How does EnConda-Bench handle repositories with pre-existing errors or ambiguous READMEs?

### Soundness
2

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
4

### Summary
This paper proposes a new dataset to advance agentic software engineering. This dataset focuses on measuring environment configuration, which is identified as a common weakness for current agents. It consists of 100 problem instances and includes process-level annotations. The dataset is constructed using LLM-assisted filtering as well as human validation.

### Strengths
* This paper identifies an important issue in agentic coding and proposes a targeted dataset and benchmark to enable future research.
* While there are no technical contributions beyond the dataset and some of the analysis, the evaluation suite does seem to offer some key benefits over previous work, in particular when it comes to the “process-level” evaluation.
* The dataset creation procedure is thorough and well-explained; I think this will be a useful resource for the community.

### Weaknesses
* There are some notable omissions in the evaluations, such a GPT-5-Codex and Claude 4.5, both of which are considered SOTA base models for coding. Furthermore, for the coding agents, why not include Codex CLI, Gemini CLI, Jules, and Claude Code? These are specifically optimized to handle novel codebases and deal with configuration issues.
* I’m not sure that ICLR is the best venue for this work. Perhaps a dedicated dataset/benchmark track would be better suited.
* Some of the figures are too small to be useful, such as Figure 5 and Figure 6. It would be better to focus on a subset of the results in the main text (relagate others to appendix) to better highlight differences.

### Questions
Why not include Codex CLI, Gemini CLI, Jules, and Claude Code? These are specifically optimized to handle novel codebases and deal with configuration.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents EnConda-Bench, a new benchmark for evaluating llm code agents. The authors collected various repositories which includes neccessary README files which can be used as a guide to correctly setup the repository environments. The author synthetically introduce errors in the README using LLMs and create the benchmark to evaluate how llm code agents can repair the issues in the README as well as to generate a correct bash script to setup the environment. The paper evaluates several llm agent baselines and provides detailed analysis on the benchmark performance.

### Strengths
- tackles a very important and currently understudied problem of using llm agents to build software development environments
- the benchmark is constructed nicely with detailed descriptions of the pipeline and manual examination, which can be used by future work
- the authors also evaluate the benchmark already on several important baselines together with detailed analysis

### Weaknesses
Missing key evaluation category:
- As the author demonstrated in the paper and from prior work, developing a script that can successfully build an environment is non-trival.
- In the paper the authors focus on the task of repairing a README with errors
- However, we can also easily use the benchmark without README with errors to evaluate given a correct README what are the performance of generating a correct environment using LLM agents.
- I think this is an interesting scenario and can allow the authors to further compare and contrast the repair results

Unclearness of the error classification class:
- From reading the paper it is unclear how the error classification is done.
- For example how does an agent know what are the different error categories? if they do not know, then do the authors still use llm-as-judge to determine that?
- I think this is a very inefficient evaluation category as it only shows how good the LLM-agents are with identifying the error type

Minor issues:
- some of the figures are extremely small with small texts that are difficult to see (e.g., Figure 4)

### Questions
1. Did the authors evaluate the base error-free results from the benchmark?
2. How does the agent perform the error classification during evaluation?

### Soundness
3

### Presentation
3

### Contribution
3
