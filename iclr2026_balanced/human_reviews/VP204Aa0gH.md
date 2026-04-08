## Human Reviewer 1

### Summary
The paper introduces OmniCode, a new benchmark designed to evaluate LLM-powered software development agents beyond the narrow scope of existing benchmarks like HumanEval and SWE-Bench. The authors argue that real-world software engineering involves a more diverse set of tasks. OmniCode addresses this gap by providing 1,794 tasks across three programming languages (Python, Java, C++) and four key task categories: bug fixing, test generation, responding to code reviews, and fixing style violations.

However, the paper's writing looks incomplete. The appendix section still has placeholder text instead of actual content. I also didn't see any specific details in the supplementary materials.

### Strengths
- The paper addresses a clear and widely recognized gap in the field. Current benchmarks focus heavily on bug fixing or single-function generation. OmniCode provides a much-needed holistic benchmark that covers a wider, more realistic spectrum of the software development lifecycle, including testing, code review, and style adherence

- The paper provides a strong set of baseline experiments. The results are insightful, such as the clear identification of Test Generation as a major weakness for current agents and the nuanced finding that code reviews help on complex Java/C++ tasks but may hurt performance on simpler Python tasks.

-

### Weaknesses
-  The paper's writing looks incomplete. The appendix section still has placeholder text instead of actual content. I also didn't see any specific details in the supplementary materials.

- The robustness of the "Test Generation" and "Code Review" tasks hinges on the quality of the synthetic "bad patches" and "review reports." The paper details how these are generated (e.g., using weaker agents or LLM-based perturbation), but a more in-depth qualitative analysis of their diversity and realism would strengthen the paper. For instance, how do we know the "bad patches" cover a truly diverse set of realistic human errors?

- The evaluation for the "Code Style" task focuses on quantifying the reduction of linter-reported issues using a specific score. However, it is not explicitly stated whether the project's functional test suite is run after the style fix. A good style fix should not introduce functional regressions, and this would be a valuable check to include for a more robust evaluation.

### Questions
Regarding the "Code Style" task: Did the evaluation process involve running the functional test suite after an agent applied a style fix? It seems critical to verify that the agent did not introduce functional regressions while refactoring the code to resolve style violations.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 2

### Summary
The authors propose OmniCode a code agents benchmark combining instances from SWE-Bench and Multi-SWE-Bench with recently mined new instances. The data sets are enhanced by using LLMs to create new task types from existing instances. More precisely, the authors add three tasks to the standard "issue resolving" task: 1) Test generation, 2) responding to code review, and 3) code style application. For 2) and 3) an LLM is used to create bad patches which are related to the ground truth patch but do not solve the task at hand. Any test generated for the test generation task has to fail for the bad patches and pass for the ground truth patch. Bad patches are also used to create "code reviews" where an LLM is tasked to generate a review of a bad patch with knowledge about the ground truth. The task is then to "respond" to the review to fix it and arrive at the ground truth solution. In the experimental section, the authors compare both the Aider and SWE-agent scaffold with Gemini 2.5 Flash and demonstrate varying performance over the different tasks.

### Strengths
* OmniCode combines multiple important tasks over multiple languages. In particular the latter is important but oftentimes overlooked. I'd love to see the authors to expand the supported languages (e.g., with JavaScript and TypeScript instances from other benchmarks that offer verified splits). 
* The manuscript is well written and easy to understand. Visualizations clearly convey the key results and experimental findings, making it straightforward to follow the authors' analysis and conclusions.

### Weaknesses
* Bad patch generation: A bad patch is defined as one that doesn't pass the golden tests. If the golden tests are too narrow or too permissive bad patches may reflect these shortcomings and the code review task would be affected by this as well.
* Quality and solvability. LLMs are used for patch and review generation. Especially since the latter builds on top of the first LLM results, the risk for decreased quality and potential impacts on solvability multiplies (LLMs on LLMs). In general it is of limited usefulness for the community to develop benchmarks based on LLM-generated inputs. If we want the agents to support humans, inputs should come from humans. Despite being trained on human preference data, typical LLMs will not write the same patches or review messages as an SDE or VibeCoder. On a larger scale this may eventually harm the field as we measure performance of coding agents on inputs that are not from the same distribution in which we'd like to use them.

### Questions
* You are already combining instances from SWE-Bench and Multi-SWE-Bench, is there a reason why you don't add instances from another verified multi-language data set like SWE-PolyBench[1]?
* Did you verify that the test cases for a given could lead to false positives (or even false negatives)?

1. Rashid, M. S., Bock, C., Zhuang, Y., Buchholz, A., Esler, T., Valentin, S., ... & Callot, L. (2025). SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents. arXiv preprint arXiv:2504.08703.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
OmniCode aims to propose a benchmark for evaluating LLM coding agents by focusing on a range of real-world SE tasks. In particular, the authors consider bug fixing, test generation, code review, and style fixing into a single, manually validated benchmark. However, the benchmark itself suffers from some critical weaknesses: the design for some of the tasks is flawed (discussed below). While OmniCode is a valuable prototype, it currently lacks the rigor to fully assess the nuanced capabilities of coding agents across SE tasks.

### Strengths
1. With growing interest and evolving design of coding agents, the setup in OmniCode presents a comprehensive way to evaluate LLM capabilities beyond over-engineered, task-specific solutions.

2. OmniBench allows for vital cross-language analyses, providing sufficient scale and diversity across the four SE tasks.

### Weaknesses
I am most not convinced by the design for some of the tasks.
1. Code review: Real-world code reviews often contain discussions centering high-level, systemic reasoning; sometimes performance optimizations; or even code deduplication. By limiting the task to merely "generate instructions" to fix bad code, the benchmark is significantly simplified.

2. Code style: An ideal design should challenge the agent's udnerstanding of idiomatic langauge features and style choices that enhance maintainability and readability. The current setup lacks the depth to measure these aspects, and are now a simple measure of an agent's ability to apply automated linting rules.

3. Test generation is brittle: While the "bad patches" strategy is interesting, its effectiveness relies entirely on the quality, diversity, and plausibility of those incorrect patches. An ideal test should focus on testing boundary conditions or invariants.

### Questions
1. Did the authors assess consistency of agent performance across tasks, i.e., does good performance on bug fixing predict success in code review responses?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces OmniCode, a benchmark for evaluating LLM-powered software development agents across four task categories: bug fixing, test generation, code review response, and style fixing. The benchmark comprises 1,794 tasks spanning Python, Java, and C++, derived from 494 base instances. The authors evaluate popular agent scaffolding frameworks (SWE-Agent and Aider) and find significant performance gaps, particularly in test generation and C++ tasks.

The idea of extending task types beyond bug-fixing or feature development is timely and addresses a real gap in existing benchmarks. The work builds on (Multi-)SWE-Bench and introduces methods to synthetically generate multiple task types from already collected instances. However, the work has several limitations that weaken its contribution: the pipeline's manual curation process limits scalability to other languages; the experimental evaluation is restricted to a single model family (Gemini); validation of synthetically generated components (bad patches, code reviews) is insufficient; statistical analysis of results is missing; and critical reproducibility information (prompts, code, containers) is not provided in the submission.

### Strengths
1. Adding code review response, test generation, and style-fixing alongside bug repair usefully widens evaluation beyond SWE-Bench-like tasks and represents a step toward more comprehensive software engineering evaluation.
2. The multi-bad-patch protocol for test evaluation is a meaningful design choice that ensures generated tests are non-trivial.
3. Using not only SWE-Bench but also Multi-SWE-Bench enables support for popular languages beyond Python.
4. The approach of bootstrapping multiple task types from base instances is compelling and enables large-scale automation.
5. The overall structure is easy to follow.

### Weaknesses
1. The paper lacks ablations and additional analyses to validate the synthetic data generation pipeline. Since synthetic data is a major component, it may be sensitive to different prompts, introduce leakage, or create unsolvable tasks. Specifically:
  - The paper states that code reviews are "informative but do not give away the complete solutions" but provides no rigorous evaluation of review quality, realism, or usefulness.
  - No validation is provided for the bad patches, are they realistic failure modes that strong models would produce?
2. The primary evaluation is quite limited. While SWE-Agent and Aider can be considered representative examples of current SWE scaffolding approaches, evaluating only with Gemini 2.5 Flash is insufficient. Comparison with leading open-source models (Qwen3-Coder/GLM-4.x/Kimi-K2) and ideally frontier models (gpt/sonnet) would provide a more comprehensive understanding of how modern LLMs perform on the proposed benchmark.
3. SWE agents' performance can vary significantly across runs, especially on small subsets like the 44 C++ instances for test generation. Reporting confidence intervals or standard errors would provide stronger statistical evidence for the performance claims.
4. The conclusions lack detail. For example, the paper states "we observe that it struggles at C++ tasks as well as Test-Generation across languages" but provides minimal investigation into underlying causes. What specific patterns emerge in test generation failures? What types of C++ bugs are most challenging?
5. The observation that reviews help for Java/C++ but hurt for Python is interesting but remains unexplained. The speculation about "distraction" lacks empirical support and appears speculative.
6. While the appendix is referenced, critical details for reproducibility are missing from the provided excerpt, so it seems that appendix itself is missing.
  - The actual prompts used for each task type are not included
  - The bad patch generation prompt is referenced but not shown
7. [Minor] The paper contains a series of typos, e.g., guage -> gauge, incomplete sentence in Sec. 5

### Questions
1. What specific patterns emerge in test generation failures? Are agents failing to understand the bug, unable to construct proper test syntax, or missing edge cases?
2. The formula (ΔFiles + Hunks + AddedLines + RemovedLines)/10 lacks theoretical or empirical justification. Could you explain how this formula was derived? Different components seem to have vastly different scales (e.g., ΔFiles typically ranges from 1-10 while AddedLines can be in the hundreds), which means their contributions are not balanced.
3. Is there a correlation between your complexity metric and task resolution rate? Does the metric actually predict difficulty?
4. The submission does not specify whether containers, prompts, bad patches, reviews, and evaluation scripts will be publicly released. Reproducibility depends critically on these artifacts. Which components do you plan to release?
5. How many instances were rejected during manual validation and for what reasons? What is the inter-annotator agreement if multiple annotators were involved?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
2

### Confidence
4