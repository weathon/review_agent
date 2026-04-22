# SWE-Tester: Training Open-Source LLMs for Issue Reproduction in Real-World Repositories

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Software testing is crucial for ensuring the correctness and reliability of software systems. Automated generation of issue reproduction tests from natural language issue descriptions enhances developer productivity by simplifying root cause analysis, promotes test-driven development (TDD) - "test first, write code later", and can be used for improving the effectiveness of automated issue resolution systems like coding agents. Existing methods proposed for this task predominantly rely on closed-source LLMs (e.g., GPT-5, Claude Sonnet), with limited exploration of open-source models likely due to their weaker performance. To address this, we propose **SWE-Tester** - a novel pipeline for training open-source LLMs to generate issue reproduction tests. First, we curate a high-quality training dataset of **41K** instances from **2.6K** open-source GitHub repositories and use it to train LLMs of varying sizes and families. The fine-tuned models achieve absolute improvements of up to **10\%** in success rate and **21\%** in change coverage on SWT-Bench Verified. Further analysis shows consistent improvements with increased inference-time compute, more data and larger models These results highlight the effectiveness of our framework for advancing open-source LLMs in this domain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes training open source models specifically on the task of repository-level test generation. They collect a large number of issues from GitHub repositories, restore a state where relevant issue-reproducing tests are missing from a state that contains issue-reproducing tests, and finetune several open-weight LLMs on creating/editing relevant tests (no finetuning on issue localization). Their evaluation demonstrates significant improvements in test generation capabilities across a range of open-weight models.

### Strengths
- I agree with the authors that test generation is underexplored and think their approach to mirror approaches like SWE-smith into test generation makes sense. Focusing only on code editing is well motivated and admissable.
- The ablations are clearly defined and interesting
- Writing and figures are clear, legible and concise

### Weaknesses
The main weakness I see is that this work is nothing fundamentally novel. However, I think there are interesting insights about the bottleneck of open-source models (i.e., that this is the editing step) and how to generate training data for test generation.

General disclaimer: I am not an expert in finetuning/training LLMs, so I may have missed crucial details in this domain.

Small nitpicks
- Line 225 could point out that filtering out swt-bench instances is not only relevant to reduce noise but also to avoid training contamination
- Line 238 should use citep instead of citet
- Typo in Line 399 "isolated"

### Questions
Can you evaluate the related works LIBRO and Zero-Shot on the respective base models evaluated? This would allow to assess whether the trained LLMs clearly outperform the respective untrained, but scaffolded base LLMs. This would also more clearly highlight the benefits of the proposed training.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the suboptimal performance of open-source models in issue reproduction by constructing a dedicated training dataset for this task. The proposed approach improves the performance of multiple open-source models on issue reproduction. However, the proposed method lacks methodological novelty, and the experiments do not include appropriate baselines for comparison.

### Strengths
- Addresses an important problem in software engineering—**bug reproduction**—and improves LLM performance on this task through targeted training.
- Conducts training across multiple models and provides detailed analyses of experimental results.

### Weaknesses
- The data construction method and reproduction pipeline are largely adapted from well-established approaches in the issue resolution literature; the work mainly applies these existing methods to the issue reproduction task, which limits its methodological novelty for a top-tier conference like ICLR.
- Focuses solely on the “edit exactly one test file” scenario, which may hurt generalizability.
- Lacks appropriate baselines. Although few prior works explicitly target issue reproduction, many **code agents** perform issue reproduction as part of the issue resolution process; comparisons with training methods designed for such agents would make the evaluation more convincing.

### Questions
- Why are different localization pipelines used for source code and test code?
- Typo in line 250: “atleast”.

### Soundness
2

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
5

### Summary
This paper introduces **SWE-Tester**, a framework for training open-source LLMs to automatically generate issue reproduction tests from natural language issue descriptions and buggy repositories. The proposed workflow follows a two-step static pipeline: (1) code localization to retrieve relevant source and test files, and (2) code editing to modify test files using a Search/Replace format. A large dataset of 41K instances from 2.6K repositories is curated, and multiple open LLMs (Qwen2.5-Coder, Llama3.1, Gemma3) are fine-tuned. The models achieve up to +10% success rate and +21% change coverage improvements on SWT-Bench Verified. The paper provides solid empirical results and a valuable dataset contribution, but its overall framework remains agentless and relies heavily on test-time scaling rather than true agentic reasoning or autonomy.

### Strengths
- The authors evaluate multiple open models of different sizes and families, analyze scaling effects in both training data and inference-time compute, and offer detailed quantitative insights.

- The dataset of 41K issue–test pairs is well-filtered and reproducible, providing a strong foundation for open-source SWE research.

- The workflow is simple and interpretable, with carefully described steps for localization, editing, and evaluation.

- The reported gains show that open-source LLMs can meaningfully improve on real-world SWE benchmarks through fine-tuning.

### Weaknesses
- The proposed framework is purely a static two-step pipeline—there is no reasoning loop, reflection, or autonomous planning. As the community rapidly transitions toward agentic SWE systems, this direction feels inherently limited and non-scalable. It lacks the ability to generalize beyond the fixed workflow or adapt dynamically to complex issue contexts.

- The performance improvements are largely achieved through sampling multiple patches and reranking rather than stronger modeling or reasoning capabilities. This kind of test-time scaling can inflate benchmark scores but does not address the underlying challenge of autonomous issue understanding or causal reasoning in code.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a two-step static pipeline for automatically generating issue reproduction tests from natural language descriptions and buggy code. The workflow first localizes relevant source and test files, then modifies the test files using a Search/Replace format. This methodology is encapsulated in a framework named SWE-Tester, for which several open-source LLMs (Qwen2.5-Coder, Llama3.1, Gemma3) were fine-tuned on a newly curated dataset of 41K instances. The resulting models demonstrate strong performance, with up to a +10% success rate and +21% change coverage increase on SWT-Bench Verified. Despite its solid empirical results and valuable dataset contribution, the framework's overall design remains agentless, depending on test-time scaling over genuine agentic reasoning.

### Strengths
1. Significant Performance Gains: The study reports substantial performance improvements, demonstrating the significant potential of open-source LLMs to effectively address real-world software engineering benchmarks.

2. Solid & Reproducible Foundation: The research is grounded in a well-curated and reproducible dataset of 41,000 issue-test pairs, establishing a solid foundation for future studies in open-source software engineering.

3. Transparent & Simple Workflow: The paper introduces a straightforward and interpretable workflow, with each step—including localization, editing, and evaluation—being meticulously detailed.

4. Comprehensive Model Analysis: A comprehensive evaluation is conducted across a diverse set of open models of various sizes and families. This analysis yields detailed quantitative insights into the scaling effects of both training data and inference-time compute.

### Weaknesses
* Superficial Performance Gains: The reported improvements are primarily driven by a brute-force approach of sampling and reranking multiple patches, rather than by genuine advancements in the model's reasoning capabilities. This reliance on test-time scaling may inflate benchmark scores but fails to address the core challenge of autonomous issue comprehension and causal reasoning in code.

* Limited and Inflexible Architecture: The framework is fundamentally a static, two-step pipeline, devoid of any reasoning loops, reflection, or autonomous planning. This rigid design is inherently limited and non-scalable, particularly as the research community shifts towards more dynamic, agentic software engineering systems. As a result, it lacks the ability to generalize beyond its fixed workflow or adapt to the complexities of real-world issues.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
