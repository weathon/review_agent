# HeurekaBench: A Benchmarking Framework for AI Co-scientist

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
LLM-based reasoning models have enabled the development of agentic systems that act as co-scientists, assisting in multi-step scientific analysis. However, evaluating these systems is challenging, as it requires realistic, end-to-end research scenarios that integrate data analysis, interpretation, and the generation of new insights from the experimental data. To address this limitation, we introduce HeurekaBench, a framework to create benchmarks with *exploratory, open-ended research questions* for experimental datasets.
Each such question is grounded in a scientific study and its corresponding code repository, and is created using a semi-automated pipeline that leverages multiple LLMs to extract insights and generate candidate workflows, which are then verified against reported findings. We instantiate the framework in single-cell biology to obtain sc-HeurekaBench benchmark and use it to compare state-of-the-art single-cell agents. We further showcase the benefits of our benchmark for quantitatively analyzing current design choices in agentic systems. We find that the addition of a *critic* module can improve ill-formed responses for open-source LLM-based agents by up to 22% and close the gap with their closed-source counterparts. Overall, HeurekaBench sets a path toward rigorous, end-to-end evaluation of scientific agents, grounding benchmark construction in real scientific workflows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces HeurekaBench, a pipeline for evaluating large language model (LLM)-based co-scientist agents with open-ended data-driven questions. The pipeline extracts insights matched with code from published studies with code, generate questions and MCQs from these insights. Using this pipeline, the authors create sc-HEUREKABENCH, a benchmark with 50 open-ended questions and 50 MCQs across 41 validated insights.

Success on the benchmark involves the agent deriving conclusions from the question and data that matches the ground-truth insights. They also propose a structured evaluation scheme using an LLM-as-a-judge (G-Eval with GPT-4o) that decomposes responses into atomic scientific facts.

### Strengths
S1. Evaluating co-scientist agents is a challenging problem especially for end-to-end analyses. The idea to leverage matched insights/conclusions as a measure is a smart way to evaluate co-scientist agents. 

S2. The authors do a decent job validating/ensuring the quality of the dataset. This includes defining what a good agent output is and including a reasonable semi-automatic pipeline to validate code, reproducibility of insights, and questions.

S3. The implementation in a concrete scientific field (single-cell biology) adds a good proof of concept.

S4. The paper is well written with good details in the appendix.

### Weaknesses
W1. I struggle with the claim of HeurekaBench as a framework. If the contribution is a framework then I would expect it to be validated in more then one domain to show how this process generalizes. Given the domain expertise required to validate code, insights, and questions in different domains, it seems like the requirements for human validation would be different. This can be further affected by the availability of code and data in different fields. Furthermore, the prompts shared are also customized to the domain of single-cell biology. Thus, if someone were to apply this framework, it does not seem to be a trivial process and essentially requires coming up with a pipeline from scratch.


While I appreciate the authors validating their dataset, the InsightExtractor having 14 weakly or unrelated insights out of 30 and the CodeMatcher matching only 75% of relevant files is worrying given the high bar for quality for an evaluation dataset. At the very least, this again means significant human effort/expertise is necessary to validate and so makes it harder to view the framework as generalizeable.

W2. The paper does not report error bars or statistical significance. Do these results change given different random seeds. Given the understandably significant effort in constructing such a dataset, I worry the number of tasks (and unique publications) is too small to show meaningful/informative performance differences.

W3. Since the evaluation hinges on LLM as a judge, it would be important to know how the LLM scores compare to human expert judgement. We do not know this right now.

### Questions
What are steps to ensure that the LLM/agent is doing the correct steps to obtain the correct insights? How do we avoid a hallucinated answer that happens to be correct?

How do we ensure that the LLM has not seen the data/publications in training, especially as new models come out? How does this impact our interpretation of results.

While the approach is different, the the focus on open-ended questions is not new [1]. It would be helpful to contextualize the motivation in the related work.

[1] Gu et al. “BLADE: Benchmarking Language Model Agents for Data-Driven Science.” arXiv preprint arXiv:2408.09667 (2024).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
HEUREKABENCH is introduced as a benchmarking framework to evaluate LLM-based agents as co-scientists by unifying prior isolated subtasks and formalizing the task as triplets (D, Q, A) grounded in real experimental datasets, open-ended research questions, and scientifically validated answers. Using a multi-LLM, human-supervised pipeline over publications, datasets, and code, the authors instantiate the framework in single-cell biology as sc-HEUREKABENCH with 50 open-ended and 50 multiple-choice questions derived from validated insights. They further create sc-HEUREKABENCH-ToolUsage to assess domain-specific tool use and present experiments on the insight-construction modules and agent ablations (e.g., removing the retriever reduces correctness on TU).

### Strengths
1. Questions are grounded in real papers, data, and code, and require multi-step analysis and evidence-based reasoning rather than pure retrieval or recall—matching the “co-scientist” setting.
2. The benchmark compares multiple single-cell agents under a common setup and systematically ablates planner/critic/retriever to quantify their impact and provide design takeaways.

### Weaknesses
1. The evaluation relies on LLM-as-judge with an atomic-facts rubric (GPT-4o), and the manuscript does not report human adjudication or agreement statistics—leaving uncertainty and potential bias.
2. MCQ choices are LLM-generated; the authors acknowledge some “incorrect” options can appear scientifically plausible, so they also report precision/recall—indicating limited answer uniqueness that may complicate assessment.
3. Methodological details about the agent/tooling side remain sparse: the paper notes a retriever that pre-selects tools and that Biomni “contains more domain-specific tools and databases,” but it does not enumerate the tool inventory or selection criteria in detail.
4. Manual “minor code edits” (loading data, mapping gene IDs, renaming variables) are routinely applied to make verification run, which weakens the claim of end-to-end automation.

### Questions
1. How many tool categories are actually involved across the environment and TU tasks (beyond the named SCENIC, CellPhoneDB, CellChat, and NMF)?
2. The retriever that selects tools before planning seems necessary—does it implement a coarse-to-fine (e.g., candidate generation + re-ranking) procedure akin to RAG pipelines?
3. Since overall system results are reported only on sc-HEUREKABENCH-Lite (≤750 MB; 22/50 OEQs and 18/50 MCQs) due to cost/stability issues, how representative is this subset and how well does its distribution align with the full benchmark?
4. Figure 1(c) illustrates the question-solving stage; could the CellAgent-style planner–executor–evaluator pipeline implement this directly, or does your setup introduce additional mechanisms beyond such frameworks—for example, a retriever that pre-selects relevant tools, software, and databases before planning?
5. The pipeline produces 10 candidate insights per paper—what motivates this fixed number, and is it sufficient to capture the diversity of findings within each study?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces HeurekaBench, a framework for creating benchmarks that evaluate LLM-based agents as AI co-scientists. The framework uses a multi-LLM pipeline to extract scientific insights from published papers and their code repositories, validates these insights through code execution, and generates open-ended and multiple-choice questions. The authors instantiate this framework in single-cell biology (sc-HeurekaBench) with 50 OEQs and 50 MCQs across 41 validated insights from 13 papers. They benchmark existing single-cell biology agents and analyze design choices including planner models, critic modules, and retrievers.

### Strengths
- The paper addresses an important gap in evaluating AI agents for scientific discovery by moving beyond simple factual recall or single-step computation tasks to assess genuine exploratory, multi-step data-driven reasoning capabilities.

- The semi-automated pipeline with human verification for insight validation is well-designed, ensuring that benchmark questions are grounded in reproducible scientific findings rather than relying solely on LLM generation capabilities.

- The experimental analysis provides valuable insights into agent design choices, demonstrating that critic modules can improve open-source model performance by up to 22% and that end-critics are more effective than plan-critics for certain models.

### Weaknesses
- The benchmark's scope is limited to only 13 papers and 50 questions in single-cell biology, raising concerns about generalizability and whether this sample size is sufficient to robustly evaluate agent capabilities across the diversity of real scientific discovery scenarios.

- The evaluation relies heavily on GPT-4o as an LLM judge for OEQs, which introduces potential biases and may favor agents using similar models, yet the paper provides limited analysis of inter-rater reliability or validation against human expert judgments.

### Questions
- As mentioned above, how sensitive are the evaluation results to the choice of LLM judge, and have the authors validated the scores against human expert assessments to ensure the rubric captures scientific correctness?

- What is the distribution of question difficulty across the benchmark, and could the authors provide more analysis on which types of insights or biological phenomena are most challenging for current agents to handle correctly?

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
This paper proposes HeurekaBench, a framework for creating benchmarks to evaluate LLM-based agents acting as AI co-scientists in data-driven, open-ended scientific discovery. Rather than relying on static knowledge retrieval or narrow computational tasks, they build benchmark  based on exploratory research questions that require multi-step analysis of real experimental datasets. They provide a semi-automated pipeline that leverages multiple LLMs to extract and validate scientific insights from published studies and their code repositories, grounding the benchmark in reproducible findings. The framework generates both open-ended questions to assess exploratory analysis and multiple-choice questions for rapid evaluation. Lastly, they instantiate the framework in single-cell biology as sc-HeurekaBench and conduct empirical experiments to evaluate state-of-the-art biological agents, analyzing the impact of key agent components like planners and critics.Their main contribution is proposing a framework for creating benchmarks to evaluate LLM-based agents acting as AI co-scientists, representing a paradigm shift toward evaluating the open-ended, data-driven reasoning required for AI co-scientists .

### Strengths
This paper proposes a novel framework for building benchmarks to evaluate LLM-based agents acting as AI co-scientists, effectively addressing the critical gap in evaluating open-ended, data-driven scientific discovery agent. Moreover, the paper is of relatively high quality, providing a comprehensive and clear exposition of its semi-automated pipeline and evaluation methodology, advancing beyond narrow task-solving, and demonstrating significant practical value of a critic module for the development of autonomous scientific agents.

### Weaknesses
The experiments in this paper are insufficient, as they lack comparative analysis with other benchmark construction methodologies[^1]. Without such comparisons, the work fails to fully demonstrate its superiority. Additionally, this study does not provide experimental analysis on the role of large language models as evaluators, such as comparing their performance with expert assessments to validate the reliability of this evaluation approach.

[^1]: Erpai Luo, Jinmeng Jia, Yifan Xiong, Xiangyu Li, Xiaobo Guo, Baoqi Yu, Lei Wei, and Xuegong Zhang. Benchmarking ai scientists in omics data-driven biological research. arXiv preprint arXiv:2505.08341, 2025.

### Questions
1.  The paper demonstrates the utility of HeurekaBench for evaluating agents, but how does the benchmark construction methodology itself compare to existing approaches, such as BaisBench , in terms of efficiency, insight quality, or scientific grounding?
2.  The study employs an LLM-as-a-judge for evaluating open-ended responses. To validate this approach, what is the correlation between the LLM's ratings and those from human domain experts, and have you analyzed specific cases where they disagree?

Suggestions:

1.  Include a comparative analysis with at least one other recent benchmark construction method (e.g., BaisBench) to more concretely position HeurekaBench's advantages and limitations within the field.
2.  Strengthen the validation of the evaluation method by conducting a small-scale study comparing the LLM-judge's scores with expert human assessments.

### Soundness
2

### Presentation
3

### Contribution
2
