# ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
Evaluating progress in large language models (LLMs) is often constrained by the challenge of verifying responses, limiting assessments to tasks like mathematics, programming, and short-form question-answering. However, many real-world applications require evaluating LLMs in processing professional documents, synthesizing information, and generating comprehensive reports in response to user queries. We introduce ProfBench: a set of over 7000 response-criterion pairs as evaluated by human-experts with professional knowledge across Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA.  We build robust and affordable LLM-Judges to evaluate ProfBench rubrics, by mitigating self-enhancement bias and reducing the cost of evaluation by 2-3 orders of magnitude, to make it fair and accessible to the broader community. Our findings reveal that ProfBench poses significant challenges even for state-of-the-art LLMs, with top-performing models like GPT-5-high achieving only 65.9\% overall performance. Furthermore, we identify notable performance disparities between proprietary and open-weight models and provide insights into the role that extended thinking plays in addressing complex, professional-domain tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ProfBench, a high-quality evaluation benchmark spanning multiple professional domains. It contains 80 tasks and more than 7,000 human-constructed response–criterion pairs, developed entirely by domain experts with PhD or MBA degrees and without any LLM assistance, ensuring authenticity and professional rigor. The authors further propose a rubric-based LLM-as-a-Judge evaluation paradigm and design three metrics to comprehensively assess assessment consistency, fairness, and efficiency. A systematic evaluation of over 40 open-source and closed-source models investigates the influence of different reasoning mechanisms, model sizes, and response lengths, and presents strategies to reduce evaluation cost and bias. Experimental results show that even the strongest closed-source model (GPT-5-high) reaches only 65.9% on this benchmark, highlighting the professional difficulty and challenge posed by the tasks.

### Strengths
In the data annotation phase, the paper adopts a rigorous expert participation mechanism. A total of 38 professionals with PhD or MBA backgrounds were recruited to design tasks and formulate scoring criteria. Multiple rounds of review and consistency verification were conducted to ensure the reliability of annotations. This process guarantees the high quality of the dataset in terms of knowledge depth and annotation accuracy.

The paper systematically compares over 40 types of mainstream models, covering dimensions such as closed-source vs. open-source, different sizes, and "thinking" settings. It also analyzes the relationships between model performance, bias, output length, and reasoning costs. The overall experiments are comprehensive, and the conclusions are credible.

During the evaluation process, the authors systematically studied the performance and cost differences of different judge models, and proposed an optimal sample allocation method and a low-cost evaluation scheme. This strategy can significantly reduce evaluation costs while maintaining high consistency, which provides valuable insights for large-scale evaluation practices.

ProfBench covers four professional domains: Physics, Chemistry, Finance, and Consulting. With diverse task types, it can comprehensively reflect the generation and judgment capabilities of large language models in professional scenarios.

### Weaknesses
1. The paper describes the annotation process and consistency metrics in detail, but lacks qualitative demonstration of controversial samples and explanations of the adjudication mechanism. Given the subjectivity of rubric-based evaluation, it is recommended that the authors supplement several typical cases to illustrate the judgment differences among different annotators and the final adjudication process, thereby enhancing interpretability and transparency.
2. The current definition of the Bias-Index relies on a limited set of reference models, and directly subtracting the Bias-Index from the Macro-F1 score to obtain the "Overall" indicator may lead to issues of dimension inconsistency and sensitivity in multi-model scenarios. It is suggested that the authors verify the robustness of this indicator on a larger model set and supplement other fairness metrics.
3. The paper’s results show obvious variations in task difficulty across different domains, yet the authors do not explain whether task balancing or weighting was performed. When comparing across domains, comprehensive scores may be affected by the distribution of domain samples, thereby impacting overall fairness.

### Questions
1. You mentioned that the annotation process underwent multiple rounds of review and reported a relatively high consistency metric. When discrepancies arise among annotators, is there a fixed adjudication process or arbitration mechanism in place? If yes, could you supplement typical cases in the appendix to help readers understand how to handle criteria with strong subjectivity?
2. Is this metric stable when the number or type of reference models changes? Have sensitivity analyses been conducted, or has the consistency of calculating the bias-index in a larger model pool been tested?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new benchmark (ProfBench) to test LLMs on PhD/MBA-level tasks that require specialized knowledge. Unlike prior benchmarks focusing on quickly verifiable problems, ProfBench focuses on open-ended tasks like writing reports that require expertise. The paper’s contributions include: (1) establishing a multi-domain, expert-annotated rubric benchmark; (2) assessing LLMs both as report generators and as judges; (3) and proposing techniques for LLM-based grading that aims to reduce evaluation cost and bias. The experiments demonstrate that LLM judges can grade responses with reasonable agreement to human experts.

### Strengths
- A major strength is the exploration of LLMs as automated judges. Building on work in rubric-evaluation, the authors propose a framework to have LLMs determine if a given response satisfies each expert criterion. The framework aims to reduce self-enhancement bias (i.e., where LLM judges would favor responses from the same model or provider), as well as API costs.
- ProfBench benchmarks LLMs as report generators in scenarios that mirror actual professional workflows, requiring multi-step reasoning and synthesizing information from multiple reference documents.
- Ablation experiments demonstrate the importance of reference documents for model performance.

### Weaknesses
- The benchmark’s scoring relies on LLM judges, and while the authors do measure agreement with human annotators, it’s shown that the best judge isn’t near perfect (<80% Macro-F1 overall). There’re risks that LLM-judges might miss nuanced criteria fulfillment or penalize creative answers. The paper doesn’t deeply discuss failure modes of the LLM-judge.
- ProfBench covers only four domains, leaving out several important domains of professional reasoning — notably, the legal, health, and engineering domains. 
- Relatedly, the paper omits some relevant benchmarks involving professional tasks in the legal and engineering domains, which are not covered by the benchmark:

      Chilton, A., Guha, N., Nyarko, J., Ho, D., Ré, C., Narayana, A., ... & Peters, A. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. Advances in Neural Information Processing Systems, 36.

      Zhou, X., Wang, X., He, Y., Wu, Y., Zou, R., Cheng, Y., ... & Zhao, J. (2025). EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving. arXiv preprint arXiv:2509.17677.

- The paper does not discuss any strategy for preventing test data leakage or overfitting. In contrast, some prior benchmarks like MMLU-Pro introduced hard, out-of-distribution questions to stay ahead of models.

- The paper’s evaluation doesn’t include an overall quality judgment beyond summing criteria. While rubric scoring is objective, it might not capture important aspects of professional work such (e.g., originality or creativity) that are hard to enumerate. This is a philosophical weakness of rubric-based evaluation in general, and acknowledging this potential limitation would improve the work.

### Questions
- Is there a risk of saturation with the best model scoring over 65%? Is the benchmark future-proof? Should there be a “hard” subset of prompts that are more adversarial in nature?

-  Could the authors clarify their rationale for selecting the four domains for ProfBench? For example, was the health or legal domains excluded due to difficulty in obtaining annotations or because existing benchmarks like HealthBench or LegalBench already cover this ground? If the goal is to provide a broad measure of professional knowledge, adding other professional domains seems valuable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a benchmark including response-criterion pairs to evaluate professional knowledge across multiple fields. It introduces an efficient LLM-as-judge evaluation framework that mitigates self-enhancement bias. The authors evaluate current LLM performance on both criterion fulfillment classification and response generation for these challenging tasks.

### Strengths
1. The benchmark covers multiple scientific domains and evaluates knowledge storage and complex reasoning capabilities. Its expert-designed criteria facilitate precise, granular assessment of LLM performance on challenging tasks.
2. The paper assesses a wide range of LLMs to provide comprehensive performance benchmarks. The experimental design encompasses comparisons across model accessibility (open-source and closed-source), scale, and reasoning capabilities.
3. The high-quality annotators group and reliable rubric creation pipeline guarantee the dataset quality.

### Weaknesses
1. In Section 4, the LLM-as-judge is used as a binary classifier, with performance evaluated by F1 score. The target LLM is used to identify whether the provided criterion fulfills all the requirements to check the quality of the response. However, for such complex tasks, the F1 score only captures misalignment between the LLM and human experts. It does not reveal the LLM's internal understanding of the task or identify specific weaknesses.
2. In the rubric creation process from Section 3, the criteria creation and review stages are not described in detail. Both stages appear heavily dependent on annotator judgment, and it remains unclear how each proposed criterion contributes to the granular assessment of response quality.

### Questions
1. While using LLM to judge the criterion-fulfillment, is there any way to extract more information from the LLM performance for further failure mode analysis? Reasoning models are allowed to generate some inference steps prior to binary predictions. How might these reasoning traces be utilized to analyze misalignment between LLM predictions and human annotations?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ProfBench, a rubric-guided benchmark of 80 expert-curated tasks spanning four domains, Chemistry, Physics, Finance, and Consulting, with more than 7000 human-written response-criterion pairs. Domain experts create the tasks as well as weighted task-specific rubrics and label three “seed” model responses (o3, Grok4, R1-0528) per criterion as Yes/No, forming the ground truth for calibrating an LLM judge. The quality of the judge is measured by a metric that takes into account also the bias towards the same model family. Each task has the structure of report-writing with grounding documents, and removing the documents leads to worsening performance. The best model reaches an average score of 65.9%, with lowest performance on Physics (49.3%) and best performance on Consulting (80%).

### Strengths
- The tasks are realistic, complex, with grounding documents and created by domain experts, with reviewer feedback.
- The LLM judges are evaluated in a clear way that takes also into account the bias towards the same model family.
- Separate re-annotations show high inter-annotator agreement. Moreover, the LLM judge is highly reliable, with a tiny difference with human-annotated scores.

### Weaknesses
- GPT-5 already achieves a high score on Consulting (~80%), while Physics lags at ~49%. This suggests that one domain might saturate sooner than the other.
- The set-up is text-only, even if tool use might be helpful for some tasks, e.g. calculators, spreadsheets, code etc.
- Despite current difficulty, the text-only format may offer limited room for improvement as models become more capable.
- Domain coverage is narrow, covering only two science and two business domains.

### Questions
- What was the rationale for choosing the four domains? Do you plan to add others?
- Are you planning on performing evaluations with models that can use tools that are relevant for the benchmark tasks?
- The LLM judge evaluation depends on three seed models. I wonder whether the judge evaluation can change when picking different models. Do you have any thoughts on that?

### Soundness
3

### Presentation
3

### Contribution
3
