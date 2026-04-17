# DexBench: Benchmarking LLMs for Personalized Decision Making in Diabetes Management

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
We present DexBench, the first benchmark designed to evaluate large language model (LLM) performance across real-world decision-making tasks faced by individuals managing diabetes in their daily lives. Unlike prior health benchmarks that are either generic, clinician-facing or focused on clinical tasks (e.g., diagnosis, triage), DexBench introduces a comprehensive evaluation framework tailored to the unique challenges of prototyping patient-facing AI solutions in diabetes, glucose management, metabolic health and related domains. Our benchmark encompasses 7 distinct task categories, reflecting the breadth of real-world questions individuals with diabetes ask, including basic glucose interpretation, educational queries, behavioral associations, advanced decision making and long term planning. Towards this end, we compile a rich dataset comprising one month of time-series data encompassing glucose traces and metrics from continuous glucose monitors (CGMs) and behavioral logs (e.g., eating and activity patterns) from 15,000 individuals across three different diabetes populations (type 1, type 2, pre-diabetes/general health and wellness). Using this data, we generate a total of 360,600 personalized, contextual questions across the 7 tasks. We evaluate model performance on these tasks across 5 metrics: accuracy, groundedness, safety, clarity and actionability. Our analysis of 8 recent LLMs reveals substantial variability across tasks and metrics; no single model consistently outperforms others across all dimensions. By establishing this benchmark, we aim to advance the reliability, safety, effectiveness and practical utility of AI solutions in diabetes care.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DexBench, a benchmark designed to evaluate large language models on real-world, patient-facing diabetes management tasks. Built from data of 15,000 individuals across type 1, type 2, and prediabetes populations, it generates 360,600 contextualized questions covering seven representative tasks derived from CGM and behavioral data. The authors also design a five-dimensional evaluation framework (accuracy, groundedness, safety, clarity, and actionability) and assess eight diverse LLMs. The study offers a framework for assessment of LLM performance in diabetes-related contexts.

### Strengths
1. The study constructs a large, diverse, and multimodal patient-facing diabetes dataset, covering comprehensive real-world management tasks and complementing existing benchmark efforts. 

2. It systematically evaluates multiple LLMs, discusses their strengths and limitations, and provides some useful insights for improving diabetes-specific model capabilities.

### Weaknesses
1. As acknowledged by the authors, the dataset lacks detailed demographic information (e.g., age) and omits important variables such as insulin use and medication data. In addition, it relies heavily on wearable and self-reported inputs, which may be sparse, noisy, and limit the robustness and representativeness of the benchmark.

2. The evaluation is primarily based on model-generated responses, which may introduce bias. Incorporating domain-specific diabetes knowledge or clinical expertise into the evaluation process could improve the rigor and reliability of model assessment.

### Questions
1. Could the authors clarify the dataset source in detail—specifically, how the real-world data were collected and whether the reported cohort of 15,000 individuals includes any synthetically generated data? ps. The prediabetes/health and wellness group may not be a true diabetes population; clarification in terminology would improve precision.

2. Although the paper notes missing demographic (e.g., age, sex) and treatment variables (insulin, medications), , the benchmark distinguishes adults and adolescents in Task 2. Could the authors provide basic dataset statistics, such as the proportion of adults vs. adolescents and the completeness rate of self-logged data? This information would help assess the representativeness and generalizability of the bench.

3. Lines 190-191 mention that a human expert manually confirms the quality of generated questions. Could the authors elaborate on the review process and criteria used to judge question quality?

4. The related work section states that previous diabetes benchmarks are clinician-facing, but some prior efforts are not strictly clinician-oriented. The authors may consider clarifying or citing those examples for completeness.

5. Given that most evaluations rely on model-based scoring, how do the authors ensure scoring fairness, especially when the scoring model itself may not be the strongest performer? What were the considerations in selecting the scoring model? In addition, for binary criteria, could a graded or probabilistic scoring scheme provide a more nuanced assessment?

6. Lines 713-715 describe accuracy as “agreement with ground-truth values within ±2 mg/dL, with no calculation errors permitted.” Could the authors clarify this definition? There seems to be a potential inconsistency. Besides, not all metrics (e.g., TIR) are expressed in mg/dL.

7. Regarding the clarity metric (Flesch-Kincaid Grade Level), could the authors explain how this measure was implemented and validated for health communication contexts?

8. The reported benchmark scores are relatively high. Does this suggest that the current benchmark may not pose sufficient challenge to newer models? How might future updates maintain or enhance its discriminative ability?

9. The paper mentions that it plans to extend DexBench to other health domains. Could the authors elaborate on how the framework could be adapted to chronic conditions such as hypertension or obesity? Have any preliminary steps been taken in that direction?

### Soundness
3

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
4

### Summary
The paper introduces DexBench, a comprehensive benchmark designed to evaluate large language models (LLMs) on patient-facing diabetes management tasks. It covers seven real-world decision-making tasks, uses data from 15,000 individuals across three diabetes populations, and evaluates models on five metrics: accuracy, groundedness, safety, clarity, and actionability.

### Strengths
The paper has several strong aspects as below
- DexBench addresses a critical gap in healthcare AI by focusing on patient-facing tasks, which are often overlooked in existing benchmarks. Crucially, the use of a large-scale dataset from real-world users provides valuable context for evaluating LLM performance in realistic scenarios.
- The evaluation framework covers multiple aspects of model performance, including accuracy, safety, and actionability, ensuring a holistic assessment.
- The paper provides extensive results comparing eight LLMs across various tasks, highlighting strengths and weaknesses for each model.

### Weaknesses
There are some key drawbacks in the dataset as below

- The dataset may lack critical information such as demographic details and specific medical conditions like insulin use, which could affect task performance.
- The reliance on synthetic data (e.g., GlucoSynth) for certain tasks raises concerns about the benchmark's real-world applicability.
- Advanced tasks like advanced reasoning and planning may require more sophisticated models to handle complex logic and context.

### Questions
- How was the dataset ensured to be diverse enough across diabetes populations, age groups, and other demographic factors?
- What steps were taken to ensure synthetic data does not skew results or limit the benchmark's generalizability?
- Why did certain diabetes cohorts (e.g., T2D) perform better on average? Are there specific reasons tied to task requirements or model capabilities?
- How was hallucination detected and addressed in the evaluation process, especially for complex tasks requiring accurate medical reasoning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a benchmark for LLMs on decision-support tasks for managing diabetes. They discuss 7 task categories: glucose math, education, simple and advanced reasoning, decision making, planning, and triage. It is noted that the performance results across all tasks are relatively high, with GPT-5-mini achieving 99.7% in safety and 98.3% in actionability.

### Strengths
- With the increasing use of LLMs in diabetes management, the problem domain of the paper is interesting. If done properly, this addresses a gap in AI benchmarking for healthcare and decision making tasks.
- The figures and detailed tasks (in appendices) are clear.
- Investigates 7 task categories in multiple criteria (accuracy, groundedness, safety, clarity, actionability)

### Weaknesses
- Limited information on human experts, annotation processes, and task coverage
- There are many confounding factors, such as experimental settings, token limits, and handling of errors/faults... which were not appropriately addressed
- Limited information on the data, even though the source code is downloadable
- Despite impressive numbers in the dataset, it is important for authors to prompt their motivations, aims and novelty of this type of research. It is relatively easy for researchers to create such a dataset, using curated data, with LLM-generated questions, and the use LLM evaluations (potential circularity?). Real impacts are far more important than getting papers accepted (even elsewhere).

### Questions
- How can the research rigour of this paper be addressed? For example, the formulation of tasks and criteria is simply developed by "human experts" - there are many methodological approaches to this.
- How can the research data be validated? What are the details on human experts and annotation processes?
- Any researcher can come up with a list of tasks, criteria, and generated questions using LLMs. What are the real values and impacts of this research work?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DexBench, a large-scale benchmark for evaluating large language models (LLMs) on context-aware, personalized decision-making tasks. Unlike prior health-AI benchmarks focused on clinician reasoning, DexBench targets user-facing reasoning that supports daily self-management of chronic conditions, exemplified through diabetes care. The benchmark defines seven task categories, from Glucose Math and Education to Planning and Alert/Triage, spanning quantitative reasoning to long-term guidance. Using 30-day CGM and behavioral data (sleep, meals, exercise) from 15,000 individuals, DexBench generates 360,600 personalized questions.
Outputs are evaluated across five binary metrics—accuracy, groundedness, safety, clarity, and actionability—using a structured pipeline that combines automated Gemini grading and expert review. Evaluating eight LLMs (Gemini, GPT-5, DeepSeek, Qwen, Llama, MedGemma), the study finds proprietary models strongest in safety and factual accuracy, while open-source models lag in groundedness and readability. Analyses of latency, modality, and thinking-budget reveal trade-offs between reasoning depth, efficiency, and safety in real-world applications.

### Strengths
- DexBench contributes not merely a dataset but a generalizable evaluation framework for multimodal reasoning and safety assessment. Its five-axis scoring scheme (accuracy/groundedness/safety/clarity/actionability) and multi-task taxonomy formalize a principled way to measure contextual reasoning quality—an under-explored but central topic in current ICLR research.
- The seven tasks form a structured reasoning hierarchy, from immediate quantitative interpretation to sequential planning and triage. This mirrors the cognitive spectrum of everyday decision-making and extends evaluation beyond static QA to dynamic, data-grounded reasoning.
- With 15,000 users, three diabetes cohorts, and longitudinal data spanning 30 days, DexBench represents one of the most comprehensive real-world datasets for patient-facing reasoning—far larger and richer than previous health benchmarks such as Diabetica or MedGPTEval.
- The authors detail transparent stages, data curation, question generation, automatic and human validation, and ethical safeguards (Appendix A.1), providing reproducibility and regulatory awareness rarely seen in benchmark design.
- The latency, modality, and “thinking-budget” experiments yield general insights about compute–accuracy–safety trade-offs in LLM reasoning. These findings speak directly to the broader LLM community, not just health applications.

### Weaknesses
- Use of synthetic glucose traces (GlucoSynth) and LLM-generated questions introduces potential bias. The paper lacks quantitative evidence of expert agreement (e.g., Cohen’s κ), which limits confidence in the reliability of the “accuracy” and “groundedness” metrics.
- The same model family (Gemini 2.5 Flash) is used for both question generation and grading, raising the possibility of alignment leakage. Independent or cross-model evaluation would strengthen fairness.
- Behavioral features are limited to sleep, meals, and exercise; omitting insulin dosage, medication adherence, or stress reduces ecological realism for decision support.
- A 0/1 metric oversimplifies nuanced criteria such as clarity and actionability. Continuous or rubric-based scoring, as in HealthBench, would better reflect performance differences.
- The related-work section should acknowledge concurrent benchmarks (MedGUIDE 2025, HELM 2.0, etc.) and clarify that DexBench’s novelty lies in its temporal, behavior-linked reasoning framework rather than being the first diabetes-oriented benchmark.
- Although Table 4 lists typical errors, the paper could offer a causal taxonomy (e.g., temporal misalignment, hallucination under uncertainty) to better guide model improvement.

### Questions
- How does DexBench’s multi-axis evaluation differ from LLM-CGM and MedGUIDE in modeling patient-facing reasoning?
- Was inter-rater reliability (e.g., κ-score) computed for expert validations?
- Could grading bias from Gemini 2.5 Flash be tested through cross-model adjudication?
- Are multimodal modalities (voice, sensor streams, graphical plots) planned for future releases?
- How well does the evaluation framework generalize to other continuous-monitoring domains (e.g., hypertension, sleep, fitness)?

### Soundness
3

### Presentation
3

### Contribution
4
