# HealthLoopQA: A Context-Aware Question Answering Benchmark for Interpreting Wearable Monitoring Data in Diabetes Care

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Medical wearables are transforming chronic disease management by enabling continuous physiological monitoring and personalised therapy, improving both clinical outcomes and quality of life. As these systems become integrated into daily care, interpreting long-term monitoring data is critical for patients and clinicians to understand health trends, detect safety-critical events promptly, and make informed decisions. However, this requires in-depth temporal reasoning that integrates domain knowledge, patient-specific conditions, and system-level behaviours, challenges that go beyond traditional time-series tasks. Recent advances in large language models (LLMs) offer new opportunities for context-aware reasoning and natural language interaction with medical monitoring data. Yet, existing question answering (QA) benchmarks lack the contextual richness, reasoning depth, and fault modelling required for realistic long-term medical monitoring scenarios. We introduce HealthLoopQA to bridge this gap. HealthLoopQA includes a hybrid closed-loop insulin delivery testbed that simulates realistic physiological and therapeutic monitoring data under varied patient activity schedules and 17 fault scenarios reflecting device failures and cybersecurity threats. The benchmark comprises comprehensive domain-specific QA templates for training and evaluating models, covering process mining, anomaly detection, and predictive reasoning, categorised by reasoning depth, ranging from purely descriptive statistics to causal and inferential reasoning. Each QA pair includes both a numerical answer and a textual rationale, enabling assessment of quantitative accuracy and reasoning fidelity. We evaluated prompt-based and agent-based baselines with state-of-the-art LLMs. Failure analysis reveals a broader phenomenon of \textit{In-Context Laziness}, where the model substitutes full computations with rough approximations and confident narrative justifications, highlighting the limitations of current LLMs for structured long-term time-series reasoning. HealthLoopQA aims to facilitate the development of in-depth and trustworthy time-series understanding in AI systems for digital health.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces HealthLoopQA, a hybrid closed-loop insulin delivery testbed and associated questions to evaluate frontier language models on their ability to synthesize holistic patient information in tandem with time-series data for insulin monitoring. This work contributes 150 question templates covering core medical time-series tasks including process mining, anomaly detection, and prediction as well as a closed loop simulation test bed associated with precise answers and reasoning rationales. Finally, they demonstrate current LLM performance and perform an error analysis over the different types of mistakes LLMs make for this novel task.

### Strengths
This paper's strengths lie in the under-explored domain application as well as the fine-grained error analysis performed. The authors do a good job with comprehensive related work, explaining the gap that this benchmark aims to fill as: "existing benchmarks lack domain-specific and fault-aware designs, while current CGM methods are limited in contextual integration and clinical-level reasoning." Further, the example of in context laziness and the associated ablation are interesting results, showing that even in settings (such as shorter time frames) where models may perform at one capacity, a practitioner cannot assume generalization over longer contexts.

### Weaknesses
The work generally struggles from weak baselines, poor formatting, and lack of clarity in both text and structure. Further, it is not clear how the findings of this work generalize to other applications, both within healthcare and to the general machine learning community at large. The results are largely empirical and the scope is relatively small, with only one model being evaluated using their proposed system. The work mentions that it evaluates GPT-5, but it doesn't clearly say that GPT-5 is the model that they evaluated in the discussion/results nor in Table 1. It is not clear where the data comes from by which the questions are generated nor which model is generating the associated questions for the benchmark. Further, there is no human baseline regarding either the capacity of people to do this task nor the realism of the synthetic question answer pairs. Throughout the paper, the wrong type of citation is used  (in-text when should be using parenthetical citations) harms the ability to parse the text. In summary, more clarity regarding the process to generate the synthetic questions along with a more robust baseline effort over the generated dataset to holistically evaluate multiple LLMs on this novel task are required. Once more models are evaluated, more discussion/experiments regarding failure modes and potential mitigation strategies would improve the significance of this work.

 Specific points of confusion are listed below in Questions.

### Questions
For in-context laziness, did you try any methods to mitigate this behavior such as prompting or explicit rules/rubrics? If you were to evaluate multiple models, would this behavior hold?

What are "cyber-physical threats" (line 53)?

How are you converting time series data to text?

Line 157 missing citation

"A fine-grained failure analysis across all questions revealed that the model failed most of the questions"—> This is stated in line 328 and it is not clear to me what this means or what the significance of this is.

What is a test bed vs an evaluation/benchmark? You use both terms but the differences don't seem distinct.

Interesting breakdown of reasoning requirements but what is the process of mapping Table 2 to your generated questions. How do you come up with those “instructions” for model input?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces HealthLoopQA, a question-answering benchmark for evaluating AI models on the interpretation of long-term wearable monitoring data in diabetes care. The authors identify critical gaps in existing benchmarks, namely the lack of therapeutic/activity context, fault modeling, and deep reasoning tasks. To address this, they construct a benchmark based on a closed-loop simulation testbed that generates realistic, 30-day continuous glucose monitoring data for virtual patients, complete with varied activity schedules and 17 types of systemic faults. Questions are systematically designed using a 2D taxonomy of task type and reasoning depth. The authors evaluate LLMs and find their performance to be significantly lacking. Crucially, they identify and analyze a novel failure mode termed "In-Context Laziness," where models avoid precise computation on long sequences, opting for heuristics and plausible-sounding assumptions. An ablation study on sequence length further supports this finding.

### Strengths
-  High-Impact Problem: The paper addresses a critical and practical problem in digital health: making sense of the overwhelming data streams from modern medical wearables like AID systems. Developing robust AI assistants for this task has enormous potential for improving patient outcomes and quality of life. 

-  Comprehensive Benchmark Design: The use of a closed-loop simulator that incorporates rich context (meals, exercise) and realistic fault models sets this benchmark far apart from its predecessors. This fault-awareness is crucial for evaluating the safety and reliability of models in a medical context.

-  Insightful Failure Analysis: The paper's main scientific contribution is the identification of the "In-Context Laziness" phenomenon. This observation captures a subtle yet critical failure mode of current LLMs when faced with long, structured inputs requiring procedural reasoning. It's a form of "reasoning hallucination" where the model simulates the process of calculation without actually performing it.

### Weaknesses
-  Limited Baseline Evaluation: The paper evaluates a "prompt-based baseline with state-of-the-art pretrained LLMs" but is vague on the specifics. A benchmark paper's strength is amplified by showing how different classes of models perform. The evaluation would be much stronger if it included:
    * A comparison across several available frontier models (e.g., GPT, Claude, Gemini, Qwen).
    * An evaluation of more sophisticated methods beyond zero-shot prompting. For instance, a tool-augmented LLM that can offload calculations to a Python interpreter seems like a natural solution to "Reluctance to Calculate."
    * Results from fine-tuning a smaller, open-source model on a subset of the benchmark data.


- Simulation vs. Reality Gap: While using a simulator is a necessary and strong design choice for control and safety, the paper could discuss the simulation-to-reality gap more explicitly. A brief discussion on how the complexity of real-world, noisy patient data might present additional challenges not captured in the simulation would strengthen the paper's positioning.

### Questions
- The concept of "In-Context Laziness" is fascinating. Do you have any hypotheses about its root cause within the model architecture? Could it be an artifact of attention mechanisms' limitations over long sequences, the nature of pre-training data, or a side effect of RLHF that rewards plausible-sounding answers over computationally intensive, correct ones?
- Table 1 reports the number of results (N) as 900 for PM, 1182 for AD, and 200 for PD. How does this map to the 150 question templates and 20 virtual patients? Is it simply templates x patients, or is there a more complex generation process?

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
4

### Summary
The paper introduces HealthLoopQA as a novel benchmark for evaluating large language models' (LLMs') ability to interpret long-term continuous glucose monitoring (CGM) data in the context of diabetes management. The authors address an important gap in existing QA benchmarks by incorporating domain-specific, fault-aware, and context-rich evaluations. Their approach includes a hybrid closed-loop insulin delivery testbed that simulates realistic physiological data with 17 predefined fault scenarios.

### Strengths
Some of the main strengths are:

- The authors effectively highlight the limitations of current QA benchmarks, such as lack of context, insufficient fault modeling, and limited reasoning depth. HealthLoopQA fills these gaps by incorporating therapeutic context, patient activity patterns, and device faults. Further, they also propose a systematic framework for assessing LLMs' capabilities, including quantitative metrics (MAE, SMAPE, classification accuracy) and qualitative analysis of reasoning patterns.
- Crucially, the use of a closed-loop simulation testbed with diverse scenarios and 17 fault types adds realism and relevance to the benchmark.
-  The inclusion of questions requiring temporal, causal, and comparative reasoning aligns well with the complexity of medical time-series data interpretation.

### Weaknesses
While the proposed dataset is of great interest, some of the main drawbacks are as follows:

- While the dataset is valuable for controlled experiments, it may not fully capture the diversity and unpredictability of real-world CGM data.
- The paper identifies "In-Context Laziness" as a significant issue where models prefer heuristic approximations over precise computations, especially with longer sequences. Building on the aforementioned weakness it would be beneficial if more diverse tasks are added to precisely rate LLMs' tendency to prefer heuristics
- The evaluation primarily uses in-house baselines rather than comparisons against other pre-trained models or established benchmarks.

### Questions
Some of the avenues for improvement are:

-  consider including more diverse and representative real-world CGM data to better align the benchmark with actual clinical usage.
- Compare the performance of HealthLoopQA against other state-of-the-art LLMs (e.g., GPT-4, PaLM) to provide a broader perspective on model limitations.
- Building upon the identified "In-Context Laziness" phenomenon by developing targeted interventions or architecture modifications to improve model reasoning accuracy.

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
5

### Summary
The paper introduces a time-series reasoning benchmark designed for wearable diabetes care applications. The benchmark simulates single-turn question-answering (QA) interactions between a patient and an intelligent healthcare assistant, grounded in long-term continuous monitoring data. Each task instance includes (i) physiological time-series data, (ii) contextual information such as insulin delivery records, patient profile, and activity logs, (iii) a natural language question about the monitoring data, and (iv) a reasoning instruction specifying the evidence-gathering process. The model (based on an LLM) is expected to output a precise answer that may be numerical, categorical, or temporal in nature. The benchmark comprises 150 question templates spanning key medical time-series reasoning tasks, including process mining, anomaly detection, and prediction. To safely evaluate model robustness and system behavior, the authors generate an automatic insulin delivery systems monitoring dataset using a closed-loop in-silico testbed with 20 virtual Type 1 Diabetes patients, allowing for controlled simulation of device malfunctions and cyberattack scenarios without risk to real patients.

### Strengths
The paper is well-written, well-motivated, timely and the task is designed very well. I really like the design of the benchmark based on 3 key reasoning abilities (description, memory, and pattern-level), the break down of the reasoning process into atomic reasoning units (e.g, quantitative calculation, etc.), and the subsequent analysis of failure modes of LLM-based models.

### Weaknesses
- Missing details and reproducibility: I believe that the authors can significantly improve the reproducibility of their paper by including important details. I listed some questions in the following section. 
- Baselines: The paper lacks baselines, which makes it hard to judge how good the proposed LLM-based model actually is. This is especially true for regression-based tasks, given that sMAPE / MAE are unbounded metrics. 
- Reluctance to calculate: This is a well-known results in LLMs, which struggle with numerical calculations. I would recommend providing citations to prior work which has shown similar results. 
- I would recommend adding descriptive captions to the figures and tables, which capture the story that they are communicating. Also, the current captions do not fully describe the different things shown in the tables, for example, it is unclear what $N$ (valid number of results) means.

### Minor
- Missing citations to prior work on time series question answering: There are a few benchmarks which have formalized the task of time series reasoning and question answering [1], which the literature work could be built on.
- The paper would benefit from proof-reading to correct minor grammatical mistakes and typos, for e.g., Srivastava et al. (2023); ?.

### References
1. Cai, Yifu, et al. "Timeseriesexam: A time series understanding exam." arXiv preprint arXiv:2410.14752 (2024).

### Questions
> We design 150 question templates covering core medical time-series tasks—process mining, anomaly detection, and prediction
- It is unclear how the question a generated from such a template. Can you provide an example of a template for each kind of time series task> 

> In addition, each question is paired with a reasoning rationale that articulates the step-by-step logic behind the answer derivation
- How are these reasoning rationales generated? Are they generated automatically?

- What is the model that is used to derive results for Table 1? I would recommend adding a description of the LLM-based system, along with its hyper-parameters, and all other details necessary to reproduce the results.

### Soundness
3

### Presentation
3

### Contribution
3
