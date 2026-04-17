# A Modular Multi-task Reasoning Framework Integrating Spatio-temporal Models and LLMs

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason’s credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems. Our code is available at: https://anonymous.4open.science/r/STReason-B0B2/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on long-contextual question answering, which requires spatio-temporal cross-task reasoning.

This paper proposes STReason, which utilizes LLM as a high-level planner and incorporates spatio-temporal models as tools/functions to deliver subtask decomposition and tool calling to handle multi-task, long-form inference and execution.

The authors introduce a new benchmark specifically for spatio-temporal tasks; both automated and human evaluations demonstrate that STReason has significant advantages over robust LLM baseline algorithms.

### Strengths
The paper proposes a new framework for spatio-temporal reasoning. Combining LLMs’ general knowledge with domain-specific small models is effective: the small models complement LLMs with specialized expertise while LLMs provide a unified framework for high-level scheduling, integrating, and analyzing information gathered by small models.

The experimental results show their superior performance, especially in the metric of Factuality score and human evaluation.

### Weaknesses
1. It would significantly strengthen the evaluation to include domain-specific baselines in spatio-temporal reasoning. Such comparisons would better contextualize the gains and clarify where the proposed approach provides unique value.

2. It’s unclear whether the function pool actually helps. Intuitively, it should, but the results in Table 3 don’t clearly demonstrate a benefit. Additionally, how to choose domain-specific functions or models for LLMs.

3. While the framework is tailored to spatio-temporal settings and has some novel elements, the overall paradigm—LLMs as tool callers invoking domain-specific small models—feels familiar. As a result, the contribution may fall a bit short of the typical acceptance bar.

4. Additionally, it is not clear how the proposed benchmark can contribute to the research field. The author states that they address long-form reasoning, but I did not see a lot of description or comparison about this.

### Questions
It appears that answers to the reasoning tasks are free-form. Could the authors provide more details about the evaluation protocol? If an additional LLM is used as the evaluator, how is the consistency and accuracy of its judgments ensured?

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
3

### Summary
The paper presents STReason, a two stage modular spatio temporal reasoning framework where an LLM emits executable ST Programs and an interpreter runs specialized modules to produce verifiable execution rationales. Experiments on a new benchmark show improvements over pure LLM baselines in constraint adherence, factuality, and forecasting accuracy; contributions include the architecture, an extensible function pool, and evaluation suite, with limitations in manual example/function curation and rare anomaly detection.

### Strengths
1. This paper proposes a new benchmark covering spatio-temporal analysis, anomaly detection, and forecasting reasoning, with unified evaluation metrics that better reflect practical task and engineering constraints.
2. It provides extensive empirical results on automatic metrics and human evaluation, and ablation studies that reveal how key design choices (e.g., in‑context examples and the Function Pool) concretely affect performance.

### Weaknesses
The baseline setup is somewhat simplistic; it lacks direct comparisons with stronger time‑series/spatio‑temporal specialized methods or advanced hybrid baselines that invoke external tools, which may lead to an overestimation of the method’s relative advantage.

The approach depends on a manually designed and maintained Function Pool (and in‑context examples), reducing automation and deployability; when handling tasks with strict constraints, repeated Function calls can cause loss of contextual information. From the second demo in the video, when asked to query weekend data, the RefineOutput function appears to forget this requirement.

### Questions
See the weakness above.

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
2

### Summary
This paper introduces STReason, a new framework that fuses LLMs with specialized spatio-temporal models to perform multi-task reasoning and execution without task-specific fine-tuning. STReason operates in the command generation stage and command execution stage. In the first stage, it generates a ST-Program based on examples, and on the second stage it executes the ST-Program sequentially using 12 modular components. The authors built a new benchmark of 150 instances from real-world datasets and evaluated the STReason on the proposed benchmark, along with baselines LLMs. Results show that the proposed STReason can improve the performance on the proposed benchmark.

### Strengths
1. The transparency and reproducibility is great. There's an open source repo and a video demo in the paper, verifying that the method is working and also ensuring it's easy to reproduce / apply the paper. 
2. The STReason proposal is novel. It combines LLM reasoning with domain-specific spatio-temporal models for multi-task inference and explanatory reasoning. 
3. The system design is simple and extensible. 
4. There are human annotators validating the accuracy of the proposed dataset.

### Weaknesses
1. The performance of the proposed approach largely depend on manually curated in-context examples. As a result, the generalization is greatly limited. 
2. The evaluation dataset is small (150 queries) and spans only among traffic and air quality, which limits the evaluation on generalization.
3. The baselines used are all general-purpose LLMs, but no comparison is made with specialized spatio-temporal neural networks. The comparison with specialized spatio-temporal neural networks will help to establish a context for the performance that LLM achieved.

### Questions
1. How are Command Interpreter Modules determined? Why do we include one module instead of another? Are those just selected by authors or there are some rational behind the selection.

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
3

### Summary
Traditional spatiotemporal models are specialized for narrow tasks (such as forecasting) and cannot handle complex reasoning, whereas general LLMs struggle to analyze ST data directly, often losing critical info when the data is converted to text. This paper presents a framework (STReason) to make LLMs into generalizable spatio-temporal reasoning systems. It uses in-context learning to decompose natural language queries into executable programs and executes these programs. A new benchmark dataset of 150 instances across three tasks (Analysis, Anomaly Detection, and Prediction & Reasoning) and an evaluation metric for long-form spatio-temporal reasoning are also proposed in the work.

### Strengths
1. Long-form reasoning for spatio-temporal data is an important and underexplored problem. 
2. ST Reason's training-free approach makes it practically advantageous.

### Weaknesses
1. Only 150 examples across three tasks. This small scale makes it difficult to believe the STBench's robustness and generalizability. And Human evaluation is conducted only on 18 questions. 
2. Only 12 pre-defined modules in the Function Pool limit generalizability, even though the paper claims "task and domain-agnostic". Function Pool requires manual intervention to add new modules.
3. Task-specific in-context examples require manual curation, which also limits generalizability.
4. The framework is benchmarked only against vanilla LLMs like (GPT-4, Deepseek-V3), they don't compare STReason with any other Spatio-temporalness aware LLMs. The experiment, as designed, demonstrates that "LLMs with ST tools" are more effective at ST tasks than "LLMs without ST tools," which is an expected outcome.

### Questions
1. How does it perform on completely new task types not seen in the three categories?
2. What is the performance if LLMs are given the ST models as tools? (No function pool or in-context examples)

### Soundness
3

### Presentation
3

### Contribution
2
