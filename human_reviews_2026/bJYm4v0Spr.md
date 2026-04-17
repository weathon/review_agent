# LFQA-E: Carefully Benchmarking Long-form QA Evaluation

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 4

## Abstract
Long-Form Question Answering (LFQA) involves generating comprehensive, paragraph-level responses to open-ended questions, which poses a significant challenge for evaluation due to the richness of information and flexible response format. Existing LFQA-evaluation benchmarks often lack reference answers and are limited in size and topic coverage, reducing their reliability. To address this gap, we introduce LFQA-E, a well-constructed, multilingual, and reference-based benchmark designed to rigorously evaluate automatic metrics for LFQA. LFQA-E comprises 1,625 questions and 7,649 pairwise comparisons across 15 topics, drawn from diverse sources such as online queries and examination questions, thereby enabling a comprehensive assessment of evaluation metrics. We examine five categories of metrics, encompassing 17 specific methods, using LFQA-E. The results demonstrate that none of the existing automatic metrics perform comparably to human judgments, highlighting their inability to capture the dense information in long-form responses. Furthermore, we present a detailed analysis of the failure cases and the generalization capacity of these metrics, offering insights to guide the future development of LFQA evaluation methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an evaluation dataset consisting of -- 1k data with question, pairwise answer, a reference answer and a human annotated preference covering English and Chinese. The paper then conducted evaluation of different evaluation metrics' correlation with the human annotation, concluding that none of them reliably model the human preference.

Overall the paper is written in a clear manner. However I do not find it to bring substantial contribution compared to prior work in the space of evaluating evaluation metrics. Please see discussion below.

### Strengths
* The paper is written in a clear manner.
* The paper conducted comprehensive experiments with different types of metrics (reference-based, reward models, etc.)
* The annotation process is conducted clearly.

### Weaknesses
* Limited contribution: the paper aims to contribute a dataset with reference-based evaluation for long-form answers, yet there are existing resources such as Feedback-Bench from [Prometheus-2(EMNLP 2024)](https://arxiv.org/abs/2405.01535). It would be good to further clarify the new contribution this new dataset introduces. Also while the benefit of having a reference answer is understandable, it is also a somewhat unrealistic setting -- many user queries would not contain a reference answer. Also it is possible that a single reference answer does not comprehensively cover all the possible answers, especially for more open-ended questions studied in this paper.
* Annotation noise: while the paper reported a cohen's kappa of 0.77 between annotators, each question is only annotated by two annotators -- this raises the question of whether the dataset capture annotators' preference, or other aspects. On a related note, it would be good to identify what are the **aspects** that these preference capture.

### Questions
* Given that there are only two annotators, how is the gold label determined when the two annotators disagree?
* The prompt for LLM evaluation (table 22) appears to be incorrect.
* Figure 2 reports performance of different comparison setting (human v.s. human, model v.s. model) and shows that evaluation metrics perform worse on model v.s. model comparison -- what's the human performance for these settings?

### Soundness
3

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
4

### Summary
- Focuses on LFQA (Long-Form Question Answering) 
 - Existing LFQA benchmarks are small, narrow and often lack reference answers.
 - LFQA-E is a multilingual, reference-based benchmark with 1,625 questions and 7,649 pairwise comparisons across 15 diverse topics.
 - The dataset covers sources like online queries and exam questions, supporting broad metric evaluation.
 - The study assesses 17 automatic metrics across five categories using LFQA-E.
 - Findings: no current automatic metric aligns well with human judgment; all struggle to capture dense, nuanced information in long-form responses.

### Strengths
The goal of effective evaluation of long responses is clearly an important for AI/NLP.

### Weaknesses
Assuming that table 1 refers to tokens (Table 1 are tokens, not chars?), 180-200 tokens is not much really, compared to what modern models can generate. 

For your LLM-based evaluation metrics, have you tries using prompt that involves generating intermediate thoughts before conclusing a final answer? 

For reasoning models, can you please see the quality metrics as a function of the thinking budget? 

There have been various benchmarks for reward modeling (input Q, and two responses with preference label: A1 > A2). These benchmarks are essentially equivalent to what you're doing here. Can you distinguish yourself against these works and what gap your work is filling? 

Make sure your paragraph heading have a consistent style (e.g., they all end with "." at the end). 

Relatedly, this an off way to format your paragraph heading. Like it's part of the subsequent sentence?? 
> **For LM-based Evaluation Metric** We observe ...

### Questions
See the previous box.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new comprehensive evaluation benchmark for evaluating long form QA systems and models. The main contributions of this paper are as follows:
1. LFQA-E a benchmark dataset for evaluation of automatic metrics for LFQA.
2. Detailed analysis of the proposed benchmark
3. Experimental evaluation on 15 evaluation metrics to show that current LFQA evaluation metrics fail to capture core information.
4. Analysis of generalisation of different metrics across languages and settings.

### Strengths
The main strengths of the paper are as follows:
1.  The paper identifies important limitations of the existing LFQA benchmarks like lack of authorized references and limited diversity and proposes a new large scale benchmark for better LFQA evaluation.
2. The benchmark is very diverse and spans 15 topics and domains.
3. Human experts based validation to makes the benchmark trustworthy.
4. Detailed comparison of standard evaluation metrics on a variety of reasoning models using frontier models like deepseek and qwen family of models.
5. Detailed benchmark contamination analysis to ensure that the evaluation data is not memorised during pre-training.

### Weaknesses
The main weaknesses of the paper are as follows:
1. The comparisons of evaluation results using previous LFQA benchmarks using frontier models are missing.
2. The paper writing and presentation can be improved.

### Questions
1. Is there a metrics used to assess difficulty of the questions in the benchmark?
2. Is there a plan to release a multilingual version of this dataset?
3. Reddit license has some restrictions, do the authors plan to release the benchmark and setup a leaderboard?

### Soundness
3

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
3

### Summary
This paper introduces LFQA-E, a reference-based multilingual benchmark for evaluating automatic metrics on long-form question answering (LFQA). The motivation is that existing automatic metrics fall short on LFQA, and also there are gaps in metric generalization. The benchmark contains questions and pairwise comparisons across 15 topics in English and Chinese, with expert-written references and a triple-choice annotation protocol (A / B / tie). The authors evaluate 17 evaluation methods spanning static, LLM-based, reward/model-based, LRM, and trained evaluation models.

### Strengths
S1) The paper addresses an important, underexplored issue: reliable automatic evaluation for paragraph-length, information-dense LFQA answers. The motivation and relation to prior small/weak benchmarks are convincing. 

S2) The paper offers a new substantial and diverse dataset for LFQA evaluation, offering 1,625 Questions and 7.6k comparisons, multilinguality (EN/ZH), and topic breadth. Statistics and careful sourcing are provided. 

S3) The annotation recipe is thorough. Using experts, double annotation, a triple-choice rubric, and inter-annotator agreement reporting improves trust in labels. 

S4) Authors test a wide range of metrics, give per-domain breakdowns, error taxonomy, contamination checks (PPL, n-gram overlap), and also try improvement (TTRL). This breadth of evaluation is useful for diagnosing metric weaknesses.

### Weaknesses
O1) The English references use the “top-ranked ELI5 answer” as candidate references. Top reddit answers are not guaranteed to be expert or correct; relying on them (even after expert review) risks reference quality variance. Although the authors state expert review, more detail is needed: how often did experts modify reddit content, and can we get quantitative measures of reference quality across sources? 

O2) Annotators are paid $2 per question (4–6 comparisons). This low pay risks rushed judgments despite training. The paper reports κ, but does not report annotator qualification statistics, and time per item distribution (they say ~7 minutes per comparison — is that per comparison or per question?). These details matter for external confidence. 

O3) Several choices need clarification or stronger controls: prompts, temperature settings, and inclusion of references differ across evaluated models. The procedural differences can advantage/disadvantage classes of metrics. The authors report a temperature ablation, but more systematic control (same prompt templates, prompt-sensitivity analysis, ensemble-vs-scalar handling) is needed to ensure fair comparisons. 

O4) Some leading recent evaluation approaches (explainable/extractive hybrid metrics, factuality-oriented measures such as FactScore variants, or evaluation ensembles) are missing or treated superficially. It’s unclear whether the 17 chosen methods fairly represent the state of the art; the paper would benefit from explicitly listing and justifying excluded but relevant baselines.

### Questions
Please address the weaknesses O1-O4.

### Soundness
3

### Presentation
3

### Contribution
3
