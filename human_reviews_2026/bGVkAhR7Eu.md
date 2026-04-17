# SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Short Answer Scoring (SAS) is a critical task in automated subjective answer grading, playing an essential role in education, standardized testing, and large-scale assessment systems. However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new dataset and evaluation framework for automated short-answer scoring (SAS). The dataset comprises short-answer questions and student answers from China’s national university entrance exam, each annotated by experts with scores and error reasons. Extensive experiments show that current LLMs struggle with consistent step-wise scoring, especially on science questions, and that providing scoring rubrics and examples improves performance.

### Strengths
- The paper addresses a gap in SAS by introducing two new evaluation metrics: step-wise scoring and error-cause annotations. While prior benchmarks only provide an only score, these new metrics allow for an evaluation of a model’s reasoning process and explanation quality.
- The SAS-Bench dataset is substantial, well-annotated, and anticipated to be open-source. It contains diverse question types (science, English, math proofs, etc.) and expert annotations. 
- The experiments include diverse comparisons (e.g., analyses of model size, reasoning approach, and training paradigm) yields important insights. For example, one insight from comparing different model sizes is that without complex prompt engineering or task-specific fine-tuning, LLMs tend to underperform smaller LMs in SAS.

### Weaknesses
- More differentiation is needed from existing SAS benchmark datasets such as the Kaggle ASAP-SAS and SemEval-2013 “Student Response Analysis” corpora datasets. The former dataset is introduced in the Related Work but not compared explicitly with SAS-Bench, and the paper associated with the latter dataset (Myroslava et al., 2013) is in the reference list but not discussed at all in the paper. The authors claim SAS-Bench is the first benchmark specifically tailored for SAS with LLMs, but they do not articulate how its novelty goes beyond combining known elements (e.g., reference answers and multidimensional scoring) in a new dataset.
- The performance tables and charts should include a measure of variance. Since LLM outputs can be stochastic, reporting single-run results without any measure of uncertainty means we can’t tell if differences between models are significant or just noise. Similarly, the authors do not report inter-annotator agreement in the form of Cohen's kappa coefficient for their expert labels.
- The SAS-Bench dataset, while diverse in subjects, is narrow in source. This raises concerns about how the benchmark’s insights would generalize to other contexts; e.g., different countries’ exams, free-form vs. structured answers, etc. The paper does not discuss whether models performing well on SAS-Bench would generalize to other curricula or languages. 
- A portion of student responses were synthetically granted by LLMs, and then annotated, to augment real answers. The paper should acknowledge that relying on LLM-synthesized data could introduce biases, for example, if LLMs recognize their own style or content in these answers.

### Questions
- Most questions in the dataset are in Chinese (except the English category). Are some of the models evaluated better suited for Chinese language questions?
- What differentiates, if anything, a “short answer” from a close-ended exact-match answer, such as those in the HLE dataset (arXiv:2501.14249v9)?
- How does the LLM-as-a-Judge framework used in the paper compare to previous works?

### Soundness
3

### Presentation
2

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
This paper introduces a new benchmark for Short Answer Scoring (SAS), called SAS-Bench. In SAS, short answer to questions seen in standardized tests are compared to a reference answer. Previous SAS benchmarks lack 2 important attributes, which this paper addresses: firstly, the assesment is not fine grained: in SAS-Bench, answers are broken down to steps using expert annotators. Secondly, each question has a predefined set of possible errors, which the expert annotators use to label the answer steps. Put together, this makes for a benchmark that allows for fine grained analysis of LLM as a judge performance for the SAS task.

### Strengths
- The paper is clearly written and well-structured.
- The figures are clean and easily understandable. 
- The evaluation is thorough; a large quantity of reasoning and non-reasoning methods are tested along varied metrics that capture both LLM and expert consistency (CCS), as well as LLM error consistency (ECS). 
- The benchmark and dataset is made up of multiple domains, with many examples: over 1,000 questions as well as over 4,000 expert annotations.

### Weaknesses
- The motivation is unclear to me: in the introduction, works like Zuang et al., 2024, Deshpande et al., 2024, and Raina et al., 2024 are cited, as works that show the shortcomings of LLMs-as-judges. As far as I can tell, none of these works deal with SAS, which results in the question: why are these works used as motivation for creating an SAS benchmark? Additionally, the gap presented in Appendix J also seems somewhat insignificant.  In general, it seems like LLMs already perform relatively well (see next point) on the benchmark, which questions the importance of SAS-Bench to the community. 
- Looking at the consistency between experts and LLMs (CCS), it seems like the top model, V3, has a score of over 74\%. This suggests that the benchmark is already nearly saturated, which puts into question the strength of the contribution.
- There seems to be some inherent ambiguity in the construction of the expert annotations. For example, line 264 mentions that experts had disagreements that were resolved in discussions to reach a consensus. It seems like the way to segment an answer into steps, score and label each step could be done in several different reasonable ways, which varies from person to person.

### Questions
1. Are there examples of biases/shortcomings that LLMs exhibit on SAS? Is it possible to quantify them?
2. Is SAS-Bench saturated, or is there still progress left to be made by LLMs on it? Can you give concrete examples of interesting cases where LLMs are wildly inconsistent with humans on SAS-Bench? 
3. Could there be differences of opinions regarding the way answers are segmented/scored/labeled from expert to expert? Other than discussions, is there a more well-defined way to resolve those differences?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new benchmark for short answer scoring/grading. The main
difference compared to existing benchmarks are that SAS-Bench contains
step-wise scoring for each answer and that errors have been categorized by
domain experts. Overall the dataset contains around 1K questions and 4K
answers. The dataset is evaluated using 16 LLMs with a focus on overall score,
step-by-step scores and error types.

### Strengths
S1: The detailed, step-by-step score annotation as well as the error type annotation is quite useful and a good addition to existing datasets.

S2: The introduced scores for overall, step-by-step and error are sensible and the evaluation in terms of number of LLMs quite extensive.

### Weaknesses
W1: The student answers seem to be mostly generate by LLMs. Some answers were generated by only six students. The distribution is not clear. How many are from students? How many of the eight generated answers per question are disregarded?

W2: The student answers are not real answers, collected by students that actually have taken the test. This means that the dataset most likely does not contain many of the patterns found in real student responses such as empty and half-completed answers.

W3: Dataset is only in Chinese. It would be more useful to have an English version as well, also to compare inter-lingual differences in terms of performance.

### Questions
Q1: Why are there only roughly 4 student responses per question? This seems very low.

Q2: What is the difference between step-by-step scores and errors? These seem highly related. A correct step will have no error and full score, whereas an error type will have a reduced score.

Q3: Can you add other languages and make the dataset multilingual?

Q4: From the paper it is not clear how many answers have been actually created by students and how many by LLMs. Also, having overall six student annotators is very little and the setting seems not realistic (i.e., these students are given the task to annotate and have not taken the exam)

Q5: Did you perform any evaluation on adversarial attacks on the LLM graders?

### Soundness
2

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
3

### Summary
This paper introduces SAS-bench, a benchmark to evaluate LLM-based short answer scoring (SAS) by providing fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams.

### Strengths
* SAS-bench aims to provide fine-grained analysis and explanations behind LLM-based SAS scoring, as well as actionable feedback. These problems are highly relevant in the EduNLP community.
* The approach to splitting answers into reasoning steps and evaluating each step for correctness and error analysis is intuitive to obtain fine-grained analysis.
* The authors release the SAS-bench publicly, containing 1030 questions from a real-world exam (China’s National College Entrance Examination(Gaokao)) with 4109 student responses.
* Auxiliary results showing in-context examples and rubrics help improve scoring performance.
* Comprehensive evaluation of 16 LLMs across different families and sizes.

### Weaknesses
* Since the primary contribution is a dataset, the synthetic nature of the student responses needs to be justified as well-aligned to responses from real-world test student takers. Reference responses are first generated by only 3 students, thereby lacking diversity. LLMs are then prompted to diversify and introduce errors to generate the final set of positive and negative responses. How well do LLMs perform at this synthetic task? Each LLM-based synthetic step should be evaluated for performance and error analysis. Prior work has shown prompting to be a poor simulator of students, with fine-tuning of real-world student responses required (SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction, EMNLP 2025).
* Generalizability to new question banks and domains: Since the error taxonomy, a critical component to get fine-grained analysis,  involves a manual intervention for consolidation and refinement, how generalizable is this scoring approach for fine-grained analysis to Q from other exams/benchmarks? Was a completely automated LLM-based approach (with few-shot human-written examples) tried out?
* What is the human performance upper bound? No inter-annotator agreement is reported on the error and scoring metrics? Without human performance and agreement, we don’t know whether this task is well defined and how much behind/ahead LLM performance is compared to humans.
* Motivation: Breaking answers into steps seems intuitive to get fine-grained analysis. However, scoring each step and then adding all scores to arrive at the overall score needs to be justified. Usually, scoring rubrics include key intermediate results needed for different scores, or which missing steps/results would lead in a deduction of points. For example, for reasoning-based domains like math and physics, there could be multiple reasoning paths with varied steps, as well as many steps (max steps for math reported is 25). Students could also include simpler steps or combine or omit them. Won’t this make an alignment between the sum of step-scores and the overall score hard? 
    * Instead of a step-score, simply using binary step correctness seems sufficient to provide fine-grained analysis as well as a useful indicator to arrive at the final score. This could also result in simplified evaluation metrics, with line 313 stating an instability in step-wise consistency evaluation. For example, why does the correct step 1 in Figure 2 get a score of 2, and the other correct steps get a score of 1? Won’t these per-step scoring rubrics be hard to design to avoid variance and ensure high agreement? For math and physics, line 474 states that step-wise labels may mislead the model.
    * Does adding step-wise evaluation help overall scoring performance (QWK)? The key aim is to improve overall scoring, since even if the fine-grained analysis is useful, if the overall scoring is poor, these models are not practically useful. Line 391 states that adding step-wise consistency introduces additional challenges to the model. Compared with SOTA overall-only scoring baselines, which omit step-wise analysis, how far ahead/behind are SAS-bench-trained models?
* What is the motivation behind the ECS metric design; does it extend a well-accepted existing metric? Does it correlate with human evaluation? When working with error distributions and frequencies, doesn’t ECS omit the ordering of steps with their errors? Including ordering seems key since an ordered list of steps with errors is being evaluated. Further, results on ECS (table 3) vary notably across different subjects and question types (line 454), showing that a different model is usually the best for different question types, and also has high variance. How would a practitioner choose the best model?

### Questions
* How is a step defined, especially in non-math/non-STEM contexts? Is there high agreement between human annotators for step decomposition?
* An education-specific evaluation needs to be performed. The major downstream application is the potential to provide useful feedback (line 205). Is this achieved by a human (teacher/student) evaluation?
* Line 385: observe positive correlation: what’s the exact number and correlation metric used?

### Soundness
2

### Presentation
3

### Contribution
2
