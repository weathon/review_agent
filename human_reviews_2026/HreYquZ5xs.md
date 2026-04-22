# Teach2Eval: An Interaction-Driven LLMs Evaluation Method via Teaching Effectiveness

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Evaluating LLMs with static, task-specific benchmarks is increasingly fragile due to contamination and saturation, and it fails to capture interactive reasoning. We introduce Teach2Eval, which reframes evaluation as teaching: a candidate model guides weaker students, and the students’ gains constitute the score. This interaction yields robustness to contamination and exposes orthogonal abilities with fine-grained metrics across Application, Judgment, Guidance, and Reflection. The framework scales automatically by exploiting natural error distributions from weak students, requiring neither bespoke rubrics nor human graders. Across 33 LLMs and 60 datasets, Teach2Eval achieves Spearman above 0.97 with human-preference leaderboards (e.g., Chatbot Arena/LiveBench), surpassing direct baselines, while offering actionable training signals (capability hierarchies, early overfitting) at low cost. We open-source our code and data at https://github.com/zhiqix/Teach2Eval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Teach2Eval reframes LLM evaluation as teaching: a candidate “teacher” guides weaker “students,” and the students’ multi-turn gains—factored into Application, Judgment, Guidance, and Reflection abilities become the score;

### Strengths
1. The paper propose an interaction-centric, contamination-resistant design for evaluation the capabilities of LLMs, and this shall be beneficial for the community.
2. The paper presents fine-grained ability decomposition that aligns strongly with human judgments;

### Weaknesses
1. Multiple interaction rounds as evaluation increase inference cost for evaluate LLMs; 
2. Evaluation similar to human preference as evaluation is probably not fair enough. Maybe should conduct a more comprehensive study on whether these LLMs are able to teach humans well. 
3. The motivation of this evalaution is only to avoid contamination; there should be better justifications and motivation to present this framework as its making evaluation more complex.

### Questions
Make the writing less GPT, there are too much GPT written paragraphs, where I frequently see ' - ' as explanations.

### Soundness
2

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
This work proposes to let large models teach small models, and to evaluate the ability of large models through measuring the improvement of the small models. They used 30 large models and evaluated their performance on tasks automatically converted through their own conversion method, finding that the correlation with human-based ranking exceeded 95%.

### Strengths
I really appreciate Figure 2 — the evaluated models and tasks in the table are very comprehensive, and running such large-scale experiments is not easy; and Figure 5 — the perspective of capability decomposition is very novel.

### Weaknesses
As for the advantage mentioned in the paper that using the teaching method to evaluate large model capability can “comprehensively” assess performance (I do not agree that it can avoid contamination — for any benchmark, as long as follow-up works release evaluation trajectories as required, those trajectories will circulate on the internet. Unless you keep the test set internally and require all later works to send their models to you for evaluation, contamination in this sense is fundamentally unavoidable), I think the advantage is not worth the increased “evaluability manipulability.”

By “evaluability manipulability,” I mean that for example, changing the teacher’s prompt — letting it perform internal multi-step reasoning to solve the problem and present that as the instructions to the small model — could greatly improve the effectiveness of the instructions. In the future, any model could easily “hack” this benchmark and achieve high scores.

### Questions
If you did not keep any private test instances, have you really avoided data contamination? Also, how do you plan to address the issue of “evaluability manipulability”?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new benchmark for assessing a model's general reasoning abilities, by evaluating their ability to guide smaller student models to solve existing verifiable benchmark tasks, in an interactive manner. This essentially transforms the arduous task of evaluating models on diverse general instructions, typically through human evaluation or highly capable expensive judge LLMs, into a automated verifiable task, which does not require expensive LLM inference (other than that of the model under evaluation, and the even cheaper student model).

The proposed approach cleverly mitigates evaluation contamination and performance saturation, without the use of an external judge, by applying indirect evaluation through a student model. The task of effectively guiding the student models to the correct answer is not a straightforward task, and likely requires a wide variety of skills, which the benchmark essentially uses as a proxy for general reasoning ability.

The results show that the benchmark has strong correlation with large-scale real-world user evaluations, significantly higher than direct evaluation of the target models on the problem set.

### Strengths
- S1. Novel idea that leverages indirect evaluation through student interaction.
- S2. High correlation with human judgements of general reasoning performance
- S3. Automated evaluation without the use of expensive judge LLMs
- S4. Robustness to contamination and saturation

### Weaknesses
- W1. Potential for saturation: while the indirect nature of the evaluation and the current results support the benchmark's robustness to saturation, the possibility of saturation is not out of the question. I.e., it is possible that the ability to guide the student model towards the answer can saturate to a certain point, leading to either (1) the student performance converging toward the teacher model's performance (AA), or (2) diminishing correlation between human preferences and the benchmark score (CA).
  - ✅ Regarding (1), Table 2 shows that the ability to guide student models does not converge at the highest levels, given the similar AA scores but dissimilar CA scores of the top 3 models.
  - ⚠️ Regarding (2), it is unclear whether the correlation between human judgements and the CA score is stable across all model capabilities. A visualization between CA and human judgement rankings, similar to Figure 5, could clarify this.
- W2. Potential for exploitation: given the nature of the questions in the problem set in Figure 3, it seems that the evaluated model could simply solve the question and tell the student model the final answer, even if the teacher model does not have access to the multiple-choice options. It seems that the only thing preventing the model from doing this is the specific wording of the teacher prompt to *guide* the student model. This may be easily exploited with specifically targeted post-training. Do the authors believe this an issue? If so, how can it be mitigated? If not, why?

### Questions
- Q1. How does the proposed benchmark's correlation to Chatbot Arena and LiveBench compare to SOTA general instruction benchmarks such as ArenaHard-v2, which use challenging user instruction sets and pairwise response comparison with LLMs?

### Soundness
4

### Presentation
3

### Contribution
3
