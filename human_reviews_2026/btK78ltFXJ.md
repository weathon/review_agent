# JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluation and mismatched question difficulty, leading to incomplete evaluations of LLM's knowledge and capability boundaries, which hinder LLM's effective application and optimization. 
To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. 
Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to call knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more complete evaluations of the LLM's knowledge boundaries. 
It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs.
Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool, and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves.
Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models.
The source code is available on https://anonymous.4open.science/r/JudgeAgent.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes the JudgeAgent framework for dynamic evaluation of model knowledge, summarized as follows:
(1) Evaluate on static base datasets to determine model proficiency.
(2) Extend from seed questions using LLM entity extraction + graph tools to other similar entities and reference texts, then generate new questions via LLMs at difficulty levels matching the model's proficiency, with difficulty estimation subject to iterative refinement; the questions are dynamic, reducing data-leakage risk of static benchmarks.
(3) LLMs provide evaluation feedback based on their question-answering history; the evaluation paradigm further supplies valuable suggestions to the model, helping it self-optimize beyond knowledge association and reasoning.

Experiments demonstrate that this framework's feedback enhances model performance on static base datasets; the authors also quantify how question difficulty changes and supply theoretical analysis.

### Strengths
(1) The paper's expressions and figures are clear.
(2) The framework design is clear and easy to understand.
(3) The authors conducted experiments using multiple models from different departments, enhancing reliability.
(4) Proposes using dynamically difficult questions to test LLMs to capture model capabilities more accurately.
(5) Introduces a knowledge-based data expansion scheme in the evaluation system to reduce the risk of data leakage.

### Weaknesses
After reviewing the experimental section, the connection between the motivation for the work and the experimental design is perplexing. As an evaluation framework, JudgeAgent's key improvements over previous dynamic evaluation frameworks lie in its broader knowledge coverage and difficulty control; the primary motivation should be to mitigate evaluation bias within the framework and avoid lack of discriminative power due to difficulty mismatch (i.e., to propose a more precise evaluation paradigm). However, the experiments primarily focus on demonstrating that the feedback provided by the framework can improve model performance on base datasets.

(1) No comparison with other evaluation paradigms (e.g., scope limitations, manual accuracy); instead, the focus is on describing that this evaluation paradigm can provide feedback to improve the model's score on the benchmark.
(2) There are no concrete metrics quantifying model performance under this evaluation framework, nor analyses of evaluation bias or discriminative power relative to other frameworks.
(3) Whether feedback improves model performance on baseline questions has no direct correlation with evaluation accuracy, since the feedback is generated in an open-book manner and numerous factors can influence model performance; the improvement in benchmark score amounts to checking for omissions and optimizing the LLM, which does not represent any effective meaning.
(4) If the authors' motivation is "to enhance model performance in knowledge question-answering through feedback," the inconsistency lies in the fact that the entire process requires a complete benchmark and answers for grading, and any eventual improvement from feedback should occur offline by expanding the model's knowledge boundaries through training; however, the framework ultimately provides feedback in an online scenario by stating it within the prompt to the model. In actual online scenarios this framework cannot be used to enhance performance.
(5) The dynamic-difficulty aspect resembles merely an instruction evolution of the benchmark; context learning is performed based on new instructions and results, which loses the original purpose of dynamic difficulty—to capture the LLM's capabilities more accurately.

Overall, the authors need to clearly articulate the framework's motivation and make corresponding revisions to the writing and experimental design.

### Questions
(1) Are manual reviews conducted to ensure that LLMs do not generate cheating information directly related to the seed question during the feedback evaluation phase (L248-249)?
(2) Why are 1/3 and 1/2 used in the difficulty threshold setting? Why is the difficulty expressed as linear?
(3) Is the work more focused on a new evaluation paradigm to accurately evaluate LLM performance, or is it more focused on how to improve the performance of LLM on a benchmark?

### Soundness
1

### Presentation
2

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
This paper proposes Agent-as-Interviewer, a novel dynamic evaluation paradigm for LLMs. Unlike static benchmark based evaluations, which often suffer from overestimation, limited scope, and fixed difficulty, this paradigm enables multi-turn, adaptive evaluations using LLM agents that dynamically generate and adjust questions based on the target model’s responses. The authors introduce JudgeAgent, a framework that consists of three core stages, 1. benchmark grading, 2. interactive extension, 3. evaluation feedback. Extensive experiments show that JudgeAgent can effectively identify capability gaps in LLMs and improve their performance through adaptive, knowledge-based feedback.

### Strengths
1. Introduces a multi-turn, dynamic evaluation paradigm enabling more adaptive and realistic assessments of LLMs.
2. Proposes JudgeAgent, a knowledge-aware evaluation framework that dynamically adjusts question difficulty based on the target model’s performance, allowing for fine-grained capability assessment.
3. Provides targeted optimization suggestions by identifying capability gaps, which lead to measurable improvements in model accuracy on follow-up tasks.
4. Conducts extensive experiments and analysis to validate the effectiveness of both the Agent-as-Interviewer paradigm and the JudgeAgent framework across multiple models and datasets.

### Weaknesses
1. JudgeAgent’s multi-turn interaction and dynamic question generation introduce significant computational overhead, requiring more resources and time than static evaluation methods.
2. Construction of context graphs, entity extraction, and multi-hop knowledge sampling are resource-intensive and pose scalability challenges, especially in large-scale deployment scenarios.
3. Compared to static benchmarks, dynamic follow-up question generation using LLMs introduces variability, which makes evaluations less reproducible and harder to standardize across repeated runs or different systems.

### Questions
1. The study focuses on medical and reasoning tasks. How well does JudgeAgent transfer to other domains?
2. Since the same questions are reused for evaluation after feedback, is there a risk that target models are simply learning the test rather than showing true capability improvement?
3. How can we verify that feedback does not contain the correct answers, and that the model is following the prompt instructions correctly during feedback generation?
4. If we provide the correct answer directly instead of feedback, how much additional performance gain is observed compared to JudgeAgent's feedback-only method?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is motivated by limits of static LLM benchmarks: saturation, data-contamination risks, restricted knowledge scope, and mismatched question difficulty. To address this, the authors propose the agent-as-interviewer method, where an agent conducts multi-turn interviews, invokes knowledge tools to broaden/deepen context for follow-ups, and adapts difficulty to the target model’s current performance. The authors implement JudgeAgent, a three-stage pipeline (benchmark grading, interactive extension, and evaluation feedback). Experiments show that agent-as-interviewer can assess target LLMs effectively and that the evaluation feedback can improve LLM performance.

### Strengths
1. This paper is clear and easy to understand.
2. Clear motivation. The paper crisply articulates why static benchmarks fall short (saturation, contamination, scope/difficulty mismatch) and why dynamic evaluation can help.
3. Knowledge-driven dynamic questioning. A context-graph sampling scheme grounds follow-up questions in wider/deeper reference texts from the benchmark knowledge base.

### Weaknesses
1. **Result Stability**: Because this work is proposed as an evaluation/benchmarking method, result stability is important. However, there is no empirical stability study across independent runs/seeds, nor any analysis of rank consistency.
2. **Quality & Diversity of Synthesized Questions**: The authors claim that their method can probe the knowledge boundaries of the evaluated LLMs, which presupposes that the synthesized questions are diverse and high-quality. However, there’s no direct assessment of diversity (coverage/novelty/duplication) or question quality (answerability, ambiguity, factuality/hallucination rate).
3. **Judge Reliability**: The framework relies on an LLM-as-judge to score synthesized questions, so the reliability of those judgments is critical. However, the paper does not calibrate the judge against human-labeled ground truth on the synthesized questions.
4. **Data Contamination**: Although dynamic evaluation methods are theoretically less vulnerable to data contamination, the synthesized questions in this work are still derived from the static datasets. Therefore, whether the proposed method can effectively avoid contamination remains unproven and should be validated with controlled experiments.

### Questions
1. Please include empirical evidence of result stability across random seeds or independent runs, as well as rank-consistency analysis
2. The work would benefit from quantitative and qualitative evaluation of the synthesized questions, including their answerability, factual accuracy, and diversity.
3. A calibration with human annotations or alternative judges would be beneficial to demonstrate judgment reliability and fairness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes JudgeAgent, an agent-as-interviewer dynamic evaluation framework that conducts multi-turn adaptive interviews with target LLMs. The system integrates 1. benchmark grading, 2. interactive extension，and 3. evaluation feedback generation..
Experiment Results on MedQA, MultiHop-RAG, and QuALITY across multiple LLMs show that JudgeAgent can find weaknesses better than traditional static evaluation and slightly improve accuracy when the model re-answers questions after receiving feedback.

### Strengths
- Addresses an important issue beyond static benchmarks to dynamic, adaptive evaluation.
- Context-graph with difficulty control is a novel way of dealing with interactive extension.
- Clear structure with intuitive figures. Easy to follow.
- Empirical improvement across datasets and models with adeqate ablations.

### Weaknesses
-The overlaps with prior dynamic / interviewer-based evaluation works makes the novelty limited 
- Evaluation quality and feedbacks are evaluated only with LLM accuracy, not expert judgment.
- Context-graph construction, difficulty labeling, and stopping criteria are not clearly explained.
- What is the cost/latency for multi-turn evaluation?
- Limited generalizatio with only QA tasks, what about reasoning or code generation?

### Questions
- How is the context graph constructed and difficulty adjusted during interaction?
- What is the cost per evaluation round and trade-off with acc? Latency for multi-turn?
- Could JudgeAgent generalize to other domains (e.g., reasoning, coding)?

### Soundness
3

### Presentation
3

### Contribution
2
