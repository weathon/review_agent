# MedReason-Dx: Benchmarking Step-by-Step Reasoning of Language Models in Medical Diagnosis

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
In high-stakes domains like medicine, $\textbf{how}$ an AI arrives at an answer can be as critical as the answer itself. However, existing medical question answering benchmarks largely ignore the reasoning process, evaluating models only on final answer accuracy. This paper addresses the overlooked importance of reasoning path evaluation in medical AI. We introduce $\textbf{MedReason-Dx}$, a novel benchmark that assesses not just answers but the step-by-step reasoning behind them. MedReason-Dx provides expert-annotated step-by-step solutions for both multiple-choice and open-ended questions, spanning 24 medical specialties. By requiring models to produce and be evaluated on intermediate reasoning steps, our benchmark enables rigorous testing of interpretability and logical consistency in medical QA. We present the design of MedReason-Dx and outline diverse evaluation metrics that reward faithful reasoning. We hope this resource will advance the development of robust, interpretable medical decision support systems and foster research into large language models that can reason as well as they respond.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MedReason-Dx, a good benchmark designed to address the critical gap in evaluating step-by-step reasoning processes of language models (LLMs) in medical diagnosis—an aspect largely overlooked by existing medical QA benchmarks that focus solely on final answer accuracy. Experiments on LLMs reveal three core findings: 1) State-of-the-art LLMs struggle with complex medical reasoning; 2) LLMs face challenges in recalling all essential medical knowledge; 3) Medical LLMs do not outperform general-purpose LLMs in complex reasoning tasks, attributed to insufficient training on reasoning-focused datasets.

### Strengths
1. MedReason-Dx fills a critical niche by focusing on step-by-step reasoning in medical diagnosis, a dimension ignored by dominant medical QA benchmarks. 
2. The five evaluation metrics provide a holistic assessment of reasoning, and experiments on 11 LLMs (covering different sizes and domains) ensure generalizable findings.
3. The paper explains technical concepts in accessible terms, with formulas and examples to illustrate key ideas. 
4. The work advances the development of trustworthy medical AI by emphasizing interpretability—essential for clinical adoption.

### Weaknesses
1. Table 1 is missing data for critical columns (e.g., CoT Evaluation, Expert Annotation) across existing benchmarks, making it difficult to fully assess MedReason-Dx’s uniqueness relative to prior work like MedReason (Wu et al., 2025a) and MedCaseReasoning (Wu et al., 2025b).
2. The paper does not report how consistent expert annotators were in creating reasoning chains and key points. This raises questions about the reliability of the ground truth, as subjective differences in clinical reasoning could introduce bias.
3. The experiments identify performance gaps but do not delve into why medical LLMs underperform or how to mitigate these issues (e.g., fine-tuning strategies using the benchmark).
4. Converting multiple-choice questions to open-ended formats using LLMs, even with expert review, may introduce subtle changes to clinical context or reasoning requirements. So how should the benchmark control the quality of the dataset?
5. The paper focuses on benchmarking existing models but does not demonstrate how MedReason-Dx can be used to improve LLMs. A good explanation is well recommended.

### Questions
In addition to the above concerns, I also have the following questions:
1. Did you compute inter-annotator agreement for the expert-curated reasoning chains and key points? If so, what were the results, and how did you resolve discrepancies? If not, how do you ensure the reliability of the ground truth?
2. Why do direct prompts outperform CoT prompts for open-ended questions in some top-performing models (e.g., GPT-4o)? Is this due to CoT introducing redundant steps or the open-ended format requiring more concise reasoning?
3. How does MedReason-Dx handle rare diseases or niche clinical scenarios?
4. Have you tested MedReason-Dx on LLMs specifically fine-tuned for medical reasoning tasks?

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
4

### Summary
The authors introduce MedReason-Dx, a benchmark to evaluate not just answer accuracy but the reasoning process of LLMs on medical diagnosis questions. The dataset spans 1,170 items across 24 specialties, each with expert-annotated, stepwise solutions and extracted key points. They propose five metrics to quantify reasoning fidelity. Baselines across general and medical LLMs show modest accuracies, with general models typically outperforming medical LLMs on complex reasoning. Evaluation relies on LLM-as-judge (GPT-4o-mini), with a small cross-judge consistency check vs. GPT-4.1-mini.

### Strengths
- This work is well motivated to evaluate the quality of reasoning paths in medical diagnosis.
- The proposed benchmark covers a broad range of specialities, with both multiple-choice and open-ended questions for comprehensive evaluation. 
- Along with the benchmark questions, this work also proposes various reasoning-focused metrics (RCS/RNS/KCS/KNS) to evaluate reasoning quality from different perspectives.
- Comprehensive evaluations are conducted on a broad range of LLMs, providing insights into their differences in medical reasoning.

### Weaknesses
- My major concern is about the potential existence of multiple reasoning paths for the same medical question. For example, if concept A connects to B and B to C, but there is also a direct link from A to C, it is unclear how MedReason-Dx accounts for such alternative but correct reasoning routes.
- It is surprising that DeepSeek R1 has a higher accuracy than V3 (both multiple-choice and open-ended) but shows a lower reasoning quality (e.g., lower RNS/RCS on multiple-choice questions, RNS/RCS/KCS on open-ended questions). A deeper analysis beyond the scores is needed to explain why such an inconsistency can happen and if the evaluation metrics for reasoning are reliable.
- While ablation of LLM judges is conducted on GPT-4o-mini and GPT-4.1-mini by comparing their evaluation scores, there is no analysis on how well they perform on the given task. Given the relatively low accuracy of all LLMs on the constructed benchmark (Table 5), it is unclear if GPT-4o-mini/GPT-4.1-mini is capable of doing such evaluations.
- The average response lengths (or number of reasoning steps / key points) also need to be reported for the evaluated LLMs, which may have a significant impact on the reasoning-related metrics

### Questions
See the listed weaknesses.

### Soundness
2

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
This paper designed a benchmarking system to evaluate medical step-by-step reasoning in LLMs. Instead of only aiming for the final correct option, the authors, along with an accuracy metric, propose evaluation metrics based on reasoning chains and key point based evaluation to assess the completeness and necessity of the generated reasoning steps. The authors construct a human-annotated dataset with 1170 questions derived from multiple choice and open-ended question datasets spanning across 24 topics.

### Strengths
- **Expert annotated dataset** to evaluate step-by-step reasoning in the medical domain which would benefit in understanding and evaluating the decision-making of models.
- **Experimental setup**. Including both general-purpose and domain-specific LLMs helps to clearly see the gaps in performance.
- **Clarity**. The paper is well-structured and easy to follow

### Weaknesses
- Although the process of data curation is well explained, it remains unclear which datasets exactly where used to construct MedReason-Dx, the paper could benefit from pointing out which datasets the authors relied on.
- **Lack of information on human experts**. No clear information on the number of human annotators and the annotation agreement at every step where humans were involved, hence, difficult to derive the results for any bias. 
- The paper heavily emphasizes quantitative metrics but provides limited qualitative discussion of failure cases or examples where LLMs’ reasoning diverges from human reasoning. Such insights would clarify *why* models fail, especially for the medical reasoning tasks.
- Using LLM-judges only from one provider (GPT-4o-mini, GPT-4.1-mini) does not allow to conclude that the evaluation is robust and fair. Perhaps authors should have considered judges from other families, including open models.

### Questions
* How many human annotators were involved in the annotation process? How were they selected and what was the inter-annotator agreement?
* Address the issues listed in the weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MedReason-Dx, a benchmark aimed at judging how medical LLMs reason, not just whether they land the right answer: each of its 1,170 questions (592 multiple-choice, 578 open-ended) comes with expert, step-by-step solution paths and key points spanning 24 specialties, enabling systematic comparison of model rationales to clinical gold standards. It introduces a suite of reasoning-focused metrics that align model outputs with expert traces and reduce sensitivity to surface wording. The dataset is intentionally multi-step to stress genuine diagnostic reasoning rather than recall. Evaluations across general-purpose and medical LLMs reveal clear separation and highlight that open-ended questions are substantially harder than multiple choice. The authors also probe evaluation robustness by comparing different LLM judges, aiming to ensure conclusions aren’t artifacts of a single scorer.

### Strengths
Rather than stopping at end-answer accuracy, the paper makes reasoning itself the object of measurement by introducing a five-facet framework with a neat key-point extraction scheme that compares model rationales to expert "must-mention" concepts, reducing surface-form bias in judging chains of thought. The benchmark is broad and realistic in scope: 1,170 items spanning 24 specialties, split roughly evenly between multiple-choice (592) and open-ended (578), with expert, step-by-step chains averaging 6.4 steps (and 27.1 key points) to keep the focus on multi-hop clinical reasoning rather than recall. Empirically, the suite exposes meaningful gaps: open-ended questions are distinctly harder and medical LLMs do not reliably beat general models on complex reasoning. Thus, the task design actually differentiates systems rather than saturating them. Finally, the authors sanity-check evaluator dependence (GPT-4o-mini vs. GPT-4.1-mini) and find closely tracking RNS/RCS scores, which bolsters confidence that conclusions aren’t an artifact of a single judge model.

### Weaknesses
While the key-point layer softens exact-string matching, the step-wise evaluation still appears to privilege a single canonical chain, leaving uncertain how genuinely different but valid routes are credited or penalized. 

The reliance on LLM judges, though partially stress-tested with two closely related models, raises residual concerns about judge bias, calibration drift, and ranking stability across models and domains. The current check is limited to two systems and two judges rather than a broader, blinded human baseline for anchoring scores.

Data curation for open-ended items involves LLM rewriting of prompts, which may subtly shift difficulty or distribution relative to naturally authored open questions; more auditing here would strengthen external validity. 

The metric design is also vulnerable to verbosity effects: KNS penalizes extra (possibly harmless) facts and RNS rewards concision, so models with terse styles could be favored independently of clinical quality. An explicit length-normalization or "necessary/sufficient proof" analysis would help. Finally, the paper reports aggregate step/key-point counts, but it does not analyze how step length interacts with each reasoning metric or accuracy (e.g., are longer chains systematically over-/under-scored?), nor does it report inter-annotator reliability for expert chains/key-points, both of which matter for reproducibility and fair comparison.

### Questions
Please address the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
