# Discerning Minds or Generic Tutors? Evaluating Instructional Guidance Capabilities in Socratic LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
The conversational capabilities of large language models hold significant promise for enabling scalable and interactive tutoring. While prior research has primarily examined their ability to generate Socratic questions, it often overlooks a critical aspect: adaptively guiding learners in accordance with their cognitive states. This study moves beyond question generation to emphasize instructional guidance capability. We ask: Can LLMs emulate expert tutors who dynamically adjust strategies in response to learners' states? To investigate this, we propose GuideEval, a benchmark grounded in authentic educational dialogues that evaluates pedagogical guidance through a three-phase behavioral framework: (1) Perception, inferring learner states; (2) Orchestration, adapting instructional strategies; and (3) Elicitation, stimulating proper reflections. Empirical results indicate that existing LLMs often fail to provide effective adaptive scaffolding when learners experience confusion or require redirection. To complement the quantitative evaluation, we conduct a detailed failure case analysis, providing an intuitive understanding of these shortcomings. Furthermore, we introduce a behavior-guided finetuning strategy that leverages behavior-prompted instructional dialogues, substantially enhancing guidance performance. By shifting the focus from isolated content evaluation to learner-centered state-aware interaction, our work advocates a more dialogic paradigm for evaluating Socratic LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces GuideEval, a benchmark to evaluate the tutoring capabilities of LLMs across three stages based on interactions with real students.
The authors use the benchmark to evaluate current commercial and open-source LLMs and find, for example, that they struggle to adapt to learner states, especially when critiquing wrong solutions.
The authors also use their data to finetune an LLM and show that finetuning improves these shortcomings.

### Strengths
- The research area is severely lacking resources that use real students (mainly due to ethical and regulatory problems / hurdles), so releasing a dataset of this scale with real students is quite impactful.

- The results show interesting observations for the future, for example, they point out the sycophancy of current LLMs which can be bad for learning.

### Weaknesses
- In general, the discussion of related work is a bit short and many findings of the paper have been discovered in prior works (though in different form). For example, Wang et al. 2023 already shows that LLMs tend to not actively engage with student error patterns and Daheim et al. 2024 show that identifying student mistakes is a challenging task even for sota LLMs. Another example is Dinucu-Jianu et al. 2025 who show that there exists a trade-off between pedagogy and giving hints (here the authors also point out that only focusing on not telling solutions leads to poor teaching). Scarlatos et al. 2025 also use DPO but this is not discussed, too. 
I don't think that this takes away from the papers findings in general but proper discussion would improve it.
The authors also do not discuss knowledge tracing but the goal of this field is precisely what the paper sets out: to adapt to current learner states. I think a discussion of such concepts could also be helpful.

- I am wondering about more details about the dataset, many seem missing, as the discussion seems limited to Sec. 2.3.
For example, no annotator agreement is reported and beyond the number of turns and dialogues there are no further statistics. I think even reporting simple statistics would be helpful. The domain is also not specified beyond saying that it consists of middle school science questions. It is also mentioned that humans verified and even edited utterances (to create negatives) but the exact process is not detailed.

- The agreement between LLM and humans is fairly high but only checks binary preference over a combination of metrics and not agreement for a specific metric which limits expressiveness for how reliable the individual metrics are.

### Questions
- How many examples were used for creating Tab. 3?

- Will you release the dataset publicly? Based on the conclusion I would assume so but I did not find it written anywhere.

## References

Wang et al., Bridging the Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math Mistakes, NAACL 2024

Daheim et al., Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Model Tutors, EMNLP 2024

Dinucu-Jianu et al., From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning., arXiv 2025

Scarlatos et al., Training llmbased tutors to improve student learning outcomes in dialogues, AIED 2025

### Soundness
2

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
3

### Summary
This paper conducted an extensive evaluation of the capabilities of LLMs in guiding learners and dynamically adapting its responses to the learners' states. To achieve this, they also collected a benchmark dataset of real multi-turn dialogues of learners from a Socratic tutoring platform.

### Strengths
- The paper conducted an extensive analysis of various LLMs and provides several insights on their capability to recognize learner states, to guide / scaffold, and to elicit further follow-ups.
- The collected dataset GuideEval can help advance the field further.
- LLM-based scoring were validated with human annotations
- The failure analysis provides useful insights

### Weaknesses
- The authors evaluated the consistency between LLM based scoring and the Human annotators using the proportion of the same labels. I am not sure if this is the right way to go about it since simply showing the proportion of agreement can be misleading, especially if there is an imbalance in the label distribution. I believe there are more appropriate inter-rater agreement metrics that account for these.
- The failure analysis categorizes the types of failures but the authors did not seem to provide the frequencies of occurrence for each type of failure category. 
- The authors only measured how the LLMs responded. For example, P-affirm and P-redirect scores are only based on how the LLM responded. But this does not tell us whether or not the LLM can recognize the learner states. It might be the case that the LLM does indeed recognize but just does not know how to respond properly.

### Questions
- How well can LLMs recognize the learner states (independent of how affirmative their responses are)?
- What is the distribution of failure types for each of the LLM models? Do they fail in similar ways?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GuideEval, a benchmark for evaluating instructional guidance capabilities of large language models (LLMs) when serving as Socratic tutors. The authors argue that existing evaluations focus primarily on question generation while overlooking adaptive guidance—the ability to dynamically adjust teaching strategies based on learners' cognitive states. The paper proposes a three-phase behavioral framework: (1) Perception - inferring learner states (accurate/erroneous/comprehension/confusion); (2) Orchestration - adapting instructional strategies through scaffolding; and (3) Elicitation - stimulating deeper thinking through strategic questioning.

The authors construct a dataset of 5,177 test samples from authentic tutoring dialogues with contrastive student state pairs, enabling controlled evaluation of model adaptivity. They evaluate 14 LLMs across 6 metrics derived from their framework, finding that models struggle with error detection, adapting to implicit cognitive cues, and maintaining consistent guidance strategies. The paper includes detailed failure pattern analysis and demonstrates that behavior-guided finetuning with Chain-of-Thought distillation substantially improves guidance performance.

### Strengths
1. The paper focuses on critical gap in LLM tutoring evaluation by focusing on adaptive guidance rather than static content quality.

2. The three-phase model is well-motivated by educational psychology literature and operationalized into measurable metrics.

3. The exp covers 14 diverse models, revealing consistent failure patterns across architectures.

4.The paper contains a detailed failure case analysis, providing an intuitive understanding beyond quantitative metrics.

5. I really like the comparative analysis of different training strategies part. The finding that outcome-only SFT degrades performance while process supervision (CoT Distillation) and pairwise preference optimization (DPO) provide substantial gains is a critical, actionable insight for the community.

### Weaknesses
1. it comes with limited scope: dataset topic - middle school science problems in Chinese. It would be more curated if you expand it to other difficulty levels and languages.

2. The cognitive modeling with 4 states (Accurate, Erroneous, Comprehension, Confusion) may be too simplified to capture nuanced learning states. As authors acknowledge, it doesn't capture individual learner profiles, misconception history, or engagement patterns

### Questions
1. Have you tested or do you plan to test this framework on other domains (e.g., humanities, programming) or age groups? What challenges do you anticipate?

2. Can you provide comparison with human tutor performance on the same benchmark? This would contextualize LLM performance.

3. The "Designed Prompt" used to generate the training data for the finetuning experiments is very explicit and rule-based. How can we be sure that the model learned a generalizable instructional guidance capability rather than just learning to mimic the explicit rules baked into the generation prompt? Have you tested or are you planning to test the finetuned model on out-of-domain tasks to see if the "skill" transfers?

4. The filtering mechanism for training data (equations on p.18) is complex. How sensitive are results to these hyperparameters (β, threshold)?

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
This paper investigates whether LLM tutors can deliver adaptive instructional guidance in Socratic dialogues rather than relying on generic questioning. The authors conceptualize guidance as a three-phase behavior: Perception (inferring the learner’s state), Orchestration (selecting an appropriate next-step strategy), and Elicitation (formulating prompts suited to the learner’s state). Building on this framework, they introduce GuideEval, a benchmark composed of real multi-turn tutoring dialogues with contrastive student states, and define phase-aligned evaluation metrics to assess model performance across these dimensions.

### Strengths
1. Clear behavioral decomposition with actionable metrics. The three-phase split translates “be a better tutor” into concrete, checkable behaviors, offering conceptual clarity and operational guidance that enable reproducible, phase-wise diagnosis across different models.

2. Useful failure taxonomy grounded in qualitative evidence. The paper goes beyond reporting average behaviors and highlights failure modes supported by dialogue snippets, providing interpretability and practical insight into model behavior.

### Weaknesses
1. Human–LLM agreement is reported without sample size or reliability statistics. In Table 3, the claim that “LLMs can serve as reliable and scalable evaluators of instructional behaviors” rests on high agreement ratios and minimal score deviations, but the paper omits sample size per metric or level, sampling protocol, number of human raters, and inter-rater reliability. Without these, chance agreement and selection bias cannot be ruled out, especially with coarse labels (binary or 3-point) that inflate raw agreement.

2. Prompt–rubric inconsistency. Generation prompts forbid giving final answers (“Do not directly provide the final answer or full solution process” in Original/Rule templates), yet the O-Advance evaluation rubric awards full credit when “the model provides the final answer.” This contradiction allows models to achieve high orchestration scores while violating generation rules. Please align the prompt and rubric definitions.

3. All models are decoded at temperature = 0.1. Please justify this setting and provide an ablation across temperatures to ensure conclusions are not artifacts of low-variance decoding.

4. Human annotators reportedly revised model outputs to create “state-edited counterparts” (e.g., answer-correctness flips, comprehension/confusion flips). Please specify the exact editing operations and provide per-operation statistics, including counts, edit distance, token-level change distribution, and human vs. synthetic proportions.

### Questions
1. What are the details of sample size per metric or level, sampling protocol, number of human raters, and inter-rater reliability statistics?

2. Generation prompts forbid giving final answers, while the O-Advance rubric rewards them. How are these conflicting criteria reconciled to ensure consistent evaluation?

3. How many items per metric were used for the LLM–human consistency analysis, and how were they sampled from the full dataset?

4. Could you replicate headline results using at least one alternative judge and report inter-judge agreement and sensitivity to confirm robustness?

5. Since training and evaluation data are drawn from the same source pool, do problem contexts (e.g., identical or near-duplicate problem IDs, stems, passages, or scaffolds) repeat across splits? If so, what is the overlap rate, and how does it affect evaluation outcomes?

### Soundness
2

### Presentation
3

### Contribution
2
