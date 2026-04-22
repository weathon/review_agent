# LLMInertia: Investigating and Mitigating Large Language Models' Unfaithfulness to Input Evidence from a Cognitive Inertia Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Large Language Models (LLMs) frequently generate output that contradicts or disregards explicit input evidence, limiting their reliability across diverse applications. We identify cognitive inertia in LLMs—the tendency to overly rely on co-occurrence associations even when confronted with new or contradictory input evidence—as an important contributing factor to such hallucinations. Through targeted experiments, we show that LLM adherence to explicit input evidence decreases as the strength of co-occurrence associations in pretraining data increases. Inspired by human counter-inertial reasoning, we propose an adaptive counter-inertial reasoning framework that probes cognitive inertia in LLMs related to the input and generates adaptive counter-inertial reminders, which are then injected into the prompt to promote more faithful and evidence-based reasoning. Experimental results in co-occurrence-induction data sets show that LLMInertia significantly reduces hallucination induction rates by 14.16\% and improves accuracy by 12.72\% on average. Comprehensive evaluations on four summarization and question-answer datasets, using three different LLM backbones, further demonstrate the effectiveness and robustness of our approach, highlighting a promising direction for developing more reliable LLM applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates a fundamental source of hallucination in large language models (LLMs), termed cognitive inertia—the tendency of models to rely excessively on learned co-occurrence patterns even when explicit evidence contradicts them. The authors systematically demonstrate that as co-occurrence strength in pretraining data increases, model faithfulness to input evidence decreases. To address this issue, they propose an adaptive counter-inertial reasoning framework (LLMInertia), which detects potential cognitive inertia in a given input and injects the dynamically generated counter-inertial reminders into the prompt to encourage evidence-based reasoning. Experimental results on controlled co-occurrence induction datasets show notable improvements. Further evaluations across multiple summarization and question-answering benchmarks, using three LLM backbones, confirm the method’s robustness and general effectiveness.

### Strengths
1. The paper studies LLM hallucination, a highly valuable research problem that is crucial for ensuring application safety and reliability.

2. The preliminary experiments are interesting and informative, effectively illustrating the issue of Cognitive Inertia in LLMs.

3. The proposed hallucination mitigation approach is simple yet effective, and demonstrates consistent accuracy improvements across different model families such as Qwen and Llama.

### Weaknesses
1. The causes of context hallucination (i.e., LLM unfaithfulness to explicit input evidence) are multifaceted and cannot be fully attributed to cognitive inertia. For example, context length is also a key factor. When the input becomes longer, models may fail due to attention dispersion. I therefore recommend that the authors include additional statistics to isolate the specific impact of cognitive inertia on hallucination. Moreover, when the input evidence conflicts with the knowledge learned during pretraining, LLMs sometimes generate responses opposite to the input.

2. The proposed hallucination mitigation method may not be fully reliable. The entity extraction and probing processes depend heavily on both the LLM’s own capabilities and the quality of the external knowledge base (Co-oc set), which could accumulate noise. In addition, injecting counter-inertial reminders into the prompt does not guarantee that the model will correctly detect and resist cognitive inertia—this may require solutions beyond prompt engineering.

3. The preliminary experiments might not be strictly necessary. As noted by the authors themselves (Lines 318–319), the influence of high-frequency co-occurrence entities on LLM reasoning is already a commonly recognized phenomenon.

### Questions
1. In Table 1, the performance of the three models under Halu Eval appears comparable to that of SFT—some better, some worse. Could the authors provide further explanation for this inconsistency?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates cognitive inertia in LLMs. The authors show that increasing co-occurrence frequency during training amplifies unfaithful, hallucinated outputs. Building on this insight, they propose LLMInertia, an adaptive counter-inertial reasoning framework that probes the model to identify potential co-occurrence biases and injects counter-inertial reminders into the prompt to encourage evidence-based reasoning.

### Strengths
1.	The paper introduces cognitive inertia as a unifying explanation for LLMs’ unfaithfulness to input evidence, framing it analogously to human inertia in reasoning.
2.	The proposed LLMInertia framework is computationally efficient. It requires no retraining and shows measurable improvements (≈ 14% hallucination reduction, ≈ 12% accuracy gain).
3.	Results on multiple datasets and against strong baselines (Prompting, SymbCoT, Lookback, SFT) indicate robustness and generalizability.

### Weaknesses
1. Prior works such as [1] and [2] already discuss the influence of co-occurrence and the interplay between parametric and contextual knowledge. The present paper could better position itself by clarifying whether cognitive inertia is fundamentally distinct or simply a reframing of these effects. 
[1] Kang, Cheongwoong, and Jaesik Choi. "Impact of co-occurrence on factual knowledge of large language models." arXiv preprint arXiv:2310.08256 (2023).
[2] Cheng, Sitao, et al. "Understanding the interplay between parametric and contextual knowledge for large language models." arXiv preprint arXiv:2410.08414 (2024).

2. While demonstrated to be effective, the proposed method works on entity-level questions. The proposed method cannot generalize to open-ended reasoning or multi-hop inference tasks where evidence is implicit, which is a big limitation.


3. The paper claims the medical pairs to build the co-occurrence induction training corpus were selected by statistically identifying high-frequency co-occurrences in the MIMIC dataset. What’s the exact process to choose these pairs? How does the author vary co-occurrence ratios to build the dataset? Those details are important for reproducibility and evaluation.

### Questions
See weakness.

### Soundness
3

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
This paper focuses on understanding the recent observations on the cognitive inertia in LLMs -- the tendancy to overly rely on the pattern in the training data, instead of focusing on new or contradictory evidence provided in the input context. 

The authors first provide empirical results showing correlation between the strength of co-occurence association in the pre-training data and the adherence of LLMs to explicit input evidence (which might contradict with the pre-training data).

This paper proposes an adaptive counter-inertial reasoning framework, which let an LLM generates counter-inertial reminder and another LLM generates answers based on the first LLMs.

### Strengths
This paper focuses on an interesting research question.
The empirical observation on the "cognitive inertia" makes sense.

### Weaknesses
The idea for the hallucination mitigation problem is not that clear.

### Questions
None

### Soundness
2

### Presentation
2

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
This paper investigates how co-occurrence frequency during pretraining contributes to faithful hallucination. They create synthetic training and test data with adjustable co-occurrence frequencies, and their co-occurrence induction experiments show that hallucination induction rate increases as LLMs are exposed to more co-occurrence associations during pretraining. They define LLM’s tendency to overly rely on co-occurrence associations as cognitive inertia. Based on this observation, they propose a prompting framework, named LLMInertia, to promote faithful output by identifying high-frequency co-occurring associations in the input context and including instructions to mitigate cognitive inertia. They evaluate their prompting framework on co-occurrence induction data, summarisation and question answering datasets to show that LLMInertia significantly reduces hallucination induction rate and improves accuracy.

### Strengths
- They conduct a detailed analysis of the impact of co-occurrence frequency on LLM generation, providing new insights into the cause of faithful hallucination
- They propose a simple and effective prompting method to improve the faithfulness of the generated outputs
- The paper is well structured and easy to follow

### Weaknesses
The experiment setup and analysis for the evaluation on question answering and summarisation benchmarks requires more clarification.

### Questions
I have some questions about the experiments on question answering and summarisation datasets.

- What is the difference between base model and naive prompting baselines? Also, it would be helpful to include a CoT baseline to better contextualise your results
- Factual consistency metrics such as AlignScore can be affected by the output length. Does the LLM tend to generate longer output with LLMInertia method?
- It is not clear how LLM reasoning is affected by the counter-inertial reminder from the case study. It would be better to include some reasoning steps or full generation outputs for clarity.

### Soundness
3

### Presentation
3

### Contribution
3
