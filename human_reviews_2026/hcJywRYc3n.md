# Do LLMs Forget What They Should? Evaluating In-Context Forgetting in Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Large Language Models (LLMs) have been extensively studied for their memory ability, yet the capacity to selectively forget during inference remains underexplored. We introduce ICF-Bench, a comprehensive benchmark for evaluating In-Context Forgetting (ICF). We define ICF as the ability of LLMs to selectively forget interference information while retaining useful knowledge in context without parameter updates. Built on high-quality datasets, ICF-Bench comprises 2k multi-turn dialogues with annotations that reflect realistic scenarios.  Extensive experiments of advanced LLMs on ICF-Bench reveal that: (1) models perform well without forgetting interference but struggle significantly when interference is present; (2) stronger memory capacity without forgetting interference does not transfer into stronger ICF capacity, highlighting an asymmetry between memory and ICF; and (3) context length has different effects on ICF capacity across scenarios. These findings expose critical vulnerabilities of current LLMs in terms of privacy protection, adaptability, and user autonomy. Our code and data will be available at https://anonymous.4open.science/r/ICF-Bench-B1C7.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ICF-Bench, a benchmark that measures Large Language Models (LLMs)' ability to follow instructions while taking into account explicit or implicit updates, called forget interference, in the conversation. The dataset was constructed by inserting LLM-generated forget interferences into existing multi-turn dialogue datasets. Various open-weight and closed models are evaluated on the benchmark, consisting of 3 scenarios (instructional forgetting, subtask revision, and dynamic preference), using 3 metrics  (NoForget Accuracy, Forget Accuracy, and SFRR). The performance of the models degrades when forget interferences are introduced. SFRR does not always increase with model size. When increasing the context size, the models behave differently depending on the scenario and the metric.

### Strengths
- The paper proposes a simple yet informative benchmark. It could help the community correctly assess the current limitations of LLMs. 
- The paper provides a lot of details about dataset creation and model evaluation. This can make replication easier. 
- A wide range of models (open-weights/closed  and of different sizes) are evaluated on the proposed benchmark.

### Weaknesses
- The analysis part could be more extensive. The paper does not go deeper in explaining the causes of the different observed phenomena. It only proposes vague conjectures (like in line 146 and line 455). For example, recency bias , which was observed and analyzed in prior work, could explain SFRR score as a function of context length on the subtask revision scenario.  
- The human evaluation is not very clear. Was it done only on one subject? Moreover, there is no statistical test on the agreement between human and automated evaluation. 
- Concerning readability, Table 1 and 2 are a bit hard to parse. Turning them into a figure (at least the average scores) could improve the readability of the paper. Also, combining the Tables might help see the trend across the metrics and avoid repeating the NA scores across the tables. The x labels in Figure 3 are not aligned.
- Related work, you might find helpful the benchmark proposed in this paper https://arxiv.org/abs/2502.13791 that is very similar to this work (subtask revision and dynamic preference scenarios).

### Questions
1 - Some times SFRR improves with model size (For example, SFRR goes from 33% for Llama3-8B to 46% for Llama3-70B), what causes the variations in SFRR across models?
2 - Can the trends be described in a more quantitative way? what is the relation between the different metrics? scenarios?
3 - Could you provide more details about the human evaluation? To what extent is human evaluation in agreement with LLM evaluation?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces "In-Context Forgetting" (ICF), defined as the ability of Large Language Models (LLMs) to selectively disregard interference information during inference, without any parameter updates. The paper's main contribution is ICF-Bench, the first comprehensive benchmark designed to evaluate this skill. The benchmark consists of 2,000 annotated, multi-turn dialogue samples covering three realistic scenarios: Instructional Forgetting, Subtask Revision, and Dynamic Preference. To measure performance, the authors use a paired-task format ("Noforget" vs. "Forget") and introduce a novel core metric: the Selective Forgetting Retention Rate (SFRR). 

Key findings from testing various LLMs show:
1. All models perform well without interference but struggle significantly when interference is present.
2. Strong memory capacity does not guarantee strong ICF ability, revealing a fundamental "asymmetry" between the two skills.
3. The effect of context length on ICF varies by scenario.
These results expose critical vulnerabilities in current LLMs regarding adaptability, privacy, and user control.

### Strengths
1. The paper poses an interesting and previously underexplored problem. It defines in-context forgetting (ICF) clearly and distinguishes ICF from parameter-based machine unlearning.
2. The ICF-Bench is thoughtfully designed around three practical, real-world scenarios. The validation of the automated evaluator (GPT-04mini) against human agreement (80-98% correlation) confirms the reliability of the experimental results.
3. SFRR provides a fair measure of forgetting ability, as it isolates failures caused by interference from a model's general inability to perform the task. TThe paper's justification for SFRR (Appendix B) is convincing. The discovery of the "asymmetry" between memory and ICF is a valuable insight.

### Weaknesses
1. The term of "Forgetting" is debatable. As noted in the case study (Appendix C, lines 1026-1028), the model's response is "You asked me to forget... so I cannot recall them." There's more "Human forbidding sth." instead of "LLMs forgetting sth." This behavior is more likely to obeying an instruction to withhold information rather than a genuine "forgetting". The current evaluation may be measuring instruction compliance more than the intended forgetting mechanism.
2. The analysis of context length's impact is restricted to a single model (GPT-4O), limiting the generalizability of the findings. More importantly, as the authors themselves suggest in Section 4.4, poor performance on long-context tasks can sometimes mimic successful forgetting. It is difficult to classify whether a model 'forgets' due to following the instruction or due to its inherent limitations in long-context recall.
3. LLMs responses rated by LLMs could introduce potential bias. The paper would be strengthened by a brief discussion acknowledging this limitation and justifying the choice over other potential evaluation methods.
4. There're many typos in this paper.

### Questions
1. In Appendix B(line 916), the text refers to "Equation ??", which is a placeholder the authors forgot to fill in. It should likely refer to Equation (4).

2. Typos: "scenaios" is misspelled three times(line 224, line 249-250, line 367-368) and it should be "scenarios". "focuse on" in line 130 should be "focuses on". "The Figure demonstrate that" should be "The Figure demonstrates that" (line 249-250). "their ability to memory" shoule be "their ability to memorize" (line 094). "response persistently follow" should be "response persistently follows" (line 1083-1084). Moreover, "non negligible" would be better to be hyphenated as "non-negligible".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies whether large language models can selectively forget information during inference, a capability the authors call in-context forgetting (ICF). Many prior works study and prioritise memory retention and long-context reasoning, however, this paper argues that real interactions also require models to discard outdated or conflicting context, with implications for privacy, personalization, and robustness. The authors introduce ICF-Bench, a benchmark of 2,000 multi-turn dialogues, and each example has a No-Forget and Forget condition. The main metric, Selective Forgetting Retention Rate (SFRR), measures how often models that answered correctly without interference remain correct when forgetting is required. 

Experiments across recent proprietary and open models show that models perform well without interference but degrade substantially when forgetting is required. The experiments find that a higher memory retention does not imply strong forgetting ability, revealing an asymmetry between remembering and forgetting.

### Strengths
- The paper identifies a meaningful gap: models can remember well but struggle to intentionally forget, which matters for privacy, preference updates, and real conversational robustness. Framing in-context forgetting as a capability pushes beyond typical long-context evaluation.

- Well designed benchmark with three representative scenarios of ICF: Instructional Forgetting, Subtask Revision, Dynamic Preference, with Forget and NoForget alternatives. This is a clean evaluation structure that avoids confounding memory quality with forgetting ability.

- The demonstrated dissociation between memory strength and forgetting ability is useful. It gives empirical proof to a hypothesis most practitioners may have suspected but not quantified.

### Weaknesses
- The paper varies context length within conversational tasks (by prepending other conversations), yet context dynamics in dialogue are only one class of real-world usage. Long-form reasoning, tool-augmented workflows, and multi-document tasks involve qualitatively different context structures and memory demands. As a result, the observed context-length effects may not generalize beyond dialogue settings. Expanding or at least discussing evaluations in these alternative settings would strengthen claims of generality.

### Questions
1) Several related work are mentioned in subsection "In-context Unlearning and Context Management." - what are the main limitations in the evaluation methods used in these papers? A more complete discussion of existing evaluation techniques in the literature would strengthen the paper and it's positioning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ICF-Bench, a benchmark for evaluating in-context forgetting (ICF) in large language models, which is the ability to discard outdated or contradicted information within a conversation. Unlike prior work focusing on memorization or unlearning, the authors design controlled multi-turn dialogues where models must selectively “forget” previous content when given new or conflicting instructions. They evaluate several open and proprietary LLMs across three scenarios (instructional forgetting, subtask revision, dynamic preference) using paired tasks with and without interference. Results show that current LLMs perform well without interference but degrade sharply when forgetting is required, illustrating vulnerabilities of current LLMs in terms of privacy protection, adaptability, and user autonomy.

### Strengths
1. The paper focuses on an important dimension of LLM behavior, the ability to forget selectively, which is particularly relevant for user privacy, adaptability, and evolving dialogue contexts.
2. The benchmark design is systematic and well-structured, covering three realistic dialogue scenarios with paired no-interference vs interference tasks.
3. The experimental evaluation is comprehensive, including multiple open-source and proprietary models, context-length variation, and a human evaluation for the automated metric. This offers convincing evidence of the claimed vulnerabilities.

### Weaknesses
1. The instructional forgetting tasks might be narrowly defined. While it provides a reasoning measurement of LLMs to follow precise forgetting instructions, it's unclear the value of optimizing this specific capability for LLMs. 
2. The paper discusses “forgetting” mainly as suppression of prior information, but less on why the model fails (e.g., attention routing, representation entanglement) and on concrete architectural remedies.
3. The ablation on the impact of context length is rather shallow. The experiment is only conducted for GPT-4o. It's unclear whether the conclusion drew here would generalize.

### Questions
1. Have the authors experimented prompt engineering to improve ICF ability even preliminarily?

### Soundness
3

### Presentation
3

### Contribution
3
