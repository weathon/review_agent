# The Dialogue That Heals: A Comprehensive Evaluation of Doctor Agent's Inquiry Capability

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
An effective physician should possess a combination of empathy, expertise, patience, and clear communication when treating a patient.
Recent advances have successfully endowed AI doctors with expert diagnostic skills, particularly the ability to actively seek information through inquiry. However, other essential qualities of a good doctor remain overlooked.
To bridge this gap, we present MAQuE (Medical Agent Questioning Evaluation), the largest-ever benchmark for the automatic and comprehensive evaluation of medical multi-turn questioning. It features 3,000 realistically simulated patient agents that exhibit diverse linguistic patterns, cognitive limitations, emotional responses, and tendencies for passive disclosure. We also introduce a multi-faceted evaluation framework, covering task success, inquiry proficiency, dialogue competence, inquiry efficiency, and patient experience.
Experiments on different LLMs reveal substantial challenges across the evaluation aspects. Even state-of-the-art models show significant room for improvement in their inquiry capabilities. These models are highly sensitive to variations in realistic patient behavior, which considerably impacts diagnostic accuracy. Furthermore, our fine-grained metrics expose trade-offs between different evaluation perspectives, highlighting the challenge of balancing performance and practicality in real-world clinical settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes an evaluation framework for the medical inquiry capabilities of LLMs. Conceptually, it is different from many existing works in terms of what information is included and what is evaluated. Empirically, it shows that medical inquiry is still a challenging problem.

### Strengths
+ medical inquiry is an important research question
+ the resource, when released, will be helpful

### Weaknesses
- It's nice to try to disentangle the inquiry and diagnosis capabilities in medical applications. In line 307, the authors use "the powerful GPT-5" as the diagnosis part and test models on the inquiry part. GPT-5 certainly isn't perfect: I wonder for the models that are more distilled from GPT-5-generated texts (which we don't know), would their outputs/inquiries be more compatible with GPT-5 as a diagnosis agent? Essentially, this could be a confounder. I wonder if any chance the diagnosis part could be replaced with an oracle diagnosis, limited in sample size and with human diagnosis perhaps, to remove the confounder and see if the patterns hold.

- Figure 3 seems to be important info about the multi-turn skills: it would be nice to have open models in the mix.

- I like that on line 374 the authors show that existing medical LMs are not doing so well on the proposed evaluation framework, thanks to its more "real-world"ness. Medical NLP is really an interesting domain: every paper always highlights how previous papers are very "un-real-world" and how we became more "real-world", while we never seem to be "real-world" enough. For the authors, if there's something about this work that's not "real-world" enough, what would that be? Will we ever get a data/evaluation that most people could agree is "real-world" enough?

- Figure 4 got me thinking: can we somehow have different models responsible for different steps? For example, one model is good at inqury do inquiry, one model is good at diagnosis do diagnosis. Or if the authors believe there should still be one model doing the best on everything.

### Questions
please see above

### Soundness
3

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
This paper presents an evaluation framework for medical agents, tackling different aspects of a conversation. A 3,000-sample dataset, MAQUE, is created for this purpose, incorporating diverse linguistic patterns, cognitive limitations, emotional responses, and tendencies for passive disclosure to simulate real-world environments. The framework includes five evaluation dimensions to assess agents' capabilities comprehensively. The authors conduct extensive experiments across a wide range of LLMs, and the results reveal that these models still fall short of the requirements for actual clinical settings.

### Strengths
1. A dataset containing 3,000 samples is constructed for medical case evaluation, which is sufficient for comprehensive assessment.

2. An evaluation framework is also proposed, covering a wide range of dimensions. Instead of simply focusing on lexical overlap, this paper emphasizes the semantic aspects of medical conversations.

3. A wide range of LLMs are evaluated, and several interesting findings are presented for reference:

- LLMs cannot compete with physicians currently.

- LLMs require long context to achieve good performance.

- Medical LLMs do not yield better performance than general LLMs.

### Weaknesses
1. It seems this paper only involves simulated patients and doctors. Despite including non-expert human evaluation, it is difficult to assess the real-world performance of these LLMs.

2. LLMs can be easily affected by variations in context. However, no case studies are conducted to illustrate how they are affected. It would be beneficial to see more detailed analysis on this aspect. Given the strong text understanding capabilities of LLMs, I would expect these variations to stem from challenging cases.

3. Line 357: "This indicates that LLMs base their diagnoses on, at most, 40% of the collected information, raising concerns about reliability." I believe most people, as well as physicians, make judgments based on partial information. Additionally, this should not serve as evidence to support "Existing LLMs are far from being effective physicians," because the LLMs' performance is compared against an upper bound rather than actual physicians.

### Questions
1. Since the dataset is simulated, have you conducted correlation analysis between expert evaluations and the proposed metrics? It would be beneficial to include such analysis as a reference for validation.

2. Additionally, how do these LLMs perform with real patients? Do they demonstrate consistent performance across simulated and real-world scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MAQUE, a benchmark for evaluating the inquiry skills of AI doctor agents. The authors argue that current benchmarks focus too much on final diagnostic accuracy and use patient simulators that are too simplistic. 

MAQUE's first contribution is a dataset of 3,000 simulated patients. A key feature is "Atomic Information Units" (AIUs), which controls information disclosure to simulate a more realistic, passive patient. Its second contribution is a 5-dimensional evaluation framework. Instead of just accuracy, it also measures Inquiry Proficiency, Dialogue Competence, Inquiry Efficiency, and Patient Experience. 

The authors tested several modern LLMs and found them to be "far from being effective physicians." Even top models fail to gather enough information, show little empathy, and struggle to balance accuracy with efficiency.

### Strengths
The paper addresses a clear gap. It rightly moves the focus from just diagnostic accuracy to the entire inquiry process and patient experience. This is a crucial step for building practical AI doctors.

The patient simulation method is a major strength. Its three-layer approach (Disclosure, Linguistics, Noise) creates a realistic testbed. However, using "Atomic Information Units" to control information flow is first proposed in MediQ (https://arxiv.org/abs/2406.00922), which the authors briefly cited in Table 1 but did not acknowledge for other high levels of similarity.

The 5-dimensional framework is thorough. It provides a holistic view of an agent's performance and captures overlooked aspects like patient communication (empathy, clarity) and inquiry skill (coverage, relevance).

### Weaknesses
* Oversimplification of Prior Work: The paper oversimplifies prior work. The claim (L066) that others ignore the "intermediate conversational process" is an overstatement. Existing work has already proposed fine-grained metrics (Schmidgall et al., 2024) and evaluated question-level quality (https://arxiv.org/abs/2502.14860). The paper fails to properly situate itself against this related work.

* Missing Related Work: The paper misses some related work. The suggestion (L133) to add more roles like a "triage nurse" overlooks research that has already explored this (e.g., https://arxiv.org/abs/2404.15155).

* The patients' personas and traits are not grounded in real-world clinical or psychological data. The categories for linguistic style, cognitive status, and emotional state seems arbitrary, and the characteristics seem randomly assigned. 

* Heavy Reliance on LLM-as-judge: The paper uses an LLM (GPT-4o-Mini) to score subjective, human-centric metrics like "empathy," "clarity," and "coherence." This is a significant weakness as LLM-judges are known to have biases and may not be reliable proxies for true human experience. The validation in Appendix G even shows a weaker correlation for "clarity" (0.66) and uses non-medical experts, which is also a limitation.

* In the main experiment (Section 4.1), the paper separates the "inquirer" (the model being tested) from the "diagnostician" (a powerful, static GPT-5). This is an artificial setup. In a real-world agent, the ability to ask good questions is driven by its own internal diagnostic reasoning. By splitting them, the benchmark isn't testing an integrated agent, so the "inquiry" scores may not reflect real-world utility.

* Superficial Simulation of Human Behavior: While the simulation is a strength, its "noise injection" (Section 3.1, Appendix D.4) is still quite simplistic. Emotions and cognitive states (like "confused" or "anxious") are assigned and their intensity is just "predicted" each turn. A truly realistic simulation would have these states dynamically evolve based on the doctor's actions (e.g., a doctor using jargon might cause a patient to become "confused" or "frustrated").

* Minor Presentation Issues:
- Section 2.2 is verbose and could be condensed.
- L198 incorrectly states that Figure 2 contains an example patient prompt; it does not.

### Questions
1. How was the diversity of the 3,000 simulated patients (especially from "Patient-Zero") ensured? How faithful is this cohort to a realistic clinical distribution of demographics, condition prevalence, and severity?

2. How was the case distribution across 21 specialties determined? Was it based on data availability, or did it model a realistic case distribution for a general practitioner?

3. What is the justification for the five evaluation dimensions? The selection seems arbitrary. Why these specific metrics and not others?

4. How reliable is the LLM-as-a-judge for subjective metrics like "empathy"? Appendix G states validation used non-experts. How does using non-experts to validate a "patient" experience score affect its validity?

5. The paper notes (L409) that emotional responses can improve agent empathy. Is there a correlation between "Patient Experience" and "Task Success"? Does a more empathetic agent gather more information (higher "Coverage") and achieve a more accurate diagnosis?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces MAQUE, a benchmark to evaluate medical multi-turn inquiry abilities of LLM-based medical agents. MAQUE include human-like behavioral such as disclosure control, linguistic variation, and noise injection. The benchmark assesses multiple complementary dimensions with various fine-grained metrics. Experiments across frontier, open-source, and medical-specific models reveal model weaknesses, behavioral sensitivity, and trade-offs between diagnostic accuracy and inquiry cost.

### Strengths
1. MAQUE introduces a three-layer simulation for patient behavior with human-like noise behavioral injection. This aligns with reality.
2. The benchmark is built on a large scale and covers 21 medical departments. Its multi-dimensional evaluation provides a comprehensive view of conversational quality. 
3. The experiment tests multiple competitive models across closed, open-source, and medical domain-specific. Inquiry strategies evaluation is interesting as it reveals failures of CoT and self-consistency on this benchmark.

### Weaknesses
1. The primary results presented in Table 2 - 4 are all point estimates. The paper does not report any measures of variance across patient agents and random seeds. This is a critical omission, as it is impossible to determine if the performance differences between models are statistically significant or merely noise.
2. Many core metrics, e.g. Patient Experience and Dialogue Competence are evaluated using an LLM-as-judge. The human validation study does not report the IAA among annotators. Without  inter annotator agreement, it's difficult to assess the reliability of their judgments or the meaning of the correlation scores (e.g. the low 0.6635 for Clarity ). Also, the human validation is conducted with non-expert and synthetic conversation, no validation with clinical staff or real patient interactions.
3. Task Success is dependent on a single, unevaluated GPT-5 used as a fixed "diagnostic agent". The paper assumes this model is "powerful" but presents no independent evidence of its diagnostic accuracy or robustness across the different medical specialties.

### Questions
1. Is there any analysis testing the robustness of the LLM-as-judge itself? For example, does the judge's scoring for metrics like clarity or empathy remain stable when the only variable changed is the patient's linguistic style or emotional state?
2. Non-expert annotators is a significant concern for a medical conversation benchmark. Do you have any plans to calibrate these subjective metrics and annotations from clinical experts or physicians?

### Soundness
2

### Presentation
2

### Contribution
2
