# ProbMedTOD: A Bayesian Network Guided Task-Oriented Dialogue System for Patient History Taking

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Task-oriented dialogue (TOD) systems for patient history-taking improve clinical workflow efficiency by collecting key diagnostic information. Most data-driven approaches for this rely on large language models (LLMs) and mimic fast, intuitive System 1 thinking. In contrast, clinicians typically reason about potential diagnoses and use that to guide the dialog.

To bridge this gap, we propose ProbMedTOD, a TOD system that combines the conversational abilities of LLMs with the probabilistic reasoning of a disease-symptom Bayesian Network (BayesNet). At each turn, ProbMedTOD extracts information from patient utterances, updates its diagnostic hypothesis over a set of potential principal diagnoses via Bayesian inference, and generates the next question using a supervised policy LLM trained on dialogue data.
The BayesNet structure is programmatically constructed from clinical documents, while its parameters are inferred automatically via self-consistent prompting of an LLM, removing the need for expert-labeled data.

We develop a patient simulator that uses patient profiles informed by the dialogue context and engages in realistic end-to-end interactions with the system, enabling evaluation of dialogue-level success. ProbMedTOD significantly outperforms LLM and retrieval-based baseline in next-question prediction and dialogue-level success, obtaining ~20 pt MRR improvement in simulation experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ProbMedTOD, a task-oriented medical dialogue system that combines an explicit disease-symptom Bayesian Network with an LLM-based policy. At each turn, an NLU module extracts structured symptoms, a Noisy-OR BayesNet updates posterior probabilities over candidate principal diagnoses, and the policy LLM uses those posteriors together with the dialogue state to choose the next question. BayesNet parameters, namely disease priors and symptom likelihoods, are estimated directly by an LLM via self-consistency prompting, which avoids reliance on patient-level data. For evaluation, the authors build a patient simulator that constructs profiles from MediTOD and MIMIC-IV, then uses an LLM to generate faithful patient responses for multi-turn interactions. Relative to supervised LLM baselines and a retrieval-augmented baseline, ProbMedTOD shows modest gains on turn-level Medical-F1 and Precision@K and larger gains on dialogue-level diagnostic ranking measured by MRR and Hit@K. An ablation that removes the BayesNet degrades performance, indicating that the explicit probabilistic layer contributes to the gains.

### Strengths
(a) The method injects explicit uncertainty modeling by feeding Bayesian posteriors to the policy, which encourages questions that reflect diagnostic likelihoods rather than treating hypotheses uniformly.

(b) The parameter estimation procedure is practical in data-constrained settings, since self-consistency with an LLM provides priors and likelihoods without patient-level annotation.

(c) The evaluation focuses on the diagnostic objective, using dialogue-level ranking metrics within a simulator that rewards information gathering rather than surface action matching.

(d) The BayesNet scaffold provides an interpretable structure whose posteriors can in principle be audited by clinicians.

### Weaknesses
(a) The novelty is limited relative to prior Bayesian inquiry frameworks. The main addition is the combination of a classical probabilistic scaffold with a modern LLM policy and LLM-based parameter estimation.

(b) Validation of LLM-estimated probabilities is thin. The paper does not compare priors and likelihoods with epidemiology or expert labels, and the reported sanity check is indirect.

(c) The single-disease assumption simplifies inference but departs from clinical reality with comorbidities, which can bias posteriors and question selection.

(d) The realism of the simulator is uncertain. LLM-generated patients may be overly cooperative or align too closely with profiles, which can inflate reported gains.

(e) Baselines are not sufficiently strong. Missing comparisons include information-gain or RL-based policies, RAG with CoT, and state-of-the-art closed models.

(f) There is no analysis of calibration, stopping rules, or escalation policies, all of which are crucial in medical applications.

(g) The engineering and inference overhead is not weighed against the relatively modest gains on turn-level metrics, and no cost-efficiency analysis is provided.

### Questions
(a) Can you validate the LLM-estimated priors and likelihoods against expert annotations or published statistics, even on a small subset, to bound estimation error.

(b) How would you extend inference beyond a single principal diagnosis, for example to a sparse multi-label posterior with approximate inference and k greater than one active diseases.

(c) Can you add stronger baselines such as information-gain policies, RAG with CoT (IRCoT, Self-RAG, ...), or a modern closed-model policy in order to clarify the performance gap.

(d) Do you have any human evaluation by clinicians of question quality and ordering or case-level diagnostic utility.

(e) How sensitive are the results to the parameter-estimation procedure, for example fewer self-consistency samples, different estimation LLMs, or uniform priors.

(f) Are posteriors calibrated, and do you have principled stopping or deferral rules when uncertainty remains high.

(g) How well does the approach transfer beyond pulmonary cases, for example on a held-out specialty or an out-of-distribution symptom mix.

(h) How robust are the results if patient responses are noisy or contradictory or generated by a different LLM, and do adversarial perturbations change your conclusions.

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
The paper proposes ProbMedTOD, a task-oriented dialogue system that integrates a disease–symptom Bayesian Network with an LLM-based dialogue policy for patient history-taking. The BayesNet updates diagnostic probabilities at each turn, which guide the question generation by the policy model. The approach is evaluated on MediTOD and MIMIC-IV datasets and shows improvements over RAG and LLM-only baselines in both turn-level and dialogue-level metrics. The method is conceptually interesting and well-motivated, though some aspects of the evaluation and interpretability remain underdeveloped.

### Strengths
# originality
The integration of explicit probabilistic reasoning into a dialogue policy framework is novel and well-motivated, bridging the gap between intuitive (System 1) and deliberative (System 2) reasoning in medical settings. The approach can be generalized to other domains involving diagnostic questioning or hypothesis-driven dialogue.

# quality

The methodology is sound and clearly implemented. The ablations demonstrate that the Bayesian component makes a meaningful contribution to the results. Although the BayesNet is not large, its incorporation provides a principled approach to uncertainty reasoning and enhances diagnostic focus.

# clarity
The paper is clear in its chosen architecture.

# significance
The problem considered is relevant and such a system can have an important impact on healthcare.

### Weaknesses
# originality
The paper could explore interpretability. This is mentioned in the paper but never explored. Additionally, other methods of encoding the output of the BayesNet should be tested, as LLMs are known for not being the best at handling numerical inputs. For example, an ordered list of the BN probabilities or semantic tags for a probability range should be tested.
 
# quality
Aspects missing from the evaluation: 
1. Evaluation with clinical experts would be beneficial.
1. An ablation assessing the impact of errors in the BayesNet.
1. Tables would benefit from the indication of stdevs
1. Impact of fine-tuning of the policy model on MediTOD. This information could be useful for other implementations.


The value of Med F1 for ProbMedTOD with the Qwen model in Table 3 is different from the homologous value in Table 1 (and largely so). Can the authors explain this?

# clarity
The paper is clear. Metrics should not be presented only as acronyms.


Future work could explore reinforcement learning or information–gain–based policies built on the same probabilistic backbone, and extend the system to multi-disease reasoning or multimodal evidence (e.g., lab tests). Expert-based validation of dialogues would be a key next step toward clinical applicability.

### Questions
See weaknesses

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
3

### Summary
This paper proposes PROBMEDTOD, a Bayesian Network–guided Task-Oriented Dialogue (TOD) system designed to assist with structured patient history taking. The system integrates a Bayesian Network (BN) for modeling probabilistic dependencies between clinical symptoms and conditions, guiding the dialogue agent’s question-asking strategy. Unlike conventional medical dialogue systems that rely solely on goal-oriented policies or LLMs, PROBMEDTOD aims to reduce unnecessary or redundant questioning, improve clinical relevance, and support more accurate patient assessment. Experiments on a simulated clinical dialogue environment show improvements in diagnostic questioning efficiency and task success rates over baselines.

### Strengths
- The idea of incorporating structured probabilistic reasoning into a medical TOD agent is theoretically sound and interesting for medical AI.
- he hybrid design leverages BN inference to guide the dialogue policy, which is well-justified for medical settings where causal and comorbidity relationships matter. This is conceptually sound and aligns with clinical exam logic, which often reasons over conditional probabilities rather than text similarity alone.
- The paper is generally easy to follow, with intuitive diagrams (e.g., the BN-TOD architecture figure) that help communicate the workflow.

### Weaknesses
- Evaluation is conducted only in a simulated setting. Although this is understandable due to data constraints, real-patient or clinician-in-the-loop evaluation would significantly strengthen the claims of clinical relevance.
- Baseline comparisons are limited. The evaluation omits comparisons with more recent LLM-based medical agents or retrieval-augmented systems. Given rapid advances in medical LLMs, comparison only with classic TOD baselines understates the challenge. The paper also does not justify the absence of stronger baselines.

### Questions
- How was the BN constructed and validated? Was any external clinical source (e.g., medical knowledge graph, clinician review) used to ensure correctness? 
- Why were modern LLM-based medical dialogue systems not included as baselines? Even small-scale comparisons would strengthen the contribution.
- Have you considered evaluating with medical professionals (even small-scale) to validate usefulness, safety, or alignment with history-taking best practices?

### Soundness
2

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
4

### Summary
The manuscript proposes ProbMedTOD, a hybrid medical task-oriented dialogue system integrating a disease-symptom Bayesian Network  for probabilistic diagnostic reasoning, and a supervised policy LLM conditioned on posterior disease probabilities. To evaluate diagnostic success, the authors introduce a patient simulator with cases from MediTOD and MIMIC-IV. Experiments show improvements over LLM baselines and RAG approaches in both turn-level metrics and diagnostic performance.

### Strengths
- The paper introduces a practical and efficient method to estimate BayesNet parameters using structured LLM prompts with self-consistency.
- Strong conceptual grounding: the paper clearly frames the problem as bridging the gap between System 1 intuitive LLM fluency and System 2 deliberate clinical reasoning.
- The system outperforms LLM and RAG baselines on both turn-level metrics and dialogue-level simulation metrics. Extra ablation study show the effectiveness of the BayesNet module.

### Weaknesses
- Limited clinical coverage and scalability concerns: the evaluation contains only to pulmonary MediTOD and small MIMIC-IV subset. This limitsgeneralizability to other medical fields which may have different reasoning patterns and disease/symptom structures. 
- In addition, the BayesNet structure was constructed via programmatic extraction followed by manual validation, where not reproducibly specified and the manual step also limits the system's applicability to new medical domains.
- Validation relies on comparing primary and other edges' likelihood averages, rather than benchmarking against an external ground truth using real-world epidemiological data or clinical expert consensus. The risk of the LLM encoding inaccurate or biased associations remains unanalyzed.
- Eq. (2) references “Equation 2.1” but equation numbering is inconsistent (Sec. 3).

### Questions
- The core inference simplification (Eq. 2)  is central of the method to make Bayesian inference tractable. However, patients in real-world clinical settings often have multiple concurrent diagnoses (comorbidities), which this model cannot naturally handle. Although model claims extensibility, is there any evidence to support this?
- The model sometimes asks non-informative questions (Sec. 7), is there any systematic analysis or mitigation strategy?

### Soundness
3

### Presentation
2

### Contribution
3
