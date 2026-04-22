# NurValues: Real-World Nursing Values Evaluation for Large Language Models in Clinical Context

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2

## Abstract
While LLMs have demonstrated medical knowledge and conversational ability, their deployment in clinical practice raises new risks: patients may place greater trust in LLM-generated responses than in nurses' professional judgments, potentially intensifying nurse–patient conflicts. Such risks highlight the urgent need of evaluating whether LLMs align with the core nursing values upheld by human nurses. This work introduces the first benchmark for nursing value alignment, consisting of five core value dimensions distilled from international nursing codes: _Altruism_, _Human Dignity_, _Integrity_, _Justice_, and _Professionalism_. We define two-level tasks on the benchmark, considering the two characteristics of emerging nurse–patient conflicts. The **Easy-Level** dataset consists of 2,200 value-aligned and value-violating instances, which are collected through a five-month longitudinal field study across three hospitals of varying tiers; The **Hard-Level** dataset is comprised of 2,200 dialogue-based instances that embed contextual cues and subtle misleading signals, which increase adversarial complexity and better reflect the subjectivity and bias of narrators in the context of emerging nurse-patient conflicts. We evaluate a total of 23 SoTA LLMs on their ability to align with nursing values, and find that general LLMs outperform medical ones, and _Justice_ is the hardest value dimension. As the first real-world benchmark for healthcare value alignment, NurValues provides novel insights into how LLMs navigate ethical challenges in clinician–patient interactions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces NurValues, a novel benchmark designed to evaluate the alignment of Large Language Models (LLMs) with core professional nursing values. The authors argue that as LLMs are integrated into clinical practice, they pose new risks, such as amplifying nurse-patient conflicts if their responses are misaligned with the ethical judgments of human nurses. To address this, the authors identified five core nursing value dimensions from international nursing codes: Altruism, Human Dignity, Integrity, Justice, and Professionalism. The benchmark is built from real-world data collected during a five-month ethnographic field study in three different-tier hospitals, resulting in 976 initial nursing behavior instances. This data was used to create a two-level benchmark. The Easy-Level dataset consists of 2,200 instances (1,100 real cases plus 1,100 LLM-generated counterfactuals) that require standard ethical judgments. Hard-level dataset consists of 2,200 dialogue-based instances derived from the Easy-level cases. These dialogues are adversarially complex, embedding contextual cues, narrator biases, and misleading signals to simulate real-world conflicts.

### Strengths
1. The benchmark's foundation in a five-month, multi-site ethnographic field study is a significant strength. This grounding in real-world nursing behaviors, rather than purely synthetic or crowdsourced scenarios, makes the dataset highly relevant and authentic.

2. The paper addresses a timely and critical gap. While other benchmarks test medical knowledge (e.g., MedQA) or general morality (e.g., ValueBench), NurValues is the first to focus specifically on the professional values of nursing, which is crucial for safe human-AI interaction in clinical settings.

3. The two-level (Easy/Hard) structure is very effective. The Hard-Level dataset is a particularly strong contribution, as it moves beyond simple statement evaluation to simulate the messy, subjective, and emotionally-laden narratives that LLMs will actually encounter in patient-facing applications.

### Weaknesses
1. The authors acknowledge this limitation, but it is a significant one. All data were collected from three hospitals in mainland China. Nurse-patient dynamics, ethical priorities (e.g., autonomy vs. beneficence), and communication norms vary dramatically across cultures. This limits the "universal" applicability of the findings and the benchmark itself without cross-cultural validation.

2. The Justice dimension, identified as the "hardest", is built on only 74 samples (3.36% of the dataset). While this may reflect real-world observational frequency, it is difficult to draw robust conclusions from such a small and imbalanced subset. The difficulty may be an artifact of the low sample size.

3. The five chosen values are foundational, but other critical nursing values, such as patient advocacy, accountability, and confidentiality, are not included. This limits the benchmark's scope in evaluating the full spectrum of nursing ethics.

### Questions
1. In Section 2.2 (Step 3) and Figure 2, you state you "leverage jailbreaking techniques for LLMs"  to create the Hard-Level dataset. Could you please elaborate on this? What specific techniques were used? How does "jailbreaking" (which is typically used to bypass safety filters) help in generating dialogues with "reasoning traps, biased framing, or plausible but misleading justification"?


2. In Table 1, could you please clarify the computation of the metrics? Specifically, what is Ma-F1? Is this the macro-average F1 score calculated across the two classes (align vs. violate) or the macro-average F1 score across the five different value dimensions?


3. Your finding that general LLMs outperform medical LLMs is fascinating. Your analysis in Appendix G suggests domain-knowledge fine-tuning is insufficient. Could you provide more insight here? Does this imply that current medical fine-tuning methods might inadvertently hinder ethical reasoning, or is it simply that ethical alignment is an orthogonal skill that is not being trained for?



4. The "Justice" dimension contains only 74 samples. How confident are you that its high difficulty  is a robust finding and not an artifact of this significant data imbalance? Did you perform any analysis (e.g., bootstrapping) to check the stability of this result?



5. Given that the data is exclusively from mainland China, what are your thoughts on how these results might generalize to Western healthcare systems (e.g., in the US or Europe), where communication norms and the legal emphasis on patient autonomy are different?


6. The related work section (Appendix B) does a good job of positioning NurValues against medical knowledge and general 3H value benchmarks. To strengthen the paper's claims about ethical and moral evaluation, I suggest you also situate it within the broader literature on moral reasoning benchmarks, such as the "MoralBench" paper [1]. This would provide a richer context for your contribution to the evaluation of LLM moral alignment.

Ref:

[1] Ji, Jianchao, et al. "Moralbench: Moral evaluation of llms." ACM SIGKDD Explorations Newsletter 27.1 (2025): 62-71.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce NurValues, a real-world evaluation for five nursing values (Altruism, Human Dignity, Integrity, Justice, and Professionalism). It has two tasks (Easy -- purely case description and Hard -- expanded dialogue from cases). The evaluation is to ask models to identify the values involved in the scenarios. Authors evaluate 23 LLMs in total to do pairwise comparison and performance comparisons in both levels.

### Strengths
S1: Good realistic dataset relating to clinical and nursing setting.
- Authors carefully curated a real-world and diverse dataset describing nursing events happened in different types of hospital (rural, urban etc) with five-month field observational studies and five licensed nurse experts.
- this resources could be very useful by being seed scenarios for many follow-up evaluations in this field

S2: (Claimed) First work in nursing field to explore important topic (value alignment).
- the value alignment topic is crucial to ensure good human-AI collaboration.
- while the benchmark is not challenging for some sota llms (e.g. claude 3.5), this benchmark is the first work exploring nursing values. this can encourage many follow-up works focusing on this topic in clinical field.

### Weaknesses
[minor] w1 Missing procedure details for deriving nursing values from principles/rules
- Since the study builds on the five nursing values summarized in Section 2, it is important to justify how the authors identified and distilled these values (see lines 145–146).
- recommend authors at least providing some examples of rules for each identified value. Ideally, they would release a dataset of rules/principles mapped with values to help readers and community to better understand.

[minor] w2 Lack of human validation on the context consistency between easy and hard (extended dialogue version of cases in easy)
-  topic consistency. Appendix D does not show the prompts used to obtain the 1–10 scale. Without this, the result may not be trustworthy.
-It remains unclear whether topic consistency alone suffices to justify that the Hard dialogue version stays on track, which is needed to justify comparing the Easy and Hard benchmarks.

[important] w3 Missing some models for testing in Table 1 to better support the arguments in discussion
- It is surprising that the hard-level benchmark is not challenging: Claude 3.5 Sonnet attains about 90%, while GPT-4o only reaches 38.05%, given that their performances are close on many benchmarks. I suggest authors can double check if models outputs have any formatting issues. If no issues, I'm curious to read some examples and error analysis on why gpt-4o did worse. Can authors specify the time version of gpt-4o they used in table 1? Also recommend authors to run a few more GPT models (e.g. latest version of GPT-4o, GPT-4.1).
- line 337-338 "This suggests that domain-specific fine-tuning improves clinical Q&A but not ethical reasoning,". I think it will be very interesting to see the analysis on the current reasoning models in this benchmark as well. And have the comparison between reasoning and non reasoning models in this clinical task.

[important] w4 Missing actionable insights for the community and connections to prior works on value-based evaluations
- Authors cited a couple of benchmark references, but I strongly recommend they consider some additional works that could inspire more interesting analyses and actionable insights for the community. Potential references and follow-ups:
- DailyDilemma (https://arxiv.org/abs/2410.02683).their system-prompt steerability experiments could be adapted to the nursing principles/rules to see if they improve model performance, enriching the CoT setting examined by authors.
- Works in values and AI safety: Emergent values https://arxiv.org/abs/2505.14633, LitmusValues https://arxiv.org/abs/2502.08640, Values in the wild https://arxiv.org/abs/2504.15236.

### Questions
Q1 line 108. why use openai o1 to generate

Q2 line 117-118: how do authors transform the case to dialogue formats?

Q3 how exactly authors identify and distill five value dimensions. can you describe the procedures of it?

Q4 line 252-253: what do you mean for jailbreaking techniques

Q5 Figure 2: altruism: 0; altruism: 1 -> what does it mean

minor:
line 212: avoid using "she" to represent the nurse expert
line 214: further details on human annotation procedure, please see App. C. => capatalize
line 309  receive a semantic similarity score ≥ . => what score? 9?
line 346: Ma-F1 ==> do you mean Macro-F1?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces NurValues, a bilingual benchmark designed to test whether LLMs uphold key nursing values in clinical communication. The dataset includes 7,635 Easy-Level dialogue-derived cases and 2,100 Hard-Level adversarial cases with labels across five value dimensions and three levels of ethical alignment. The authors evaluate multiple general and medical LLMs, in zero-shot and in-context learning settings, using accuracy, macro-F1 and McNemar tests. Results show sizeable gaps on adversarial items and that general LLMs often outperform medical LLMs on value-sensitive judgments.

### Strengths
* Clear problem framing. The benchmark is built around established nursing codes, providing strong domain grounding and clear construct definitions.

* Good quality, realistic data with adversarial challenge cases. Easy-Level cases come from real nurse–patient dialogues. Hard-Level role-play and counterfactuals probe failure modes that is often omitted in other datasets. This two-tier design improves ecological validity and adversarial robustness assessment.

* Careful annotation and reliability reporting. Reported inter-annotator agreement values shows a good practice  (however, number of samples used can potentially be increased, and a stratification for each sub-category of questions) [1].

* Novelty and Relevance.The work fills a gap between general-purpose benchmarks and clinical-safety datasets, expanding the landscape of value-alignment evaluation [2-4]

[1] Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for
categorical data. Biometrics, 33(1), 159–174. (https://academic.oup.com/biometrics/article-abstract/66/4/1185/7333578)
[2] Ren, Y., et al. (2024). ValueBench: Towards Comprehensively Evaluating Value
Orientations and Value Understanding in LLMs. ACL. NurValues Review 3 (https://arxiv.org/pdf/2406.04214)
[3] Zhao, W., et al. (2024). WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural Value Awareness of Language Models. LREC-COLING. (https://aclanthology.org/2024.lrec-main.1539.pdf)
[4] Huang, K., et al. (2024). FLAMES: Benchmarking Value Alignment of LLMs in Chinese. NAACL. (https://aclanthology.org/2024.naacl-long.256.pdf)

### Weaknesses
* Taxonomy coverage and balance. The benchmark focuses on five nursing value dimensions but several widely used nursing codes (e.g. privacy and confidentiality, advocacy, safety) seem to be missing [1-3].

* Adversarial data generation limitations. Most adversarial cases come from a single frontier model, which may introduce stylistic artefacts and attack-surface bias tied to that model (as shown by [4]). 

* Evaluation metrics could be richer. Accuracy and macro-F1 on imbalanced, ordinal-like labels may hide clinically relevant errors. 

* Limited comparisons to adjacent benchmarks. The paper cites but does not transfer-test or cross-validate on ValueBench, WorldValuesBench, FLAMES or medical-safety frameworks like MedSafetyBench. Such comparisons would clarify what NurValues uniquely captures in healthcare ethics.

[1] International Council of Nurses. (2021). ICN Code of Ethics for Nurses. International Council of Nurses. (https://www.icn.ch/sites/default/files/2023-06/ICN_Code-of-Ethics_EN_Web.pdf) (accessed: during review period ICLR’26)
[2] Nursing and Midwifery Council. (2024). The Code: Professional standards of practice and behaviour for nurses, midwives and nursing associates. NMC.(https://www.nmc.org.uk/standards/code/) (accessed: during review period
ICLR’26)
[3] American Nurses Association. (2025). The Code of Ethics for Nurses. ANA. https://codeofethics.ana.org/ (accessed: during review period ICLR’26)
[4] Wang, Y., et al. (2024). A survey on natural language counterfactual generation. EMNLP. (https://aclanthology.org/2024.findings-emnlp.276.pdf)

### Questions
1. Could you clarify the rationale for selecting only five value dimensions and omitting others commonly emphasised in nursing codes (e.g. privacy/confidentiality, advocacy, patient safety)? Would you be willing to add a brief justification or including these categories in future iterations?

2. Since adversarial cases rely heavily on a single frontier model, how do you mitigate potential stylistic bias or attack-surface overfitting? Would you consider multi-model generation or human-seeded adversarial prompts to strengthen robustness?

3. How does NurValues empirically differ from or complement ValueBench, WorldValuesBench, FLAMES or MedSafetyBench?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper explores the ability of LLMs to track nursing values from a situational summary. Authors conduct numerous evaluations on their chosen "values"

### Strengths
The authors describe various ablations or slices to evaluate the performance. It is commendable that the authors collected real world data and extensively annotated.

### Weaknesses
1. Significance:
- Why is such a benchmark required?
- Since all of the data situational not conversational, is the challenge different?

2. Novelty:
- There has been similar exploration (https://pmc.ncbi.nlm.nih.gov/articles/PMC12099337/, https://arxiv.org/pdf/2505.04152, https://arxiv.org/abs/2409.15188) around LLMs for care and clinician-patient interaction. Or other 
hospital agent (https://dl.acm.org/doi/10.1145/3699765, https://arxiv.org/pdf/2401.05654). Where does a benchmark like this add value? Authors could consider distinguishing the benchmark and positioning it better?
- Implications of how others might use the benchmark needs to be better addressed

3. Counterfactuls:
The validity of counterfactuals is unclear. For an alternate situation, did the manual coders also check for ecological validity (or practicality) of the new situation?

4. Choice of models:
The authors focus a lot on medical LLMs. Since the task is more trait or affective, this focus needs to be further justified to even consider why medical LLMs "may" encode this knowledge or one might want them to.

5. Metrics:
The annotation allows for multiple positive labels. How are the scores calculated in such cases?

### Questions
Included in weakness section

### Soundness
3

### Presentation
2

### Contribution
2
