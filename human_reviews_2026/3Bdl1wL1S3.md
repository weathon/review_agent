# Can LLMs Move Beyond Short Exchanges to Realistic Therapy Conversations?

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent incidents have revealed that large language models (LLMs) deployed in mental health contexts can generate unsafe guidance, including reports of chatbots encouraging self-harm. Such risks highlight the urgent need for rigorous, clinically valid evaluation before integration into care. However, existing benchmarks remain inadequate: 1) they rely on synthetic or weakly validated data, undermining clinical reliability; 2) they reduce counseling to isolated QA or single-turn tasks, overlooking the extended, adaptive nature of real interactions; and 3) they rarely capture the formal therapeutic structure of sessions. These gaps risk overestimating LLM competence and obscuring safety-critical failures. To address this, we present \textbf{CareBench-CBT}, the largest clinically validated benchmark for CBT-based counseling, unifying thousands of expert-curated items, realistic multi-turn dialogues, and formal CBT structural alignment. Evaluating 18 state-of-the-art LLMs reveals consistent gaps: high scores on public QA degrade under expert rephrasing, vignette reasoning remains difficult, and dialogue competence falls well below human counselors. Recognizing that long-horizon context management limits multi-turn performance, we further propose Hierarchical Therapy Memory (HTM), a training-free inference framework that structures dialogue history into global states and episodic summaries. HTM consistently improves session-level therapeutic coherence while reducing computational latency. Together, CareBench-CBT and HTM provide a rigorous foundation for advancing the safe and responsible integration of LLMs into mental health care. All code and data are released in the Supplementary Materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CareBench-CBT, a large-scale, clinically validated benchmark designed to evaluate the counseling competence of LLMs based on Cognitive Behavioral Therapy (CBT). The authors identify critical gaps in existing benchmarks, namely their reliance on synthetic or single-turn QA data and their failure to capture the formal structure of therapeutic conversations.

CareBench-CBT addresses this by unifying three complementary components:
- Knowledge-based QA: 640 items (430 public, 210 expert-rephrased "private" items) to test factual CBT knowledge and robustness against "teaching to the test."
- Case vignette classification: 60 expert-authored clinical vignettes to evaluate diagnostic and clinical reasoning abilities.
- Multi-turn counseling dialogues: 256 complete, anonymized counseling sessions (totaling 7,680 turns) annotated by 21 licensed professionals. These dialogues are aligned with the formal structure of CBT.

The authors use this benchmark to evaluate 18 state-of-the-art LLMs. Key findings show that while models perform well on public QA, their scores degrade significantly (avg. 21-point drop) on the private, rephrased set. Furthermore, nuanced clinical reasoning (vignette classification) remains a major challenge for most models, and even the best-performing LLMs fall substantially short of human counselor performance in long-form, multi-turn dialogue competence.

### Strengths
- The benchmark's core strength is its high-quality, clinically-grounded data. 21 licensed professionals validated all items using robust protocols (CTS-R, MITI) and achieved high inter-rater reliability (Cohen's K > 0.89). This rigor is applied to evaluating "process-level competence" using 256 complete, multi-turn dialogues aligned with formal CBT structure.

- The design of the Knowledge-based QA task is a significant contribution. By creating a "private" set of 210 expert-rephrased items and comparing performance against the 430 public items, the authors provide clear, quantitative evidence of performance degradation. The average 21-point accuracy drop effectively demonstrates that high scores on public benchmarks are often a result of memorization, not robust, generalizable clinical knowledge.

### Weaknesses
- **Overly Broad Title and Limited Scope**: The paper's title is misleading. The benchmark is exclusively focused on CBT. While CBT is a widely-used modality, its conversational structure is not representative of all mental health counseling (e.g., psychodynamic, humanistic, DBT). The benchmark's findings and utility are therefore limited to CBT and cannot be generalized to "realistic therapy conversations" as a whole, as the title implies.

- **Data Provenance Ambiguity**: The paper states the 256 dialogues were "provided by 21 licensed mental health counselors" and "collected and anonymized from authentic therapeutic sessions." However, the exact source and nature of these sessions are not detailed. This ambiguity makes it difficult to assess potential sampling biases (e.g., in patient demographics, severity, or cultural diversity), which is a critical detail for a benchmark claiming high clinical validity.

- **Circular Human Baseline Evaluation**: The main "Human" baseline score in the dialogue evaluation (Figure 5) is derived from the LLM-as-a-Judge ensemble scoring the ground-truth human therapist responses. This means the benchmark used to demonstrate the "gap" between LLMs and humans is itself an LLM-generated metric. This is a circular methodology. A more robust baseline would involve human experts scoring the human therapist responses to set a "gold standard" bar, rather than relying on an LLM to judge a human to set the bar for other LLMs.

### Questions
Can the authors provide more detail on the source of the 256 authentic counseling dialogues? Were they from a specific set of clinics, training archives, or multiple private practices? This information is crucial for understanding potential sampling biases (e.g., in patient demographics, problem severity, or cultural diversity).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors introduce **CareBench-CBT**, a clinically grounded benchmark for evaluating LLMs in mental-health counseling that addresses three gaps they identify: synthetic data, lack of long-form interaction, and absence of formal therapeutic structure.  CareBench-CBT combines 8,142 curated items, validated by 21 licensed professionals, with 256 counseling sessions (≈7,680 turns).

The benchmark spans three task types: (1) **knowledge-based QA** (mix of public items and clinician-rephrased items), (2) **case-vignette classification**, and (3) **multi-turn counseling dialogues**.  Evaluation for QA/vignettes uses Accuracy and F1, while multi-turn dialogues use a two-stage framework: an **LLM-as-judge ensemble** (GPT-5, Gemini-2.5 Pro, Claude 4.1 Opus) scored with a clinician rubric to produce per-turn “Turnscore” and a holistic “Wholescore,” and a **targeted human validation** to confirm judge reliability.  

Across 18 models, the results show high scores on public QA items but a significant drop on clinician-rephrased items, indicating an overreliance on web-familiar phrasing. In multi-turn counseling, all models fall short of human counselors, with deficits in empathy, pacing, and structured interventions, despite reasonable factual accuracy.  The authors position CareBench-CBT as, to their knowledge, the largest clinically validated, CBT-structured, multi-turn benchmark, and they release the dataset and code.

### Strengths
This paper discusses a critical and essential topic: Mental health and LLM, which not many researchers explore.
Strengths:

1- Clinically validated CBT benchmark that pairs authentic, multi-turn therapy sessions (256 sessions; ~7,680 turns) with structured CBT process labels and clinician-reviewed single-turn tasks—explicitly targeting process-level counseling competence rather than only factual QA. 

2- Two multi-turn evaluation protocols (model history, human history) plus an LLM-judge ensemble validated against human raters—meant to provide a scalable yet clinically anchored assessment. 

3- Quality-assured data pipeline with inter-rater reliability, content validity, and behavioral audits (CTS-R, MITI), and item-level QA cards—positioned as raising the clinical fidelity bar for mental-health LLM evaluation.

4- Broad task coverage across knowledge QA, case vignette reasoning, and multi-turn dialogues that captures complementary capabilities.

5- Transparent positioning and research-use release that support replication and follow-on work.

6- One of the first to propose a three-dimensional data framework, filling a critical methodological gap in mental health and AI.

### Weaknesses
* Reliance on proprietary LLMs as judges risks style bias toward those model families (using the same family to generate results and as a judge); rubric anchoring, adjudication procedures, and resolution of judge disagreements are not fully documented.
* “Largest” claim must remain explicitly scoped to largest with clinician validation and multi-turn CBT structure, so readers do not conflate scope with raw size; by raw size, larger public sets exist, such as Psy-Insight dataset and HOPE dataset.
* Inclusion of unpublished or anonymous datasets in comparison tables reduces clarity and verifiability; either justify their presence or remove them until a stable citation exists (MentalBench-10).
* Research ethics and governance details are under-specified; the paper should state IRB or REB approval status, consent basis, de-identification method, data retention, human recruitment, and access controls
* Appendix examples are long and unbalanced across models; provide a consistent number of representative dialogues per model or explain the selection rationale so it's clear.
* Model responses in showcased dialogues are longer than human references, which is a major drawback. You should enforce matched token budgets, decoding settings, and prompt templates to enable fair comparisons.
* Provenance of multi-turn conversations needs clearer specification; explicitly detail sources, authoring process, de-identification steps, licensing terms, and any transformations.
* Human-grounding for the evaluation is insufficient for validating LLM-as-judge; include 35–40 full human-rated dialogues per model to test judge–human equivalence, enable calibration, and quantify residual bias. In general, an LLM as a judge to evaluate the results is not sufficient in this field.
* The researchers in this field increasingly agree that domain-specific metrics (e.g., empathy, safety, appropriateness) are more meaningful than generic scores like accuracy or F1. The current work does not clearly operationalize or report such customized therapeutic metrics; it should state whether these were considered, how they were defined and validated, and why they were (or were not) used alongside— or instead of—generic classification metrics.
* Section 5.4 is insufficiently explained and remains unclear.
* In Table 1, you mention that the  8,142 conversations are annotated by Experts. Can you clarify this part?

### Questions
Please review the Weaknesses and provide an answer for each point.

### Soundness
4

### Presentation
3

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
This paper introduces CareBench-CBT, a validated benchmark by 21 licensed practitioners for the CBT-based counseling. It involves three tasks (knowledge based QA, case vignette classification and multi-turn counseling dialogues). In multi-turn counseling dialogues, authors developed two settings (model-based history and human-based history). They evaluated 18 SOTA LLMs to show that models perform worse on vignette reasoning and dialogue competence than human counselors.

### Strengths
S1: Developed multi-turn therapy dialogue and knowledge QA dataset. It is a huge effort to collaborate with many licensed practitioners. This dataset is definitely to be useful for the CBT therapy evaluation and training for AI.

S2: Authors aim to address very important topic and loophole (i.e. whether AI can create realistic conversation) existed in many existing evaluations for AI mental Heath chatbot. 

S3: Carefully design the human validation pipelines on many elements and parts during the dataset construction. We should definitely encourage people/researchers to focus on the data quality that they collected when constructing/publishing dataset/benchmark work, rather than just showcasing a model training/finetuning technique to boost some scores in their built benchmarks.

### Weaknesses
[important] W1 Potential flaw on the methodology for multi-turn counseling dialogues evaluation
- I am not entirely sure if I understand the evaluation methodology: I assume authors collected some real-world counselling conversations. And then they developed two settings (model-based history to simulate the therapist; and human therapist from the collected dialogues). For both settings, the same client utterences ($u_x$ from (1) and (2) in p.6) from the collected dialogues were used. It seems to have potential flaws in the model-based history since what client says is independent of the model gives during evaluation. The conversation may not be consistent in content -- e.g. may even talking completely different topics when the conversations go for many turns. There may be unfair comparison and assessment on LLMs performances if the simulated conversation does not coherent in content.
- if the client utterences are dependent of models' responses, perhaps authors need to change the notation in formula (1) and (2) to better clarify. Another potential issue (that need human validation) is to ensure the client (if it is also simulated by LLM) is consistent to present the topic/mental health issue during the conversation.
- this makes me uncertain on the performances/results shown in section 5.3


[important] W2 Wrong choices of statistics in section 5.4
- Authors use spearman correlation to show the consistency between llm-as-a-judge and human ratings to justify the validity of judge model use in evaluation
- However, the score calculated by spearman correlation may not be good indicator since it is less sensitive to outliers. More importantly, the models' performances are likely clustered since model-model is likely to have higher similarity than model-human performance (also shown in fig 6 b) and c)). The score calculated is largely influenced by the judge models performances themselves rather than the difference between judge model and human (which is what authors claim to prove). 
- A better statistic could be using the F1 score (treating human rating as gold labels for each dialogue)
e.g., in appendix p,18, authors mentioned they have used three models (GPT-5, Claude-4.1 Opus, and Gemini 2.5 Pro) to independently evaluate the counseling response. Can you provide a f1 score (treating human rating as gold labels) for each model and also their interrater score (kappa alpha) between models? 
- I'm also wondering if authors provide the interater score on the human rating labels for this particular judge dataset (I may miss it somewhere).

[minor] W3 Demonstrate examples in main text & image annotation for better readability of paper
- I appreciate authors provide many details (especially the QA/human validation and dataset collection) in main text and also many evaluation examples in appendix. However, I found they are a bit overloaded and I am not sure how to understand those before actually introducing the data. Authors can consider restructure the paper to make it more readable. I think it will be even better if authors can provide one/two-line examples, especially during discussion. 
- e.g. line 369-370: public items vs. private items
- also the figure 4, can authors make the models name to be aligned in the same direction (rather than having some to be upside-down e.g. llama -3.1-8b)
- the figure 6: the model names are really small.

### Questions
[minor]
Q1 line 199-200, any examples for the systematcially restructured the question wording to prevent models from replying memorization

Q2 line 214-215,  "256 complete therapist–client counseling interviews provided by 21 licensed mental health counselors, totaling 7,680 conversational turns". Do the interviews match with 60 anonymized case vignettes mental health counselors provided? 

Q3: line 216-217, To ensure clinical fidelity and quality, every full session underwent peer cross-review within the counselor cohort. Can authors provide more details on the cross-review apart from the a.2 e.g., the instruction provided for counselor?

Q4 line 235: normalized multi-turn transcripts. what does normalized mean?

Q5 line 240-242: the communication quality is verified through random audits (20% of dialogues): how are the Fleiss' K between the two human verifiers? Do you only remove data that do not fulfill the inter-rater agreement within the 20% subset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CareBench-CBT, a benchmark intended to evaluate large language models in clinically grounded, CBT-aligned settings. It spans three task types: (1) knowledge-based QA drawn from public sources and expert-rephrased items, (2) case vignette classification mapped to CBT-relevant categories, and (3) multi-turn counseling dialogues collected and anonymized from clinician-provided sessions with structure aligned to CBT phases (rapport, exploration, intervention, closure). The dataset is described as clinically validated through inter-rater reliability, content validity indices, and behavioral audits (CTS-R and MITI). The authors evaluate 18 models and report: strong model performance on public QA items but a sharp drop on expert-rephrased items; modest accuracy on vignette classification; and substantially lower multi-turn dialogue scores relative to human counselors. They argue the benchmark addresses gaps in prior work by emphasizing data reliability, multi-turn realism, and formal therapeutic structure.

### Strengths
- The focus on multi-turn dialogues with CBT structure and the attempt to quantify empathy, therapeutic alignment, and intervention quality are important and timely.
- Clear articulation of why standard single-turn QA benchmarks are insufficient for counseling scenarios, and why process-level metrics matter.
- Inclusion of two evaluation protocols for dialogues (model-based vs human-based history) is a thoughtful way to separate error accumulation from intrinsic per-turn competence.
- The use of inter-rater reliability, content validity indices, and behavioral scales (CTS-R, MITI) reflects awareness of clinical measurement standards.
- Comparative results across 18 models provide a useful snapshot of current capabilities and gaps.

### Weaknesses
- Content validity threshold inconsistency. Appendix A.4 (lines 846–849) sets S-CVI/Ave ≥ 0.90 as the standard, but Appendix A.5 (lines 879–880) treats S-CVI/Ave = 0.89 as sufficient by referencing a 0.78 cutoff. This weakens measurement rigor; a single, predeclared criterion should be used and interpreted consistently.
- Ambiguity in ICD-11 anchoring for non-diagnostic categories. Section 3.3 (lines 228–229) states all diagnostic and category assignments are anchored to ICD-11, yet Appendix A.5 (lines 885–914; 906–914) includes non-diagnostic/problem-focused labels (e.g., Academic Overload, Poor Time Management). The scope and mapping of these categories need clarification.
- Implausible effect sizes. Section 5.1 (lines 370–372) reports Cohen’s d ≈ 11.6 (Acc) and 12.8 (F1) for the public-to-private drop, which is unrealistically large given typical scales and likely indicates a calculation or scaling error.

### Questions
- Which content validity standard is authoritative? Please justify using S-CVI/Ave ≥ 0.90 versus ≥ 0.78, and under the chosen criterion, recompute and report S-CVI/Ave and item-level I-CVI with the number of experts and rating distributions (lines 846–849, 879–880).
- How are non-diagnostic categories anchored to ICD-11? Please provide a mapping of diagnostic labels to ICD-11 codes and a clear policy for non-diagnostic/problem-focused categories (source taxonomy, inclusion criteria, and any bridging to ICD-11) (lines 228–229, 885–914, 906–914).
- How exactly was Cohen’s d computed? Please specify the formula (e.g., pooled SD), sample sizes, means/SDs for Acc and F1, the measurement scale used, and re-report corrected effect sizes if applicable (lines 370–372).
- If S-CVI/Ave = 0.89 is below the prespecified ≥ 0.90 threshold, what remediation is planned (e.g., item revision, additional expert rounds, expanded sampling), and how would this affect conclusions?
- Will you release de-identified raw model outputs and human rating sheets to allow independent recomputation of effect sizes and validity indices, facilitating reproducibility?

### Soundness
2

### Presentation
3

### Contribution
3
