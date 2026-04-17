# MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark

- Decision: Accept (Poster)
- Scores: 2, 4, 8

## Abstract
Arabic remains one of the most underrepresented languages in natural language processing research, particularly in medical applications, due to the limited availability of open-source data and benchmarks. The lack of resources hinders efforts to evaluate and advance the multilingual capabilities of Large Language Models (LLMs). In this paper, we introduce MedAraBench, a large-scale dataset consisting of Arabic multiple-choice question-answer pairs across various medical specialties. We constructed the dataset by manually digitizing a large repository of academic materials created by medical professionals in the Arabic-speaking region. We then conducted extensive preprocessing and split the dataset into training and test sets to support future research efforts in the area. To assess the quality of the data, we adopted two frameworks, namely expert human evaluation and LLM-as-a-judge. Our dataset is diverse and of high quality, spanning 19 specialties and five difficulty levels. For benchmarking purposes, we assessed the performance of sixteen state-of-the-art open-source and proprietary models, such as GPT-5, Gemini 2.0 Flash, and Claude 4-Sonnet. Our findings highlight the need for further domain-specific enhancements. We also explore QLoRA fine-tuning on LLaMa-3.1-8B-instruct to assess our dataset's viability. We release the dataset and evaluation scripts to broaden the diversity of medical data benchmarks, expand the scope of evaluation suites for LLMs, and enhance the multilingual capabilities of models for deployment in clinical settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MedAraBench, a large-scale Arabic medical QA benchmark of 24K multiple-choice questions across 19 specialties and five difficulty levels. The dataset was digitized from Arabic medical exams, manually filtered, and evaluated by human experts and through an “LLM-as-a-judge” framework. Eight LLMs (open-source and proprietary) are benchmarked in zero-shot settings. The goal is to provide a specialty-diverse Arabic medical QA benchmark for evaluating LLMs.

### Strengths
1. Significant manual data cleaning and digitization effort, which adds credibility and quality to the dataset.

2. Diverse specialty coverage and structured annotation across difficulty levels, ensuring representativeness within the medical domain.

3. Inclusion of a human expert evaluation component, which is commendable and adds qualitative depth to the study.

4. Contributes to Arabic NLP, a domain with limited existing benchmarks and resources.

### Weaknesses
1. Unjustified selection of evaluator LLMs (Section 3.2.2)
The paper provides no justification for the selection of the three LLMs used as evaluators in the LLM-as-a-judge setup. There is no discussion of why these particular models were chosen, nor any rationale for excluding medical or Arabic-specialized LLMs, such as BiMediX (arabic+medical) or Fanar(arabic) or medgemma(medical+multilingual) etc .
 A more rigorous approach would have been to compare multiple candidate evaluators and measure their correlation with human expert scores (e.g., Pearson or Spearman coefficients) to identify which LLM aligns best with human judgment.

2. Missing mention of existing Arabic medical benchmarks (Table 1, lines 058–059). 
The comparison table omits BiMediX (arXiv:2402.13253), which already provides Arabic translations of MedQA and MedMCQA.

3. Unsupported validation and overstated conclusions (lines 360–362, Table D2)
The conclusion that “ the potential of LLMs to be used for data quality evaluation in the medical domain” (lines 362–363) is overstated and empirically unsupported. The only evidence presented is a superficial similarity in average scores between human and LLM evaluations (Table D2). However, this does not constitute proof of agreement or reliability.
Several methodological issues invalidate the comparison:
- Different evaluation scales: Human experts used a binary high/low rating, while LLMs used a 1–5 Likert scale, making numerical averages non-comparable.
- Different samples: Humans and LLMs did not evaluate the same subset of data
- No agreement metrics: No statistical measure of correlation or agreement between human evaluators and the LLM judges (e.g., Pearson, Spearman, or Cohen’s κ) is reported. 
- Moreover, the statement in lines 360–362 that : “While they are not directly comparable due to varying evaluation scales, we note that the results of LLM-as-a-judge and expert quality evaluation are comparable.” is internally contradictory comparability cannot be claimed if the scales and samples differ.
The clause “pending further alignment with medical standards” implicitly acknowledges this weakness, but does not substitute for empirical validation.

4. Unjustified use of the Likert 1–5 scale 
The 1–5 scale is applied without defining intermediate values (2–4) in the prompt (Appendix C), and no rationale is provided for using a 5-point scale instead of a binary one matching the human evaluation. This undermines comparability and interpretability.

5. Absence of medical Arabic LLM baselines (Section 3.3)
Although the work benchmarks several proprietary and open models (GPT-5, Gemini, Claude, etc.), no Arabic or medical LLMs are tested, despite the availability of models like BiMediX (Arabic and medical), medgemma (medical and multilingual) etc.

6. Invalid cross-benchmark and cross-model comparisons and flawed analysis of model progress (discussion lines 384–394; Table D1)
In the discussion (lines 384–394), the authors claim to observe “evolution of model performance across generations”.
 This analysis is methodologically invalid, as it compares different models on different benchmarks (MedArabiQ vs. MedAraBench).
 Because neither the models nor the datasets are constant, performance differences cannot be attributed to either factor.
Table D1 seems intended to show that MedAraBench might provide a more informative or challenging evaluation, but the comparison is not correctly designed, and the caption does not clarify what the table represents.
 The two columns correspond to different model generations, making them not directly comparable.
This is a missed opportunity:
 If the authors had evaluated the same models on both MedArabiQ and MedAraBench, they could have shown whether the new benchmark is more challenging and thus more valuable.
 Alternatively, testing different generations of the same model family (e.g., GPT-4 vs GPT-5) on MedAraBench would have allowed a valid analysis of progress over time.
 As it stands, the setup conflates dataset variation with model advancement, so conclusions about “model evolution” are unsupported.
 Both the discussion and Table D1 should be revised: either clarify that the comparison is descriptive or conduct controlled, same-model evaluations.

7. Invalid comparison of models (lines 365–367)
The authors conclude that “proprietary models outperform open-source models,” yet the proprietary models are orders of magnitude larger than open ones. Such comparisons are meaningless without controlling for scale.

8. Dataset imbalance (Figure 2a, Table A2)
Over 56 % (Figure 2a, Table A2) of questions are Year-1 level and only 5 % Year-5, resulting in a dataset dominated by basic-science items. This imbalance likely makes the benchmark less challenging and limits its capacity to assess advanced reasoning.

9. Suspicious perfect accuracies without explanation (Table 3)
In Table 3, several models report perfect accuracies (1.0) for the ABCDEF configuration, while scores on other subsets remain between 0.55–0.77.
This sudden jump to perfect accuracy across models is highly suspicious and atypical for medical QA tasks.
No explanation or investigation is provided. The authors should have clarified whether the ABCDEF subset:
- contains very few items (inflating accuracy), 
- includes only Year-1 questions (simpler), or 
- whether the addition of letter choices (A–F) helped models guess the correct answer (e.g., positional or formatting cues).
Without such clarification, the results appear unreliable and raise concerns about evaluation validity.

10. Limited novelty and under-utilization of the dataset
While the dataset is valuable for Arabic medical NLP, the contribution is incremental rather than conceptual, there is no new evaluation framework or modeling insight beyond prior work (MedArabiQ). The paper advertises ~24K MCQs, yet only ~4.9K test items are actually used in experiments; the ~20K training split is never explored (no fine-tuning, few-shot, or human/LLM evaluation on train). As a result, the empirical scope is limited to the test set, leaving the benchmark largely under-utilized. The most tangible contribution remains the digitization and manual curation of Arabic medical MCQs.

### Questions
Questions:

1. Did you test Arabic or medical-specialized LLMs as potential judges?

2. How is the “Average (Fraction of 5)” metric in Table 3 calculated?

3. How do explain the fact that several models reach perfect (1.0) accuracy in the ABCDEF configuration?

4. What does Table D1 intend to represent, benchmark comparison or model evolution?

5. Could you share the prompts used to evaluate the benchmarked models, including input format, language setup, and answer extraction method?

Remarks:

1. Misplaced or unclear citation (line 099).
The citation to the GPT-4 technical report (Achiam et al., 2023) does not logically connect to the preceding sentence. If the authors meant to refer to GPT-4 being evaluated on translated benchmarks, the sentence should be rephrased for clarity.

2. Missing cross-reference (lines 170–171)
In Section 3.1, methodological details are discussed without referencing the appropriate subsection (Section 4.1), reducing readability.

3. Lack of explanation for evaluation platform (line 215)
The authors mention that expert evaluations were conducted using Qualtrics, yet they do not explain what it is nor provide a footnote or citation.

4. Incomplete sentence (line 331)

5.  Invalid link (line 452): 
The repository link (https://anony-mous.4open.science/r/medarabench-3BE4/) is inaccessible.

### Soundness
1

### Presentation
2

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
This paper introduces MedAraBench, a benchmark for evaluating LLMs on Arabic medical MCQ task. The dataset contains about 24k questions across 19 medical domains and 5 difficulty levels, manually digitized from Arabic medical school materials. The authors conduct both clinical expert evaluation and LLM-as-a-judge assessments to measure data quality, finding moderate agreement and generally acceptable accuracy. They then benchmark open-source and proprietary models like GPT-5, Gemini 2.0, and Claude 4, showing that proprietary models outperform open-source ones but still fall short of expert level accuracy. The paper provides a structured resource for testing Arabic medical tasks but is mainly limited to multiple-choice formats and zero-shot evaluations.

### Strengths
- The manual digitization and expert validation of data from non-digital academic sources shows significant effort and ensure the dataset’s authenticity and reliability.
- The dataset spans 19 medical specialties at various difficulty levels, offering a structured framework that supports fine grained evaluation of LLM performance across various domains of medical knowledge for the Arabic language.

### Weaknesses
- For validating data quality using LLM-as-a-judge, the authors employ GPT-4, Gemini 1.5 Pro, and Claude 3.5 Sonnet. However, there is no justification provided for selecting these specific models, Are they known to outperform others in Arabic understanding? Moreover, the prompt instructs the models to act as medical education expert but does not account for the Arabic language aspect of the task. The capability of these models in Arabic medical understanding needs further evaluation.
- The literature review and experimental comparisons are not comprehensive. Open-source models like BiMediX [1] have benchmarked Arabic medical tasks and have released translated and verified datasets. Additionally, compare with more medical open-source models like Apollo [2], Med42 [3], Meditron [4]. Proprietary models like Gemini 2.5 Pro [5] and Flash are also missing from the evaluations.
- If LLM as a judge is an automated framework to evaluate the quality of the proposed dataset. Extending this validation to the training set could help further filter and enhance the quality of the data. Currently, the validation is limited only to the test set.

[1] *Pieri, Sara, et al. "Bimedix: Bilingual medical mixture of experts llm." arXiv preprint arXiv:2402.13253 (2024).*

[2] *Wang, Xidong, et al. "Apollo: A lightweight multilingual medical LLM towards democratizing medical AI to 6B people." arXiv preprint arXiv:2403.03640 (2024).*

[3] *Christophe, Clément, et al. "Med42-v2: A suite of clinical llms." arXiv preprint arXiv:2408.06142 (2024).*

[4] *Chen, Zeming, et al. "Meditron-70b: Scaling medical pretraining for large language models." arXiv preprint arXiv:2311.16079 (2023).*

[5] *Comanici, Gheorghe, et al. "Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities." arXiv preprint arXiv:2507.06261 (2025)*

### Questions
Please address the above weaknesses. 
- How were the medical specialties determined for the data samples? Was this an automated process or done with the help of clinical experts?
- Line 197: “No Cueing: options do not provide clues to other answers.” Is it meant to be “Clueing”?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a benchmark for medical knowledge in Arabic. The dataset comprises multi-choice questions in 19 medical specialities, extracted and refined manually from exams and lecture notes of medical schools in the Arabic-speaking world, then quality-checked by experts and LLM-as-a-judge scheme. The paper presents benchmarking results for a number of SOTA LLMs.

### Strengths
The paper presents a useful resource for Arabic medical understanding, and benchmarking results for a number of SOTA models.

### Weaknesses
Some missing details about the construction of the dataset and the implementation of the benchmarking are mentioned in Questions below

### Questions
- How is it that Arabic is underrepresented in the medical domain because of its rich morphology and dialectal variation? And why the uneven linguistic landscape calls for medical LLMs? The motivation of this work in Sec.1 should be revised.

- The reference to Appendix A is missing in Sec. 3.1

- MedArabiQ is missing from Table 1

- How were lecture notes converted to meaningful MCQs? This is an important detail that deserves to be discussed.

- The LLMs chosen for benchmarking do not include any Arabic-focused model. Why not? The differences in the benchmarking results could be due to the models' inherent limitation in Arabic itself rather than in medical knowledge, where bigger, proprietary models have an edge, but this angle is not explored.

- Line 331 (Sec 4.3) is truncated.

- How big were the subsets of MCQ? Guessing from Fig. E3. subset ABCDEF has only 2 questions in the test set and 9 question overall, so it might be better not to be considered at all as a separate category.

- In Sec. 5 (Discussion) and Table D1, how can MCQ scores of different benchmarks be at all comparable?

- How did the authors homogenize medical terminology in their dataset given the lack of standardization in the Arabic medical domain?

- In Appendix A, how comes the average question length is just over 8 characters?

- How was the MCQ benchmarking implemented? e.g. using logit ranking or post-processing of model answer for choice character?

### Soundness
3

### Presentation
3

### Contribution
3
