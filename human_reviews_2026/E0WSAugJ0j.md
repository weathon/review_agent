# LiveClin: A Live Clinical Benchmark without Leakage

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
The reliability of medical LLM evaluation is critically undermined by data contamination and knowledge obsolescence, leading to inflated scores on static benchmarks. To address these challenges, we introduce LiveClin, a live benchmark designed for the approximating real-world clinical practice. Built from contemporary, peer-reviewed case reports and updated biannually, LiveClin ensures clinical currency and resists data contamination. Using a verified AI–human workflow involving 239 physicians, we transform authentic patient cases into complex, multimodal evaluation scenarios that span the entire clinical pathway. The benchmark currently comprises 1,407 case reports and 6,605 questions. Our evaluation of 26 models on LiveClin reveals the profound difficulty of these real-world scenarios, with the top-performing model achieving a Case Accuracy of just 35.7\%. In benchmarking against human experts, Chief Physicians achieved the highest accuracy, followed closely by Attending Physicians, with both surpassing most models. LiveClin thus provides a continuously evolving, clinically grounded framework to guide the development of medical LLMs towards closing this gap and achieving greater reliability and real-world utility. Our data and code are publicly available at https://github.com/AQ-MedAI/LiveClin.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
LiveClin tackles the problem of LLM evaluation in healthcare settings. Recognizing the need for benchmarks that are dynamic, to prevent inflation of scores due to contamination, and multi-turn, to evaluate multi-stage clinical reasoning, the authors propose a benchmark they call LiveClin, a novel biannually updated clinical benchmark of over 1,000 multi-question cases. The authors build the benchmark from the last six months of PubMed case reports and evaluate in-depth across LLMs.

LiveClin is generated in a human-AI agentic workflow and constructed to cover comprehensive disease clusters. Evaluation is comprehensive and informative. The authors discover that while individual question accuracy can be high for most models, overall case accuracy is low and decreases with model updates. The detail of the dataset allows for an understanding of model performance by clinical area.

### Strengths
-	The benchmark is constructed to be very comprehensive across medical topics. The taxonomy is clear, well explained and well evaluated. 
-	The authors include a clear description and depiction of the benchmark generation process and evaluate with creative ablations.

### Weaknesses
-	There could be further discussion of the fact that the accuracy for specific questions is high for most models, but that total case accuracy is low. This is an interesting finding of the benchmark consistent with other claims that LLMs lack longitudinal reasoning.
-	It would also be helpful to argue or explain that cases from PubMed will be sufficient for evaluation. What kinds of cases are typically published in this forum? Does this bias the benchmark?
-	There are very few medical specific LLMs tested. This is surprising given the authors claim that domain specific models may be the solution to low overall performance by general models that decreases with newer versions.
-	The authors should compare to other recent medical reasoning benchmarks using case reports including: MedCaseReasoning Wu et al 2025, McDuff et al 2025 (NEJM CPC), and CaseReportBench Zhang et al 2025.

Additional notes (not a part of recommendation)
-	Line 116: Figure is missing an A in NARRATIVE.
-	Line 185: COnstruction
-	Line 266: experts is written twice.
-	Line 273: cut-off in figure could be improved.
-	Table 1: I think it should be “Cost ($).”
-	It is odd that the related work is in the appendix.
-	What is the horizontal line in Figure 7B?

### Questions
How will this benchmark be prevented from being contaminated if the data is on open access PubMed? There could be further flushing out of the live generation methods. Will old cases be dropped once they have become available for long enough for models to be trained on? Where will this benchmark be managed and who will have access? How will over 200 physicians be employed regularly to maintain this benchmark?

### Soundness
3

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
The paper’s primary work is developing LiveClin, a dynamic and contamination-resistant evaluation benchmark for medical large language models (LLMs), addressing the limitations of static benchmarks in clinical relevance and anti-leakage. Built on contemporary peer-reviewed case reports from PubMed Central (1,407 cases, 6,605 questions), LiveClin integrates multimodal data (e.g., CT scans, pathological slices, tables) and simulates the full clinical pathway—spanning initial assessment, diagnostic testing, treatment planning, and long-term management—to assess models’ sequential reasoning capabilities.
 The benchmark is validated through comprehensive testing of 26 mainstream models (proprietary, open-source general, open-source medical), revealing distinct reasoning weaknesses across model classes.

### Strengths
1. The overall logic is fairly clear, and the writing is relatively well-structured.
2. Built from the latest peer-reviewed clinical cases and updated biannually, it effectively mitigates data leakage and knowledge obsolescence issues that plague static benchmarks, ensuring long-term clinical relevance of evaluations.
3. It simulates the patient care process (from initial consultation to long-term management) and integrates diverse multimodal data (images, tables, etc.),  reflecting real-world clinical reasoning scenarios.
4. The AI-human collaborative (Generator-Critic-Judge) pipeline, combined with rigorous review by 239 physicians, balances clinical accuracy, construction efficiency, and question challenge—solving the trade-off between scalability and quality in medical benchmark development.

### Weaknesses
Major Comments

1.The benchmark is constructed using case reports from the first half of 2025 in the PubMed Central (PMC) Open Access subset. Could there still be potential data leakage risks for some newly released models such as GPT-5? 

2.Does multiple physicians verify the same piece of data? If yes, what is the inter-annotator agreement (e.g., Cohen’s Kappa coefficient) among different physicians? 

3.The paper proposes "updating the benchmark biannually" but lacks details on the specific cost and efficiency of the update process. Will all test data be completely replaced during updates? If so, is it necessary to retest all models entirely after replacement? 

4.No human baseline based on physicians’ performance is provided. 

5.It appears that a single prompt was used for testing with a temperature setting of 0, and no multiple rounds of testing were conducted. This may lead to accidental errors due to randomness. 

Minor Comments

1.Details about the review process involving 239 physicians are insufficient. What is the distribution of their professional fields? Is there a risk that some disease types lack review by specialized physicians? 

2.Although 41.9% of the questions require multimodal interpretation, relevant details are lacking. 

3.Some statements are overly absolute or exaggerated. For example, the claim that "the era of 'free lunch' is over" is based on only two model examples, and "faithful replication of clinical practice" overstates the benchmark’s alignment with real clinical scenarios.

### Questions
None

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
5

### Summary
The authors present a new dataset and a generation method to assess the medical knowledge and capabilities of LLMs. The method proposes to update the benchmark twice a year to ensure updated and uncontaminated evaluations. They generate multiple-choice questions based on open-access cases and evaluate 26 AI models on the generated benchmark. They show that models struggle to answer the questions and obtain low scores compared to commonly used evaluations such as MedQA.

### Strengths
Contamination and the rapid evolution of medical knowledge are significant concerns for the evaluation in the medical domain. This approach presents a solution to both of these issues. The dataset is sufficiently large, and the multistep approach is a welcome addition compared to previous evaluations that test zero-shot knowledge on a complete vignette. The dataset is also multimodal, integrating imaging, labs, and other signals. It was also validated by a large number of clinicians, which strengthens the method's validity.

The findings regarding model performance are interesting and demonstrate the need for more thorough validation for safe and effective clinical use.

### Weaknesses
While the method is solid, I am concerned by the reliance on case reports, as, by definition, case reports are published to communicate unusual or rare cases to the medical community. This reliance on unusual/rare cases induces a bias in the knowledge and reasoning capabilities of the models. I am also concerned about the lack of a physician baseline to compare the accuracy of models with what is expected of an attending physician.

The reported metric for case accuracy scores appears too strict and not representative of the models' actual capabilities, as a single error causes the models to obtain a score of 0 on that case. A rubric-based assessment would strengthen the evaluation and enhance the interpretability of mistakes and areas for improvement in these models.

The reliance on MCQs also weakens the benchmark, considering the identified limitations of this testing methodology.

### Questions
# Major concerns

1) The authors should at least discuss the limitations of using case reports that are likely not representative of clinical workflows. A subset containing more common cases would help clarify whether the errors occur due to out-of-distribution cases or if they result from intrinsic shortcomings of LLMs.

2) A baseline of physicians on a subset (100 cases), including residents and attendings, would help with the interpretation of the results. At the moment, 35% seems relatively low, but if attendings score 20% it would demonstrate that LLMs may already be ready for clinical decision support. As a physician myself, I would not be surprised if I obtained a low score due to the nature of the cases included.

3) A more balanced scoring methodology beyond simple accuracy would help to identify the issues. For instance, scoring based on the severity of the error, for example, suggesting the second-best exam should not carry the same weight as sending a patient with a STEMI home. HealthBench, for example, weights rubrics differently [1].

4) The reliance on MCQ should be acknowledged as a limitation, as it has been identified that LLMs exploit patterns, even more so when the question generator is also an LLM [2].

# Minor

1) The authors should discuss the biases of case reports that are likely not representative of medicine worldwide, as publications are very US-centric, with minimal case reports from low-resource settings. 

[1] HealthBench: Evaluating Large Language Models Towards Improved Human Health (Arora et al. Preprint 2025)

[2] Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine (Griot et al., ACL 2025)

### Soundness
2

### Presentation
4

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
This paper introduces LiveClin, a dynamic medical benchmark addressing data contamination and knowledge obsolescence through biannually updated case reports from PubMed Central. The benchmark comprises 1,407 cases with 6,605 questions spanning the entire clinical pathway, revealing that even top models achieve only 35.7% case accuracy, with distinct failure modes across model classes.

### Strengths
* Pilot study convincingly shows 10-point performance drop on post-cutoff data.

* The three-tier taxonomy (ICD-10 chapters, disease clusters, individual codes) enables multi-resolution analysis while ensuring broad disease representation.

* The 239-physician verification pipeline with both annotation and inspection phases demonstrates exceptional attention to clinical validity.

* The multimodal integration is high quality, where images are naturally embedded in clinical workflow.

* Reveals newer models don't always outperform predecessors; identifies distinct failure patterns.

### Weaknesses
* The ablation study shows AI generates more "challenging" questions (lower trivial ratio), but doesn't validate whether this difficulty stems from genuine clinical complexity or artifacts of the generation process.

* he zero-shot, conversational evaluation may disadvantage models not optimized for this specific format. It's possible that the performance differences reflect not clinical reasoning ability but the adaptation to the evaluation format. Adding few-shot experiments would help.

* Maintaining physician review for biannual updates seems resource-intensive. While the paper reports $42K for initial construction, the long-term sustainability of this approach remains unclear. More details on this would be helpful to strengthen this claim.

* Despite being a core motivation, the paper doesn't empirically demonstrate that its approach prevents contamination better than decontamination methods or how quickly new cases might enter training corpora.

### Questions
1. How do you ensure the case reports selected are representative of disease distributions in real clinical practice? Does collecting data from PMC result in higher probability of rare cases?

2. To support the claim of the paper better, can you demonstrate empirically that LiveClin remains contamination-free over time? For instance, tracking whether newly released cases appear in web crawls or model training data?

3. How does model performance correlate between LiveClin and real clinical decision-making tasks or other clinical benchmarks?

### Soundness
2

### Presentation
3

### Contribution
3
