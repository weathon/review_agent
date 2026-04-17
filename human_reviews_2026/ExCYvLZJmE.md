# When Case Gets Rare: A Retrieval Benchmark for Off-Guideline Medical Question Answering

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Across medical specialties, clinical practice is anchored in evidence-based guidelines that codify best studied diagnostic and treatment pathways. These pathways work well for the majority of patients but routinely fall short for the long tail of real-world care not covered by guidelines. Most medical large language models (LLMs), however, are trained to encode common, guideline-focused medical knowledge in their parameters. Current evaluations test models primarily on recalling and reasoning with this memorized content, often in multiple-choice settings. Given the fundamental importance of evidence-based reasoning in medicine, it is neither feasible nor reliable to depend on such memorization in practice. To address this gap, we introduce OGCaReBench, a long-form retrieval-focused benchmark aimed at evaluating LLMs at answering clinical questions that require going beyond typical guidelines. Extracted from published medical case reports and validated by medical professionals, OGCaReBench contains long-form clinical questions requiring free-text answers, providing a systematic framework for assessing open-ended medical reasoning in rare, case-based scenarios. Our experiments reveal that even the best-performing baseline (GPT-o3-mini) correctly answers only 51% of our benchmark with open-source models only reaching 36%. Augmenting the models with retrieved medical articles improves this performance to up to 75% (using GPT-5) highlighting the importance of evidence-grounding for real-world medical reasoning tasks. OGCaReBench thus establishes a foundation for benchmarking and advancing both general-purpose and medical language models to produce reliable answers in challenging clinical contexts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a new benchmark and dataset derived from published case reports. The benchmark is filtered by physicians to ensure validity and relevance to the task defined by the authors. The resulting evaluation is then used to obtain a baseline for a variety of models including closed and open models. The authors then augment the models by using RAG over the case reports dataset. Different embedding models are used and compared. Overall, this work shows that adding RAG improves the performance on the OGCAREBENCH evaluation compared to the models without RAG.

### Strengths
The dataset curated by the authors has merit and the question generation pipeline is robust. The experimental results are clear and demonstrate the relevance of RAG for improving performance on under represented data such as rare medical cases. Including multiple embedding models supports the need for adapted models and suggest that focusing on improving RAG may be a more viable approach than training LLMs for domain specialization.

### Weaknesses
The assumptions made by the authors on what the goals of care are and how clinicians practice medicine are questionable. Rare cases are by definition rare and often have non-specific presentations shared with more common diseases [1]. The assumption that the best next step for a known rare case is the best next step in a real-world encounter is incorrect. For example, if a 20 year old male presents to the ER with shortness of breath and imaging shows a pneumothorax, the optimal diagnostic step is not to look for FLCN genetic mutations whereas that would be the case if in a Birt Hogg Dube case report [2].

This misalignment raises questions about the applicability of a system focusing on rare cases. For rare kidney diseases, a biopsy is the gold standard to make the diagnosis [3], would using this system in real world settings suggest a biopsy for every patient presenting with proteinuria or altered kidney function? Does the system rely on the clinician's intuition of what is or isn't a rare case? What are the cost and outcome implications of a system proposing advanced interventions too quickly?

This misalignment adds to the existing disconnect between lab research and the reality of clinical practice. While the experience is interesting, I do not find the method, system, and results to be novel enough to compensate the lack of real-world relevance and insufficient framing as is.

Considering the baseline and comparison, having a baseline of classic inference is necessary but insufficient, to justify using this RAG approach, it should be compared to other RAG systems, GPT-5 with web search for example, deepsearch, and OpenEvidence would be more accurate comparisons.

The inclusion of OpenBioLLM is a major issue for me [4]. A model released without any information, paper or data description making bold claims of achieving SOTA should not be included in any scientific work especially given the results reported in this paper.

# References

[1] Rare inherited kidney diseases: challenges, opportunities, and perspectives (Devuyst et al. 2014)

[2] Birt-Hogg-Dube Syndrome (Crane et al. 2023)

[3] The Kidney, (Brenner., and Rector. 2019)

[4] aaditya/Llama3-OpenBioLLM-70B (2024)

### Questions
# Suggestions/Questions

1) Include in the baseline the performance of base models with web search usage (GPT-5 Search, Claude with web use, deepsearch).

2) I recommend removing OpenBioLLM from the paper and in general to avoid models from unknown sources without data/technical report.

3) Evaluate the performance of the system on common cases, does it make rare suggestions? Could you quantify the incidence of the cases in the evaluation dataset and quantify the pre- and post-test likelihood of the correct answer? In addition, I would like to see a benefit/risk assessment beyond accuracy. For example, missing causa equina syndrome has more impact than missing a birt hogg dube diagnosis. Likewise, performing a kidney biopsy on a healthy patient is more risky than a blood sample. Finding the optimal benefit/risk ratio is the primary objective in clinical practice, not accuracy [1].

4) An expert baseline on the benchmark would help put into perspective the performance of the system.

5) Was any method used to ensure the absence of contamination between the evaluation cases and the cases included in the RAG dataset?

# References

[1]  Comparing diagnostic tests on benefit-risk (Pennello et al., 2016)

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
3

### Summary
This paper introduces OGCAREBENCH, a benchmark designed to evaluate large language models (LLMs) on rare or off-guideline medical cases. The benchmark is constructed from over 53,000 open-access medical case reports and includes 235 curated question–answer pairs validated by physicians. The dataset focuses on long-form, retrieval-based question answering where guideline-based reasoning is insufficient. Experiments compare several general-purpose and medical LLMs—both with and without retrieval augmentation (RAG)—demonstrating that even advanced models like GPT-5 struggle without retrieval (≈45–50% accuracy) but achieve substantial gains (up to 75%) when augmented with relevant case report retrieval. The paper argues that reliable clinical LLMs must move beyond memorized knowledge and toward retrieval-grounded reasoning to handle real-world, rare patient scenarios.

### Strengths
**Novel focus on off-guideline cases**:
The paper highlights an important but underexplored problem—how LLMs perform when faced with rare, atypical medical scenarios not covered by standard clinical guidelines. This makes the benchmark highly relevant to the deployment of LLMs in clinical support settings.

**Robust dataset design and validation**:
The authors construct OGCAREBENCH using a clear multi-step pipeline involving filtering, controlled question modifications, and expert validation. The attention to medical plausibility and domain fidelity adds credibility to the dataset’s reliability.

**Comprehensive experimental evaluation**:
The paper thoroughly evaluates multiple general-purpose and domain-specific models under both baseline and retrieval-augmented conditions. The inclusion of 15 retrieval models, from BM25 to biomedical-specific retrievers, offers a valuable empirical contribution to RAG research in healthcare.

**Clear demonstration of retrieval importance**:
Results convincingly show that RAG significantly enhances reasoning accuracy for rare medical cases, underscoring a key insight: parametric memory alone is insufficient for safe and effective medical reasoning.

### Weaknesses
**Limited methodological novelty**:
The work primarily focuses on dataset creation and empirical benchmarking rather than proposing a new retrieval or reasoning framework. This makes it somewhat engineering-heavy and evaluation-oriented, which may not align well with ICLR’s focus on algorithmic or representational innovation.

**Scale and representativeness concerns**:
Despite using over 50,000 case reports as the retrieval corpus, the final benchmark contains only 235 validated instances. This relatively small size raises questions about statistical robustness and whether the benchmark adequately covers the diversity of real-world rare cases.

**Evaluation dependency on GPT-based judging**:
The reliance on GPT-4o as an automatic evaluator introduces potential bias and inconsistency in clinical correctness judgments. The limited physician cross-validation (45 samples) may not be sufficient to confirm reliability across all cases.

**Shallow analysis of model failure modes**:
While quantitative results are extensive, the paper lacks qualitative insights into why models fail on certain rare cases—whether due to retrieval errors, reasoning gaps, or hallucinated procedures. This limits the interpretability of the findings.

### Questions
**Venue suitability (ICLR relevance)**:
The contribution centers on dataset construction and empirical evaluation, not on learning mechanisms or model training. It may better fit NLP or medical informatics venues (e.g., ACL, EMNLP) rather than ICLR, which emphasizes theoretical and representational advances.

**Benchmark longevity and updateability**:
Since medical knowledge evolves rapidly, the benchmark may require frequent updates to remain relevant. The paper does not discuss mechanisms for versioning, reannotation, or handling outdated clinical knowledge.

**Clinical validation and real-world utility**:
While the dataset is well-validated at construction time (Section 3.1 Step 1~4), it remains unclear how the benchmark correlates with actual clinical decision-making outcomes. Without external validation in real medical workflows, practical impact remains speculative. Furthermore, the authors did not provide any information about the annotators (i.e., three physicians).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
**Problem & Motivation:**
The paper identifies a gap between the training of most medical LLMs and the demands of real-world clinical practice. Current medical LLMs are primarily trained on and evaluated against common, "guideline-focused" medical knowledge, often using multiple-choice question formats. This reliance on parametric memorization is insufficient for the "long tail" of clinical care, where physicians encounter rare or "off-guideline" cases not covered by standard pathways. In these scenarios, evidence-based reasoning, which requires dynamically consulting external sources (like case reports), is essential. The authors argue that current benchmarks fail to test this critical skill, rarely evaluating whether models can generate expert-level, long-form answers grounded in retrieved evidence for complex, rare cases.

**Method:**
To address this gap, the paper introduces OGCAREBENCH, a long-form, retrieval-focused benchmark designed to evaluate LLMs on realistic, "off-guideline" clinical questions derived from medical case reports. The dataset's creation involves a four-step, semi-automatic process:
- *Corpus & Case Filtering:* A retrieval corpus of 53,617 case reports was collected from PubMed Central. From this, a subset of 28,219 reports was filtered by excluding older reports (published ≤ 2022) and those with high citation counts (indicating the case may have become standard knowledge).
- *Q&A Extraction:* An LLM (GPT-4o) was used on a 1,100-report subset to extract a timeline of the case and its significant contribution (e.g., a novel diagnosis, rare treatment). A question-answer pair was then generated, with the question detailing the case up to the decision point and the answer being the contribution (the novel step taken).
- *Question Modification:* To simulate a realistic clinical scenario where a new patient resembles but is not identical to a published case, the extracted questions were modified by another LLM (Claude 4 Opus). This involved adding "distractors" like altered demographics, comorbidities, or synonymous medical terms, making the question distinct from the source text.
- *Physician Validation:* The final modified Q&A pairs were validated by physicians, who rated them on medical alignment and difficulty. Only pairs requiring expert-level knowledge (rated 4 or 5 out of 5) were retained, resulting in the final benchmark of 235 validated cases.

**Experimental Setup:**
The benchmark is evaluated in two settings:
- Baseline (Memorization): Models answer questions using only their parametric knowledge.
- RAG (Retrieval-Augmented): Models are provided with relevant case reports retrieved from the 53k corpus to ground their answers.
A mix of general-purpose (e.g., GPT-5, GPT-03-mini, Llama 3.3) and medical-specific (e.g., MedGemma, Llama 3-Med42) models were tested.

**Results & Findings:**
The results demonstrate the benchmark's effectiveness in highlighting the limitations of memorization.
Without retrieval, even the best-performing model (GPT-03-mini) answered only 51.5% of questions correctly. Open-source models were lower, with MedGemma at 36.2%. This confirms that current models have not memorized this long-tail, rare-case knowledge.
When augmented with retrieved case reports, performance increased significantly. The top-performing model (GPT-5 with RAG) achieved 75.3% accuracy.

### Strengths
- The task of answering rare, off-guideline clinical questions is highly relevant.
- Timeline extraction and question reformulation by presenting all procedures preceding the decision point.
- A broad spectrum of models—domain-specialized and not—is benchmarked, including 8 LLMs, 14 semantic retrievers, and BM25.

### Weaknesses
- *Limited methodological novelty.* The primary contribution is the use of case reports as a source, not the development of a novel benchmark methodology. The resource is constructed by filtering and sampling (with poor control) an existing PubMed Central corpus, applying simple semi-automatic LLM-based extraction, and performing (incomplete) manual verification.
- *Superficial comparison to existing work.* For a resource-centric paper, the comparison to the existing literature is insufficient. It lacks a detailed, quantitative comparison table positioning OGCAREBENCH against other benchmarks (especially those already using case reports) across key dimensions (e.g., scale, task, validation rigor). This makes it difficult to assess the true novelty or "delta" provided by this work.
- *Questionable sampling and unbalanced dataset.*
   - Sampling. A "pure random sampling" of 1,100 reports (from which only 235 are finalized) is inadequate. Stratified sampling (e.g., by specialty, contribution type) would have been more rigorous.
   - Scale. The final expert-verified dataset of 235 questions is extremely small.
   - Balance. The dataset is highly unbalanced, with 70% of cases coming from only two specialties, a distribution that does not reflect the source corpus. The authors also fail to quantify the distribution of "contribution types" (diagnosis, treatment, etc.), a dimension they claim is a key part of their novelty.
- *Critically insufficient expert validation.* The validation process, particularly for a sensitive medical domain, is inadequate and falls well below standard scientific practice.
   - The utility of case reports is justified anecdotally via "informal interviews with 10 physicians," lacking any structured evidence or detailed findings.
   - The "distractor" modifications are a delicate process where errors could invalidate the answer. The complete list of modification types is not reported in the main paper, only two examples are reported. The authors state this stage was verified by three physicians for only an unspecified subset of questions, not the entire dataset. The subset size, selection criteria, provided instructions, verification results, and inter-annotator agreement are not provided.
   - Some details about instructions and annotations criteria are provided only for step 4 but, again, they appear incomplete and poorly designed. It seems that each report has been evaluated by one physician only. One quality dimension only (realism). As for the 1-5 Likert scale, the authors only provide definitions for 1 (unrealistic) and 5 (realistic), leaving intermediate values (2, 3, 4) to subjective interpretation.
- *Over-reliance on LLM-as-a-Judge.* The primary evaluation metric is an "LLM-as-a-Judge" (GPT-4o) for equivalence. Physician validation of this judge was performed on a very small subsample (45/235). While the 93% agreement is noted, the process, again, lacks detail.
- *Predictable RAG findings due to circular experimental design.* A primary conclusion is that RAG enhances performance. Assuming a successful retrieval and considering the nature of the applied modifications in QA generation, this finding is mostly an anticipated consequence of the experimental setup, where the benchmark's questions were derived directly from the retrieval corpus.
- *Questionable novelty of QA format.* The paper criticizes multiple-choice question benchmarks (e.g., MedQA, MedMCQA, PubMedQA) while advocating for its long-form questions and open-ended answers. However, the "open-ended" answers are exceptionally short (29 tokens on average), resembling simple verbalizations of a multiple-choice correct option. This raises doubts about whether this format truly evaluates the "open-ended reasoning" the authors claim is necessary. As other researchers have already created open-ended versions of benchmarks like MedQA by verbalizing correct answers with and without LLMs, the QA format, as implemented, does not appear to be a significant novel identity factor for this resource.
- *Insufficient depth in model and retriever analysis.* A deeper breakdown of error modes (e.g., where RAG fails despite correct retrieval, or which case types defeat open-source models) would be valuable.
- *Missing technical details.* No prompt or context engineering discussion, LLM decoding strategies information (crucial for results interpretation), and any statistical significance tests to validate the results.
- *Presentation quality.* The manuscript's quality is limited, featuring low-resolution, non-vectorial figures with default Draw.io colors, poorly organized tables, inconsistent notation (e.g., use of commas for thousands), and repeated acronym definitions.
- *Missing license.* The authors state the dataset and code will be publicly released but do not specify the license, which is a critical omission for a resource paper.
- *Domain drift and dataset maintenance.* In Appendix, the authors recognize that the benchmark's relevance may erode as some rare cases become standardized, yet there is limited technical prescription for maintaining dataset relevance or tracking drift over time.

### Questions
- How robust is the LLM-based evaluation metric to changes in prompting, underlying judge model, or model drift over time?
- Can the authors expand the analysis of error modes—for example, cases where retrieval finds the correct document but LLMs misinterpret, or vice versa?
- What is the observed impact of distractor additions on model/retriever confusion rates? Can the authors share specific ablations or examples where distractors led to errors, to clarify how realistic modifications challenge current systems?
- What steps could be taken to dynamically update or maintain the relevance and challenge of rare-case benchmarks as some cases become incorporated into guidelines?

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
4

### Summary
This paper introduces OGCaReBench, i.e. a benchmark to evaluate LLM and RAG performance on questions derived from rare clinical case studies. To this end, a dataset is constructed semi-automatically by mining publicly available case reports, converting them into QA format and altering non-relevant parts of the report. A suite of LLMs and retrievers is evaluated, suggesting that LLMs alone struggle with the benchmark, as evaluated by a (validated) judge LLM, but retrieval improves performance - with access to original documents which the QA pair is derived from, accuracy is very high, suggesting that the problem is mostly that of retrieval.

### Strengths
The paper is well written, i think overall the research is largely well executed. I appreciate the validation of LLM results with human annotators.

### Weaknesses
I find no big weaknesses with the execution of the research, only a few remarks:

- it would be great to have more details regarding the protocols used for human validation (both the validity of the QA pairs as well as the LLM as a judge).

- It seems like the problem is mostly with retrieval - given that the larger LLMs have very big context windows, I'm not sure why the RAG experiments reported in table 7 stop at 5 documents. It would be good to see results if the context window is maxxed out for each LLM.

- Given the rather small dataset size, it would be good to have statistical significance reporting, in form of statistical significance tests as well as confidence intervals, to contextualise the amount of statistical uncertainty 

That being said, I do question the motivation and therefore the overall contribution of the paper a bit. 

The use case appears to be very applied, therefore it should be of interest for domain experts who could use such technology (i.e. physicians). But then, QA is only a proxy of the actual task at hand, that is helping to find the appropriate course of action for a patient, which shouldn't be evaluated in QA format but rather as the actual task, where physicians are (or aren't) supported by technology (such as LLMs) to inform their action. Probably, the findings would be of more interest for audiences of medical (informatics) journals, who could also more rigorously judge the significance and validity of the research, rather than an AI conference. 

Looking at the QA formulation, the findings are somewhat plain - LLMs struggle to answer questions about rare cases and using RAG (and retrieving very similar cases) improves performance. I don't find this finding particularly exciting. In order to tease interest for more AI/CS relevant audiences, it would be good to see analyses investigating the root causes, failure modes and potential avenues for or actual demonstrated methods of improvement.

### Questions
Please address my minor remarks stated in the weaknesses.

Also, did you tamper with the submission template? The margins are smaller compared to the official template.

### Soundness
3

### Presentation
3

### Contribution
2
