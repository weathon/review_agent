# MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Large language models (LLMs) have saturated standard medical benchmarks that test factual recall, yet their ability to perform higher-order reasoning, such as synthesizing evidence from multiple sources, remains critically under-explored. To address this gap, we introduce MedMeta, the first benchmark designed to evaluate an LLM's ability to generate conclusions from medical meta-analyses using only the abstracts of cited studies. MedMeta comprises 81 meta-analyses from PubMed (2018--2025) and evaluates models using two distinct workflows: a Retrieval-Augmented Generation (Golden-RAG) setting with ground-truth abstracts, and a Parametric-only approach relying on internal knowledge. Our evaluation framework is validated by a well-structured analysis showing our LLM-as-a-judge protocol strongly aligns with human expert ratings, as evidenced by high Pearson's r correlation (0.81) and Bland-Altman analysis revealing negligible systematic bias, establishing it as a reliable proxy for scalable evaluation. Our findings underscore the critical importance of information grounding: the Golden-RAG workflow consistently and significantly outperforms the Parametric-only approach across models. In contrast, the benefits of domain-specific fine-tuning are marginal and largely neutralized when external material is provided. Furthermore, stress tests show that all models, regardless of architecture, fail to identify and reject negated evidence, highlighting a critical vulnerability in current RAG systems. Notably, even under ideal RAG conditions, current LLMs achieve only slightly above-average performance (~2.7/5.0). MedMeta provides a challenging new benchmark for evidence synthesis and demonstrates that for clinical applications, developing robust RAG systems is a more promising direction than model specialization alone.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduced a benchmark to evaluate the LLM's ability to draw conclusions based on medical papers. The benchmark is built based on existing benchmarks from PubMed. Multiple models with various methodologies are compared on the benchmark.

### Strengths
* All data in the benchmark are from existing medical papers.
* Multiple models and methodologies are evaluated and compared.

### Weaknesses
* Several details regarding the benchmark's construction require further clarification. Please see the accompanying questions for more information.

* One of the claims presented in the Results section is not well-supported by the exp results. Please refer to the specific questions for details.

* The evaluation methodology presents a significant concern, as the models under test (Gemini 2.5 Pro, O4 mini, and Qwen3) are also employed as the judges. The paper provides no justification to address the potential for self-enhancement bias, where a model might unfairly favor its own outputs. Furthermore, while the 'LLM-as-judge' paradigm has been applied in other ML tasks, its soundness in high-stakes, medicine-related applications remains questionable. The limited scale of the human validation mentioned in the paper is insufficient to alleviate these concerns regarding the metric's reliability.

### Questions
## Benchmark Construction

* Line 152: The criterion for retaining meta-analyses—requiring that "all cited primary studies are retrievable in PubMed with available abstracts"—raises some concerns about **selection bias**. This filter could exclude numerous high-quality journal articles that are not indexed in PubMed. The authors should justify this strict inclusion criterion and the potential consequent bias to the final dataset's representativeness.

* The decision to include only the abstracts of the primary studies, rather than their full text, is a notable limitation. This design choice restricts the information available to the models, while recent works (like [1]) have used techniques like sliding windows to process full-text documents.

[1] Wang, Jianyou, et al. "Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark." arXiv preprint arXiv:2411.18831 (2024).


* Lines 188-192: The "feasibility filtering" step, which relies on Gemini Flash 2.5, lacks robust validation. The authors' justification, citing "robustness" that is itself derived from other LLM-judged scores, is methodologically circular. While full human annotation may be infeasible, the authors need to validate this automated step. I strongly recommend reporting the inter-annotator agreement between the LLM filter and human experts on an appropriately sized development set to substantiate this filtering approach.

* Line 195: There appears to be a typo. The text refers to "Figure 2," but the context and linked content strongly suggest it should be "Table 2."


## Evaluation

* Table 1: The validation of the "expert" annotations is weak on two fronts. First, the paper provides insufficient detail about the **annotators' qualifications**. Meta-analysis is a highly specialized task, and it is unclear if the annotators possess the requisite domain expertise. Second, a **validation sample size of N=20 is far too small** to draw statistically significant conclusions about the annotation quality or the reliability of the "LLM-as-judge" metric.

* Table 2: The highest reported performance (approx. 3.17) is substantially lower than the scale's upper bound. Is this low score due to the **inherent limitations of current LLMs** for this complex task, or is it an **artifact of the benchmark's design** (e.g., providing only abstracts)? To properly contextualize these results, the paper needs a **human expert baseline**. What score would a human expert achieve under the identical constraints (i.e., using only abstracts and being scored by the LLM judge)?

* Negated-RAG (N-RAG): The N-RAG task, as described, is highly artificial. I found how negated conclusions are generated in the appendix, but it is unclear whether the model is informed of this negation during the evaluation. If the model is unaware that the provided conclusion is false, the task is simply RAG with contradictory evidence, not a test of identifying deliberate negation. In this case, even human experts can hardly draw the expected conclusion provided with contradicting evidence. This ambiguity undermines the claims in Lines 342-349, as the task setup may not align with the authors' interpretation of the results. The authors need to clarify the exact setup.

### Soundness
2

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
The paper introduces MedMeta, a benchmark of 81 PubMed meta-analyses to test LLMs’ ability to synthesize meta-analysis conclusions from multiple primary-study abstracts, under parametric vs. RAG workflows (incl. oracle and adversarial settings). RAG consistently beats parametric reasoning; even with oracle retrieval, the best model averages, and models readily absorb fabricated ("negated") evidence. The evaluation uses an LLM-as-judge panel validated against medical annotators.

### Strengths
**S1)** Clear problem framing and well-specified benchmark construction - I think the motivation and approach is well organized and clearly described. Authors position their work as multi-source conclusion synthesis and find that it is under-evaluated based on prior literature. Overall I found the paper to be well written. 

**S2)** Empirical findings are actionable (RAG > Parametric); domain fine-tuning adding little evidence; and models failing under adversarial negation. These are all valuable findings. 

**S3)** Evaluation is reasonably rigorous (e.g. high r, negligible bias).

### Weaknesses
**W1)** My primary critique is that methodically, the originality in this work is very limited. While positioned as a "first" for meta-analysis conclusion synthesis, the core premise of RAG vs parametric contrast (including LLM-as-a-judge) seems to be built on widely used ideas (including in some of the related work authors cite). This makes the conceptual novelty feel very incremental and applicative to a point where this may be better suited for a workshop. 

**W2)** 81 meta-analyses across 24 specialties, while useful, is still small for broad claims of generality in medicine. 

**W3)** In table 3 authors report correlation and bias, but this does not quite quantify inter-model bias transfer or failure-mode overlaps. Even with human eval, using LLMs to grade peers can introduce correlated viases. 

**W4)** I'm not sure if the adversarial setup here (negated-RAG) is realistic enough. How do authors evaluate this setup (i.e. how representative it is) against real world retrieval of noise or bias patterns.

### Questions
**Q1)** Following up from W4 above, beyond the prompt how did you ensure negated abstract remain *clinically plausible* (e.g. independent clinician screening, exclusion of self-contradiction etc)?

**Q2)** Statistical testing coverage: Tab 4 reports paired t-tests for H2/H3 (L424–L431). Were cross-model differences (e.g., Gemini vs. O4 vs. Gemma) also tested with correction for multiple comparisons?

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
This work (1) introduces a benchmark of filtered meta-analysis from pubmed and (2) benchmarks the impact of RAG and CoT reasoning on LLM-written conclusions.

### Strengths
This work contributes to an assessment of LLM capability and strategy in a specialized but important medical domain. It's well written and provides a well-scoped evaluation of RAG with a clear set of hypotheses and experiments. This is novel as existing work on synthesis evaluation does not explicitly assess evidence retrieval and generation.

### Weaknesses
* The operationalization of "synthesis" in this work is not clear to me. In systematic reviews, synthesis implies an integration of multiple study findings and explicit evidence weighting. The rubric used here primarily assesses relevance and correctness of results, not integration. This weakens the claim that this benchmark can be used to measure synthesis capabilities. 

* Related to the first point, one of the contributions of this work is a benchmark for meta-analysis, however omits important prior word on multi-document evidence synthesis in the medical domain (e.g., cites below [1] [2] [3]). It’s unclear how this benchmark differs from existing datasets and papers explicitly evaluating synthesis.

* I'm a bit unclear about the methodology described for mitigating data contamination (L144: “ articles between 2018 and 2025 ... mitigating contamination from model pre-training corpora”). Since models have different pre-training cut-off dates, it's unclear whether these were considered on a per-model basis. Given how close the scores are across methods, this factor could meaningfully affect the results. A single year-based filter without model-specific cut-offs is unlikely to adequately control for contamination.

* I believe the first finding is an over-claim (L299; The Value of Structured Reasoning). In table 2, many of the RAG improvements are not fully supported: 95% confidence intervals overlap substantially across settings, implying differences are suggestive rather than significant. This should be made clear in the writing for the first set of results. 

* There are important missing reliability measures to validate the human assessments. This work reports correlation between humans and LLM judges, but there is no Fleiss'/Cohen's kappa reported amongst the annotators themselves. Without this, it is unclear how consistent human "gold labels" are. 

[1] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents (Qiu et al, 2025; assessment of retrieval steps and oracles for SRs) 
[2] Do Multi-Document Summarization Models Synthesize? (DeYoung et al, 2024; see Cochrane dataset and related dataset references) 
[3] Summarizing, Simplifying, and Synthesizing Medical Evidence Using GPT-3 (with Varying Success) (Shaib et al, 2024; evaluation rubric that delineates between synthesis and summarization)

### Questions
1. Can you clarify how MedMeta differs from existing datasets for medical, multi-document synthesis like Cochrane? 

2. How is synthesis operationally defined and distinguished from summarization in your evaluation? 

3. What are the inter-annotator agreement scores? 

4. Can you clarify the validity of the 2018-2025 filtering strategy/sampling across models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces MedMeta, a benchmark designed to evaluate large language models’ ability to to generate meta-analysis–style conclusions from primary study abstracts. The authors curate 81 medical meta-analyses spanning 24 specialties and test LLMs under six workflows, including parametric inference, chain-of-thought prompting, and retrieval-augmented generation. The benchmark uses an LLM-as-a-judge evaluation protocol, which is compared against nine annotators.

### Strengths
* This work addresses an important and clinically relevant problem of multi-study evidence synthesis, which is under-explored in current LLM evaluation.

* The dataset spans multiple medical specialties and uses real meta-analyses.

### Weaknesses
While I strongly believe this work addresses an important and relevant clinical problem, I believe this work has several major methodological concerns that need to be addressed.


**1. Abstract-only synthesis is not a faithful surrogate for meta-analysis**

The benchmark assumes meta-analysis conclusions can be reconstructed from abstracts alone, but real meta-analyses rely heavily on full-text data (effect sizes, confidence intervals, inclusion criteria, heterogeneity stats). The LLMs are effectively expected to perform meta-analytic reasoning despite having access only to abstracts without **all** the underlying statistical data or methodological details required to reproduce an actual meta-analysis. As such LLMs can’t perform effect size aggregation, heterogeneity assessment, bias & quality scoring (e.g. PRISMA, GRADE) or even a simple sensitivity analysis. As a consequence, the dataset does not accurately represent evidence synthesis requirements and results may reflect artifacts of a flawed task design, not model capability in the actual task.

Suggestion: Improve your data selection to include full-text articles. For example, you can access them via PMC-OA. Prior work has done something similar to this recommendation.

**2. Using LLMs to determine feasibility introduces bias**

The authors claim to perform a feasibility check to determine whether abstracts contain sufficient info to infer the conclusion. However, it was done using another LLM (Gemini). The authors themselves acknowledge that using an LLM to screen feasibility introduces bias and present a diagram as a mitigation strategy. This reasoning is flawed and incorrect. I can think of a least two major flaws:


* Model-dependency bias:  The same class of models determines dataset inclusion and is then evaluated on that dataset. 


* self-confirmation bias: papers are selected where an LLM believes synthesis is possible, which incorrectly filters the dataset, as shown by the author's own results and prior literature models can’t perfectly solve this task, how can they effectively filter this dataset?


Suggestion: Perform an actual feasibility check with actual medical experts to validate the feasibility of the selected meta-analysis. This should reduce LLM bias. There is extensive literature documenting the risks of using LLMs to curate or evaluate data. For instance, LLMs acting as judges have been shown to exhibit model bias, such as GPT-4 systematically favoring outputs produced by GPT-4. Simply adding a diagram does not constitute a mitigation strategy. This represents a major limitation of the study design.



**3. Dataset Size Is Limited (81 Questions):**

Eighty-one items are likely insufficient to capture the breadth of clinical SR conclusions, risking sampling bias.

Suggestion: Expand to 500–1K question. Make sure feasibility is filtered using medical experts with experience. 

Additionally, very simple statistics (commonly reported in other medical datasets) are missing. For example, what is the distribution of questions per medical specialty (not by topic)? . Furthermore, the stratification by topic  (shown in Table 3) is not informative. "Diagnosis" as a topic doesn't tell me much (basically every disease has a diagnosis or prognosis; it's a very general term. I really recommend adding a clinical annotator  (or at least talking to one) to improve these systematic issues. 


**4. LLM-as-judge evaluation risks validity issues**

Even though validated with correlation, there are at least three big major flaws:
Judges are frontier models (Gemini, O4, Qwen) that are also evaluated — not evaluation-model-neutral.


Correlation against 9 relatively junior annotators (pharm, biologists, master’s)  without medical experience does not reflect true clinical expert agreement.


Pearson correlation alone is insufficient to establish clinical evaluation reliability.
As a  consequence the claims of “reliable proxy for medical experts” are overstated and incorrect.

**5. Potential data contamination remains**

Although only 2018–2025 abstracts were used, many models were trained on dumps up to 2023. for instance, GPT-4o’s cutoff is October 2023. As a consequence, the premise “post-cutoff data mitigates memorization” is weak. What mitigations are the authors taking to try to mitigate this?


**6. Novelty claim is overstated**

Several works already explore multi-document synthesis, medical RAG, judgment tasks, and safety under adversarial evidence. The authors should more clearly describe how their work positions itself relative to Cochrane automation efforts and existing biomedical multi-document reasoning benchmarks. For example, how does this work compare to prior studies that are closely related (e.g., MedREQAL, MedEvidence) or that evaluate similar aspects (e.g., ConflictBench), just to name a few?

**Additional Feedback:**

It is stated that in order “To validate our automated metrics [...] nine annotators with medical backgrounds” were selected; however, by looking at the table 0 annotators have medical experience, as such I worried about the quality of these evaluations. A common standard in medicine is to count years of experience **after** medical school; none of the annotators meet that criterion, and as such, the phrase "medical backgrounds" is a huge overstatement.

There is no analysis of Inter-annotator agreement (Kappa), Judge disagreement variance, or sensitivity to prompt design. I recommend calculating this once the dataset is fixed.


Overall, I agree this is a very important problem; however, given these systematic issues, the work requires further iteration to reduce bias and better reflect the real task.

### Questions
questions and suggestions are provided above ^

### Soundness
2

### Presentation
3

### Contribution
2
