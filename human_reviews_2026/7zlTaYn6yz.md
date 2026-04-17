# MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
As Large Language Models (LLMs) are increasingly deployed in healthcare settings, accurate error detection and correction in generated or existing text becomes critical, as even minor mistakes can pose risks to patient safety. Existing methods for error detection and correction, including automated checks and heuristic-based approaches, do not generalize well across unseen datasets. In this paper, we propose MedGuards as a medical safety guardrail, which is a new framework that treats medical error detection and correction as a multi-agent in-context learning task. Specialized agents separately detect, localize, and correct errors, while a confidence-guided arbitration mechanism resolves disagreements using reasoning traces and confidence scores. This design enhances interpretability, robustness, and adaptability, without requiring additional training of the base LLMs. Additionally, we introduce the Keyword-Prioritized Correction Score (KPCS), a new evaluation metric that considers whether critical keywords within the reference text are generated correctly, providing a more comprehensive assessment than conventional metrics. Experiments across four multilingual medical datasets consisting of clinical notes demonstrate significant improvements by the proposed framework across several metrics and models. Our aim is to enable safer deployment of LLMs in real-world healthcare applications. For reproducibility, we make our code publicly available at https://anonymous.4open.science/r/MedGuards-52F2/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes "MedGuards" -- a framework which employs a multi-agentic system for medical report generation error detection and correction. The paper additionally proposes a new metric KPCS which aims to overcome the shortcomings of alternative popular metrics by assigning a higher weight to more important words in the generated text. The methodology demonstrates substaintial gains in performance over baseline methods with propriatary LLMs.

### Strengths
* The work is timely and important. A good solution could have an important impact on real-world environment as healthcare. Agentic AI is a growing and popular area of research.
* The proposed methodology provides good performance increases with a relatively simple implementation.
* The paper is generally written to a good standard, but some key information and citations are missing (see weaknesses).
* Thorough ablation studies

### Weaknesses
* **The writing is lacking sufficient evidence for claims. The writing can be notably improved** For example (this is just the first page):
    * "Existing applications of LLMs in medical text generation often lack robust error correction mechanisms, a gap that is particularly critical in high-stakes clinical settings." (cf. line 39) -- citation required
    * "Current quality control methods predominantly focus on assessing outputs on the semantic level, which is insufficient for addressing errors that require deep structural understanding and clinical reasoning, both essential in medical scenarios (Li et al., 2024)." (cf. line 41) -- the citation is a benchmark for evaluation Q&A ability of LLMs in medicine. This is insufficient for the claim the current QC methods predominantely focus on outputs on semantic level, for example a quick google scholar shows methods such as deferral [1]. Additionally, you must provide evidence evidence for the claim "essential in medical scenarios"
   * "medical text often involves subtle errors" (cf. line 46) -- citation required, "...which is naturally aligned with the scenarios paradigm and the idea of Chain-of-Though" -- why? (+citation required)
  * "complex tasks are more tractable when decomposed into interpretable intermediate steps" (cf. line 49) -- citation required
  * "medical applications demand high reliability, which a single model output cannot guarantee" (cf. line 50) -- citation required. How are you defining reliablity? This is a very bold claim (that a single model output cannot guarantee). There are many examples of a single model being deployed in healthcare (of which we can assume after passing strict regulations & laws, are sufficiently reliable)
* **It would be nice for a central figure to help readers understand the pipleline of the proposed methodology**
* **Some information may be missing**:
     * What are the confidence scores? How are they derived? Are they calibrated?
     * How are the domain-specific keywords for KPCS determined/derived? Is it just a fixed list of words? What happens if some text contains a critical word not on this list?
* **No LLMs applicable for the use in healthcare are benchmarked**. I think it's important to place the analysis in the context of the authors' proposed setting of healthcare. Here, there are stringent regulations regarding data privacy to protect patients and health care providers. In my own experience, it would be impossible to use such proprietary LLMs used in the analysis of this paper for these reasons. A better evaluation would additionally benchmark against open-source medical LLMs.     
* **Consideration of application in healthcare**. I think the paper sorely misses discussing the application of the proposed method in healthcare, of which comes with strigent regulations. The authors motivate the setting of healthcare as it's important to mitigate errors here (for patient safety), but then do not discuss the implications of the proposed method on patient safety. 
* **Technical novelty** - I am somewhat concerned that the technical novelty may not pass the high bar required for ICLR

Others:
* The paper claims the proposed method increases ``reliability'' (of which I'm assuming the authors are defining as the reduction of errors in generated medical text -- though no definition is given). In doing so, I really think the authors should be reporting variability in performance over the multiple runs. 

Minor:
* "multiple specialized agents that compute predictions independently" (many places in the paper) I think a better word is "separately". Independently has mathematical connotations of which I do not think apply here. 


Citations: 
[1] Strong et al. 2019 "Trustworthy and Practical AI for Healthcare: A Guided Deferral System with Large Language Models" (AAAI-25)

### Questions
1. What are the confidence scores? How are they derived? Are they calibrated?
2. How are the domain-specific keywords for KPCS determined/derived? Is it just a fixed list of words? What happens if some text contains a critical word not on this list?
3. Can you comment on the applicability of MedGuards for the real-world deployment in healthcare? What are the computational/monetary costs? Is the error detection/localization/correction performance sufficient enough for use? Do you not need to benchmark against open-source LLMs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MEDGUARD, a multi-agent framework designed to enhance the performance of error detection and correction in medical generated text. The authors also address a limitation in existing evaluation metrics, noting that they prioritize general semantic similarity and can assign high scores even when critical medical keywords are omitted. To address this gap, the paper proposes a novel metric, the Keyword-Prioritized Correction Score (KPCS).

### Strengths
- The framework offers a resource-effective and time-efficient approach to enhancing model reliability by improving performance without requiring additional fine-tuning of the base LLMs.
- The framework enhances interpretability by generating reasoning traces during each stage.

### Weaknesses
- The multi-agent self-consistency mechanism appears flawed as it only triggers arbitration in cases of disagreement. The paper fails to specify a protocol for low-confidence consensus, where both agents agree on a decision but both report very low confidence scores. Relying on such an agreement is potentially risky.
- The justification for using precisely two agents for the detection and localization steps is not provided. An ablation study is necessary to explore how performance varies with the number of agents to justify this specific design choice. As the authors already mentioned, employing multiple agents would be expensive, yet without a proper ablation, it remains unclear whether using all agents provide a justifiable performance gain relative to its cost.
- The validation for the proposed KPCS metric appears insufficient, primarily because the binary nature of its K() function is problematic. As this function assigns a full score of 1 if at least one keyword is present, a correction could receive a high score while still omitting other critical medical keywords.
- The description of the datasets is somewhat brief. While MEDEC is referenced, a more detailed breakdown of the MedErrBench dataset, particularly regarding its construction, validation, and characteristics, would be beneficial. We couldn't find out the source of MedErrBench, which appears to be an in-house benchmark without explicit clarification. Moreover, if the English, Chinese, and Arabic versions are merely translations of the same dataset, they may not adequately capture language-specific clinical nuances, thus limiting the validity of the claimed multiple benchmark evaluation.

### Questions
- The current method inputs the entire note, extracts an erroneous sentence, and then uses string similarity to align it with the original text. Did the authors consider a sentence-by-sentence processing approach? This could potentially eliminate string alignment step. Furthermore, the per-sentence reasoning traces from such a process could be directly fed into the correction agent to provide more localized context.
- Regarding the "critical keywords" used in the KPCS metric: how are these defined and extracted? Are these keywords extracted from the reference sentence Sr automatically or are they manually annotated? Does this check require a strict exact string match, or does the mechanism also account for semantic equivalents?
- The citation order appears inconsistent, with some listed chronologically and others in reverse order; it would be preferable to standardize the format throughout the paper.
- The explanation of the self-consistency procedure is somewhat insufficient. While the code suggests that multiple prompts were used for a single model to ensure consistency, this point is not clearly described in the main text and should be elaborated for clarity.
- It would be helpful to clarify how the proposed metric differs from MEDCON [1]. In particular, instead of using a binary (0/1) scale, adopting MEDCON's keyword-level checking mechanism might provide a more informative comparison.
- The value of α in KPCS metric is not specified. Is there an empirically optimal α that shows the highest correlation with human evaluation? Including such details would strengthen the metric's interpretability.
- In line 249, the comma before "of other agents" appears to be a grammatical error or typographical mistake.
- In line 306, and 351, "appendixA.2" and "appendixA.3" should be consistently capitalized. The inconsistent use of lowercase "appendix" should be corrected.
- In line 449, the figure reference seems incorrect; it should likely refer to Figure 1 rather than Figure 3, since Figure 3 presents MedErrBench, not MEDEC.
- In line 465, the figure number is missing and should be specified for clarity.
- In line 736, "and5" lacks a space and should be corrected to "and 5."
----
*[1] Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation*

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework (called MedGuards) that treats medical error detection and correction as a multi-agent in-context learning task. additionally, the paper introduces an evaluation metric (called KPCS) that assigns greater weights to critical clinical entities.

### Strengths
1. The paper's methodology section is very clear and contains lots of relevant information that might be useful to understand the framework being proposed in this work.

2. being able to validate the proposed framework (medguards) using human evals is a plus point.

3. The work builds on theoretical framework and then validate it using experimental setup across multiple datasets

4. This work is useful in medical setup where medical errors can be automatically deducted. hence there is a real life implication

### Weaknesses
1. The paper has shared the anonymous github link, however it would be useful to have the prompts in appendix for quick reference

2. The paper does not mention the false positives or negatives by the agents at each stage. It would be interesting to see (qualitatively) if there is any patterns where a specific llm fails (and perhaps other llm did not fail).

### Questions
1. Can you clarify whether each human evaluator rated all cases or only a subset? If each evaluator rated all the cases, I believe it might be a good idea to include inter-rater agreement to show how much humans agree.

2. The table 1 title does not specify which dataset results are being shown? It took me a while after reading to infer but it would be helpful to have that information right in the title itself (similar to table 2 wher you mentioned the dataset name)

3. What do you mean by "best MedGuards-enhanced model" in table 1's caption? I think some clarity is needed here.

4. I understood that one single LLM is being utilized in three agents. However, would it be possible to see how the mixture of models would change the performance? For instance you can create a permutation and combination of all models for all the stages (detection, localization and correction). I do see Table 7 doing that but it is (a) limitated to one dataset (b) not showing all possible combinations.

### Soundness
3

### Presentation
2

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
The paper introduces MedGuards, a novel multi-agent system designed to enhance reliability in medical error detection and correction (MEDEC) within LLM-generated clinical text, addressing the limitations of single-model approaches in high-stakes domains. MedGuards employs a CoT-guided decomposition into detection, localization, and correction stages, utilizing specialized agents coordinated by a confidence-guided ICL-based arbitration mechanism that resolves disagreements using reasoning traces. To complement the framework, the authors propose the Keyword-Prioritized Correction Score (KPCS), a domain-specific evaluation metric that emphasizes the fidelity of critical clinical entities. Extensive experiments across four multilingual medical datasets demonstrate robust and significant performance gains over strong LLM baselines and state-of-the-art MEDEC methods, validating the system's effectiveness and clinical safety orientation.

### Strengths
- The paper introduces a highly original multi-agent self-consistency framework using ICL-based arbitration specifically designed for the safety-critical task of medical error correction.
- Extensive empirical evaluation across four diverse and multilingual datasets consistently shows significant improvements over strong LLM baselines and specialized prior systems.
- The proposed Keyword-Prioritized Correction Score (KPCS) is a meaningful methodological contribution that successfully aligns evaluation with clinical safety requirements by prioritizing critical entities.
- The methodology is presented clearly, grounding the system design in established principles like CoT decomposition, which contributes to the interpretability and robustness of the pipeline.

### Weaknesses
- The reliance on a multi-agent architecture necessitates significant computational overhead and latency, which may hinder its practical deployment speed in time-sensitive clinical settings.
- The paper does not clearly define the source or methodology for extracting the "critical clinical entities" used in the Keyword-Prioritized Correction Score (KPCS).
- The error localization step heavily utilizes simple character-level string similarity (LCS/SequenceMatcher) to align the predicted erroneous sentence with the original text, which may fail for purely semantic errors or paraphrasing.
- The ablation study presented in Table 4 is incomplete as it focuses only on ICL components and omits ablating the core multi-agent structure versus a single high-performing agent pipeline.
- The system's robustness is predicated on the LLM backbones reliably generating structured outputs, including specific XML-like tags for confidence and reasoning, which can be brittle.
- While detection and localization use arbitration, the final error correction stage relies on a single agent prediction, potentially reducing the robust error mitigation intended by the overall framework.
- The parameter $\alpha$ used in the KPCS weighting is described as tunable, but the paper does not specify which value was used for the reported results, impacting reproducibility and metric context.

### Questions
- Could the authors provide details on the specific criteria and process used to identify and extract the "critical keywords" required for calculating the KPCS score across the diverse datasets?
- Given that the multi-agent system inherently increases the number of API calls, what is the measured increase in inference latency and computational cost relative to a high-performing single-LLM CoT baseline?
- The final weighted KPCS score depends on the hyperparameter $\alpha$; what value of $\alpha$ was used for the results presented in Tables 2 and 3, and how sensitive is the model ranking to changes in this weighting?

### Soundness
3

### Presentation
3

### Contribution
3
