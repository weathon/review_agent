# Bidirectional Collaborative Medical Report Generation via Concept-level Interaction

- Decision: Reject
- Scores: 2, 2, 4, 6, 8

## Abstract
We introduce the first bidirectional collaborative medical report generation framework to reduce physicians' workload and enhance trustworthiness through targeted physician-AI interaction, where physicians provide feedback only on the most critical parts, and the Vision-Language Model (VLM) propagates these to finalize the full report. The core challenge lies in defining the optimal unit of interaction. We propose the Anatomy-Finding Concept Unit (AFCU), a minimal, clinically grounded semantic statement (e.g., ``left lobe: hypoechoic nodule''), satisfying three key principles: atomicity, lack of ambiguity, and anatomical anchoring. To extract AFCU, we use a Large Language Model (LLM) guided by predefined clinical templates followed by information bottleneck clustering to group lexically diverse but semantically equivalent anatomical concepts (e.g., “left and right lobe” to “both lobes of the thyroid gland”), eliminating redundancy while preserving diagnostic fidelity. To prioritize physician intervention, we introduce the Concept Risk Score (CRS), quantifying behavioral inconsistency (concepts generated regardless of image content) and semantic instability (inconsistent associated findings under image perturbations) via occlusion-based visual grounding. Finally, we propose Holistic Semantic Match (HSM), a concept-based metric that correlates strongly with human judgment (Pearson’s r = 0.846, $p < 0.05$). Experiments show our framework improves semantic quality by 9.13\% HSM across four organs by correcting only one AFCU with high error risk per report -- a minimal, clinically feasible intervention, enabling efficient and trustworthy physician-AI collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper motivates that a new collaborative radiology reporting paradigm is required, as clinician oversight is an any case still required for AI-generated reports. They define “Report Collaboration Levels” in the current literature, arguing current works fall in level 1 and 2, which do not allow interactive/bidirectional collaboration. They propose a collaborative approach, where AI and radiologist interact on a “Anatomy-Finding-Contept Unit”, which they propose as a tuple of anatomy and finding, which they extract using DeepSeek, followed by semantic clustering using Sentence Transformer embeddings. They propose the “Concept Risk Score” as a metric to detect which of their concepts are too stable or not stable enough and therefore likely wrong and the “Holistic Semantic Match Metric” as a new report generation metric measuring IoU over the concepts they defined combined with cosine similarity of findings. The results show one round of correction improves the report generation result.

### Strengths
- Interaction on an abstracted level seems promising for reducing interaction overhead  
- Minimal physician action leads to notable gains in performance on report generation  
- Rating error risk and deciding accordingly what to let the clinician correct makes sense and can reduce workload and the proposed uncertainty score seems to work better than other methods

### Weaknesses
- Contribution claims: The five-fold contribution is over-claimed. I would say the main novelty lies in the concept-level interaction together with the uncertainty estimation \- the finetuning, and the metric as well as improved results should not be listed as unique contributions  
- Clarity:  
  - The authors define a lot of new names for rather simple concepts, making the paper hard to follow and overstating the contribution  
  - Figures 2 and 3 are very unclear  
  - Its not clear what table 1 shows and the results are not discussed in the result section, only shortly in the introduction  
- Method:  
  - The collaborative process is not really explained in the methodology, even though this is the main promise of the paper  
  - It's unclear if the fine-tuning process described in section 3.2 is needed. It seems like today's LLMs would be able to adapt a report given a flagged error without fine-tuning  
  - It seems like the only “interaction” is once correcting one finding that their risk score is classified as most probably wrong. That does not solve the problem of radiologists needing to double-check reports because there remains a lot of unchecked content in the report nevertheless. Why not allow correcting more than one finding if more findings are uncertain?  
- Results:  
  - It's unclear how the results of the proposed method are collected \- is a radiologist   
  - It is not surprising that performance improves when correcting mistakes. The method is only compared to one pass report generation methods without correction \- this does not really show that the proposed way of interaction is better than other interaction modes. A comparison of basic interaction modes in both time it takes as well as reached accuracy (judged by an external reader instead of an arbitrarily defined score) would be needed in my opinion  
  - It’s not clear why a new report generation metric is needed as opposed to using a similar concept as e.g. proposed in GreenScore.  
  - It’s unclear how the method was evaluated / how the interaction was simulated to get the corrected report.

### Questions
- Why are finding concepts “clinically discriminative”? Also there ambiguity exists, e.g. “enlarged heart” vs “cardiomegaly”  
- How is the theoretical upper bound determined? Why is it not 1 for a fully correct report?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a bidirectional collaborative framework for medical report generation that enhances model trustworthiness while reducing physicians’ workload. It leverages information bottleneck theory for concept-level feedback refinement and introduces the Concept Risk Score to prioritize high-impact diagnostic corrections. Additionally, it establishes concept-grounded semantic metrics and demonstrates a 9.13% improvement in report quality with minimal physician intervention.

### Strengths
- The move to concept-level feedback refinement is original and relevant for real-world applications.
- The use of semantic similarity to go beyond simple overlap metrics is well motivated and adds depth. Also comparison with other metrics is beneficial. 
- The correlation analysis between the proposed metrics and human judgments / LLM evaluations is a strong plus and a meaningful evaluation.
- Report a quantifiable improvement (9.13%).

### Weaknesses
* The authors claim a five-fold contribution, but the paper really delivers two main contributions: the bidirectional collaborative interaction, uncertainty estimation and maybe the new metric.  
* Authors redefine existing definitions, making it hard to read the paper and confusing.  
* The presentation is uneven: mixing languages in figures, and a methodology section that interleaves high‐level design with implementation details and discussions, hurts readability.  
* It would be nice to see the proper evaluation of each part of the method, to understand where exactly the contribution is and what helped to improve the performance.   
* The paper has to be rewritten.  
* There are some interesting results in the tables, but they are not properly discussed or explained.

### Questions
If the metric goes as a contribution, it would be worth comparing it to other semantic metrics like RaTEScore, GreenScore, GEMA‑Score and etc.

The authors introduce a collaborative interaction mechanism, yet they do not compare it to other methods under the same setup. Since giving one method the possibility to correct its mistakes makes it obvious that it will perform better, the more meaningful question is: if other competing methods were given the same correction-capability setup, would the proposed method still outperform them?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new interactive medical report generation method. Intuitively, authors reduce physician workload in medical report generation by identifying areas of uncertainty in images. To some extent, this is reasonable.

### Strengths
1. The writing of this paper is good and easy to follow.

2. This paper has a certain degree of innovation, introducing the Concept Risk Score (CRS) to help models to capture the unstable regions of images.

### Weaknesses
1. Figures 2, 3, 6, 9, 10, and 12 contain a lot of non-English content, which is not friendly to reviewers.

2. I suggest that the authors can add relevant data on clinical efficacy (CE) metrics in Table 1, 2, 3, 10, 11, 12 and 13, as the CE metrics of the medical report generation comes first.

3. Most of the comparison algorithms are too old, making it difficult to demonstrate the effective of the method proposed in this paper. I suggest the authors add two more algorithms from the last two years for comparison.

4. The authors proposed the CRS indicator, but its reliability has not been verified. It is recommended that the authors provide a theoretical analysis of CRS.

5. The LLM, acting as a trainee, points out uncertain areas based on its own capabilities. However, the LLM itself lacks the ability to reason [1,2], making it difficult to pinpoint the few truly difficult areas and tends to point out the majority of uncertain areas. This does not reduce the workload of physicians.

[1] Are Large Language Models Really Good Logical Reasoners? A Comprehensive Evaluation and Beyond

[2] Understanding social reasoning in language models with language models

### Questions
Please see the weaknesses.

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
2

### Summary
This paper proposes a new medical report generation framework, RCL-3 (Report Collaboration Level-3), which aims to improve report credibility and reduce physician burden through a bidirectional collaboration mechanism between physicians and AI. The core concept is that AI first generates a draft report, then automatically detects and labels high-risk misconceptions. Physicians then manually correct only these "Anatomy-Finding Concept Units" (AFCUs). Finally, the AI regenerates the report based on the feedback. The paper's main modules include: AFCU concept extraction, which extracts anatomy and finding concepts from reports using DeepSeek-V3; Information Bottleneck Clustering, which merges semantically similar concepts to reduce redundancy; Concept Risk Score (CRS), which calculates visual-semantic instability through occlusion perturbation to prioritize physician intervention; and the Holistic Semantic Match (HSM) metric, which measures the semantic consistency between model output and human annotations. The paper reports an average HSM improvement of approximately 9.13% on four organ datasets (thyroid, breast, liver, and ovary), and claims that the report quality can be significantly improved by modifying only one AFCU.

### Strengths
This study offers a new perspective by framing report generation as a human-machine collaborative problem rather than a purely generative task. The main strength of this paper lies in its directional insights and conceptually valuable attempt to shift medical report generation from one-off automation to collaborative AI design. The proposed AFCU granularity is intuitive and reasonable, providing a middle ground between sentence-level and token-level semantics. Multi-organ experiments also demonstrate some generalization capabilities.

### Weaknesses
1. Validation relies solely on machine-translated English datasets (from Chinese), risking distortion of nuanced clinical semantics; no tests on native English datasets (e.g., MIMIC-CXR) leave adaptability to English workflows uncertain.
2. The Concept Risk Score only detects high-risk concepts the model generates (e.g., incorrect nodules) but misses omitted critical concepts (e.g., unmentioned pleural effusions), limiting report safety.
3. The "single AFCU intervention" design lacks exploration of 2–3 high-risk AFCU corrections (for better efficiency-accuracy balance) and no discussion of low-risk error accumulation in practice.

### Questions
1. How is bidirectional collaboration actually realized? Did any physicians participate in providing real feedback, or is it fully simulated?
2. How does the proposed CRS compare quantitatively with standard uncertainty estimation methods (entropy, ensemble variance, semantic entropy, etc.)?
3. What criteria were used to set the Information Bottleneck coefficient β? How sensitive are the results to this hyperparameter?
4. How stable is the correlation between HSM and established metrics (BLEU, CheXbert, RadGraph-F1) across datasets or random seeds?
5. The paper assumes that “correcting one AFCU” suffices—how realistic is this in multi-pathology cases where multiple findings coexist?
6. Would the framework still function effectively if the base model (e.g., LLaVA, Gemini, or Qwen) were replaced?
7. Has the study quantified the physician interaction cost or time savings through any real or simulated user experiments?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper propose the Anatomy-Finding Concept Unit (AFCU), a minimal, clinically grounded semantic statement satisfying three key principles: atomicity, lack of ambiguity, and anatomical anchoring. To extract AFCU, we use a Large Language Model (LLM) guided by pre-
defined clinical templates followed by information bottleneck clustering to group lexically diverse but semantically equivalent anatomical concepts. To prioritize physician intervention, the paper introduce the Concept Risk Score (CRS), quantifying behavioral inconsistency and semantic instability. Experiments show the proposed framework improves semantic quality by 9.13% HSM across four organs.

### Strengths
1. Echocardiography report generation with a human-in-the-loop is both compelling and clinically important.

2. The framework is original; the proposed modules—Anatomy–Finding Concept Unit (AFCU), concept risk scoring, and holistic semantic matching—are well-motivated and valuable to the community.

3. Extensive experiments demonstrate the method’s effectiveness, and the paper includes sufficient implementation details to support reimplementation.

### Weaknesses
1. The paper states (Line 259) that no retraining is required and that the model can follow intervention prompts to refine its outputs. Please clarify whether this capability is inherited from the base VLM (e.g., Qwen) or emerges from your framework. If it mainly comes from the base model, consider adding a small supervised fine-tuning (SFT) set with refinement prompts to improve controllability and consistency. A comparison among (i) base Qwen with prompts, (ii) your method without retraining, and (iii) your method with light SFT would isolate the contribution of each component.

2. The table is difficult to interpret because the definitions of “phrase-level,” “sentence-level,” and “report-level” SFT are not specified (task formulation, supervision granularity, data size). The current takeaway appears to be that AFCU-derived concepts outperform GPT-generated concepts—which is plausible—but report-level SFT yields the strongest results overall. Please explain the rationale behind these trends, control for data volume, and state what the table is intended to demonstrate (e.g., the value of concept granularity versus holistic supervision).

### Questions
1.  Qwen2.5-VL-3B outperforms the 7B model on many metrics. What factors drive this (e.g., instruction tuning mismatch, optimization stability, decoding settings, or domain overfitting)?

2. Would you consider releasing code, prompts, and (where permissible) data specifications to facilitate replication and downstream use?

### Soundness
4

### Presentation
4

### Contribution
4
