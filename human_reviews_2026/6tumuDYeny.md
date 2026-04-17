# DATR: DDI-Aware Therapeutic Structure Reconstruction for Safer Medication Recommendation

- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Medication recommendation systems play a critical role in clinical decision support, where ensuring both predicting accuracy and safety, particularly drug-drug interaction (DDI) avoidance, is essential. While recent studies have explored drug molecular structures to enhance accuracy, they often overlook the semantic gap between chemical structures and therapeutic outcomes, leading to suboptimal recommendation. Moreover, existing DDI mitigation strategies typically operate in a post-hoc manner, limiting their ability to proactively prevent adverse interactions. In this work, we propose DDI-Aware therapeutic Structure Reconstruction (DATR), a novel framework that jointly models drug structures, therapeutic intent, and safety profiles. DATR conditionally encodes drug structures based on ATC-derived therapeutic labels, enabling intent-aware representation learning, and introduces a selectivity potential DDI constraint to proactively reduce interaction risk. Experiments on two real-world datasets and evaluations by clinical experts demonstrate that DATR achieves superior performance in recommendation accuracy and DDI reduction. Code is available at https://anonymous.4open.science/r/DATR-7EA8.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces DDI-Aware Therapeutic Structure Reconstruction (DATR), a framework aimed at improving medication recommendation systems by simultaneously enhancing accuracy and safety, specifically by reducing drug-drug interactions (DDIs). The framework integrates molecular structure information with therapeutic intent, using a novel therapeutic structure reconstruction method and a proactive DDI constraint. Experimental results on two real-world datasets show that DATR outperforms existing methods in both accuracy and DDI reduction.

### Strengths
1. The integration of therapeutic intent with molecular structure is an innovative approach. Explicitly considering the clinical significance differences between interacting drugs is an interesting idea. The model is theoretically well-founded.
2. Extensive experiments on two real-world datasets with ablation studies and case analysis demonstrate strong performance in both effectiveness and safety.
3. The paper is well-written and well-organized, making it easy to follow. And the appendices gives reproducible details.

### Weaknesses
1. The paper does not provide sufficient insight into how the model makes decisions, particularly how the integration of molecular structures and therapeutic intent influences the final recommendations. Explainability is crucial for making clinical decision.
2. The paper does not discuss how the framework might perform across different therapeutic areas or for drugs with less common usage patterns. It would be helpful to see more discussion on its adaptability or limitations in diverse clinical contexts.
3. The datasets used (MIMIC-III and MIMIC-IV) are widely used in the healthcare community, but their biases (e.g., patient demographic distribution or disease-specific contexts) might limit the generalizability of the model. Addressing potential biases and testing on more diverse datasets could improve the model’s applicability to different patient populations.

### Questions
1. Can you illustrate how specific molecular fragments contribute to disease-specific recommendations through the proposed therapeutic structure reconstruction? Are there examples where the molecular features directly correlate with therapeutic outcomes?
2. The experiments did not report how many drugs the framework recommends on average. Could you provide this information and discuss how it might affect clinical decision-making and the adoption of the framework in real-world settings?
3. The variant without the substructure-level representation of DATR underperforms the original model, even worse than some baselines. Could you clarify why the substructure-level representation is crucial and whether the improvements come solely from the addition of BRICS or from other factors? Could alternative explanations be explored for this observed performance drop?

### Soundness
3

### Presentation
4

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
This paper proposes DATR (DDI-Aware Therapeutic Structure Reconstruction), a framework that jointly models drug molecular structures, therapeutic intent (ATC-4), and DDI (drug–drug interaction) knowledge. The method not only improves the overall performance of medication recommendation, but also reduces DDIs, thereby mitigating potential adverse impacts of drug–drug interactions during recommendation.

### Strengths
1. The authors introduce a DDI-constraint mechanism that simultaneously reduces the risk of drug–drug interactions and improves recommendation performance.

2. On two real-world clinical datasets, MIMIC-III and MIMIC-IV, the model achieves state-of-the-art results.

3. The study further involves 20 clinicians to subjectively evaluate the recommended drug combinations, taking into account medications in the patients’ histories that might be appropriate yet previously overlooked, which provides clinical validation of the approach.

### Weaknesses
In Appendix D2, the paper presents detailed case analyses for individual patients (e.g., patients X and Y). However, Section 5.3 does not sufficiently document the details of the 20-clinic expert evaluation (e.g., patient conditions, criteria for judging effectiveness, and the underlying rationale). The paper would benefit from adding this analysis or providing more comprehensive expert-evaluation information in the appendix.

### Questions
1. In many clinical settings, patients have limited visit histories; thus cold-start scenarios (first several visits) are particularly important for medication recommendation. What's the performance stratified by the number of visits?

2. When the DDI loss is removed (i.e., $\gamma=0$), the predictive performance decreases rather than increases. What primarily drives this drop? Please clarify if the DDI constraint offers any direct benefit to recommendation accuracy beyond mitigating interaction risk.

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
4

### Summary
This paper introduces DDI-Aware Therapeutic Structure Reconstruction (DATR), a framework for medication recommendation that jointly optimizes for accuracy and safety. The authors attempt to address two key issues: the semantic gap, where a drug's molecular structure doesn't capture its specific clinical use, and the post-hoc nature of existing DDI-avoidance strategies. DATR's solution involves first applying Therapeutic Structure Reconstruction, which learns drug representations by encoding their molecular structure conditioned on their ATC category. Second, it introduces a Potential DDI Constraint, an asymmetric penalty that identifies interacting drugs and suppresses the one with lower therapeutic relevance to the patient's current condition, preserving only the most critical treatment. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that DATR significantly outperforms all baselines, achieving state-of-the-art accuracy while simultaneously recording the lowest DDI rates.

### Strengths
The problem is relevant, and the proposed approach is reasonable. The presented results significantly improve performance when compared to competitor models. Ablation studies provide sufficient evidence for the importance of many of the components of the model, and the robustness to the selection of hyperparameters is well demonstrated.

### Weaknesses
## General 

While the method seems sound and efficient, the presentation of the paper requires more work. Many paragraphs are unclear and convoluted, and following the presented argumentation is at times very difficult. Moreover, mathematical notation is often inconsistent and does not help understand the problem formulation and the proposed solution. Some crucial modeling choices are not properly justified by referencing the relevant literature or claiming authorship of those ideas. Finally, competitor methods are only briefly discussed. 

It seems that the paper suffers from a lack of space to develop certain ideas due to the page limit. Please note that some minor issues are not necessarily wrong passages, but suggestions of parts that could be reduced to leave space to improve the discussion in other sections. 

## Major 

* Pieces of the Introduction and Related Works sections are too dense and high-level. Examples, concrete cases, figures and a clearer explanation are necessary. Main examples of these issues can be found below, but others may also be present, so the paper would benefit from additional proofreading by the authors.

* Line 45: This alleged gap needs better justification. The cited paper discusses computer vision representation learning. It's not at all clear how this could be extended for molecular representations. Overall, it would be important to have stronger evidence of the existence of such gap, i.e., why only relying on the global structure is not sufficient. 

* Line 130: the authors say "VAEs have recently..." and proceed to cite a paper from 2015. My understanding is that VAEs are not a recent model, at least by machine learning standards. Other than that, I believe the original VAE paper by Kingma and Welling (2013) [1] would be a better reference for this paragraph. 

* Line 132: The sentence here may make a reader believe that the other models discussed in this section do not utilize gradients, which is not the case. 

* In "Deep-learning-based molecular representations" there's no discussion on molecular transformers, even though this class of models has shown significant results [2]. 

* Line 154: It is said that A is a binary matrix and afterward it is said that it is calculated as the amount of known interactions between the medications. Both definitions are contradictory. 

* Figure 1 does not correspond to the textual description in the main text. For example, it does not show how categorical features are used to quantify the relevance of a drug category to the health condition of the patient. 

* Figure 2 has several problems: the VAE structure is not illustrated, the arrows corresponding to the Potential DDI constraint do not correspond to what is written in the text and there's no mention about CA standing for Cross-Attention. Overall, these problems make it difficult to follow the general architecture of the model.

* Some references seem to have errors. For instance, the reference for "Attention is All You Need" only mentions Vaswani as the author, while there are others. 

* Line 295: references to the previous works should be included in this part.

* Lines 214-257 present material that largely summarizes well-established concepts. There is no need to go in-depth into the math, citing the original paper that derived these equations may be enough. This section could focus on explaining the practical steps employed to generate the reconstructions in the context of the model, which is not clear as it stands now. 

* My major issue with the paper lies on the many details missing to understand the overall structure of the model and the experimental settings. In addition to the examples already provided above, it is unclear what the actual input of the Reconstruction module is and how the authors choose to input the ATC label into the model. 

* The paper doesn’t explain how the dataset was handled apart from mentioning the training and testing split. It’s not clear, for example, whether this split was done by patients, that is, whether the model was tested on patients who were not included in the training set.

* One of the claimed contributions of the paper is to address a “semantic gap” by using ATC labels. However, the ablation study does not include an experiment that tests whether this is actually addressed by DATR.

## Minor 

* Line 97: this part is very confusing. If I understood it correctly, I would suggest something like "Furthermore, the model can avoid the dependency on specific drug pairs in the training data because of its global consideration of all drug pairs for potential interacting risks". 

* Lines 111-123: expanding the explanation of instance-based approaches could be beneficial. 

* Line 161: use of calligraphical M while before it was normal (see line 151). 

* Equation 1: epsilon is not described in the text. 

* Equation 7: It may be better to use one equation per line to improve readability. Also, $E_d$, $E_p$, $E_m$ and $T(\cdot)$ are not explicitly defined in the text.

* Lines 268-269 needs some work. It would be good to change it to something like "The medication taken by the patient in the previous time point is denoted by...".

* Table 1: column DDI has no runner-up. Also, it may be beneficial to push down the table.

* Line 434: Notation "R->T" needs to be introduced before it is used. 

* Appendix C1 is unnecesary.  

* D.3.1: could be enriched by adding sources on this feature of VAEs and limitations in the transfer learning. 

## Typos and Language 

* Line 63: Reconstruction 

* Line 66: label 

* Line 101: bridge 

* Line 172: obtains 

* Line 184: medication, predictions, constraints 

* Equation 2: comma in formula 

* Line 267: linearly 

* Line 983: be 

* Line 1113: they are 

* Line 1127: utilize 

References

[1] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

[2] Luong, K. D., & Singh, A. (2024). Application of transformers in cheminformatics. Journal of Chemical Information and Modeling, 64(11), 4392-4409.

### Questions
* Line 130: what does the authors mean by "prime factors"? 

* Line 159: Is the DDI graph A or D? Or are there two matrices describing these interactions? 

* Line 189: is the idea of using global and substructure-based representation new? 

* Line 336: is there a difference between the process for choosing the hyperparameters for DATR and for competitor models? If so, why? 

* Equation 12: only $L_{DDI}$ has a weight coefficient? It is common practice in VAEs to also ponder the reconstruction loss. Is there a reason for this choice?

### Soundness
3

### Presentation
1

### Contribution
2
