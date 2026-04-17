# HeLoM: Progressive Disease Detection with Heterogeneous and Longitudinal EHRs via Memory-Augmented LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
Recent developments in large language models (LLMs), have significantly advanced healthcare applications, especially the electronic health record (EHR) processing, and demonstrated great potential in disease prediction. EHR are digital records of patients’ medical data, including historical visits, diagnoses, lab tests, and treatments, organized across hospital visits for clinical and research use. Despite LLMs' great potentials, previous methods to predict disease with EHRs based on LLMs face several persistent challenges: (1) they often concatenate short and fixed number of EHR visits (e.g., the latest five) from individual patients and then feed it to LLMs due to either limited input context length or LLMs' capabilities to understand long context, which limits the disease prediction with longitudinal EHR; (2) most prior work focuses on clinical note and overlook EHR's inherent nature like heterogeneity; and (3) EHR are characterized by heterogeneous patterns of missingness (e.g., the missingness of various vital signs). To tackle these problems, we propose a novel progressive memory-augmented framework HeLoM that consists of three key steps: For the first challenge, in a current EHR visit, HeLoM first adaptively fetches previously refined memory (i.e., the patient's previous visits) most relevant to the current disease prediction and then refine this visit to update its memory bank. For the second challenge, we incorporate the heterogeneous data, vital signs, from EHR to enhance the prediction performance. For the third challenge, we introduce two imputation strategies to handle missing data: one leverages LLMs to generate plausible values, and the other applies linear interpolation algorithms to estimate the missing value. By collecting a real-world longitudinal EHR data on Type-2 diabetes from the hospital of our institution, we show the superior performance of HeLoM in disease prediction in terms of both prediction accuracy and early detection. Comprehensive ablation studies underscore the importance of generating missing values from heterogeneous sources, and provide insights into building reliable systems for real-world EHRs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HeLoM, a memory-augmented framework for progressive disease prediction that learns from longitudinal electronic health records (EHRs). The method processes patient visit records iteratively, maintaining a memory bank to store and update summaries of past visits, while integrating heterogeneous data (such as clinical notes and vital signs) through LLM-based or interpolation-based imputation strategies. The framework can operate during the inference stage without requiring retraining, and it was evaluated on real-world clinical data comprising 354 patients with diabetes.

### Strengths
1. Zero-shot methods adapt to varying patient history lengths, avoiding costly retraining through adaptive memory retrieval and refinement.
2. Addressing data heterogeneity and missing values, systematically comparing LLM interpolation with interpolation methods.
3. Clinically meaningful evaluation metrics, proposing Average Prediction Visit to measure early detection capability.

### Weaknesses
1. There is a major issue that needs to be addressed regarding T2D. Most T2DM records are from patients who have already been diagnosed with diabetes when they visit the hosipital. Using these post-diagnosis records directly for prediction is problematic. Instead, records prior to the onset of diabetes should be considered for prediction.

2. In the Related Work section, the definition of 'Memory' is very vague. It should be discussed in relation to existing methods like RAG and what the specific differences are. Additionally, regarding what the authors mentioned: 'LLM. For the current visit, LLMs will utilize the past visits from the "memory bank" that are important for the current prediction as the context.' - what is the actual mechanism?

3. The paper should supplement with experimental results comparing the effectiveness of using RAG methods versus memory, which would better highlight the distinctiveness of using memory in this paper.

4. T2M includes comorbidities, and if memory truncation is applied directly, it may overlook symptoms or behaviors from earlier visits, capturing only the most recent highlights. This is meaningless for diagnosis, as it seems the model is merely guessing whether T2M symptoms have occurred based on the latest key points.

5. The experimental results in Table 2 show anomalies with PromptEHR: Llama3.1 + PromptEHR has a Recall of only 0.057? Also, given that the authors use a large amount of data, they should include fine-tuning methods for comparison.

6. The authors use longitudinal EHR data spanning ten years and employ LLMs for imputation along with linear interpolation. However, formula (2) assumes that vital sign changes are linear, but many physiological indicators exhibit non-linear changes (such as blood glucose). The reasonableness of this assumption is not discussed, and imputation quality assessment and sensitivity analysis are needed.

### Questions
If the summarized memory of early visits contains errors, how will these inaccuracies influence the predictions for subsequent visits?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
To tackle three persistent challenges, inefficient utilization of previous visits, overlook of EHR's inherent nature and the missingness of vital signs, this paper proposes HeLoM to adaptively fetch previously refined memory, incorporate vital signs and utilize two imputation strategies to handle missing data.

### Strengths
- The motivation is clear.
- The paper is easy to read and understand.

### Weaknesses
- Limited novelty. The core contributions—using a memory mechanism introduced in prior work and incorporating vital signs with standard imputation during prediction—are incremental. As presented, the methodological novelty does not meet the typical bar for ICLR.
- Insufficient experimental analysis. The paper lacks comprehensive experiments to substantiate the claims. More thorough evaluations are needed to demonstrate the method’s effectiveness and robustness.

### Questions
- Novelty is limited. The manuscript adopts a memory-augmented inference paradigm already explored in prior work, and the two imputation strategies are standard. As implemented, the pipeline reads more like pre-/post-processing around a base model than a new framework. Please clarify the distinct technical contribution.
- No validation of the generated memory. Provide a quantitative validation of memory quality: e.g., factual consistency checks, information coverage/omission rates. Report correlations.
- “Complementary” imputations. "Two complementary imputation strategies", explain in what sense the two imputation methods complement each other.
- Scope of evaluation. A single dataset/task is insufficient to demonstrate generality. Please include additional EHR datasets (e.g., MIMIC-III/IV) and multiple tasks. 
- Positioning vs. few-shot LLMs and retraining baselines. Clarify the comparison target: if the claim is “test-time adaptation via contextual enrichment,” focus baselines on prompt-based and retrieval-augmented methods. "Unlike approaches that rely on continuous retraining, our method performs test-time adaptation purely through contextual enrichment." Here the "approaches" are deep learning methods? Your method targets at LLMs, where prompt engineering does not need model retraining. Why compare with deep learning methods?
- Baseline coverage. Current comparisons (two baselines) are not enough. Adding general-domain and medical-domain baselines is needed. 
- Backbone selection & metrics. The backbones should vary with its targeted domain and model size. In the medical setting, report AUROC and AUPRC (AUPRC preferred for class imbalance). Since LLMs can output probabilities, evaluate on calibrated scores rather than only text outputs.
- Statistical testing. Including confidence intervals and statistical significance will validate the effectiveness of the method.
- The experiment "AVERAGE VISIT INDEX FOR DISEASE DETECTION" does not compare with the baselines. So what is the point of this experiment? Limited by context length? There are LLMs with enough context length.
- Model-specific claims. Statements such as "The exception is for Llama3.1, which is partly due to its limited capabilities in handling heterogeneous data." must be supported with objective evidence.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper aims to address (early) disease detection in longitudinal electronic health records (EHRs) using a memory-augmented large language model (LLM) framework. The proposed approach, called HeLoM, iteratively processes a patient's visits and dynamically incorporates relevant prior visits as memory context for disease prediction (demonstrated on Type-2 diabetes). It integrates heterogeneous data by including structured vital sign measurements alongside clinical notes. To handle missing values in vital signs, two complementary imputation strategies are used: an LLM-based imputation and linear interpolation. Experiments on a newly collected 10-year EHR dataset show improved early detection and higher recall/F1 scores over baseline methods.

### Strengths
- From my personal perspective, the proposed work seems novel and highly relevant to both healthcare and llm communities. This work proposes a inference framework that allows an LLM to handle long-term patient histories adaptively. Unlike prior methods that truncate or fix the number of visits, the iterative memory bank enables the model to dynamically incorporate relevant context from arbitrary-length EHR sequences without fine-tuning. Integrating structured vital sign data with unstructured notes via prompt engineering further sets this approach apart from existing studies that focus solely on clinical text.

- Well-motivated. The heterogeneity-aware prompting and dual imputation strategies are well-motivated, and strengthen the work robustness to irregular data. 

- The experimental evaluation is comprehensive. And results look good

### Weaknesses
- Perhaps a straightforward weakness is that the study is evaluated on a single institution’s dataset for one disease (Type-2 diabetes) (I agree this is important and meaningful data), which may limit the generality of the conclusions.

- The experimental comparison focuses on prompting-based LLM approaches, maybe including conventional or fine-tuned models for EHR disease prediction would be more thorough and interesting

- Statement of ethics seems missing

### Questions
Please see weaknesses above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Thank you for your great ideas on incorporating the memory-augmented inference-based framework, handling missing data, and especially, integrating multimodal heterogeneous data sources to enhance disease prediction (T2D).

### Strengths
Comprehensive analysis from the author with a different LLM backbones

### Weaknesses
Unfortunately, I find that the current stage of the work needs to be improved in the two major points:
First, the key point of this work is the incorporation of memory-augmented inference-based methods; however, it lacks a comprehensive analysis of the effectiveness of HeLoM in terms of the time elapsed from time t to the previous memory point, as well as the effectiveness of memory as a function of temporal distance. Specifically, the proposed HeLoM framework does not evaluate how performance varies with different memory spans, e.g., from very recent contexts (t–1 to t–5) to moderately distant (t–5 to t–10) or long-term memory (t–10 and beyond). This leads to two critical gaps:
i. Temporal sensitivity is not quantified: No ablation or sensitivity analysis examines how the elapsed time between the current timestep t and the retrieved memory point influences model performance.
ii. The memory–performance trade-off is not explored: Without experimental evidence showing performance gains as memory grows, it is unclear whether long memory windows are beneficial or if diminishing returns occur after a particular horizon.

Second, even a comprehensive analysis from the author with a different LLM backbone; however, it was run only on a single dataset. The reviewer wonders what the findings will be for other datasets. 


Minor:
1. Please provide a short description of the ethical approval for the obtained dataset (even if it is with anonymous information)
2. A trade-off between evaluated metric performance and computational analysis should be made, so that the reader can expect to consider the limitations of the shared GPU from the hospital, including the time it takes to conduct the experiments. 
3. Detailed hyperparameter setting up should also include to support the reproducibility analysis.

### Questions
Besides that, the following concerns need to be discussed or provided with more clarification or justifications:
1. Please provide a table that shows the comprehensive advantages and disadvantages of the proposed approach (HeLoM) compared to the discussed SOTA works.
2. Page 7: Is there any analysis to justify why HeLoM improves Recal compared to PromptEHR? 
The authors confirm: “These findings suggest that the principled design of HeLoM —memory augmented, heterogeneity, and missingness handling – enables it to achieve a more clinically desirable balance between sensitivity and reliability, offering more consistent identification of high-risk patients compared to baseline methods.”
Unfortunately, it is not clear enough to conclude with a convincing justification. Could you please provide further clarification on this point?
3. Section 4.3. Missing values: what happens if the missing values for the vital signal at time t are significantly different from the previous times in case of an emergency visit ===> This will lead to a strong bias in the prior vital sign, especially in terms of the Interpolation approach, which strongly relies on recent and prior visit results.
4. It is quite confusing from the obtained results from Fig. 2 vs Table 4? Fig. 2 confirms that linear interpolation outperforms the LLM-based imputation. Please provide clarification for these points; otherwise, it leads to a contradiction between the findings from the two experiments. And if it is the case for linear interpolation, I will return to the question raised in point 3 above.
If necessary, please also provide the complete results from linear imputation, as shown in Table 4 for LLM-based imputation.

### Soundness
3

### Presentation
2

### Contribution
2
