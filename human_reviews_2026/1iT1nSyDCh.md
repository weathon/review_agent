# Intercept Cancer: Cancer Pre-Screening with Large Scale Healthcare Foundation Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Cancer screening, leading to early detection, saves lives. Unfortunately, existing screening techniques require expensive and intrusive medical procedures, not globally available, resulting in too many lost would-be-saved lives. We present CATCH-FM, CATch Cancer early with Healthcare Foundation Models, a cancer pre-screening methodology that identifies high-risk patients for further screening solely based on their historical medical records. With millions of electronic healthcare records (EHR), we establish the scaling law of EHR foundation models pretrained on medical code sequences, pretrain compute-optimal foundation models of up to 2.4 billion parameters, and finetune them on clinician-curated cancer risk prediction cohorts. In our retrospective evaluation comprising of thirty thousand patients, CATCH-FM achieves strong efficacy, with 50\% sensitivity in predicting first cancer risks at 99\% specificity cutoff, and outperforming feature-based tree models and both general and medical LLMs by up to 20\% AUPRC. Despite significant demographic, healthcare system, and EHR coding differences, CATCH-FM achieves state-of-the-art pancreatic cancer risk prediction on the EHRSHOT few-shot leaderboard, outperforming EHR foundation models pretrained using on-site patient data. Our analysis demonstrates the robustness of CATCH-FM in various patient distributions, the benefits of operating in the ICD code space, and its ability to capture non-trivial cancer risk factors. Our code will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes CATCH-FM, a foundation model for identifying high-risk cancer patients from electronic health records (EHRs) to enable early intervention. The authors establish a scaling law for EHR foundation models by pretraining compute-optimal models up to 2.4 billion parameters. Experimental results show that CATCH-FM outperforms existing baselines and achieves state-of-the-art performance on the EHRSHOT few-shot leaderboard.

### Strengths
- The motivation is clear.
- The paper is easy to understand.

### Weaknesses
- The novelty is limited. The NHIRD-Cancer dataset is curated from publicly available data, and the model architecture largely follows standard transformer decoder designs without substantial methodological innovation.
- The experimental evaluation is insufficient. The paper lacks comprehensive comparisons and deeper analysis of model behavior.
- The citation format is inconsistent and does not follow a standard academic style (e.g., see the first two paragraphs of the introduction).
- The paper appears to violate the double-blind review policy by including self-revealing statements such as:
“Our interpretability analyses Gao et al. (2024) reveal that CATCH-FM identified not only known cancer risk factors but also non-trivial markers discovered in recent medical research.”

### Questions
- Novelty of CATCH-FM:
Please clarify and emphasize the unique contributions of CATCH-FM. Currently, it seems to apply a standard transformer decoder for cancer risk prediction, which limits its novelty. Are there architectural or training innovations that distinguish CATCH-FM from existing EHR-based transformers?

- Baselines:
The baselines are outdated and incomplete. While it is useful to include both traditional machine learning and deep learning models, the most recent comparison (a 2020 paper) does not reflect current progress.
Please include comparisons with recent general-domain LLMs such as LLaVA [1] and Llama 3 [2], as well as medical-domain models such as LLaVA-Med [3] and Med-R1 [4]. In addition, direct cancer risk prediction baselines should be included to contextualize performance improvements.

- Interpretability:
Interpretation and medical insight should be a key aspect of the paper but are underdeveloped. The related section, “Interpretability Experiments,” and "RISK FACTORS CAPTURED" provide limited medical insight; 
Please conduct more experiments on interpretability and deep medical insights, such as the clinical significance of the RISK FACTORS, or interpret the proposed method and its results with real-world validation. Ideally, include qualitative or case-based examples that link interpretability findings to real-world medical understanding.

[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023

[2] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

[3] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36: 28541–28564, 2023a.

[4] Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, and Xiaofeng Yang. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. arXiv preprint arXiv:2503.13939, 2025.

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
4

### Summary
This paper demonstrates that next token pretraining is an effective pretraining strategy for improving cancer prediction.

### Strengths
- AUROC and AUPRC are good evaluation metrics

### Weaknesses
- It's unclear what the contribution is? This is a reasonable case study of applying next token pretraining, but I'm unsure what new information this adds for the ICLR audience.
- Lack of code limits my ability to review, which is especially important when the most important experiments are on-non public datasets. 
- Evaluation on only cancer is a weakness, it would be good to measure the methods on a variety of outcomes
- Cumulative duration matching for controls is incorrect, as it leaks information from the future into the training/test population. 
- I am quite concerned about the complete failure to get RETAIN and StageNet working. There is no reason why those models should fail this badly?
- No discussion of hyperparameter tuning for baselines?
- EHRSHOT is a very small dataset, especially the diagnostic parts. I am concerned about the statistical validity of your comparisons. AUPRC in particular is known as high variance. Please include confidence intervals with respect to the test set to show how much your results would vary if the test set was resampled.

### Questions
See above

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CATCH-FM, a transformer-based foundation model for predicting cancer risk directly from EHR data. Trained on millions of patient records from Taiwan’s NHIRD, it identifies high-risk individuals for cancers such as pancreatic, liver, and lung without costly screening. CATCH-FM achieves up to 70% sensitivity at 99% specificity, outperforming selected baselines and demonstrating strong generalization across populations

### Strengths
1. The model achieves strong and consistent performance across diverse datasets, demonstrating clear advantages over traditional and prior foundation model baselines.

2. The demonstration of scaling laws for EHR-based foundation models is impressive, providing valuable insights into how model size and data scale influence healthcare AI performance.

3. The proposed approach shows strong generalization across different populations and healthcare systems, highlighting its robustness and real-world applicability.

4. The open-sourcing of models and benchmark datasets promotes transparency and reproducibility, helping to advance future research in medical AI.

### Weaknesses
1. Some experiment settings are questionable. For instance, Figure 3 and Table 5 compare Qwen-2.5-500m with CATCH-FM-2.4b, where notable parameter differences exist. More fair comparison should have been conducted with comparable amount of parameters. Moreover, it will be beneficial if larger-scale Qwen model can be used in experiment, as the 500m variant is not very common and may fail to serve as a competitive baseline. 

2. Some experiment results are strange without any explanation. For example, commonly-used baselines, i.e., Retain and StageNET achieve extremely poor performance according to Table 4. Any discussion on these results can help with potential confusion raised by readers.

3. The joint usage of specificity and sensitivity in Table 4 could confuse readers. A more intuitive alternative might be replacing them with recall as they share the same definition.

### Questions
I am curious how the authors will share the data. Although the authors have been granted with the permission of accessing the data, it is still unknown whether re-sharing the data is allowed. In the case of MIMIC dataset, this kind of data distribution could be infeasible, causing accessing issue to the dataset.

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
4

### Summary
The paper proposes CATCH-FM an EHR-based foundation model that can identifies high-risk cancer patients for screening. CATCH-FM is a decoder-only Transformer model pretrained with EHR data from 3M patients. It exhibits better performance on three cancer prediction tasks in finetuning and few-shot setting.

### Strengths
- The paper conducts extensive experiments and ablation studies to show the scaling law of the foundation EHR model on multiple tasks and two datasets.

- The proposed model outperforms multiple baselines including ML, DL and LLM models.

### Weaknesses
- The methods are not quite different from other previous works such as CLMBR. Although it is pretrained a different dataset, the contribution is limited. 

- The details of preprocessing the EHR dataset are not clearly stated (patient sample filtering and splitting). Some statistics of the dataset read counter-intuitive. (1) The prevalence of pancreatic cancer should be much less than lung and liver cancer (5-10 times less). However, Table 2 shows the positive rates are quite close among three types of cancers. (2). How is First/Subsequent Target Cancer Cohort defined? Why does the "subsequent" have less patients in total than the "first"? It will be helpful to include a flowchart on how each cohort is selected at each step.

- The PR-curve in Fig. 3 looks too good to be true. On pancreatic cancer, the precision maintains ~100% at 25% recall. It suggests the model can predict 25% of pancreatic cancer patients without any false positives. This achieves the accuracy of diagnosing with biopsies, which is unexpected for forecasting tasks. There can be potential leakage (e.g. patients with related ICD /medication codes are not excluded).

- Table 5 and Figure 4 miss the performance on pancreatic cancer. 

- The risk factors only highlights the general diseases which does not support such high precision in Fig. 3. It will be interesting to compare the prevalence of cancers among patients who have these diseases with the precision of prediction.

### Questions
What licenses are required to get the pretrained weights and NHIRD data? How to apply for the license?

### Soundness
2

### Presentation
3

### Contribution
2
