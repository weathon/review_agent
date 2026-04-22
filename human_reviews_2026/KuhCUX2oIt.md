# LEAD: Large Foundation Model for EEG-Based Alzheimer’s Disease Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Electroencephalography (EEG) provides a non-invasive, highly accessible, and cost-effective approach for detecting Alzheimer’s disease (AD). However, existing methods, whether based on handcrafted feature engineering or standard deep learning, face two major challenges: 1) the lack of large-scale EEG-AD datasets for robust representation learning, and 2) the absence of a dedicated deep learning pipeline for subject-level detection, which is more clinically meaningful than the commonly used sample-level detection. To address these gaps, we have curated the world's largest EEG-AD corpus to date, comprising 2,255 subjects. Leveraging this unique data corpus, we propose LEAD, the first large-scale foundation model for EEG analysis in dementia. Our approach provides an innovative framework for subject-level AD detection, including: 1) a comprehensive preprocessing pipeline such as artifact removal, resampling, and filtering, and a newly proposed multi-scale segmentation strategy, 2) a subject-regularized spatio–temporal transformer trained with a novel subject-level cross-entropy loss and an indices group-shuffling algorithm, and 3) AD-guided contrastive pre-training. We pre-train on 12 datasets (3 AD-related and 9 non-AD) and fine-tune/test on 4 AD datasets. Compared with 10 baselines, LEAD consistently obtains superior subject-level detection performance under the challenging subject-independent cross-validation protocol. On the benchmark ADFTD dataset, our model achieves an impressive subject-level Sensitivity of 90.91\% under the leave-one-subject-out (LOSO) setting. These results strongly validate the effectiveness of our method for real-world EEG-based AD detection. Source code: \url{https://anonymous.4open.science/r/LEAD-3B51}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LEAD, a large-scale foundation model for EEG-based Alzheimer's Disease (AD) detection. The authors address two primary challenges in the field: the lack of large-scale EEG-AD datasets and the common focus on sample-level detection rather than the more clinically meaningful subject-level detection.

### Strengths
1. The proposed methodological additions are directly target the subject-level problem.
2. The paper convincingly shows that its domain-relevant pre-training strategy (using ~730 hours of data) can outperform foundation models pre-trained on much larger, more general EEG corpora (LaBraM at ~2,000 hours, CBraMod at ~9,000 hours). This highlights the importance of data selection for pre-training.

### Weaknesses
1. The paper frames itself as the 'first foundation model for EEG-based AD detection'. This premise is arguably a contradiction in terms. Foundation models (such as the baselines LaBraM and CBraMod ) are intended to learn general, robust, and task-agnostic representations from massive, broad data. Specific tasks, like AD detection, are meant to be downstream applications solved by fine-tuning these general models. The authors' approach, curating a large but highly specific pre-training corpus of 'domain-relevant' neurological disorders, is a form of specialized transfer learning. It does not constitute the creation of a new 'foundation model' in the accepted sense. It is, rather, a highly effective domain-specific model, and this framing overstates the generality of its contribution.

2. The core technical contribution beyond the (non-trivial) data engineering appears to be incremental. The proposed model is an adaptation of a spatio-temporal transformer, and the key novel components (subject-level loss, index shuffling) are application-specific engineering solutions. 

3. More concerningly, the value of the architectural changes is directly contradicted by the paper's own ablation study.
* The "Vanilla Backbone" alone achieves a strong subject-level F1 score of 73.11%. The full LEAD pipeline, with all modules, achieves 75.77%, a marginal improvement of only ~2.66%.
* Adding "Half-Overlapping" segmentation causes performance to drop from 73.11% to 72.12%. Later, adding "Multi-Scale Segmentation" also causes a performance drop from 75.49% to 74.96%, which suggests the architectural components are, at best, offering a negligible and inconsistent benefit. The vast majority of the gain  comes from "+Sample/Subject-Level Pre-training", reinforcing that the contribution is in the pre-training data strategy, not the LEAD architecture itself.

4.  For a tool aimed at Alzheimer's Disease, the most vital clinical task is early detection, which requires distinguishing MCI from HC and AD. The model's performance on this task is weak.
* On the multi-class datasets that include MCI, the subject-level F1 scores are low (~64.3% and ~59.6%, respectively).
* On the largest dataset, CAUEEG (1,379 subjects), the LEAD model's subject-level F1 score (59.61%) is statistically indistinct from, several baselines, such as LaBraM (58.92%).

5. The abstract states the model achieves a "Sensitivity of 90.91%" on the LOSO benchmark , but Table 9 and Appendix F.2 refer to this 90.91% value as the "Subject-level Accuracy". Could the authors please clarify which metric this value represents?

### Questions
Please check above. I will reconsider my assessment after checking authors' response.

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
4

### Summary
This paper proposes LEAD, a large-scale, subject-regularized spatio-temporal transformer for EEG-based Alzheimer’s disease detection. It achieves state-of-the-art subject-level performance across four AD datasets.

### Strengths
S1. The authors curate the largest unified EEG–AD dataset to date, comprising 2,255 subjects from heterogeneous sources, significantly expanding the available data scale for clinical modeling.

S2. They introduce a subject-regularized spatio-temporal transformer with group shuffling and a subject-level loss to directly optimize subject-level consistency.

S3. The proposed domain-relevant contrastive pretraining effectively enhances representation learning and downstream AD detection performance.

### Weaknesses
W1. Signal frequency and channel unification may cause information loss. To handle heterogeneous datasets, all recordings were resampled to 128 Hz and mapped to 19 channels. Although this ensures format consistency, it inevitably sacrifices high-frequency components and temporal resolution.

W2. The COGNISION dataset (7 channels) was spatially interpolated to 19 channels, which achieves formal consistency but weakens the neurophysiological grounding. In addition, Figure 5 and the channel-importance visualization section do not specify their derivation or correspondence to AD-related regions, making the interpretation resemble a simple visual description rather than a physiologically meaningful analysis.

W3. The model processes EEG in the time domain, no frequency-specific representations or analyses are provided. It remains unclear whether the learned patterns align with known AD-related spectral alterations.

W4. The contrastive learning component shows limited methodological innovation. The subject-level contrast appears to simply cluster samples from the same subject rather than capture disease-specific dynamics, and the sample-level contrast largely reuses existing SimCLR/MoCo augmentations. 

W5. A few citation styles appear incorrect at sentence endings. The authors may consider using the \citep{} format.

### Questions
Q1. Given the variability in diagnostic criteria and MMSE thresholds for AD/MCI/HC among institutions, how did the authors ensure consistent labeling across datasets such as AD-AUDITORY (https://openneuro.org/datasets/ds005048/versions/1.0.0) and ADFTD (https://openneuro.org/datasets/ds004504/versions/1.0.8)? Were any normalization or re-annotation steps applied?

Q2. What is the rationale for fixing the configuration at 128 Hz and 19 channels? Was this choice made for engineering convenience, or was it supported by empirical analysis showing optimal performance?

Q3. Since LEAD’s pretraining remains disease-oriented rather than task-agnostic, in what sense should it be regarded as a foundation model rather than a cross-disease mixed pretraining framework?

### Soundness
2

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
4

### Summary
The paper introduces a foundation model trained to process EEG data from pathological individuals, with a focus on Alzheimer’s disease detection downstream tasks at the subject level. A transformer architecture with separate temporal and spatial branches is first pretrained using contrastive objectives at the sample and subject levels. The model is then finetuned to perform Alzheimer’s disease-related classification tasks such as pathology type or stage classification. For this, a corpus of public and private datasets (2255 subjects, 442h of recording) was curated and used to pretrain and evaluate performance of the model. Results suggest that in most cases the proposed model outperforms other pipelines based on handcrafted features, fully supervised architectures, or “generalist” EEG foundation models.

### Strengths
Originality
* The paper presents the first EEG foundation model pretrained specifically on pathological data with a focus on Alzheimer’s disease. While there are multiple papers in the literature focused on building EEG foundation models, I am not aware of any that are specifically focused on this subtype of data.
* Similarly, the curation and unified preprocessing of a large corpus of AD-specific datasets appears novel.

Quality
* The proposed method is compared to several relevant baselines (handcrafted features approach, supervised models, and two other “generalist” EEG foundation models), on 4 Alzheimer’s disease datasets from different sources. These comparisons suggest that the proposed model almost always outperforms baselines, which supports the claim of foundation model training.
* The provided ablation studies suggest adding pretraining steps yield the largest performance increase, which supports the use of a pretraining-finetuning pipeline. The ablation study on the pretraining corpus is also informative and supports the use of the curated corpus.

Clarity
* The paper is overall well written and easy to follow, and technical decisions are well supported.

Significance
* Based on the reported strong results on the 4 downstream datasets, this paper is likely to be relevant for the EEG and Alzheimer’s disease prediction community. More generally, this can provide a strong point of comparison for evaluating “generalist” EEG foundation models on pathological EEG detection tasks.

### Weaknesses
1. While the baseline approaches are diverse, comparisons with architectures that have been specifically designed and/or tested on AD data in the past appear to be missing. Appendix A lists a few of them, which I believe are not considered in the results of Tables 3 and 7. Most notably, the last model to be listed, ADformer, presents many similarities to the proposed work both in terms of architecture and evaluation protocol. Similarly, it would be interesting to compare performance to existing EEG foundation models that have also tested their models on similar tasks, for instance [1]. See Q1.

[1] Yuan, Zhizhang, et al. "Brainwave: A brain signal foundation model for clinical applications." arXiv preprint arXiv:2402.10251 (2024).

2. The reason for focusing on subject-level evaluation is well explained in the manuscript. However, it seems to me that the proposed approach relies on the same strategy as previous work to yield subject-level predictions, i.e. majority voting from sample-level predictions. Along with the limited performance increase yielded by the subject cross-entropy loss (Table 5), this weakens the listed contribution of building a novel subject-level detection pipeline.

3. The contrastive learning strategy at the core of the paper is taken from a previous paper (Wang et al., 2023), which limits the novelty of this work on the algorithm side. Of note, this reference is only mentioned in the appendix, whereas it seems central enough that it should be cited in the main text.

4. The preprocessing pipeline relies on heavy preprocessing steps, i.e. ICA and rejection of artifactual components, and requires adapting electrode montages to a common 19-channel montage. Given existing foundation models have shown promising results with limited or no preprocessing steps applied (e.g. LaBraM and CBraMod), with flexible montage support, this limits the flexibility of the proposed model in comparison (see Q2). Additionally, filtering the signals below 45 Hz may get rid of relevant information (see Q3).

### Questions
1. How does LEAD compare to existing deep learning approaches specifically designed for AD-related classification tasks?
2. Have the authors evaluated the importance of the artifact correction step?
3. Cutting off information in higher frequency bands (above 45 Hz) may be limiting as information in the gamma band has been reported as relevant for AD classification, e.g. in Wang et al. (2017). Why was this choice made?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LEAD, a large-scale model for EEG-based Alzheimer’s Disease detection that is pre-trained on 3 AD datasets and 9 non-AD datasets by employing both sample- and subject-level contrastive learning, followed by fine-tuning on 60% of subjects from 4 AD datasets with joint sample- and subject-level cross-entropy losses, while the remaining 40% are reserved for validation and evaluation. Their main contribution is integrating a multi-scale segmentation strategy, a subject-regularized spatio–temporal
transformer with both sample- and subject-level cross-entropy losses, and an indices group-shuffling algorithm to reduce differences between sample-level performance  and subject-level detection. The authors fine-tuned and tested the model on four datasets,  mainly report sensitivity and F1-scores and report performance improvements in comparison to 10 baseline results. In addition, the authors conducted an ablation study to evaluate the effectiveness of each domain-relevant pre-training dataset and an ablation study on the ADFTD dataset to assess the contribution of each module.

### Strengths
* Performance comparison among different groups on the ADFTD dataset 
* The paper provides an ablation study of modules, and an ablation study of pre-training datasets. 
* binary and multi-class results are provided
* Each evaluation is repeated with five random seeds reporting the mean and standard deviation
* creating larger, open-source datasets from heterogeneous clinical populations  is important to address  the lack of large-scale EEG datasets and is essential to investigate cross-institutional generalisability.

### Weaknesses
* In addition, Leave-One-Subject-Out Analysis is provided, however it is limited to one dataset (ADFTD) and only for the LEAD model.
* The information in the repository does not match that in the document. For example repository: 9AD and 7 non-AD, 5 AD for finetuning; Paper: 7AD, 9 non-AD, 4 AD for finetuning.
* The authors compare the performance of the LEAD model to 10 baselines. Two models are foundation models, however the other models are not designed for AD detection. 
* A direct comparison of the results to previously published state-of-the-art results on the four datasets is not provided, nor is any information given about the typical range of classification performance for AD detection.
* For a direct comparison, please provide the sensitivity for 95% specificity, and also other performance metrics, such as accuracy, balanced accuracy, specificity, PR curves, ROC curves, AUCs, etc.
detailed analysis is focused on the ADFTD dataset, no detailed analysis for the other three evaluation sets
* Overall, multi-class classification performances (Sensitivity and F1-score) of LEAD are in the range of 60-77%, and for binary classification in the range of 60-93%. On the benchmark ADFTD dataset, the  model achieves a subject-level sensitivity of 90.91% under the leave-one-subject-out (LOSO) setting.  However, it is unclear how this compares to previous state-of-the-art results.
* The baselines used in this work are mostly designed for other EEG tasks: 
  - EEGNet: BCI
  - TST: multivariate time series datasets (heartbeat, Facedetection, etc; no EEG)
  - EEGInception: Motor Imagery BCI
  - EEGConformer: BCI motor imagery and emotion recognition paradigms
  - BIOT: pre-trained on 6 datasets, fine-tuned on abnormal EEG detection
  - Medformer: included AD detection, multi-class performance on ADFTD: 97.50% F1 score
  - MedGNN:  medical time series classification: sample-based F1score ADFD dataset: 98.30%
  - LaBraM: pre-trained foundation model evaluated on various EEG detection tasks, such as including abnormal detection, event type  classification, emotion recognition, and gait prediction
  - CBraMod: pre-trained foundation model for various  BCI tasks
	
* A statistical evaluation is missing.
* It remains unclear what clinical improvements this approach provides, especially compared to previously introduced approaches.

### Questions
* Why are the ADFTD, CNBPM, Cognision, CAUEEG chosen as final dataset for fine-tuning and evaluation and not AD-Auditory, BrainLat and P-ADIC?
* What are the criteria for selecting the baseline? Why did you not use DNN designed for AD detection, such as SteadyNet, MNet or ADformer?
* Are the baselines trained using the subject-level cross-entropy loss and indices group-shuffling?
* To what extent is the imbalance of the dataset taken into account in relation to the loss function?
* How does the performance of LEAD compare to other models on datasets that have not been pre-trained on multiple datasets?
* Combining multiple EEG datasets, there are several key factors that have to be considered, including data heterogeneity and acquisition variability, institutional biases (distribution mismatch), demographic disparities (dataset imbalance, physical characteristics), data governance and handling. What kind of methodological and technical solutions (e.g. data harmonisation) are applied to address these challenges?  

Suggestions: 

* Include a discussion of the challenges faced when combining multiple EEG datasets and how they were addressed in this work.
* For a more robust comparison, please include baselines that have been applied for AD detection, such as SteadyNet, ADformer or MNet, as well as a detailed comparison to previous published results on these four datasets.

### Soundness
2

### Presentation
3

### Contribution
2
