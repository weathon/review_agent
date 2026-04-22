# AhaTrans: A Hierarchical Adaptive Transfer Learning Framework for Cross-City Traffic Flow Prediction

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Accurate prediction of urban traffic flow is essential for optimizing traffic management, enhancing urban planning, and promoting the development of smart cities. Due to the difficulty of data acquisition in many cities, data scarcity arises, significantly impeding the practical application of deep learning techniques. Consequently, researchers have turned to transfer learning for mitigating data scarcity through cross-city knowledge interaction. However, existing transfer learning methods lack precision and discrimination in spatio-temporal feature extraction, thereby restricting the predictive performance. Moreover, these approaches frequently fail to adequately account for the disparities between the source and target cities, resulting in the loss of essential knowledge and, at times, the introduction of detrimental knowledge into the target city. To overcome these challenges, we novelly introduce **A** **h**ierarchical **a**daptive **Trans**fer Learning Framework (**AhaTrans**), which ensures precise feature learning as well as effective, non-detrimental knowledge transfer in cross-city traffic flow prediction by focusing on three key levels: model architecture, feature representation, and data adaptation. Specifically, AhaTrans consists of the following three core modules: i) Guarded Transfer Experts Network (GTEN), which clearly distinguishes between shared and city-specific experts, enabling the target city to access beneficial knowledge from the source city while preventing harmful knowledge; ii) Spatial-Temporal Contrastive Embedding Module (STCE), which enhances the representation of spatio-temporal features through contrastive learning; iii) Transfer-Based Reweighting Module (TBR), which dynamically adjusts source city samples to extract knowledge most relevant for the target city's traffic patterns. Extensive experiments demonstrate that AhaTrans significantly outperforms existing methods, substantially improving the accuracy of traffic flow prediction while exhibiting excellent robustness and generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AhaTrans, a method for transfer learning between cities for the task of spatio-temporal forecasting. The method contains three main designs. First, the authors design a method based on multi-head networks, which involves a source-specific head, a target-specific head, and a shared head. This enables separation of specific and general knowledge. Second, the authors design a method for contrastive learning for better spatial and temporal feature discrimination. Finally, the authors design a re-weighting based mechanism that aims to align the distribution of source city data with target ones by ruling out those irrelevant source knowledge. Extensive experiments are conducted on real-world datasets, where the proposed AhaTrans achieves performance improvements compared to a wide range of baselines.

### Strengths
1. **Valid problem to study** The problem studied is valid. Indeed, there are cases where traffic data is scarce in certain occasions, e.g. newly established services. The motivation of individual technical designs, e.g. the separation of heterogeneous knowledge between cities, the enhancement of spatio-temporal discriminability, and the alignment of source & target data distributions are all sound and correspond well to the technical solutions. 
2. **A wide range of experiments**. Experiments are done against a wide range of baselines and real-world datasets. For example, cross-city transfer learning methods are compared. Recent spatio-temporal pre-trained models (UniST, etc.) are compared. The proposed AhaTrans shows good accuracy improvements.

### Weaknesses
1. **Technical solutions and insights are established heavily upon existing efforts**. While the focus of this work is valid, it is not completely new and relies heavily upon existing efforts. The following points are non-exhaustive. 
    - First, the two main motivations in the abstract, i.e. the lack of precision and discrimination in spatio-temporal feature extraction, and the disparity between source and target cities, are not completely new. The idea of discriminative spatio-temporal feature learning has been studied in works like ST-SSL (Ji et al. 2023), CL4ST (Tang et al. 2023), etc. The idea of source and target disparity is also studied. More specifically, CrossTReS has a very similar figure as Figure 1 in this paper, and tells similar stories of harm knowledge from source cities. Therefore, the two main motivations of this paper are not completely new, and the authors may need to justify what the additional insights of this paper are. 
    - Second, the technical solutions of this paper are also built heavily upon existing efforts. For example, the separation of source-specific and target-specific networks is more or less similar to PR-UIDT (Ding et al. 2020) which splits city-specific and general POI embeddings and user embeddings. The contrastive learning part is more or less built upon existing efforts like the Rank-n-constrast. The re-weighting technique is more or less similar to CrossTReS. Therefore, the additional values and insights provided by this paper may seem a little bit limited. 
   
2. **Experimental evaluations did not fully uncover the unique contributions of this work.** While the comparison with existing efforts is extensive, the evaluation part falls short in demonstrating some essential designs of AhaTrans. The following list is again not exhaustive. 
    - Regarding the contrastive learning part, as there have been several efforts in bridging spatio-temporal forecasting with contrastive learning, the authors may need to demonstrate why the proposed contrastive learning method is better than other designs.
    - Similarly, as the re-weighting mechanism is more or less similar to CrossTReS, a direct comparison (i.e. replacing the proposed re-weighting algorithm with CrossTReS) may be needed to show that the proposed mechanism has indeed (by itself) outperformed CrossTReS. 

3. Minor issues. I find the notation usage and the presentation throughout the paper vague and inconsistent. The following list is again, not exhaustive. 
(1) In Eqn. 1, the authors use $|g_{t-1} \notin r_{i, j}...|$, while it should be indicating the size of a set, so it should be $|\{g_{t-1} \notin r_{i, j}...\}|$
(2) The authors use different notations to indicate the prediction loss, the prediction, and the ground truth between Eqn. (2) and (6). 
(3). The description of theorem 3.2 is very vague. It is not written in a clear, verifiable way (like a theorem), but instead "a sufficient number of training samples", "appropriate learning rate", "have sufficient discriminative power". 


(Ji et al. 2023) Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction, AAAI2023. 

(Tang et al. 2023) Spatio-Temporal Meta Contrastive Learning, CIKM2023

(Ding et al. 2020) Learning from Hometown and Current City: Cross-city POI Recommendation via Interest Drift and Transfer Learning. ACM IMWUT 2020

### Questions
Please refer to the "Weaknesses" part. 

In addition, another 2 questions. 

(1) Regarding the number of source cities. I notice that the authors seem to assume a single source city. However, utilizing multiple cities is becoming increasingly common for cross-city transfer learning. Can the proposed AhaTrans be extended to the case with multiple source cities, and how?
(2) In the contrastive learning part, as the source city has a dominant amount of data, how to ensure that the contrastive learning does not bias towards the source city?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AhaTrans, a hierarchical adaptive transfer learning framework for cross-city traffic flow prediction. It integrates three complementary modules: a Gated Transfer Expert Network (GTEN) that disentangles shared and city-specific knowledge to prevent negative transfer; a Spatio-Temporal Contrastive Encoder (STCE) that enhances representation learning through contrastive objectives across space and time; and a Target-guided Reweighting module (TBR) that adaptively selects relevant source samples for the target domain. Extensive experiments on multiple real-world datasets demonstrate consistent performance improvements across diverse transfer scenarios and data availability conditions.

### Strengths
(1) The study addresses the practically important yet challenging problem of cross-city traffic flow transfer, which involves heterogeneous spatial distributions and data scarcity, an area of growing relevance for intelligent transportation systems.
(2) Experiments on multiple cities (NYC, DC, Chicago) using both bike and taxi datasets demonstrate consistent and substantial performance gains across different transfer directions and data-sufficiency settings.
(3) The inclusion of formal generalization and convergence analyses for all three core components (GTEN, STCE, TBR) meaningfully strengthens the framework’s credibility beyond empirical results, offering clear theoretical motivation for the design choices.

### Weaknesses
(1) The differentiation from existing spatio-temporal contrastive/self-supervised frameworks (e.g., ST-SSL, STGL) is not fully articulated. Clarifying what is genuinely novel beyond Rank-N-Contrast would strengthen the contribution.
(2) External variables like weather, holidays, or events are excluded. While this simplifies the setup, it limits real-world generalization; even a brief discussion or experiment would improve completeness.
(3) The paper lacks analyses quantifying training time, memory consumption, or inference latency compared to lighter baselines. Without such evaluation, it remains unclear whether the method is practical for real-time or large-scale deployment in urban traffic systems.

### Questions
(1)What are the training and inference costs (time, FLOPs, GPU memory) compared to CrossTReS and TransGTR?
(2)If exogenous variables (e.g., weather) are introduced, how would GTEN distribute them between shared and city-specific experts?
(3)How robust is TBR under severe target scarcity or noisy labels? Any regularization or clipping strategies applied?
(4)To better evaluate the robustness and generalization ability of AhaTrans, could the authors conduct cross-modality transfer experiments (e.g., bike → taxi or taxi → bike) within the same city pairings?

### Soundness
4

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
5

### Summary
This paper proposes AhaTrans, a hierarchical adaptive transfer learning framework designed to solve data scarcity in cross-city traffic prediction. It addresses the key challenges of imprecise spatio-temporal features and negative transfer from source cities. The framework integrates three core modules. At the feature level, a Spatio-Temporal Contrastive Embedding module learns more precise representations. At the model level, the Guarded Transfer Experts Network decouples knowledge into "source," "target," and "shared" components, using a gating mechanism to adaptively fuse them and block harmful information. Finally, at the data level, a Transfer-based Reweighting strategy assigns weights to source samples based on their relevance to the target city. Across six real-world datasets, AhaTrans was shown to significantly outperform 14 baseline methods.

### Strengths
S1. The paper's motivation is sufficient, using the NYCBike dataset to illustrate the limited discrimination of existing methods in extracting spatio-temporal features. The presentation is also clear.
S2. In the experimental section, it compares against 14 SOTA methods from four major categories, ranging from traditional ARIMA to the latest foundation models.
S3. This paper proposes a Guarded Transfer Experts Network, which is based on performing selective knowledge transfer through expert decoupling and gated fusion.

### Weaknesses
W1. Source City Selection is crucial for the practical application of transfer learning. The paper relies on manual specification, which limits the framework's automation and ease of practical deployment.
W2. The effectiveness of the TBR module depends on feedback from the target city's validation set to guide the reweighting of source city samples (Line 278). However, the paper's core premise is that target city data is "extremely scarce." Using what might be a very small and unrepresentative validation set to guide the weighting of large-scale source data raises questions about the stability and robustness of this approach.
W3. There are a few typos/grammar errors; for example, on Line 475, "Framework" should be lowercase.
W4. The provided link (https://anonymous.4open.science/r/AhaTrans-A37F)) currently has an issue where it displays "The requested file is not found." The files are still downloadable, but this error should be fixed.

### Questions
See the weakness above

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
This paper presents AhaTrans, a hierarchical adaptive transfer learning framework for cross-city traffic flow prediction. The authors argue that existing transfer learning approaches struggle with two persistent issues—(1) imprecise spatio-temporal feature extraction and (2) negative knowledge transfer when distributions between source and target cities differ. Extensive experiments on six public datasets show consistent and significant improvements over both traditional and recent SOTA baselines. The paper also includes ablation, sensitivity, and generalization studies, as well as a theoretical analysis of generalization bounds and convergence.

### Strengths
1. The paper derives generalization bounds and convergence guarantees that support the model’s design choices.
2. The paper is easy to follow.
3. The experimental setup is rigorous, with diverse datasets, multiple transfer directions, and consistent metrics.

### Weaknesses
1. The novelty of this paper is limited. Cross-city transfer learning has been extensively studied in previous literature. The proposed modules (expert-based decomposition, contrastive representation learning, and sample reweighting) are all adaptations of existing approaches.
2. All datasets are from similar, grid-based urban mobility domains (NYC, DC, Beijing, Chengdu), without irregular graphs or cross-modality settings (e.g., road networks, weather, events).
3. The study omits larger-scale benchmarks (e.g., METR-LA, PeMS), and comparisons with emerging foundation models (e.g., UrbanGPT, UniST) are shallow and not integrated under consistent fine-tuning settings.
4. No analysis of negative transfer or failure cases is provided, which is crucial given that the paper’s primary claim is to “mitigate harmful transfer.”

### Questions
See in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
