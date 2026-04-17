# Inter-Domain Sensor Alignment for Unsupevised Domain Adaptation of Wearable Multivariate Time Series

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Unsupervised domain adaptation (UDA) for multivariate time-series (MTS) data in the wearable domain transfers knowledge from a labeled source to an unlabeled target, typically with signals collected from multiple body-worn sensors. Although existing UDA methods devote substantial effort to modeling temporal shifts, they often rely on simple spatial alignment across domains, thereby limiting their capacity for effective adaptation. 
Real systems in the wearable domain exhibit \emph{sensor-wise domain shift}, including changes in placement or orientation, which necessitates the explicit consideration of inter-domain spatial sensor relations. 
Therefore, we introduce \textbf{IDSA}, \textit{\textbf{I}nter-\textbf{D}omain \textbf{S}ensor \textbf{A}lignment for wearable MTS-UDA}, a plug-in module that augments any base UDA loss with two complementary components: (i) an \textit{inter-domain sensor transport} that learns a cross-sensor relation matrix from domain-specific sensor embeddings and transports target channels toward the source, and (ii) a \textit{channel decorrelation} regularizer that sparsifies intra-domain graphs to suppress redundant or noisy couplings. Our sensor transportation loss is shown to be equivalent (up to a constant) to the discrete $1$-Wasserstein objective. When used as a plug-in with Deep CORAL or CLUDA, IDSA achieves consistent gains across five HAR and sEMG benchmarks compared to recent baselines in activity classification accuracy, achieving a performance enhancement in most scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles the Unsupervised domain adaptation (UDA) problem for multivariate time-series in wearable sensor domain.   
It motivates from the sensor-wise domain shift, where sensor placement/orientation or other factors might change for each individual in the same domain. Furthermore, it mentioned the potential intra-domain noisy or redundant information in each domain.   

Based on two above motivations, it proposes Inter-Domain Sensor Alignment (IDSA) with two modules: (1) Inter-domain sensor alignment and (2) Intra-domain Sensor Decorrelation. (1) is implemented to align with the distance correlation of sensors, while (2) introduces a self-similarities w.r.t. identity matrix loss. 

Experiments are conducted on activity recognition and surface electromyography datasets. Several methods are compared, with two being selected as the baseline to plug the proposed IDSA. In general, IDSA improves the two selected baseline methods by a big margin.

### Strengths
## Strengths
- The motivation is reasonable and well discussed and investigated, e.g., Sensor-wise Domain Shift can be demonstrated in Fig. 2. And with the motivation this paper defines the Sensor-wise Domain Shift as variations in sensor configurations, such as differences in sensor placement or orientation across domains. 
- A inter-domain alignment module is designed to solve the Sensor-wise Domain Shift in Eq. (2). 
- A Channel Decorrelation Loss is introduced to compress redundant or noisy information. 
- Built on two baseline methods, the proposed model improves them by a big margin.

### Weaknesses
## Weaknesses
- Although the motivation is reasonable and well illustrated, the solution is relatively straightforward and has been explored intensively elsewhere.   
> E.g., Optimal transport for domain adaptation has been widely-used, such as [R1-R3] and to name a few. No relevant papers are discussed or compared either in Related Work or Experiments. 
> The idea of using distance among sensors as guideline or matching target is also well studied in different areas such as traffic flow prediction [R4]. 
- The introduction of Channel Decorrelation Loss is simply repelling the self-similarities, which is also a common technique in graph [R6] or contrastive learning [R5]. 
- Other modules are simply graph neural networks. 
- Experimental baseline are relatively out-dated methods from 2023 or earlier. The proposed modules are applied to two baseline methods Deep Coral (Sun & Saenko, 2016) and CLUDA (Ozyurt et al., 2023), which are from 2016 and 2023. No optimal-tranport based UDA method is compared, which is highly relevant to this paper. 
- No visualization of the transport map or self-correlation matrix is shown. 


[R1] Kerdoncuff, Tanguy, Rémi Emonet, and Marc Sebban. "Metric learning in optimal transport for domain adaptation." International joint conference on artificial intelligence. IJCAI, 2020.

[R2] Courty, Nicolas, et al. "Optimal transport for domain adaptation." IEEE transactions on pattern analysis and machine intelligence 39.9 (2016): 1853-1865.

[R3] Aritake, Toshimitsu, and Hideitsu Hino. "Unsupervised domain adaptation for extra features in the target domain using optimal transport." Neural Computation 34.12 (2022): 2432-2466. 

[R4] Zheng, Chuanpan, et al. "Gman: A graph multi-attention network for traffic prediction." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 01. 2020.

[R5] Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction." International conference on machine learning. PMLR, 2021. 

[R6] Ma, Yuchen, Yanbei Chen, and Zeynep Akata. "Distilling knowledge from self-supervised teacher by embedding graph alignment." arXiv preprint arXiv:2211.13264 (2022).

### Questions
What is the difference of this method compared with OT-based UDA method? 

What does the transport map or self-correlation matrix look? 

Others please see above Weaknesses.

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
4

### Summary
In this study, the authors design a plug-in module for unsupervised domain adaptation on sensor data. The application scenarios tested include human activity recongiont and sEMG. On the technical part, the key component of this paper is the plug-in module that does not need to change existing model architecture. The experiments are conducted on 5 real-world datasets, and the authors have demonstrated the performance gain compared to multiple baseline models. Overall, the problem studied is interesting, but there are concerns that need to be addressed carefully. Please see the following comments in detail.

### Strengths
[1] The topic studied in this paper is interesting, and could have broad applications in the real world. 

[2] Most figures are well-designed, which benefits the understanding of this paper. The reviewer appreciates the efforts from the authors. 

[3] The experiments are conducted on two tasks with multiple real-world datasets.

### Weaknesses
In the abstract, the authors mentioned the improvement of the WISDM dataset. The reviewer understands this emphasis is trying to highlight the effectiveness of the proposed method. However, it might mislead the readers in the future. Since this is not the only dataset used in this study. Also, the improvement on this dataset might not represent the improvements on all datasets tested by the proposed method. 

The last sentence in the opening paragraph of the intro is hard to understand. 

In line 50, does the phrase “various domains” has the same meaning like source and target domains? 

What is the “MTS” data in line 53. It would be great to explain all abbreviations when they first appear. 

Figure 2 introduces many concerns. First, it is challenging to understand the whole setting. Also, if the sensor here refers to IMU sensor, there are multiple channels for each IMU sensor, how does the “similarity” actually calculated remains unclear. Third, the sensors could rotate and different channels or sensors from source and target at the same position might have significant differences in terms of reading, this is intuitive. In line 168, the authors mentioned the “functuional consistency”, which is not easy to understand. 

In the appendix, could you explain why there are only 4 subjects and 4 classes of activities considered for this study? It seems the task has little challenge. 

For baselines, while there are many advanced activity recognition models in recent years, the baselines selected are general models for time series, which might raise concerns about the technical advancement.  Here are some recent examples of human activity recognition: "NeurIPS 2024, UniMTS: Unified Pre-training for Motion Time Series", "Ubicomp 2024, CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining". 

It will be easier for readers if the tasks of experiment could be introduced  in the main content. For example, what are the input and output. 

The writings could be further enhanced. For example, in the opening para of Section 3. This long sentence makes it difficult to understand. 

In line 816, the color could be changed back to black.

### Questions
In the abstract, the authors mentioned the improvement of the WISDM dataset. The reviewer understands this emphasis is trying to highlight the effectiveness of the proposed method. However, it might mislead the readers in the future. Since this is not the only dataset used in this study. Also, the improvement on this dataset might not represent the improvements on all datasets tested by the proposed method. 

The last sentence in the opening paragraph of the intro is hard to understand. 

In line 50, does the phrase “various domains” has the same meaning like source and target domains? 

What is the “MTS” data in line 53. It would be great to explain all abbreviations when they first appear. 

Figure 2 introduces many concerns. First, it is challenging to understand the whole setting. Also, if the sensor here refers to IMU sensor, there are multiple channels for each IMU sensor, how does the “similarity” actually calculated remains unclear. Third, the sensors could rotate and different channels or sensors from source and target at the same position might have significant differences in terms of reading, this is intuitive. In line 168, the authors mentioned the “functuional consistency”, which is not easy to understand. 

In the appendix, could you explain why there are only 4 subjects and 4 classes of activities considered for this study? It seems the task has little challenge. 

For baselines, while there are many advanced activity recognition models in recent years, the baselines selected are general models for time series, which might raise concerns about the technical advancement.  Here are some recent examples of human activity recognition: "NeurIPS 2024, UniMTS: Unified Pre-training for Motion Time Series", "Ubicomp 2024, CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining". 

It will be easier for readers if the tasks of experiment could be introduced  in the main content. For example, what are the input and output. 

The writings could be further enhanced. For example, in the opening para of Section 3. This long sentence makes it difficult to understand. 

In line 816, the color could be changed back to black.

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
This paper proposes IDSA, an unsupervised domain adaptation module for wearable multi-sensor data. It targets sensor-wise domain shifts caused by differences in sensor placement and orientation. IDSA introduces a spatial transport loss, formulated as an optimal transport problem to align sensor spatial correlations across domains, and a channel decorrelation loss to reduce redundant intra-domain correlations. The module can be easily integrated into UDA backbones and achieves consistent performance gains on five HAR and sEMG benchmarks.

### Strengths
1. The paper addresses an important challenge in wearable sensing. Misalignment at the sensor level is meaningful and relevant to real-world deployment, which has attracted recent research interests.

2. The proposed module is a plug-in module that can be integrated into existing UDA pipelines, making it very practical.

3. The presentation is clear, figures are intuitive, and mathematical derivations are easy to follow.

### Weaknesses
1. While the paper presents the idea of “inter-domain sensor alignment” as new for multi-sensor time-series data, there is extensive prior literature on multivariate time-series domain adaptation using structurally similar techniques, such as correlation or covariance alignment (e.g., Time Series Domain Adaptation via Sparse Associative Structure Alignment, AAAI 2021, and related works). These works address domain shifts in the spatial dependencies of multivariate time series. It seems that the main difference between these works and the proposed method lies in the conceptual distinction between “multi-sensor” and “multi-variate” alignment, making the originality of this paper appear incremental. In addition, the paper in AAAI 2021 also considers sparsity.

2. The authors emphasize handling differences in sensor "placement or orientation", yet the experimental setup appears to use datasets with identical sensor configurations across subjects, which does not actually reflect such spatial misalignment. This weakens the empirical justification of the claimed contribution.

3. The paper is vague about the meaning of the channel dimension (N) in the learned representation Z. According to Section 5.1, the sensor embedding P has the same dimension N×T as the input MTS data, implying that each of the N channels corresponds to an axis or variable rather than a complete sensor unit, as each sensor usually has multiple axes. If this is correct, then the modeling operates at the channel/variable level rather than per sensor, which questions whether the alignment is truly “sensor-wise.”

4. It is unclear whether IDSA can handle non-identical sensor deployments, such as different positional configurations, or only applies to the same sensor deployment for source and targets.

### Questions
1. Is the proposed method limited to domains with identical sensor deployments (same sensor types, positions, and count)? If not, can the authors provide additional experiments or analysis demonstrating robustness to heterogeneous or partially misaligned sensor configurations?

2. In Section 5.1, the representation dimension N is stated to “match the input dimensions of MTS data.” Does N correspond to individual sensor channels or sensors as composite entities? This clarification is important because if each channel corresponds merely to one axis of a sensor, the problem reduces to conventional multivariate variable alignment.

3. Could the authors explicitly differentiate their formulation from existing multivariate time-series adaptation approaches that align correlation or covariance matrices of multiple variables? Beyond naming the entities “sensors,” what is the core modeling difference or advantage of IDSA in capturing inter-sensor relations compared to these works? And can this be evidenced by a thorough experimental comparison?

### Soundness
2

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
5

### Summary
The paper proposes IDSA (Inter-Domain Sensor Alignment), a plug-in module for unsupervised domain adaptation (UDA) on wearable multivariate time-series data. Unlike previous approaches focused mainly on simple spatial alignment, IDSA tackles sensor-wise domain shift caused by differences in sensor placement and orientation. It introduces a spatial transportation loss, formulated as a transport problem for inter-domain sensor alignment, and a channel decorrelation loss to reduce redundant intra-domain correlations.

### Strengths
1.	IDSA is compatible with existing UDA frameworks such as Deep CORAL and CLUDA, requiring minimal architectural changes and demonstrating performance gains.

2.	The method is theoretically grounded, formulating inter-domain sensor alignment as a transport problem and proving equivalence to the discrete 1-Wasserstein.

3.	IDSA shows strong empirical performance, achieving notable gains on cross-subject HAR and sEMG benchmarks, indicating its effectiveness in addressing sensor-wise domain shifts.

### Weaknesses
1.	The evaluation setup in Table 1 seems limited, as the selection of specific source-target domain pairs could be less fair and does not fully test generalization. A fairer and more comprehensive evaluation would be a leave-one-subject-out (LOSO) or leave-one-group-out (LOGO) protocol, where each subject or subject group serves as the target domain in turn. The authors cited some references for this selection, but the selected pairs of this paper differ from those in the referenced papers without a proper explanation.

2.	The proposed decorrelation loss for sparsity is fairly standard and widely used in prior works. While the ablation results indicate that combining it with the transport loss improves performance, it contributes less novelty compared to the main inter-domain sensor alignment component. Also, the connection between the two losses is not clearly explained, making it uncertain whether the observed performance gain results from their genuine interaction or simply from incremental improvements contributed by each loss independently.

3.	My biggest concern lies in the relation of this work to a line of existing works [1,2]. The inter-domain transport formulation shares conceptual similarities with existing methods that align correlation or covariance matrices between the source and target domains of multivariate time-series data, making the degree of novelty somewhat incremental rather than completely novel. For example, [1] also uses sparsity and enforces structural alignment for inter-variable relations.

[1] Time Series Domain Adaptation via Sparse Associative Structure Alignment, AAAI 2021.

[2] Transferable Time-Series Forecasting Under Causal Conditional Shift, TPAMI 2024.

### Questions
1.	Regarding the evaluation setup, how were the specific source-target domain pairs in Table 1 selected? Why are they different from the papers cited in section 6.2, paragraph 1?

2.	Concerning the relations of the two loss functions, could the authors clarify the connection between the spatial transport loss and the decorrelation loss? Is there any explicit interaction or dependency between them, beyond their independent contributions? And is there any further evidence for such a connection?

3.	In relation to prior work, have you directly compared IDSA against existing methods that align correlation or covariance matrices between source and target domains of multivariate time-series data (by changing their concept of multi-variate to multi-sensor for your setting)? Beyond the conceptual difference (multivariate to multi-sensor), what are the substantive differences or advantages that distinguish IDSA from these earlier approaches?

### Soundness
2

### Presentation
3

### Contribution
2
