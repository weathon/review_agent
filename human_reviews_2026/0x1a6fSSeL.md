# Modality Matters: Universal Time Series Modeling via Channel Dependency Search

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
The expanding development of wireless and mobile devices results in a proliferation of multivariate time series data, enabling various analytical tasks, e.g., forecasting, classification, and anomaly detection. Most existing time series modeling methods are dedicated to developing task-specific models due to the heterogeneous dimensionalities, resulting in inefficient resource utilization and limited cross-domain transferability. To address this issue, this study achieves a unified paradigm transcending task boundaries and proposes a universal modality-aware Time series modeling framework leveraging Channel Dependency Search named TimeCDS. Specifically, TimeCDS innovatively identifies a certain number of representative features by projecting the heterogeneous time series features into the hierarchical spaces and dynamically modeling their inter-channel relationships to alleviate the heterogeneity issue. A novel time series imaging method is then proposed to automatically introduce the image modality from sequences, facilitating the comprehensive temporal-spatial pattern extraction. Further, a dual-branch architecture is designed to process the sequential data and the visual representations simultaneously, exploiting the complementary cross-modal features through the proposed Cross-Modal Attention and Dynamic Weighted-Averaging. Extensive experiments across different analytical tasks demonstrate the consistently superior performance of TimeCDS, outperforming existing state-of-the-art baselines by up to 15.9%. The code of TimeCDS is publicly available at https://anonymous.4open.science/r/TimeCDS/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TimeCDA, a novel framework designed to address the *heterogeneous dimensionality* problem among different channels in multivariate time series. TimeCDA introduces several key components, including a Channel Dependency Search (CDS) module and a dual-branch architecture, to model inter-channel relationships and fuse numerical and visual representations. The method is evaluated across multiple domains and benchmarks, demonstrating promising performance.

### Strengths
1. While most prior works assume *channel independence*, this paper’s decision to explicitly model inter-channel relationships is refreshing and conceptually meaningful.
2. The proposed Dual-Branch Encoding, together with CMAM and DWAM, achieves a well-balanced integration between the *numerical view* and the *visual (image-based) view* of time series.

### Weaknesses
1. The current discussion lacks a deeper justification for why inter-channel modeling offers advantages over the dominant *channel-independent* approaches (e.g., PatchTST, TimeLLM). A detailed analysis or empirical study highlighting this superiority would strengthen the contribution.
2. The choice of baselines could be broader. Recent strong models, such as iTransformer and N-HiTS, should be included. Additionally, works that similarly combine sequence modeling with image-based representations, such as TimeVLM and DMMV, should be discussed or compared.

### Questions
1. The motivation for introducing the image-based encoder is not sufficiently clear. Could other architectures (e.g., LLM-based encoders) achieve similar benefits?
2. The proposed dual-branch encoding might lead to additional computational overhead. It would be useful for the authors to include a discussion or analysis of time complexity and efficiency.
3. Some reported numbers in Table 1 appear inconsistent with those in the cited references. Were the experiments reproduced using different input windows or settings?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces TimeCDS, a framework for universal time series modeling that combines temporal and spatial representations using multimodal fusion. It addresses challenges in dimensional heterogeneity and captures both temporal dynamics and spatial correlations. The framework features a unique channel dependency search for selecting representative features and a dual-branch encoding architecture. Evaluations across forecasting, anomaly detection, and classification tasks show TimeCDS outperforming existing methods.

### Strengths
1. The paper provides a clear and structured explanation of its methodology, especially the channel dependency search and dual-branch encoding.
2. The framework outperforms state-of-the-art methods in multiple tasks, including forecasting, anomaly detection, and classification.

### Weaknesses
1. The paper introduces many methods but does not clearly explain the motivation behind converting time series data into images. 
2. It would be beneficial to include additional experiments to verify which features image-based methods are particularly good at extracting from time series data.

### Questions
Q1: A couple of confusion regarding channel dependency search:
(1) I’m curious about why we specifically choose K channels. In multi-channel time series, the relationships between the channels are interconnected and serve different purposes. How can we determine that there is always one or several channels that are "most representative"?
(2)  Is the number of channels, K, treated as a hyperparameter in the paper? Considering that the relationships between channels can change dynamically with events, would it make sense for K to also change over time? And could the value of K vary depending on the specific dataset or domain?

Q2: I wonder what the effect of shuffling channels would be on time images encoder. Since images have inherent spatial relationships, do the channels of the multivariate time series also carry spatial significance? Given that different datasets may have channels that represent different physical meanings, would it still be reasonable to convert them into time images to capture spatial information?

Q3: In Section 3.4.4, it is suggested to consider comparing TimeCDS with other image/CNN-based methods, such as the classic InceptionTime, Rocket, and others, to highlight the advantages of the proposed time image encoder.

Q4: It is recommended that the figures 1,2, 3 in the paper be presented as vector images to enhance the clarity and resolution, especially for better scalability in different viewing formats.

### Soundness
2

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
3

### Summary
This paper proposes a channel dependency search module to model time series data under a unified scenario. The proposed framework handles different tasks at the same time and empirical results show its effectiveness.

### Strengths
1. Time series analysis is critical research field with solid motivation, especially under a multi-task scenario.
2. The proposed channel aware searching is reasonable to flexibly adapt into different tasks in a unified time series modeling framework.

### Weaknesses
1. Overall format needs significant improvement. Table is too small to read and figures are squeezed too much. They affect the readability of this draft.
2. It is hard to tell the proposed framework is novel enough, which is more like a combination of previous methods.
3. The term "search" of the proposed method is a little confusing. There is no searching operation, it is more like a graph learning concept.
4. That will be great if the comparison methods can be referred to corresponding papers in the experimental tables.

### Questions
Please check the above section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TimeCDS, a universal, modality-aware framework for multi-task time-series analysis. It attempts to overcome the dimensional-heterogeneity problem across datasets by (i) selecting a fixed number of representative channels via HNSW-based channel-dependency search, (ii) converting the reduced series into an “image” representation, and (iii) fusing temporal and spatial features through a dual-branch Transformer/CNN encoder plus a cross-modal attention module. Extensive experiments on forecasting, anomaly detection and classification report consistent gains over ten or more baselines.

### Strengths
(1) Tackling heterogeneous variable counts and multiple tasks with one model is an important open problem.
(2) Channel-dependency search with soft distance-weighted fusion is new, and the imaging pipeline (periodicity + relation matrix + phase-amplitude) is creative.
(3) Best average MSE/MAE on 8 forecasting sets, highest F1/AUC/PATE on 5 anomaly sets, and top accuracy/F1 on 10 UEA classification sets; ablations show each module contributes.
(4) Well-organized structure, easy-to-follow notation, comprehensive appendix.

### Weaknesses
(1) Dual-branch Transformer+CNN with cross-attention resembles prior vision-language or multimodal models; novelty is mostly in the channel-search and imaging steps.
(2) No justification why 20 channels suffice, no analysis of information loss after discarding N−K channels.
(3) HNSW search + dual-branch forward pass + Cross-Modal Attention Mechanism (CMAM) is heavy; runtime/memory vs. baselines not reported.
(4) Foundation-model baselines (Timer, UniTS) were fine-tuned on individual tasks, whereas TimeCDS uses joint pre-training on UTSD-4G; comparison is therefore slightly favorable to TimeCDS.
(5) Only “w/o branch” and “w/o CMAM” tested; no study on patch size, imaging choices, or HNSW hyper-parameters.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
