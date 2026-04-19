# Harnessing Spatial Dependency for Domain Generalization in Multivariate Time-series Sensor Data

- Decision: Reject
- Scores: 3, 5, 6, 5

## Abstract
Multivariate time-series (MTS) data from multiple sensors often vary across domains due to factors like sensor misalignment, reattachment, or individual differences, posing significant challenges for domain generalization (DG). Existing methods inadequately address the alignment of domain-specific spatial dependencies across different domains in MTS data, as they often assume a unified invariant spatial structure and overlook the distributional discrepancies arising from varying sensor relationships. To address this limitation, we propose ASAM (Adaptive Spatial Dependency Alignment in MTS Data for Domain Generalization), a novel framework that adaptively aligns spatial dependencies across domains. ASAM proposes a DG layer with domain generalization loss function and two-view regularization loss functions to align spatial dependencies between domains adaptively. We adopt a two-phase approach to align different sets of domains effectively. An input-aware graph generation process and a GNN-based DG layer, coupled with the domain generalization loss function, adaptively align the spatial dependencies learned in the second phase with those from the first phase, ensuring a more precise alignment. We additionally incorporate a two-view reg- ularization method to effectively capture underlying spatiotemporal information comprised of spatial decorrelation loss and Gaussian kernel loss. Our theoretical analysis demonstrates that ASAM effectively assimilates information bottleneck, ensuring robustness across diverse distributions. Extensive evaluations of the four real-world datasets show ASAM outperforms ten recent baselines. To the best of our knowledge, this work is among the first to explore DG approaches for MTS data by focusing on spatial dependency alignment. Our code is available at https://anonymous.4open.science/r/ASAM.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes dynamic contrastive learning method for time series representation learning, which  harness every time step in a sequence as positive and negative pairs. Experiments are carried on clustering and classification tasks.

### Strengths
1. The paper is well-structured and clear.
2. theoretical insights and codes are provided.

### Weaknesses
1. The abstract is too long and lack readability.

2. The novelty and contribution of this work is unclear and need more clear presentation

3. Comparison and discussion of some relevant methods are needed, including:

   [1] SEA++: Multi-Graph-based High-Order Sensor Alignment for Multivariate Time-Series Unsupervised Domain Adaptation[J]. arXiv preprint arXiv:2311.10806, 2023.

   [2] CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series[C]//Forty-first International Conference on Machine Learning.

   [3] Source-free domain adaptation with temporal imputation for time series data, ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023

4. What is the motication of Channel Decorrelation Loss? Variable-wise correlations commonly exist in MTS, and they are representative properties.

5. More datasets should be evaluated in the experiments, e.g., SleepEDF, Boiler ...

6. The experimental setting is unclear.

7. Model computational effort analysis and experiments should be provided.

### Questions
Please refer to weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents a model named ASAM to address the domain generalization setting for multivariate time-series data analysis tasks. ASAM addresses these challenges by introducing a DG layer and a loss function to align spatial dependencies across domains. The method utilizes a two-phase approach: first capturing spatiotemporal features and then aligning spatial dependencies using a GNN-based architecture. Then a two-view regularization method is introduced to further capture underlying spatial dependency in both phases and is comprised of spatial decorrelation loss and Gaussian kernel loss. The model is evaluated on four real-world datasets, demonstrating superior performance over recent baselines in tasks like gesture recognition and human activity recognition.

### Strengths
S1. The paper introduces ASAM, a model specifically designed to address spatial dependency alignment in domain generalization setting for multivariate time-series data. The proposed model is claimed to be the pioneering work in this topic.

S2. The paper provides a theoretical analysis demonstrating how the DG layer assimilates invariant information, which provides foundation for domain generalization in multivariate time series data.

S3. Experiments show that the proposed model outperforms ten recent baselines across four datasets on two tasks.

### Weaknesses
W1. I am not quite convinced by the motivation of the problem statement. Domain generalization is typically critical in scenarios where no data is accessible for target domains. However, in this paper's tested scenarios, including gesture and human activity recognition, datasets seem to be reasonably accessible. I doubt whether domain generalization use case really applies to these two tasks.

W2. While the proposed model is claimed to address domain generalization for multivariate time series data. The evaluation is conducted on datasets which mainly focused on human activity and gesture recognition tasks. Broader validation on a wider variety of domains could strengthen the generalizability claims of the framework. For example, multivariate time series from traffic sensors also exhibit spatial dependencies. 

W3. For the source domain, it is not clear how the spatial dependency graph is constructed for the source domain data. If it is a KNN graph, it seems meaningless to derive the spatial dependencies for the target domain. We just need to use this rule to derive the graph for the target domain.

W4. The impact of the spatial and temporal modeling layers on the proposed model is not sufficiently explored. The paper uses basic GNN + LSTM architectures, but it would not clear how the model performs different variants of spatiotemporal models, such as more advanced models for human activity recognition.

W5. There seems to be a conflict between the function of the spatial layer and the channel decorrelation loss. The spatial layer employs message passing among different nodes (sensors) according to the spatial dependency graph. However, the representations, which encode the spatial correlations, are processed to approximate identity matrix after inner product. How can the inner product produce identity matrices for entries that are already produced by considering spatial dependencies?

### Questions
Please clarify the comments in W1-W5.

### Soundness
3

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
The authors proposed a simple but interesting framework to incorporate spatial information into domain generalization. To align the spatial information between different domains, the authors split the domains as two parts. A model is trained with one part and further tuned by another part. The input-aware graph generation is designed for the tuning process.

### Strengths
1.	To align the spatial information between different domains, the authors split the domains as two parts. A model is trained with one part and further tuned by another part. The input-aware graph generation is designed for the tuning process. Though it can be further improved, the idea is interesting.

2.	Experiments are comprehensive.

### Weaknesses
1.	The definition of spatial information is unclear. In this work, the authors align the sensor features between domains by reducing their distances, so it seems that the authors treat the node features as the spatial information given they are updated by propagation. However, in my opinion, the spatial information should include both the sensor features and the spatial structure between these sensors. In this method, the spatial structure, i.e., their edges, cannot be aligned explicitly.  

2.	Fig 1 is not informative. What do you mean by a, b, c, d and A, B, C, D. How do you argue 'ours' is better than (a) and (b). These are all not clear.

3.	Regarding Eq. (5), I dont understand why reducing the loss can alleviate the redundant data. D = ZZ^T should be like the correlations between sensors. I think it would remove the correlation information  when reducing the loss. Can the authors explain this?

4.	Relevant work is not discussed [1], which also transfer spatial information during alignment, though it was designed for UDA.

5.	Is there any rule to make better split for phase 1 and 2?


[1] Wang Y, Xu Y, Yang J, et al. Sea++: Multi-graph-based higher-order sensor alignment for multivariate time-series unsupervised domain adaptation[J]. IEEE transactions on pattern analysis and machine intelligence, 2024.

### Questions
See abve

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduces a framework called ASAM that enhances the generalization of multivariate time-series sensor data across different domains by adaptively aligning spatial dependencies. ASAM employs a two-phase approach, including a domain generalization layer and loss functions designed for spatial alignment, to capture and align spatial dependencies across domains. Experimental results demonstrate that ASAM outperforms several recent baseline models on multiple real-world datasets, showing its effectiveness in domain generalization tasks.

### Strengths
**Innovation and Originality:** This paper focuses on the domain generalization problem in multivariate time series sensor data, particularly addressing spatial dependency alignment. This is a noteworthy research topic.

**Theoretical Analysis:** This paper provide a theoretical analysis of how the domain generalization layer and loss functions promote domain generalization, which enhances the credibility of the approach and offers a theoretical foundation for domain generalization.

**Experimental Design and Results:** The experimental section covers multiple real-world datasets and compares the method against several baseline models, demonstrating the effectiveness and superior performance of ASAM across various datasets.

**Code and Resources:**  A link to the code is provided, which aids the community in reproducing and further exploring this work.

### Weaknesses
**Weaknesses:**

1. **Motivation and Insight:** The concept of "alignment" as presented seems somewhat unconvincing. Given the heterogeneity of time series data, can spatial dependencies across different domains truly be aligned? What specific issues does the alignment process aim to address? Does it imply an assumption that the source domain should contain distributions similar to those in the target domain?

2. **Writing Clarity:** While the proposed method is theoretically innovative, certain sections, especially regarding the algorithmic implementation details and explanations of mathematical formulas, could be more clear and detailed. Symbols should be well-defined upon first appearance. For instance, in Section 3.2, what does $F$ represent in $C \in \mathbb{R}^{F \times T}$? Additionally, why do the so-called representations $Z \in \mathbb{R}^{N \times T}$ share the same dimensions as the input data $X$? Typically, representations exist in a hidden space of dimension $D$. The appearance of subscripts and superscripts should also be explicitly explained to clarify their role in the workflow. Such issues make the complex algorithm harder to understand. Significant improvements in the writing of the methods section are necessary.

3. **Additional Concerns:**
   - Is the so-called DG layer merely a simple GCN? As presented, it is unclear how they differ.
   - Furthermore, does simultaneously optimizing both the graph structure and the GNN lead to potential bi-level optimization issues? [1][2][3]
   - The authors could consider conducting additional generalization experiments or discussions to more objectively evaluate the "alignment" strategy and better position this work. For example, under what circumstances (e.g., when there is a substantial gap between the source and target domains) might the alignment approach become ineffective?

[1] Learning Discrete Structures for Graph Neural Networks  
[2] Discrete Graph Structure Learning for Forecasting Multiple Time Series  
[3] Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3
