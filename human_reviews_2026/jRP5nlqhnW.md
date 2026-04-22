# Multi‑view Adaptive Partitioning with Global Association for Graph Anomaly Detection

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Graph anomaly detection has emerged as a fundamental technique for detecting anomalies in complex relational data across diverse domains.
A key challenge in graph anomaly detection is the difficulty deep models face in representing tabular node attributes. Unlike image and text data, whose samples usually reside on a single, smooth, and differentiable manifold, tabular samples scatter across fragmented manifolds lacking manifold connectedness. This fragmentation violates the local-smoothness assumption that most deep networks rely on, leading to degraded performance. 
To address the above challenge, a Multi-view Adaptive Partition Encoder (MAPE) is proposed. Multiple complementary adaptive partition operators are introduced in MAPE to discretize the feature space and assign learnable embeddings to the resulting subspaces, thereby reducing manifold connectedness.
Furthermore, sharing a sub-space is treated as evidence of high-order affinity between nodes, forming the basis of the proposed Multi-Pattern Global Association (MPGA) module for capturing global dependencies.
Extensive experiments across 10 benchmarks demonstrate that the proposed method consistently outperforms 27 competitive baselines, including recent state‑of‑the‑art models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Multi-view Adaptive Partition Encoder to address the challenge of modeling fragmented tabular node attributes in graph anomaly detection by discretizing the feature space into learnable subspaces, thereby enhancing manifold connectedness. Then, a Multi-Pattern Global Association (MPGA) module is designed to capture the high-order global dependencies.

### Strengths
1. The proposed method achieves good performance compared with 27 baseline methods.
2. The paper is well-organized and the presentation of this paper is good.

### Weaknesses
1. My major concern of this paper is that the motivation of this paper is not sufficiently convincing. The authors claim that the main challenge in graph anomaly detection lies in representing tabular node attributes. However, this issue has already been discussed and addressed in GAAP [1]. Although GAAP does not explicitly use the term tabular node attributes, Figure 1 in that paper conveys a similar concept. Therefore, the authors should clarify why this problem remains a key challenge and how their work differs fundamentally from GAAP in addressing it. Furthermore, the second challenge this paper tries to address is the long-range dependency, which has also been addressed by many existing methods, such as GADAM [2] and UniGAD [3], as mentioned by authors in the related work. 
2. The novelty of this paper is limited. The Multi-Pattern Global Association is a simple combination of local node similarity and global node similarity, a similar idea presented in many existing work, such as GAAP [1], GADAM [2] and UniGAD [3]. 
3. In the related work (line 196), Why does single-relation mechanism, representation-agnostic formulation limits the capability? Is there any existing work supporting this statement?
4. There are many typos in the paper.
- In line 377, "GAAPA" should be "GAAP". In addition, "GAP" in the table 1 should be "GAAP".
- In line 239 "Eq. equation 1-equation 2" -> "Eq. 1-2".
- In line 248 "Eq. equation 1" -> "Eq. 1".
- In line 250 "Eq. equation 2" -> "Eq. 2".
- In table 1, the performance of GAP and MAPGA are both 99.69 on the T-soc dataset, while GAP is underlined as the second best method.
5. In Figure 2, the authors should visualize the t-SNE embedding of other stronger baseline methods, such as GAAP, DGA-GNN, XGBGraph. MLP and XGBoost are weak baseline methods based on the experimental results shown in table 1 and they are not even designed for GAD task.
6. As shown in Table 3, the node feature type of some datasets (e.g., Reddit, Weibo, and Questions) is text embedding. The proposed method is designed for tabular features. How does the proposed method deal with text embedding feature?
7. In the experiment evaluation, the authors should report the standard deviation as well.
8. The experimental setup lacks sufficient clarity. For example, the paper does not specify how the training and test data are split, making it difficult to assess the validity and reproducibility of the reported results.

[1] Mingjiang Duan, Da He, Tongya Zheng, Lingxiang Jia, Mingli Song, XinyuWang, and Zunlei Feng. Global attribute-association pattern aggregation for graph fraud detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 11616–11624, 2025.

[2] Jingyan Chen, Guanghui Zhu, Chunfeng Yuan, and Yihua Huang. Boosting graph anomaly detection with adaptive message passing. In The Twelfth International Conference on Learning Representations, 2024a.

[3] Yiqing Lin, Jianheng Tang, Chenyi Zi, H Vicky Zhao, Yuan Yao, and Jia Li. Unigad: Unifying multi-level graph anomaly detection. Advances in neural information processing systems, 37: 136120–136148, 2024.

### Questions
1. In equation 6, I assume both $N_{out(m)}$ and $N_{in(m)}$ refer to the set of neighbors, but what are exactly $N_{out(m)}$ and $N_{in(m)}$?
2.  As shown in Table 3, the node feature type of some datasets (e.g., Reddit, Weibo, and Questions) is text embedding. The proposed method is designed for tabular features. How does the proposed method deal with text embedding feature? Since these datasets are not tabular attributes graph, why does the proposed method outperforms other baseline methods on these datasets?
3. In section 2.1, the authors mention that without manifold connectedness, the local smoothness prior yields ill-conditioned optimization dynamics and results in slow or unstable convergence. Then, what is the training time or time complexity of the proposed method?

### Soundness
2

### Presentation
3

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
This paper addresses the challenges in GAD, specifically the manifold fragmentation of tabular node attributes and the difficulty in modeling global dependencies in graph topology. The authors propose MAPGA, a framework comprising a Multi-view Adaptive Partition Encoder (MAPE) to discretize the feature space and restore manifold connectedness, and a Multi-Pattern Global Association (MPGA) module to capture long-range dependencies via representation-based and behavior-pattern graphs.

### Strengths
1.The multi-view discretization mechanism is technically novel and directly addresses manifold disjointedness.

2.This paper offers a novel perspective on why GNNs struggle with tabular attributes, enhancing the paper's conceptual contribution.

3.Extensive experimental results demonstrate the effectiveness of the method.

### Weaknesses
1. The writing needs further improvement. For example, the introduction introduces the challenges of modeling tabular data, but does not adequately discuss its relationship to GNNs. Why use GNNs to model tabular data? How is tabular data constructed as a graph? Where do the edges of the original graph come from?

2. The description of the method is not detailed and clear enough. For example, how does MAPE instantiate K independent views? Why is this design used?

3.A large number of views and partitions are generated, which may result in large computing resource requirements and affect the deployment of large-scale datasets.

### Questions
Please refer to Weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses graph anomaly detection by tackling two key challenges: (1) the difficulty of modeling tabular node attributes due to manifold fragmentation, and (2) capturing global dependencies in graph structures. The authors propose MAPGA, which consists of two main components: a Multi-view Adaptive Partition Encoder (MAPE) that discretizes the feature space through learnable partitions to establish manifold connectedness, and a Multi-Pattern Global Association (MPGA) module that captures global dependencies through representation-based and behavior-based association graphs. Extensive experiments on 10 benchmarks against 27 baselines demonstrate consistent improvements, with an average AUROC of 90.63% (+2.02pp over the best baseline) and particularly strong gains in AUPRC (average 19.6% improvement).

### Strengths
1. Clear Problem Motivation The paper provides a novel theoretical perspective by explaining the fundamental difficulty in modeling tabular node attributes through the lens of manifold connectedness. The visualization in Figures 1-2 effectively demonstrates how tabular data exhibits fragmented manifolds compared to perceptual data (audio, image, text), which violates the local-smoothness assumption required by deep networks.

2. Well-Designed Methodology
* MAPE employs learnable adaptive partitions to restore manifold connectivity through discrete-semantic embeddings
* MPGA captures global dependencies from complementary sources: representation-based co-assignment and behavior-based neighborhood patterns
* The two modules exhibit strong complementarity, as evidenced by ablation results

3. Comprehensive Experimental Evaluation
* 10 public benchmark datasets covering diverse application scenarios (social networks, financial fraud, cryptocurrency, crowdsourcing)
* 27 competitive baselines including recent SOTA methods (GAP, DGA-GNN, ConsisGAD, GGAD)
* Three evaluation metrics (AUROC, AUPRC, Rec@K) addressing different aspects of anomaly detection performance
* Thorough ablation studies examining both architectural components and hyperparameters

4. Consistent and Significant Results MAPGA achieves best or second-best performance across all datasets, with particularly impressive results: 99.20% AUROC on Elliptic (+5.58pp over best baseline), 99.54% on Weibo, and 99.96% on T-Social. The average improvements (+2.02pp AUROC, +1.93pp AUPRC, +2.98pp Rec@K) demonstrate robust superiority.

5. Effective Visualization The t-SNE projections in Figures 1, 2, and 5 provide intuitive evidence for the manifold fragmentation problem and demonstrate how MAPE successfully bridges disconnected clusters to form a connected manifold.

### Weaknesses
* The paper lacks a formal theoretical justification for why adaptive partitioning restores manifold connectedness. What mathematical properties guarantee this restoration?
* While the conclusion acknowledges computational overhead from multiple parallel partition operators, detailed time/space complexity analysis is missing
* The impact of decision tree initialization on final performance is not thoroughly analyzed

### Questions
* Can you provide a theoretical analysis or at least an intuitive explanation for why adaptive partitioning restores manifold connectedness? How do you formally define and quantitatively measure manifold connectedness? Are there any theoretical guarantees on the quality of the learned partitions?
* How is the decision tree depth determined during initialization? Does it significantly affect final performance?
* In Eq. 1, what specific smooth surrogate is used for the Heaviside function during backpropagation? How is α chosen?

### Soundness
3

### Presentation
3

### Contribution
3
