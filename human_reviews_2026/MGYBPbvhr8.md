# Combining Euclidean and Hyperbolic Representations for Node-level Anomaly Detection

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Node-level anomaly detection (NAD) is challenging due to diverse structural patterns and feature distributions. As such, NAD is a critical task with several applications which range from fraud detection, cybersecurity, to recommendation systems. We introduce Janus, a framework that jointly leverages Euclidean and Hyperbolic Graph Neural Networks to capture complementary aspects of node representations. Each node is described by two views, composed by the original features and structural features derived from random walks and degrees, then embedded into Euclidean and Hyperbolic spaces. A multi Graph-Autoencoder framework, equipped with a contrastive learning objective as regularization term, aligns the embeddings across the Euclidean and Hyperbolic spaces, highlighting nodes whose views are difficult to reconcile and are thus likely anomalous. Experiments on four real-world datasets show that Janus consistently outperforms shallow and deep baselines, empirically demonstrating that combining multiple geometric representations provides a robust and effective approach for identifying subtle and complex anomalies in graphs. We publicly release our source code at https://anonymous.4open.science/r/JANUS-5EDF/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce Janus, a framework that jointly leverages Euclidean and Hyperbolic Graph Neural Networks to capture complementary aspects of node representations for unsupervised graph anomaly detection. Given experiments show the effectiveness to some extent.

### Strengths
1. The authors introduce a framework that jointly leverages Euclidean and Hyperbolic Graph Neural Networks for unsupervised graph anomaly detection.
2. Given experiments show the effectiveness to some extent.

### Weaknesses
1. The design should be further explained. For example, in the algorithm, the authors tend to use the loss as the anomly score for each node, but during the training procedure, the minimization of loss will lead to a low score for all the nodes, which makes it difficult for the framework to learn useful signals. Furthermore, directly setting the coefficient of adjacency reconstruction to 0 during inference requires reasonable explanations. 
2. The compared baselines are not comprehensive enough. The authors mainly consider the old baselines in the unsupervised anomaly detection area. They should include novel baselines, such as [1-3]. 
3. The compared datasets are not representative enough. The included datasets can not show the effectiveness of the framework as there are sevearl real-world anomaly detection datasets in [4]. The authors should conduct further experiments on those datasets, especially the largest ones, to simulate the real deployment. 
4. The authors should provide hyperparameter analysis. Without hyperparameter analysis, it is hard to see the influence of different hyperparameters. 
5. The authors should explain how they chose hyperparameters. As shown in Table 4, they utilize a grid search technique for the training procedure. However, it is difficult for unsupervised learning to decide which hyperparameters should be chosen. Besides, they should also explain how they choose hyperparameters for new datasets. 

[1] Hezhe Qiao, Guansong Pang. Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection. NeurIPS 2023. 

[2] Jingyan Chen, Guanghui Zhu, Chunfeng Yuan, Yihua Huang. Boosting Graph Anomaly Detection with Adaptive Message Passing. ICLR 2024. 

[3] Xiangyu Dong, Xingyi Zhang, Yanni Sun, Lei Chen, Mingxuan Yuan, Sibo Wang. SmoothGNN: Smoothing-aware GNN for Unsupervised Node Anomaly Detection. WWW 2025. 

[4] Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. NeurIPS 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes a node-level graph anomaly detection method called Janus. It combines a Euclidean GNN and a hyperbolic GNN with a cross-geometry contrastive regularizer. The loss combining alignment across views and feature/adjacency reconstruction is used both for training and as the anomaly score. Experiments on four real-world datasets show improvements over shallow and deep baselines. Ablation studies are presented to show contributions of both components.

### Strengths
1. This paper takes an interesting approach: using mixed-curvature representations to improve node-level anomaly detection. The design of using two geometries is well-motivated and shown to be effective.
2. The real-world datasets, including a large, dense finance graph, make the problem meaningful in practice.
3. The inclusion of algorithmic pseudocode and a link to the implementation enhances reproducibility and transparency.

### Weaknesses
1. While the combination of Euclidean and hyperbolic spaces for anomaly detection is interesting, mixed-curvature and product-space ideas have already appeared in other contexts. The anomaly detection framework is also not new. The novelty here lies primarily in applying the mixed-curvature design to the existing node-level anomaly detection framework and is thus limited.
2. Additional ablation can be done to check the contribution of each geometry, say considering Euclidean-only and hyperbolic-only variants of the model.
3. The authors do not justify why hyperbolic space is appropriate for the chosen datasets. Evidence such as hierarchical graph structures, degree distributions, or hyperbolicity analysis could better motivate the use of non-Euclidean embeddings.
4. Eq (12) is specifically for GCN, not a general GNN.
5. This paper does not have sensitivity analysis for the key hyperparameters.
6. The baseline methods seem weak for the chosen datasets. Several baseline methods yield ROC-AUC $< 0.5$ or $\sim 0.5$, implying either inadequate tuning or that the methods are not suitable for the datasets. For instance, CONAD and CARD were originally evaluated on different datasets and achieved much higher performance. To strengthen credibility, the authors should either introduce stronger baseline methods or include more datasets as presented in CONAD and CARD.

### Questions
1. [W2] Can you perform ablation studies to check the contribution of each geometry?
2. [W3] Why is hyperbolic space suitable for the datasets you consider?
3. [W5] Can you perform sensitivity analysis for the key hyperparameters?
4. [W6] Can you either introduce stronger baseline methods or include more datasets as presented in CONAD and CARD?

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
The paper proposes the Janus framework, which claims to improve the performance of node-level anomaly detection by jointly using Euclidean and hyperbolic graph neural networks to capture complementary features of node representations. Its core design involves constructing two views of raw features and structural features for each node, embedding them in Euclidean and hyperbolic spaces respectively, and then aligning the embedding space with the graph autoencoder combined with the contrast learning object. Finally, the nodes that are difficult to coordinate with the views are determined as anomalies.

### Strengths
1.	This paper explores the application of “dual geometric spaces” in the NAD task, offering a novel technical approach for this field.

2.	The paper is well-structured and contains no obvious errors.

### Weaknesses
1. Motivation Lack of Necessity Argument: The paper does not address the core question of "why a single geometric model cannot meet the needs of NAD".

2. The baseline model selection is outdated and has not been compared with the latest methods.

3. The paper does not provide a model diagram, making it difficult to gain a clear understanding of the overall model architecture.

### Questions
1.	Can the author provide a stronger theoretical justification for why combining Euclidean and hyperbolic spaces is effective for anomaly detection tasks?

2.	The introduction of hyperbolic space will increase the computational cost. Please compare the parameters, training time, and inference time between Janus and the baseline model, explain the trade-off between the model's efficiency and performance, and prove its applicability in real scenarios.

### Soundness
2

### Presentation
1

### Contribution
2
