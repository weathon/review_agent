# Escaping the Homophily Trap: A Threshold-free Graph Outlier Detection Framework via Clustering-guided Edge Reweighting

- Decision: Accept (Poster)
- Scores: 2, 2, 4, 8

## Abstract
Graph outlier detection is a critical task for identifying rare, deviant patterns in graph-structured data. 
However, prevalent methods based on graph convolution are fundamentally challenged by the ''Homophily Trap'': the aggregation of features from neighboring nodes inadvertently contaminates the representations of normal nodes near anomalies, blurring their distinctions. 
To overcome this limitation, we propose a Clustering-guided Edge Reweighting framework for Graph Outlier Detection (CER-GOD), which jointly optimizes a self-discriminative masking spoiler with an adaptive clustering-based outlier detector. 
The masking spoiler learns to selectively weaken the influence of heterogeneous neighbors, preserving the discriminative power of node embeddings. 
This process is guided by the clustering detector, which generates pseudo-labels in an unsupervised manner, thereby eliminating the need for predefined anomaly thresholds. 
To ensure robust optimization and prevent class collapse—a failure mode exacerbated by the homophily trap—we introduce a diversity loss that stabilizes the clustering process. 
Our end-to-end framework demonstrates superior performance on multiple benchmark datasets, establishing a new state-of-the-art by effectively dismantling the homophily trap.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a Clustering-guided Edge Reweighting framework for Graph Outlier Detection (CER-GOD), which jointly optimizes a self-discriminative masking spoiler with an adaptive clustering-based outlier detector. Experiments show their improvement to some extent.

### Strengths
1. The authors propose a cluster-based edge reweighting framework for unsupervised graph outlier detection. 
2. Experiments show their improvement to some extent.

### Weaknesses
1. Although the authors provide a time complexity, the N-squared complexity might be a hindrance for real deployment. It will be better if the authors can provide an experimental comparison. 
2. The included datasets are relatively small compared to the newest anomaly (outlier) detection datasets, such as those in [1]. The authors should include the real-world datasets in their comparison. 
3. The included baselines are not comprehensive, which questions the effectiveness of the framework. The authors should include SOTA works such as [2], [3], and [4].
4. Figure 3 shows that the method can be very sensitive to the variation of hyperparameters. As stated in Appendix G, they adopt grid search for finding the hyperparameters. It can be a question of how to utilize grid search for unsupervised learning without any ground truth labels. Furthermore, it can be difficult for the method to choose proper hyperparameters to operate on new datasets. 
5. The authors should show the finetuned hyperparameters and how many times they conduct the experiments to search for the hyperparameters. 

[1] Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. NeurIPS 2023. 

[2] Hezhe Qiao, Guansong Pang. Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection. NeurIPS 2023. 

[3] Jingyan Chen, Guanghui Zhu, Chunfeng Yuan, Yihua Huang. Boosting Graph Anomaly Detection with Adaptive Message Passing. ICLR 2024. 

[4] Xiangyu Dong, Xingyi Zhang, Yanni Sun, Lei Chen, Mingxuan Yuan, Sibo Wang. SmoothGNN: Smoothing-aware GNN for Unsupervised Node Anomaly Detection. WWW 2025.

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
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces CER-GOD, a graph outlier detection framework designed to mitigate the "Homophily Trap" in graph convolutional networks (GCNs), where aggregating features from anomalous neighbors contaminates normal node representations. The method combines a self-discriminative masking spoiler that adaptively re-weights edges to reduce heterogeneous influences, guided by a learnable clustering layer that generates pseudo-labels without predefined thresholds. A diversity loss prevents class collapse during clustering. The overall objective balances reconstruction, clustering, distribution repulsion, and diversity terms. Experiments on six datasets show superior AUC performance over baselines. Contributions include: (1) analysis of the homophily issue with a masking mechanism, (2) threshold-free pseudo-labeling via clustering, and (3) a regularization to stabilize optimization.

### Strengths
The paper creatively combines edge re-weighting with unsupervised clustering for outlier detection, extending the "Homophily Trap" concept (He et al., 2024) into a joint optimization framework. The diversity loss is a novel tweak to address a specific failure mode in clustering under homophily constraints, potentially applicable beyond outliers.

### Weaknesses
1. The paper's core innovation—adaptive edge masking guided by binary clustering—feels incremental rather than transformative, building too closely on existing ideas without sufficient novelty. For instance, the masking spoiler resembles attention mechanisms in GAT or adaptive topology learning, but lacks a rigorous comparison showing why global MMD guidance (Eq. (8)) outperforms local feature-based weighting. 
2. The "self-discriminative" claim is undersold: it primarily enforces intra-cluster aggregation via pseudo-labels, which is akin to DEC (Guo et al., 2017) but without justifying why two clusters suffice for diverse anomaly types (e.g., point vs. structural outliers). 
3. Proposition 1, while neat, is a straightforward extension of over-squashing bounds (Topping et al., 2022) and does not uniquely motivate the framework—empirical validation in Fig. 1 is limited to one dataset (Email) and single-layer GCN, ignoring multi-layer or heterophilic graphs.

### Questions
1.	Can the authors clearly explain how the proposed masking differs mathematically from the attention coefficients in GAT or the relation-strength function in ADA-GAD?
2.	How sensitive is the model to the initial K-Means clustering used for pseudo-label initialization?
3.	Does the method scale to larger graphs (e.g., >100k nodes)? Have the authors considered mini-batch or approximate MMD computation?
4.	Can the authors provide quantitative evidence (e.g., correlation coefficients) showing that the learned edge weights actually correlate with homophily or anomaly boundaries?

### Soundness
2

### Presentation
2

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
This paper proposes a learnable clustering-guided edge reweighting framework that jointly optimizes a self-discriminative masking module and an adaptive clustering-based outlier detector. The clustering process generates pseudo labels in an unsupervised manner, which are then utilized to guide the training process.

### Strengths
(1) The method is straightforward to understand, but the paper’s writing quality needs improvement.  am particularly confused about the role of the pseudo labels. 

(2) The proposed method achieves the best performance across six small-scale graph anomaly detection  datasets.

### Weaknesses
(1) The authors should provide a clear technical definition or explanation for the newly introduced concepts “multi-hop away from anomalies” and “1-hop away from anomalies.” For instance, a 1-hop-away node should refer to a normal node that is directly connected to at least one anomalous node.
Additionally, the description of “data distribution” in the framework is misleading — it should be clarified as the anomaly score distribution or MMD score distribution if you use the histogram. Moreover, the current explanation of the framework implies that only two clusters are considered in the diversity loss. The role of the pseudo-labeling process is also not clearly reflected in the framework diagram, and the legend is missing. The type of latent embedding *z* is not consistent with the main paper. The authors should revise the figure and explanation to make these components explicit.

(2)Most of the benchmark datasets used are relatively small, making the experimental results less reliable—especially on Enron, which contains only five anomalies. I suggest that the authors include larger-scale datasets, covering both more injected or real-world data with real anomalies, to better demonstrate the robustness and effectiveness of the proposed method.

(3) The approach appears to focus primarily on the reconstruction process and the learning of discriminating representations.  The unsupervised clustering module is learnable, and the inference relies on the distance to the centroids. In this case, it is unclear why a reconstruction component is still necessary. Why not directly apply clustering to the graph representations learned by the GCN encoder? Moreover, the ablation study is not sufficiently comprehensive—additional variants, like directly applying the learnable clustering, should be included for a more thorough evaluation

(4)Determining the appropriate number of clusters for each dataset is challenging. How do the authors decide on the cluster count in practice? Additionally, the reconstruction process is computationally complex, and the inclusion of clustering further increases the overall computational cost. Besides, the performance with varying the number of clusters should be provided.

(5) As you mention in lines 244-247, ”Then we first designate the cluster containing a relatively larger number of samples as the normal cluster, and temporarily treat all nodes within it as normal candidates. Conversely, the remaining clusters are considered the anomalous candidate cluster“.  How many large clusters are treated as normal nodes? In real GAD datasets, the data distribution often consists of multiple normal clusters with relatively large sample sizes, along with multiple anomalous clusters and some isolated outlier points.

### Questions
See above **Weaknesses**

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes CER-GOD, a novel graph outlier detection framework to address the '' Homophily Trap'', which is a critical issue where graph convolutional operations blur the feature representations of normal and anomalous nodes that are neighbours. The proposed method synergistically integrates two main components: a self-discriminative masking spoiler that learns to reweight graph edges to suppress contaminating information flow from heterogeneous neighbours, and a clustering-based outlier detector that generates unsupervised pseudo-labels to guide this reweighting process. To ensure stable training and prevent clustering collapse, a diversity loss is introduced as a regularization term. Extensive experiments on multiple benchmark datasets demonstrate that CER-GOD significantly outperforms a wide range of state-of-the-art baselines.

### Strengths
1. The paper addresses a critical and well-articulated problem in GNN-based anomaly detection: the "Homophily Trap". The authors provided clear motivation, supported by insightful empirical analysis (Figure 1), highlighting how neighbourhood aggregation can contaminate node representations and fundamentally hinder outlier identification.
2. The proposed CER-GOD framework is methodologically sound and the idea is novel. The main innovation lies in the synergistic joint optimization of a self-discriminative masking spoiler and a clustering-based detector. This design creates a powerful feedback loop where pseudo-labels from clustering guide the edge reweighting, and the refined graph structure, in turn, yields more discriminative embeddings for improved clustering.
3. The comparison against a comprehensive set of baselines fully validates the effectiveness of CER-GOD. Additionally, the authors provided convincing qualitative evidence, such as t-SNE and mask visualizations, to further enhance the  persuasiveness and interpretability of the approach.
4. The authors provided the implementation code of the proposed method, which increases the reproducibility.

### Weaknesses
1. The choice of the Chebyshev distance for the MMD kernel calculation should be elaborated. While the intuition is provided, the paper would be strengthened by an empirical comparison against a more conventional Euclidean-based RBF kernel to justify this specific design.

2. The diversity loss $l_{diversity}$ introduces a crucial hyperparameter $\epsilon$ to control the minimum proportion of samples per cluster. However, there is no discussion regarding the setting of $\epsilon$ or any corresponding parameter sensitivity analysis.

3. A comparison with existing graph rewriting baselines is missing, which would help validate the method's effectiveness.

### Questions
1. Please also include a parameter sensitivity analysis examining how the distribution repulsion loss performs when applied to different graph convolutional layers.
2. The paper distinguishes the edge reweighting mechanism from graph rewriting methods. Could the authors further elaborate on the fundamental advantages of such learnable masks over heuristic-based hard edge removal or addition? A deeper discussion would be valuable.
3. The framework relies on a feedback loop where pseudo-labels guide the mask optimization. In the early training phase, these pseudo-labels might be noisy, which potentially leads the model to a suboptimal solution. How does the model ensure stability during the training phase? Is there a warm-up phase, or does the joint optimization naturally navigate towards a good solution?

### Soundness
3

### Presentation
4

### Contribution
4
