# JacobiGAD: Jacobi Polynomial–Powered Heterogeneous Graph-Level Anomaly Detection

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Heterogeneous graph-level anomaly detection is vital for applications such as fraud detection and drug discovery, yet remains challenging due to mixed features, complex structures, and severe class imbalance. This paper introduces JacobiGAD, a unified framework that addresses these challenges through three key innovations. First, learnable multiscale filters based on Jacobi Polynomials adapt to different node and edge types, fusing multiple graph views to enhance anomaly signals. Second, these polynomials enable efficient approximation of targeted functions and naturally encode diverse geometries. Third, a Ricci Flow-inspired loss amplifies gradients for rare anomalies, mitigating class imbalance without distorting graph embeddings, ensuring stable convergence. Extensive experiments on real-world benchmarks show JacobiGAD outperforms the best baseline by up to 2.79\% (AUROC), 7.78\% (AUPRC), 7.11\% (Recall@k), and 5.96\% (F1-score) on average.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
JacobiGAD is a framework for heterogeneous graph-level anomaly detection that integrates a Jacobi-polynomial–based graph neural network (JPGNN) with a Ricci Flow Adaptive Curvature Enhancement (RFACE) loss.

### Strengths
* The use of Jacobi polynomials as a learnable graph filter is new; theoretical analysis (Theorems 2–7) proves convergence, information preservation, and bounded approximation error.

* Multi-view fusion via Jacobi filters yields both injectivity and signal-to-noise amplification proportional to the number of views.

* Links Jacobi bases to Laplace–Beltrami eigenfunctions across Euclidean, spherical, and hyperbolic geometries, suggesting geometric generality.

### Weaknesses
* The paper is extremely difficult to follow. There is no clear narrative flow or visual guidance — the entire text consists mainly of dense equations, scattered theorems, and tables of numbers. While the method itself actually follows a relatively straightforward pipeline (JacobiGAD = Feature Alignment + Multi-view Fusion + JPGNN + RFACE Loss), the paper fails to convey this structure clearly. Including an overview figure or intuitive illustrations would make it much easier for readers to grasp. The numerous proofs and theoretical claims should be placed in the appendix rather than interrupting the main story.

* The authors claim that “most existing anomaly detection models can only handle homogeneous graphs,” which is an oversimplified and somewhat misleading statement. Ironically, their own results show that heterogeneous models underperform homogeneous ones, without offering any explanation for this phenomenon. Moreover, the related work section cites multiple heterogeneous graph classification methods, yet the paper asserts that such models cannot handle heterogeneity — this contradiction undermines the central motivation. In reality, many heterogeneous graph-level anomaly detection approaches already exist (e.g., [1–4]), and the authors should clearly position their work among them.

* The authors have not released their code, making it impossible to verify whether the reported training procedures truly match the described algorithm. Additionally, evaluation is limited to performance metrics (AUROC, AUPRC, etc.) without consideration of computational aspects such as training time, memory consumption, or scalability. For a model introducing high-order polynomial filters, such analysis is essential for a fair comparison.

[1]HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks.
[2]Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection.
[3]FiGraph: A Dynamic Heterogeneous Graph Dataset for Financial Anomaly Detection
[4]Deep Graph Anomaly Detection: A Survey and New Perspectives.

### Questions
See weaknesss

### Soundness
2

### Presentation
1

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
This paper studies heterogeneous graph-level anomaly detection, a task complicated by mixed node/edge types, irregular structures, and extreme class imbalance. The authors propose JacobiGAD, a unified framework leveraging learnable multi-scale filters based on Jacobi polynomials to adaptively capture diverse graph patterns and fuse multiple structural views. The polynomial design also enables efficient approximation of targeted functions across different graph geometries. Additionally, a Ricci-Flow-inspired loss is introduced to strengthen gradients on scarce anomalies while preserving stable embedding optimization. Experiments on multiple real-world benchmarks demonstrate improvements over strong baselines.

### Strengths
1. Well-written, no obvious typo.
2. With many theories to prove the effectiveness.

### Weaknesses
1. unsufficient experiment: This paper only report the main experiment in main text and even don't have ablation study.

2. Too many theory: I sincerely admit the importance of propose a theory to explain the effective of the method from the perspective of math, but too many theory seems not appropriate in ICLR, maybe it's suit for AISTATS or some conference focus on theory.

3. Anomaly detection in Heterogeneous Graph seems not a new task and have done by many works. [1,2]

[1] Fast memory-efficient anomaly detection in streaming heterogeneous graphs

[2] Thgnn: An embedding-based model for anomaly detection in dynamic heterogeneous social networks.

### Questions
see Weaknesses

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
3

### Summary
This paper introduces JacobiGAD, a novel end-to-end framework for heterogeneous graph-level anomaly detection (GAD). JacobiGAD proposes two core technical contributions. First, it employs a spectral Graph Neural Network (GNN) that uses learnable Jacobi Polynomials as filters. These filters are designed to adapt to different node/edge types, fuse information from multiple graph views, and capture diverse geometric patterns (Euclidean, Spherical, Hyperbolic). Second, the paper introduces a Ricci Flow-inspired loss function (RFACE) to combat class imbalance by dynamically amplifying gradients for rare anomalous classes. The authors provide a suite of theoretical results to justify their design choices, covering aspects like information preservation, approximation efficiency, and loss convergence.

### Strengths
1. This paper is well-structured and clearly articulates its core problem.
2. The authors test their model on an impressive 15 datasets, including a private industrial dataset, which demonstrates its applicability to real-world problems.

### Weaknesses
1. The method is trained in a supervised strategy, which optimizes for known anomalous modes but offers no explicit mechanism for handling unseen anomaly types or distribution shifts. As a result, the detector may overfit to the labeled anomaly patterns and fail to flag novel or rare patterns at test time. 
2. Theorem 2 claims the "optimal choice" of basis is Jacobi Polynomials. The proof sketch suggests it's an excellent choice due to its flexibility and orthogonality, but calling it "optimal" for any graph distribution is a very strong claim that may not hold universally. It is recommended that the authors provide a more detailed proof.
3. Several datasets and transformations are unclear. For node classification datasets converted into graph-level via BFS, key details (BFS depth, subgraph size, sampling strategy, balancing, multiple seeds) are missing. The listed biological datasets (MCF-7, MOLT-4, etc.) and the very high numbers of node/edge types raise questions about provenance and preprocessing.

### Questions
1. Regarding Theorem 2 (Optimality): The argument for Jacobi Polynomials being "optimal" relies on assumptions about the optimization landscape and spectral density. Could you elaborate on the conditions under which this optimality holds?
2. Could you provide a clearer intuitive comparison between RFACE and Focal Loss? Both seem to achieve a similar goal of up-weighting hard/rare examples. What is the key advantage of the proposed dynamic adjustment based on the loss gradient over a simpler modulation factor based on prediction confidence like in Focal Loss?

### Soundness
2

### Presentation
3

### Contribution
2
