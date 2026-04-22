# SHAKE-GNN: Scalable Hierarchical Kirchhoff-Forest Graph Neural Network

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Graph Neural Networks (GNNs) have achieved remarkable success across a range of learning tasks. However, scaling GNNs to large graphs remains a significant challenge, especially for graph-level tasks. In this work, we introduce SHAKE-GNN, a novel scalable graph-level GNN framework based on a hierarchy of Kirchhoff Forests, a class of random spanning forests used to construct stochastic multi-resolution decompositions of graphs. SHAKE-GNN produces multi-scale representations, enabling flexible trade-offs between efficiency and performance. We introduce an improved, data-driven strategy for selecting the trade-off parameter and analyse the time-complexity of SHAKE-GNN. Experimental results on multiple large-scale graph classification benchmarks demonstrate that SHAKE-GNN achieves competitive performance while offering improved scalability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies large-scale graph-level tasks. SHAKE-GNN is proposed, which is a subgraph sampling method using Kirchoff Forests. A data-driven strategy is used for selecting the trade-off parameter.

### Strengths
S1: The writing is clear.

S2: The method is easy-to-follow.

S3: Hyperparameters are listed clearly.

### Weaknesses
W1: The motivation is not clear. Using subgraph sampling/coarsening/sparsification/condensation is common in scalable gnn, but the motivation of using Kirchhoff Forests for graph-level task is unclear.

W2: Some statements need formal proof. See questions.

W3: Experiments require considerable effort to improve.

### Questions
Q1: Why use Kirchhoff Forests for graph-level task? Is there any special reason that it will work better on graph-level tasks than standard node-classification?

Q2: Line 377, this statement is made: "Since $r(q)<1$, the total complexity of SHAKE-GNN is strictly lower than that of a standard GNN". Can you provide a formal proof on that? This does not seem trivial.

Q3: This paper studies large-scale graph-level tasks, but the datasets used in the experiments are small. The average number of node per graph is only 25.5 for the MolHIV datasets. Can you also provide statistics of each datasets as a table?

Q4: No proper baselines are selected. The proposed method is only compared with the vanilla GCN. Scalable graph learning methods should be added as baselines.

Q5: Is there setting difference between this paper and the official ogb leaderboard? It seems like lots of works achieve 80+ score in early 2021.

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
2

### Summary
This paper studies the problem of scaling graph neural networks for large graph classification tasks and proposes a hierarchical framework based on Kirchhoff random forests. The method constructs multi-resolution graph representations by progressively coarsening the graph using probabilistically grounded Kirchhoff Forests, and performs message passing from coarse to fine levels. Experiments on multiple real-world datasets show that SHAKE-GNN significantly reduces training time while maintaining the performance of the original model.

### Strengths
1. The idea of Kirchhoff Forest provides a theoretically grounded way to generate multi-resolution graph structures, which can potentially inspire future studies.
2. The proposed strategy for choosing the parameter q is interesting and allows the framework adaptive to different datasets.

### Weaknesses
I don’t have too many questions regarding the methodology but have some concerns on the experiments:

1. All experiments are conducted using GCN backbones and it is unclear how this framework will perform across different GNN variants.
2. The experiments are primarily conducted on standard graph classification benchmarks; there is no evaluation on more challenging or large real-world industrial graphs, nor on other tasks such as graph regression or link prediction.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SHAKE-GNN, a hierarchical graph neural network architecture that leverages Kirchhoff Forest (KF) decomposition for multi-resolution graph representation learning. The method constructs a hierarchy of coarsened graphs by sampling random spanning forests with varying resolution parameters q, then applies message passing at each level.

### Strengths
1. The paper correctly identifies the limitations of message-passing GNNs on large graphs and situates SHAKE-GNN within this context.

2. Employing KFs for hierarchical decomposition is an interesting idea grounded in spectral graph theory, providing a theoretically interpretable framework for stochastic multiresolution representation.

3. The analytical derivation of expected node/edge counts and asymptotic runtime are comprehensive.

### Weaknesses
1. The technical novelty is modest. The paper heavily builds on existing Kirchhoff-Forest decomposition methods: Bressan & Vigna (2023) and Barthelmé et al. (2025). Tikhonov smoothing for q-selection builds on Tremblay et al. (2023). The only distinct contribution is the data-driven q-selection, which, while useful, is an incremental refinement rather than a conceptual breakthrough.

2. I would suggest the authors clearly mention the motivation for the proposed design, even though the techniques are borrowed from other references. For example, why k(q) are defined like that on line 218? Is this from a Graph Signal Denoising problem?

3. Another concern is about the complexity. As mentioned, "one can perform a single eigen-decomposition of L and Le, after which the quantities for all q are obtained by simple per-eigenvalue evaluations." However, the Laplacian decomposition will have an approximate cubic complexity with respect to the number of nodes at worst, which is unrealistic to implement in practice. And this part is not mentioned in the complexity analysis section.

4. Following the previous point, empirical efficiency evaluation is necessary, especially on large-scale graphs.

5. The major problem is the evaluation of the paper. The authors compare only against a vanilla baseline GNN trained on original graphs. They completely fail to compare against:

(1) Other hierarchical GNN methods (DiffPool, minCUT pooling, SAGPool)

(2) Graph coarsening baselines (METIS, which they cite in related work)

(3) Other scalability-focused methods (PPRGO, GraphSAINT, ClusterGCN)

6. Since this coarsening is stochastic, the results in the table only have one single value, which is not promising, and the reproducibility is not clear. The mean and standard deviation are required.

7. Despite the "scalable GNN" claim, SHAKE-GNN is only evaluated on graph-classification benchmarks. There are no results on node classification, link prediction tasks, which are crucial to demonstrate scalability and generality.

8. The study also lacks analysis of how each component contributes to performance.

### Questions
Pease refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors follow a specific scenario where a graph is known, and values are missing in some of the vertices (or specifically, in some dimensions in some of the vertices). They then propose an imputation algorithm based on some optimization of the discrepancy between missing values and non-missing values multiplied by the graph spectral composition (Eigen vectors of the laplacians). They then compare themselves to MDFI methods (which was never defined in the paper, but based on the methods they cite that they refer to standard tabular data) and GCN of all kinds

They then claim better imputation accuracy on six datasets (and indeed they are almost always - although here also in table 2 (which is after table 3) they claim to best in everything in their colors,  but it is imprecise..

### Strengths
They propose a novel graph-based imputation.
The imputation is solved  by optimizing a pre-defined loss function, which is always good.
They compare to some baselines.
They address an important problem of missing values in the presence of a graph.

### Weaknesses
The text is terribly unclear (had to read it 3 times and I am still not sure what they are technically doing)
The comparison to baseline is very problematic, tabular methods ignore the graphs, and GCNs are not built for imputation. There are message passing methods to address this specific issue.
The comparison and discussion on the algorithm are very limited.

### Questions
I would strongly suggest comparing to graph based imputation (and not to graph based classical machine learning).
Should define all acronyms and all concepts in advance.
Their method is spectral, it is important to explain how well does it scales, and the expected run time.
The measure is on accuracy, but the more interesting measure is probably the effect on ML accuracy.

### Soundness
2

### Presentation
2

### Contribution
2
