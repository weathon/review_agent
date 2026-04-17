# Cluster Attention for Graph Machine Learning

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Message passing neural networks have recently become the most popular approach to graph machine learning tasks, however, their receptive field is limited by the number of message-passing layers. To increase the receptive field, using graph transformers with global attention has been proposed, however, global attention does not take into account the graph topology and thus lacks graph-structure-based inductive biases which are typically very important for graph machine learning tasks. In this work, we propose an alternative approach: cluster attention (CLATT). We divide graph nodes into clusters with off-the-shelf graph community detection algorithms and let each node attend to all other nodes in each cluster. CLATT provides large receptive fields while still having strong graph-structure-based inductive biases. We show that augmenting message-passing neural networks or graph transformers with CLATT significantly improves their performance on a wide range of graph datasets including datasets from the recently introduced GraphLand benchmark representing real-world applications of graph machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a mechanism named Cluster Attention (CLATT), which applies attention over graph clustering results to bridge the gap between traditional message passing and global attention. However, the work has room for improvement in terms of theoretical depth, novelty margin, and comprehensiveness of baseline comparisons.

### Strengths
1. The motivation is reasonable; using graph clustering algorithms as a preprocessing step and applying attention is an intuitive and rational intermediate solution.
2. The experimental datasets cover a wide range.

### Weaknesses
1. Lack of novelty. Information aggregation over clusters/communities is not a new concept in graph learning [1]. The paper does not discuss its connections to or distinctions from related work such as Graph Coarsening, hierarchical GNNs, or hypergraph studies.
2. Lack of theoretical analysis. Although the method design is intuitive, it would be beneficial to include a section (even informal one) discussing the underlying mechanisms that explain why CLATT works.
3. Insufficient experiments. The evaluation lacks analysis of computational overhead and the impact of different clustering algorithms. Additionally, several relevant baselines are missing, such as the classic GAT and recent state-of-the-art methods.
4. Some parts of the writing are overly informal (e.g., lines 186–203), and the overall presentation lacks clarity and structure.

[1] Bianchi F M, Grattarola D, Alippi C. Spectral clustering with graph neural networks for graph pooling[C]//International conference on machine learning. PMLR, 2020: 874-883.

### Questions
See the weaknesses mentioned above.

### Soundness
3

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
3

### Summary
This paper introduces CLATT, a graph neural network component that allows nodes to attend to all other nodes within the same cluster, as determined by community detection or graph clustering algorithms. The key motivation is to provide an inductive bias that leverages graph structure for expanding receptive fields, striking a balance between pure message passing, which is limited to local neighborhoods, and global transformers, which ignore graph structure. The authors show how CLATT can be integrated into both MPNNs and global graph transformers, then empirically evaluate the approach on 11 diverse datasets, demonstrating generally improved performance.

### Strengths
1. Proposes a clear and well-motivated approach that uses graph clustering to define mid-range attention, addressing the limitations of both local message passing and indiscriminate global attention. This structural bias is conceptually appealing for tasks where communities/clusters are meaningful.

2. Experiments cover a wide range of datasets including both homophilous and non-homophilous graphs, sparse and dense graphs, and several realistic/industrial datasets, not just citation benchmarks.

3. The results in Table 2 demonstrate consistent improvements across several strong baselines, including GCN, GraphSAGE, Local Graph Transformers, and two types of global graph transformers with various positional encodings. For example, performance improvements for GCN-CLATT and GraphSAGE-CLATT are evident and statistically significant in most cases.

### Weaknesses
1. While the paper goes into some mathematical detail on the CLATT mechanism, there is almost no theoretical analysis of its representational or generalization properties. For instance, there is no discussion of what information might be lost or gained under various clustering regimes, no bounds on expressivity, and no framework for reasoning about which graph types CLATT is provably suitable for.

2. The concatenation of multiple cluster-based attention outputs for each node may cause the embedding size to scale linearly with the number of clusterings. This could introduce issues with memory and computation for large graphs or when many clustering algorithms are used.

3. While the key equations for CLATT are present (Eq., Pages 4-5), their notation is sometimes ambiguous. For example, there’s no clarification about the treatment of singleton clusters, solution for clusters of widely varying size, or masking strategy when clusters overlap (if allowed). Additionally, the use of learnable parameters per clustering ($\mathbf{W}_{q}^{C}$, etc.) has unclear implications for parameter sharing and scalability—none of these architectural choices are ablated/justified.

### Questions
1. How robust is CLATT to cluster quality? Have the authors tried intentionally poor clustering (e.g., random partitions) to probe sensitivity?
Does the concatenation-based combination of multiple clusterings risk dimensionality inflation or feature redundancy, and are there alternatives (e.g., learned aggregation) that trade off better between expressivity and complexity?

2. Are there meaningful differences in which types of graphs (homophilous, highly modular, assortative/disassortative) benefit most from CLATT, and can the authors provide a diagnostic or criterion to recommend when CLATT should or shouldn’t be applied?

3. Can the authors provide more insight or empirical results for tasks where CLATT fails or offers no benefit?

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
The paper introduces Cluster Attention (CLATT), an attention layer for graphs where each node attends over nodes grouped by precomputed clusters rather than only 1-hop neighbors (MPNNs) or the entire graph (global transformers). The authors instantiate CLATT with four candidate clustering methods. They select a small subset of clusterings that are mutually dissimilar, apply CLATT separately to each selected clustering, and concatenate the outputs. They report gains across diverse benchmarks (including GraphLand) when inserting CLATT into GCN/GraphSAGE and local/global transformers.

### Strengths
The paper identifies a pragmatic middle ground between local and global aggregation; it offers a general plug-in, reports results on many datasets, and discusses engineering choices (handling different clusterings, concatenation and projection). The candidate set of clusterers covers assortative, disassortative, and feature-space partitions, which is a sensible coverage of regimes encountered in practice.

### Weaknesses
1. Although all four candidate algorithms are present in the paper, they are not enumerated in one place with assumptions and hyperparameters. Please add a compact summary of them. 

2. The paper states a desire to pick diverse clusterings. You do show similarity matrices and mention the Correlation Coefficient (CC) as the similarity index; however, the selection rule (how CC informs which clusterings are kept) is not formalized. Please specify:

* the metric used,
* the threshold or diversity objective (e.g., greedy selection maximizing minimum pairwise dissimilarity),
* whether similarity is computed on the full graph or a validation subset, and its computational cost.

3. Combining clustering with GNNs is not a novel idea [1,2]. Position CLATT more directly against:

- GNN-based clustering and end-to-end formulations [1];
- Hierarchical / coarsened MPNNs and hierarchical message passing [2];
- global transformers with structural encodings (e.g., Laplacian/positional encodings, shortest-path/centrality biases).
  Clarify where CLATT provides a new capability (e.g., multi-clustering selective aggregation) rather than a re-packaging of hierarchical pooling.

4. “Is attention necessary?” Because CLATT attends within clusters, a simple baseline is to pool each cluster (mean/max/gated pooling) and pass pooled summaries to nodes (or to use cluster-level tokens with cross-attention). Without this, it is hard to attribute gains to attention rather than to the cluster prior itself. [2]

5. The paper’s motivation is that global attention lacks graph-topology bias. CLATT’s topology-based clusterers do inject such bias, but the inclusion of feature-only k-means weakens the story. Please either:
(i) frame CLATT as a general prior layer that can combine topology and feature clusterings, and analyze when each helps; or
(ii) restrict main claims to topology-derived clusterings and move feature-only variants to ablations.

6. Compute and memory. Report wall-clock and peak memory for base vs. CLATT (including clustering and selection overhead) on the largest datasets. This is important for practitioners.

### Questions
1. Please confirm the four candidate methods (Leiden/CPM; Bayesian planted partition; disassortative hierarchical SBM with smallest non-trivial level; k-means on ResMLP embeddings) and provide their hyperparameters and implementations used.

2. How exactly do you quantify differences between clusterings? How are diversity thresholds chosen, and how does this choice affect accuracy and cost?

3. Why not learn clusters with a GNN-based clustering module [1], potentially allowing backpropagation of the task loss into the clustering? Could CLATT benefit from such end-to-end training?

4. Is attention necessary? Please add ablations replacing CLATT with cluster pooling (mean/max/gated) or cluster tokens without attention, to show the specific value of attention beyond the cluster prior. See [2] for hierarchical message passing baselines.

5. Can you provide analyses showing that CLATT improves structure-sensitive metrics (e.g., performance vs. assortativity, conductance, role similarity) when using topology-only clusterers, and contrast that with feature-only k-means?

6. What is the delta when re-selecting the clusterings per backbone?

7. How sensitive is CLATT to the resolution (average cluster size) across the topology-based methods?

[1] Graph clustering with graph neural networks, JMLR, 2023. 

[2] Hierarchical message-passing graph neural networks, DMKD, 2023.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Cluster Attention (CLATT): run one or more fast graph clustering/community-detection algorithms as a preprocessing step, then perform dense attention within each cluster (optionally for multiple clusterings and concatenate their outputs), and combine this with standard message passing (or with global graph transformers). The goal is a “middle ground” between purely local MPNNs and fully global attention—capturing longer-range dependencies while retaining graph-structural inductive biases.

### Strengths
1. This paper strikes a practical design point between edge-local MPNN and full all-pairs GGT -- computing graph attention on edges and nodes within each cluster.
2. Despite relying on one clustering algorithm, the proposed method considers various different clustering algorithms and combine the different types of in-cluster attention.
3. This paper covers diverse types of graph datasets to validate the effectiveness.

### Weaknesses
1. The complexity is $\sum |C_i|^2$ where $|C_i|$ is the number of cluster i. However, sometimes most of the nodes in a graph will be clustered into one cluster. For example, when the graph has a majority connected component and many small unconnected parts. In this condition, the computational complexity is $O(|V|^2)$ and the proposed method will be very similar to graph Transformer.
2. Other case where the proposed method will not work well include grids.
3. In inductive settings, unseen nodes during training may significantly change the clustering structure of graphs. And thus the learned attention may not be appropriate for the real underlying clustering structure.
4. There are already some previous works [1,2,3] combining graph clustering and graph attention, the authors should discuss the difference between their proposed method and previous literature.
5. The authors are encouraged to do some ablation studies to discuss the contribution of each different clustering algorithm.
6. The authors are proposed to compare the attention scores learned by the proposed method with vanilla graph attention and graph Transformer, to illustrate the effectiveness of the proposed method.


---
Reference:

- [1] Differentiable Clustering for Graph Attention.
- [2] Attention-based Graph Clustering Network with Dual Information Interaction.
- [3]. Transforming Graphs for Enhanced Attribute Clustering: An Innovative Graph Transformer-Based Method.

### Questions
Refer to "Weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3
