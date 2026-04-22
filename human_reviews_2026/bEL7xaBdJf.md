# Centrality Graph Shift Operators for Graph Neural Networks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 8

## Abstract
Graph Shift Operators (GSOs), such as the adjacency and graph Laplacian matrices, play a fundamental role in graph theory and graph representation learning. Traditional GSOs are typically constructed by normalizing the adjacency matrix by the degree matrix, a local centrality metric. In this work, we instead propose and study Centrality GSOs (CGSOs), which normalize adjacency matrices by global centrality metrics such as the PageRank, $k$-core or count of fixed length paths. We study spectral properties of the CGSOs, allowing us to get an understanding of their action on graph signals. We confirm this understanding by defining and running the spectral clustering algorithm based on different CGSOs on several synthetic and real-world datasets. We furthermore outline how our CGSO can act as the message passing operator in any Graph Neural Network and in particular demonstrate strong performance of a variant of the Graph Convolutional Network and Graph Attention Network using our CGSOs on several real-world benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper directly extends the parameterized graph shift operator (PGSO) from using degree based normalization into a general graph centrality (pagerank, L-walk, and k-core) based and develops centrality based graph shift operator (CGSO). The authors provide related mathematical properties associated with the related formulation. The work explores the impact of CGSO on spectral clustering and node classification tasks and observed improvements against GCN/GAT baselines, especially on heterophilic graphs. However, the novelty is limited since it is an incremental extension of PGSO. Moreover, the performance of CGSO does not show systematic advantages over the degree based PGSO in most graph datasets. Despite the extra computational cost brought by obtaining the centralities for each node, which may be non-trivial for large graphs, the work lack systematic analysis (theoretical or experimental) on why and how the centrality based GSO can provide the mentioned benefit comparing to the simpled degree based GSO. The heavy mathematics provided in the main context and appendix does not directly help understanding the behavior difference between GSO/PGSO and CGSO on the clustering and classification tasks. How does CGSO affect the graph spectrum properties from a signal processing perspective for homophilic/heterophilic graphs? This is a critical question discussed by literatures, but it is missing in this paper.

### Strengths
1. Uses several graph centralities to substitute degree in GSO/PGSO and induces global information into the message passing process. 
2. Obtains performance improvements on node clustering and classification tasks against classic baselines. 
3. Provides the performance landscape over the key trainable parameters (e2 and e3) for spectral clustering.

### Weaknesses
1. It is an incremental extension of PGSO. 
2. The PageRank, k-core, and L-walk based CGSO do not show general advantage over degree based, which is basically PGSO, on node classification tasks (table2 and 3). I can accept that the paper does not compare the performance of CGSO with SOTA models like FAGCN, ChebNet, TEDGCN, GPRGNN, LINKX, ACM-GCN, and GloGNN, but CGSO should significant improvement over PGSO (CGSO-D in this paper), otherwise, what's point of spending extra computation on the other centralities?
3. The paper lacks systematic theoretical and empirial insights on how the centralities changes the behavior of GSO, how that is related with the observed performance changes, and what the interplay between the graph structure and the favored centralities is. Simply saying globol information is used is far from enough. 
4. Propositions 1 to 4 do not help understand the behavior of CGSO on the clustering and classfication tasks. 
5. Graph spectral analysis from the signal filtering perspective is not provided. This is an important topic, or say a focus, discussed in key literatures, such as [1], [2], and [3]. 
6. In line 431, the authors state "an observation that we have not previously seen in the literature". Actually this phenomenon is analyzed in [3]. 
7. For k-core and L-walk based CGSO, how to choose the hyper parameters of k and L? No guidance, theoretical or empirical, is provided.
8. Lemma 4.1 on the average degree of BA scale-free network is a classic well-known conclusion, not the contribution of this paper. Also, I don't see how this can help undertand the behavior of CGSO. 
9. No sufficient explanations are provided on why CGSO tends to do better on heterophilic graphs. 

[1] Analyzing the expressive power of graph neural networks in a spectral perspective. 
[2] Beyond low-frequency information in graph convolutional networks. 
[3] From Trainable Negative Depth to Edge Heterophily in Graphs.

### Questions
1. In lines 354-355, by "homophilic" and "heterophilic", do the authors actually mean "homogeneous" and "heterogeneous"?
2. For clustering on Cora, how to obtain the ground truth node membership?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Centrality Graph Shift Operators (CGSO), by normalizing the adjacency matrix with different centrality metrics (e.g., degrees, PageRank score, k-core numbers). The authors provide spectral analysis on the properties of these CGSOs, with empirical validation in synthetic and real-world graphs. The authors further incorporate CGSOs in graph neural networks to demonstrate their utility on several real-world datasets.

### Strengths
1. It is interesting to study the class of generalized graph shift operators, and combine them with graph neural networks.

2. The paper is mostly well-written.

### Weaknesses
1. The motivation of this work remains unclear. Are the class of CGSOs intended to inject more global information to graph representations? If so, why not encoding these centrality metrics as node features? Or perhaps the proposed CGSOs enjoy provable guarantees on some problems (e.g. spectral clustering)? However, the established theoretical properties do not explicitly answer why or in what settings these CGSOs are preferred.

2. Limited novelty and effectiveness when incorporating CGSO in GNNs. The authors largely reuse the parameterized CGSO framework from Dasoulas et al. (eqn 3), merely changing the degree matrix normalization to other centrality metrics. More importantly, for the real-world benchmark evaluations (Table 2), out of 7 benchmark graphs except arxiv-year, none of the proposed new CGSOs outperform the original degree matrix normalization for GCN-based baselines (3rd block); similar trend can be observed for GATv2 baseline. In additiona, combining local CGSO (degree normalization) and global CGSO do not always improve performance over using local CGSO only (see Table 3). This raises concerns on how effective these newly proposed CGSOs are for real-world graph tasks.

### Questions
1. Based on the spectrum properties of CGSO (e.g. Prop 3.3, 3.4), can the authors compare the three proposed CGSO (using k-core, PageRank score, and Walk Count) from their spectrum properties and discuss when or why they outperform one another?

2. Can the authors compare their proposed CGSO with the baseline of using standard degree matrix normalization (CGCG w/ D) plus encoding the centrality metrics as node features? Intuitively this provides a easy way to incorporate (global) centrality features in the graph representations, and avoid possible numerical instability for inverting the centrality matrix.

3. For many figures (e.g. Fig 1., 4, 5), the color bars across subplots are not standardized to the common range. This renders comparison across different subplots difficult and confusing. Can the authors fix this and discuss the figures on a comparable basis?

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
This paper introduces a novel class of graph shift operators (GSOs), termed Centrality-based Graph Shift Operators (CGSO). To address this, the authors propose integrating various node centrality metrics (e.g., betweenness, closeness, eigenvector centrality) into the design of GSOs. Specifically, they define CGSOs as a centrality-weighted version of the adjacency matrix, where each edge is scaled based on the centrality of its incident nodes. The paper presents theoretical analysis of CGSO properties (spectral behavior, stability, interpretability), and empirically evaluates CGSO on tasks like node classification and graph signal forecasting.

### Strengths
The idea of incorporating global node importance into the shift operator is simple, well-motivated

The paper provides strong theoretical support for the CGSO design, including propositions on spectral characteristics and signal propagation.

Experiments on node classification show that CGSO offers consistent performance gains.

### Weaknesses
Although improvements are consistent, the gains are sometimes marginal (especially on Pubmed)

While the authors propose to use node centrality as a way to encode global importance, there is no principled theoretical justification for why these particular centrality metrics (e.g., betweenness, closeness, eigenvector) are most suitable for constructing GSOs.

While CGSO is integrated into models like GCN, this integration is surface-level

### Questions
Could the centrality scores be learned or updated during training, instead of being precomputed?

Is there a way to automatically choose or combine centrality functions based on data or task type?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Centrality Graph Shift Operators (CGSOs) based on the normalization of the adjacency matrix with different centrality metrics. The authors analyze spectral properties of the CGSOs and validate them via spectral clustering on both synthetic and real-world datasets. They further use CGSOs as message passing operators in Graph Convolutional Network and Graph Attention Network v2, demonstrating strong performance across benchmarks. Overall, the paper is well-motivated and clearly-structured with both theoretical and empirical support.

### Strengths
1. The idea and construction of CGSOs are simple but broadly applicable.
2. The paper provides theoretical properties of Markov Averaging Operators to explain when/why CGSOs should separate clusters.
3. The empirical results (many shown in Appendix) are comprehensive and presented with clarity.

### Weaknesses
1. The proof of Proposition 3.2 seems to assume no self-loops in the graph. In Section 3.2, the new parametrized CGSO adds self-loops to the adjacency matrix, so a brief discussion on how the theories/propositions can be extended would improve completeness. Also, it would
be helpful to clearly state the assumptions and constants in propositions.
2. The hyperparameters in Appendix A.5 are from a grid search on the classical GCN, which might under-tune others. Although this is designed for comparison, it would be helpful to see the model performances with hyperparameters tuned for CGCN too.

### Questions
1. Could you add all assumptions and constants in the statements of propositions for clarity?
2. For the walk count node centrality matrix, you use l=2 in all experiments. Would different l affects the result? How sensitive are results to l?

### Soundness
3

### Presentation
3

### Contribution
3
