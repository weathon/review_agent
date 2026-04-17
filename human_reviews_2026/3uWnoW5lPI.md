# Adaptive Granularity Graph Rewiring via Granular-ball for Graph Clustering

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Graph clustering aims to partition a graph into homogeneous groups of nodes, capturing the graph's node features and connectivity structure. Graph neural network-based approaches excel in node clustering by leveraging the homophilic assumption, which posits that neighboring nodes are likely to share similar characteristics, but low-homophily edges can introduce noise, potentially reducing clustering accuracy. Previous work rewires connections by estimating homophily at an overly fine granular, primarily based on the similarity of connected nodes. Nevertheless, they largely neglect the fact that homophily is distributed across multiple granular levels within the graph. Considering the multi-granular nature of the homophily distribution, we could better differentiate between homophilic and heterophilic nodes at the optimal granularity. To this end, we propose a novel Adaptive Granular Graph Rewiring method (AGGR) that adaptively identifies homophilic regions at appropriate granularities and subtly enhances homophily within the graph structure through graph rewiring, significantly improving GNN performance and clustering outcomes. Specifically, AGGR introduces an Adaptive Granular-Ball graph refinement mechanism to capture homophilic structures within graphs. In addition, a Multi-Granularity Graph Rewiring method is further proposed to add highly homophilic social relations intra-homophilic domains and cut low homophilic relations inter-them. Moreover, we propose a Multi-Task Homophily Refinement Learning framework to integrate the optimization of graph rewiring with graph clustering. Extensive experiments conducted on benchmark datasets demonstrate that AGGR outperforms the state-of-the-art method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel graph clustering method, Adaptive Granularity Graph Rewiring via Granular-Ball for Graph Clustering (AGGR). The core innovation of AGGR lies in incorporating the multi-granularity distribution of homophily into the graph rewiring process, enabling an adaptive granular approach. This method adaptively refines the graph’s homophilic structure, subtly enhancing homophily and thereby improving clustering performance. Furthermore, the Multi-Task Homophily Refinement Learning module is developed to effectively utilize clustering information, further advancing the performance of both graph rewiring and graph neural networks. Experimental results indicate that AGGR demonstrates competitive clustering performance metrics across multiple datasets in low-homogeneity social networks compared to existing methods.

### Strengths
1. The paper introduces a novel graph clustering method, Adaptive Granularity Graph Rewiring (AGGR), which optimizes graph structures by identifying and rewiring multi-granularity homogeneous regions. 

2. The paper clearly explains the theoretical foundation and implementation details of the AGGR method, including the identification of multi-granularity homogeneity distribution and the application of adaptive granular-ball mechanisms.

### Weaknesses
1. The paper does not explain the related work on graph rewiring, making it difficult to assess the novelty of the proposed graph rewiring method.

2. The experiments for RQ2 and RQ4 were conducted on only two datasets, which is not sufficiently convincing.

3. RQ1 should be compared with other graph rewiring methods to demonstrate its superiority over existing approaches.

### Questions
1. What are the specific distributions P and Q in the KL divergence loss presented in Equation (13)?
2. The paper appears to apply a graph rerouting method to graph clustering, but it is unclear what the necessary conditions and motivations are for using graph rerouting in graph clustering.
3. What are the innovations of this paper compared to other graph rerouting methods?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a graph modification algorithm named AGGR, designed for graph clustering tasks. The paper emphasizes the multi-granular nature of graphs, suggesting that homophily is distributed across multiple granular levels, and that homophilic nodes should be distinguished at the optimal granularity. Based on this idea, the authors introduce an adaptive granular graph rewiring method. AGGR can adaptively identify homophilic regions at appropriate granularities. It introduces a graph refinement mechanism based on adaptive granular spheres and employs a multi-granular graph rewiring strategy. Experimental results demonstrate that the proposed approach achieves strong performance.

### Strengths
The methodology of the paper is clearly presented and easy to follow. Moreover, the proposed AGGR method seems to perform well in the experiments.

### Weaknesses
1. It is unclear to me how the multi-granular nature of graphs benefits the clustering task. This aspect appears to be central to the paper, yet unfortunately it is only briefly discussed in the introduction. A more thorough theoretical and empirical analysis is expected.

2. Some of the claims in the paper, such as those concerning the relationship between graph granularity and homophily, lack theoretical justification, empirical validation on real datasets, or supporting references. As a result, these statements are not fully convincing.

3. Moreover, I am not convinced that such multi-granular properties are widely present in real-world graph data.

### Questions
1. Is it necessary for us to introduce two lambdas in Eq. 17, or would setting one of them to $1$ already be sufficient, thus saving a hyperparameter?

2. In Section 3.3, why is the ratio for adding edges related to the number of nodes, expressed as $\delta * |V|$, while the ratio for deleting edges is related to the number of edges, expressed as $\gamma * |E|$?

3. In Section 4.6, does the setting where $\gamma$ is set to 1 mean that all edges are removed? That seems unreasonable — please clarify. Do the experimental results suggest that we should always choose the largest possible $\gamma$?

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
The paper proposes MGGR for graph classification, focusing on multi-granularity subdomains via granular-ball decomposition. It adaptively partitions graphs ($\sqrt N$ initialization, recursive binary splits) using a quality criterion that combines structural measures and label purity, then applies a hierarchical encoder with intra-domain structural feature aggregation and inter-domain GNN modeling. Experiments on several benchmarks show consistent accuracy gains over strong baselines and robustness to label noise.

### Strengths
- Interpretable multi-granularity decomposition that preserves intra- and inter-subdomain structure for finer, structure-aware representations.

- Effective hierarchical encoding (intra-domain structural features + inter-domain GNN) capturing local/global patterns, with strong accuracy and noise robustness.

- Adaptive splitting with quality criteria and local computations improves scalability; validated by comprehensive ablations and sensitivity analyses.

### Weaknesses
- Dependence on label purity in splitting limits applicability to unsupervised or weakly supervised settings.

- Heuristic partitioning ($\sqrt N$ centers, highest-degree seeds, binary splits) may be sensitive to degree skew/topology, with limited theoretical guarantees.

- Hand-crafted intra-domain structural features are not end-to-end learned, potentially limiting expressiveness and task adaptivity.

- Nontrivial overhead for computing eigen/centrality/diameter; graph-level readout is under-specified and runtime/memory analysis is limited.

- Related work gap: several hierarchical graph representation learning approaches are not sufficiently discussed [1, 2].

[1] Galaxy Network Embedding: A Hierarchical Community Structure Preserving Approach.

[2] Hierarchical community structure preserving network embedding: A subspace approach

### Questions
- The splitting criterion relies on label purity. How does MGGR operate when labels are unavailable or highly noisy?

### Soundness
2

### Presentation
3

### Contribution
2
