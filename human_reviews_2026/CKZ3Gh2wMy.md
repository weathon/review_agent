# ATLAS: Adaptive Topology -based Learning at Scale for Homophilic and Heterophilic Graphs

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We present ATLAS (Adaptive Topology - based Learning at Scale for Homophilic and Heterophilic Graphs), a novel graph learning algorithm that addresses two important challenges in graph neural networks (GNNs). First, the accuracy of GNNs degrades when the graph is heterophilic. Second, the iterative feature aggregation limits the scalability of GNNs to large graphs. We address these challenges by extracting topological information about the graph communities at different levels of refinement, concatenating the community assignments to the feature vector, and applying multilayer perceptrons (MLPs) on this new feature vector. By doing so, we inherently obtain the topological data about the nodes and their neighbors without invoking aggregation. Because MLPs are typically more scalable than GNNs, our approach applies to large graphs—without the need for sampling. Our results, on a wide set of graphs, show that ATLAS has comparable accuracy to baseline methods, with accuracy being as high as 20 percentage points over GCN for heterophilic graphs with negative structural bias and
11 percentage points over MLP for homophilic graphs. Furthermore, we show how multi-resolution community features systematically modulate performance in both homophilic and heterophilic settings, opening a principled path toward explainable graph learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ATLAS, a topology-based learning method aiming to handle both homophilic and heterophilic graphs by incorporating community detection information into MLP features. The authors argue that this approach scales better than GNNs while maintaining comparable accuracy on various graphs.

### Strengths
The topic of homophily and heterophily learning on graphs with community detection is relevant and potentially interesting.

### Weaknesses
While the topic is relevant and potentially interesting, the paper suffers from several critical issues. 

1. The literature review is insufficient, and the background and related work section is extremely weak. Key surveys and foundational works on heterophilic graph learning are missing, such as Zheng et al., “Graph Neural Networks for Graphs with Heterophily: A Survey” (arXiv:2202.07082, 2022), along with many subsequent heterophilic GNN studies. 

2. The contribution lacks novelty, as the idea of combining community information with simple MLPs has already been explored (e.g., LinkX: Large Scale Learning on Non-Homophilous Graphs, NeurIPS 2021). The technical innovation over existing baselines is minimal. 

3. Moreover, the experimental evaluation does not convincingly demonstrate superiority over recent heterophilic GNNs. 

4. Lastly, the manuscript presentation is unpolished—the font and formatting differ noticeably from standard NeurIPS submissions, suggesting a lack of careful preparation.

### Questions
No more questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ATLAS, a novel graph learning algorithm aimed at improving scalability and performance across both homophilic and heterophilic graphs. Instead of relying on neighborhood aggregation as in traditional GNNs, ATLAS extracts multi-level community features that encode topological structure and combines them with node features for classification using multilayer perceptrons (MLPs). This design eliminates costly message passing while preserving structural information, enabling efficient learning on large graphs. The authors also introduce a theoretical framework using normalized mutual information (NMI) to adaptively refine communities based on the degree of homophily, achieving strong interpretability and adaptability.

However, the experimental evaluation has notable weaknesses. On the OGBN-Products dataset, the set of comparison baselines is too limited and includes mostly outdated methods, making it difficult to assess ATLAS’s true competitiveness against modern scalable and heterophily-aware GNNs. Furthermore, while the method is conceptually sound, the problem it addresses—heterophilic rpoblem and graph scale—has been explored in prior research, reducing its overall novelty.

### Strengths
This work is well-structured and clearly written. The paper effectively connects theoretical analysis to practical implementation. This paper considers two important issues in the GNN, heterophilic problem and graph scale.

### Weaknesses
The theory part is hard to follow, and how this part helps the methododolegy is not clear. The experimental part needs more recent baslines.

### Questions
From my perspective, these baselines are before 2023, is that possible to compare with other recent works?

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
4

### Summary
This paper proposes a simple yet powerful framework to overcome the scalability and heterophily limitations of traditional Graph Neural Networks. The method constructs multi-resolution community features using adaptive community refinement guided by Normalized Mutual Information (NMI), concatenates them with node features, and feeds them into an MLP—eliminating neighborhood aggregation and enabling adjacency-free inference.

### Strengths
The graph categorization based on structural bias is interesting.

### Weaknesses
See below

### Questions
1. "Accurate classification requires two orthogonal pieces of information–(i) the features at each node, and (ii) the connections between the node and its neighbors." These two components are not necessarily orthogonal to each other, and sometimes you need to disentangle their relation to investigate their impacts on GNN performance [1].

2. Section 3 is not well developed and need to be polished. It starts with some notations that are not properly introduced before (e.g. three hyperparameters, modularities), which is confusing. Also, it's better to emphasize its connection to the previous sections and the main storyline of this paper.

3. Is the community assignment algorithm the main contribution of your paper? How does it compare with other existing algorithms?

4. How does your "multi-resolution community" address the heterophily problem? You can provide some insights if there is no theoretical evidence.

5. Does the partitions of communities be aware of the label distribution? Does your proposed method add more hyperparameters to tune?

6. In table 2, you hightlight all results of your proposed method, however it is not the best among the baselines in all tasks.

7. You should introduce the definition of high/low/negative structural bias before using them.

8. Missing comparison with some baseline models, e.g. ACM-GCN and Bernnet [2,3]. Ablation study on the impact of different resolutions of communities is needed.


[1] What is missing for graph homophily? disentangling graph homophily for graph neural networks. Advances in Neural Information Processing Systems, 37, 68406-68452.

[2] Revisiting heterophily for graph neural networks. Advances in neural information processing systems. 2022 Dec 6;35:1362-75.

[3] Bernnet: Learning arbitrary graph spectral filters via bernstein approximation. Advances in neural information processing systems, 34, 14239-14251.

### Soundness
2

### Presentation
2

### Contribution
2
