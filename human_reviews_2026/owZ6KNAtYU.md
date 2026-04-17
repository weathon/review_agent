# Graph homophily booster: Reimagining the role of discrete features in heterophilic graph learning

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Graph neural networks (GNNs) have emerged as a powerful tool for modeling graph-structured data, demonstrating remarkable success in many real-world applications such as complex biological network analysis, neuroscientific analysis, and social network analysis. However, existing GNNs often struggle with heterophilic graphs, where connected nodes tend to have dissimilar features or labels. While numerous methods have been proposed to address this challenge, they primarily focus on architectural designs without directly targeting the root cause of the heterophily problem. These approaches still perform even worse than the simplest MLPs on challenging heterophilic datasets. For instance, our experiments show that 23 latest GNNs still fall behind the MLP on the Actor dataset. This critical challenge calls for an innovative approach to addressing graph heterophily beyond architectural designs. To bridge this gap, we propose and study a new and unexplored paradigm: directly increasing the graph homophily via a carefully designed graph transformation. In this work, we present a simple yet effective framework called Graph Homophily Booster (GRAPHITE) to address graph heterophily. To the best of our knowledge, this work is the first method that explicitly transforms the graph to directly improve the graph homophily. Stemmed from the exact definition of homophily, our proposed GRAPHITE creates feature nodes to facilitate homophilic message passing between nodes that share similar features. Furthermore, we both theoretically and empirically show that our proposed GRAPHITE significantly increases the homophily of originally heterophilic graphs, with only a slight increase in the graph size. Extensive experiments on challenging datasets demonstrate that our proposed GRAPHITE significantly outperforms state-of-the-art methods on heterophilic graphs while achieving comparable accuracy with state-of-the-art methods on homophilic graphs. Furthermore, our proposed graph transformation alone can already enhance the performance of homophilic GNNs on heterophilic graphs, even though they were not originally designed for heterophilic graphs. Our code is publicly available at https://github.com/q-rz/ICLR26-GRAPHITE .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GRAPHITE, a simple graph transformation that explicitly increases homophily before running any GNN: it adds a set of feature nodes (hubs) and connects each original node to the hubs matching its discrete features, so that nodes with similar features become two hops apart and message passing becomes homophilic.

### Strengths
The motivation is straightforwad and the presentation is easy to understand.

### Weaknesses
Some motivation and explanation need more solid justifications.

### Questions
1. It is found that higher homophily does not always indicate a better graph for GNNs, i.e. mid-homophily pitfall [1]. How does the motivation of your proposed homophily booster align with this conclusion?

2. More experiments on large scale datasets used in [2].

3. What is the motivation or intuition behind the definition of shortcut connection, which connect nodes that share at least one feature dimension?

4. Naive homophily booster (NHB) oversimplifies the distribution of features. As shown in [3], there exist both good and bad features for message passing, and if you connect two nodes when they share a bad feature, you will not obtain a graph with higher homophily.

5. To my understanding, the number of feature nodes is the same as the number of feature dimension. For the tasks they have very large feature dimensions, this will significantly increase the number of nodes in the graph. What is the computational complexity of your algorithm?

6. Will the connection to "hub” nodes cause over-squashing problem?

7. Missing baselines in your comparison, e.g. ACMGNN [4], FSGNN [5]


[1] When do graph neural networks help with node classification? investigating the homophily principle on node distinguishability. Advances in Neural Information Processing Systems. 2024 Feb 13;36.

[2] Finding global homophily in graph neural networks when meeting heterophily. In International Conference on Machine Learning, pp. 13242–13256. PMLR, 2022.

[3] Let Your Features Tell The Differences: Understanding Graph Convolution By Feature Splitting. InThe Thirteenth International Conference on Learning Representations.


[4] Revisiting heterophily for graph neural networks. Advances in neural information processing systems. 2022 Dec 6;35:1362-75.


[5] Simplifying approach to node classification in graph neural networks. Journal of Computational Science, 62, 101695.

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
4

### Summary
This paper proposes GRAPHITE (GRAph homoPHIly boosTEr), a novel framework that directly enhances graph homophily on heterophilic graphs through an explicit structural transformation. Unlike prior methods that rely on architectural modifications, GRAPHITE introduces “feature nodes” as intermediaries to indirectly connect original nodes sharing similar discrete features—significantly boosting homophily with only a modest increase in graph size. The authors provide theoretical guarantees for this homophily gain and demonstrate through extensive experiments that GRAPHITE substantially outperforms state-of-the-art methods on heterophilic benchmarks, while also improving the performance of standard homophilic GNNs when applied to the transformed graph.

### Strengths
1. Paradigm shift:  GRAPHITE directly enhances graph homophily through structural transformation, offering a fundamental and effective new perspective. Moreover, its design is simple and efficient: by using feature nodes as hubs, it avoids the O(|V|²) edge explosion and achieves provably improved homophily with only O(|V|) added nodes and O(|E|) added edges.

2. It elegantly decouples feature similarity from the original graph structure to enable semantics-aware message passing: by introducing feature nodes that explicitly connect nodes sharing discrete features, GRAPHITE preserves the original topology while adding semantic similarity-based propagation paths—allowing GNNs to naturally integrate structural and semantic homophilic signals without architectural changes.

3. Comprehensive and convincing experiments: GRAPHITE consistently outperforms more than 25 state-of-the-art methods across four heterophilic datasets and significantly boosts the performance of standard homophilic GNNs when applied to the transformed graph.

### Weaknesses
1. Although the authors list the hyperparameter search ranges and training configurations in Section 4.1 and Appendix B.3, the paper lacks sensitivity analysis of these parameters, which weakens the robustness argument of the method.

2. The homophily definition used in the theoretical analysis (based on feature intersection) is inconsistent with the experimental metrics (feature/adjusted homophily). Although the two are intuitively positively correlated, the paper lacks rigorous derivation or empirical validation—e.g., is the theoretical homophily strongly correlated with the experimental homophily metrics across all datasets?

3. Feature nodes only connect nodes that share the same discrete features, which may cause the GNN to aggregate neighbors based on "shared keywords" rather than "same-class labels." Can case studies or ablation experiments demonstrate that feature nodes indeed promote semantic or label-level homophily, rather than merely feature-level aggregation?

### Questions
See Weakness.

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
This paper introduces the GRAPHITE method, which improves the performance of Graph Neural Networks (GNNs) on heterophilic graphs by introducing feature nodes to increase the graph's homophily. Unlike existing approaches, GRAPHITE directly enhances homophily through graph transformation rather than architectural changes. The method is theoretically proven to be effective and is validated through extensive experiments, demonstrating significant improvements on heterophilic datasets. Results show that GRAPHITE outperforms state-of-the-art methods on heterophilic graphs while remaining competitive on homophilic graphs.

### Strengths
1)	Introduces a graph transformation method that effectively boosts homophily on heterophilic graphs.
2)	Both theoretical analysis and empirical experiments support the superior performance of GRAPHITE on heterophilic graphs.
3)	Demonstrates good compatibility and enhancement effects with existing GNN architectures.

### Weaknesses
1)	Limited Novelty in Core Idea: While the paper introduces the GRAPHITE method to boost homophily in heterophilic graphs, the fundamental concept of adding feature nodes or shortcut connections between similar nodes is not entirely novel. Many prior works have explored similar strategies for improving graph neural network (GNN) performance, such as feature augmentation or graph transformation methods. The core idea lacks significant innovation, which might reduce its contribution to the broader research community.

2)	Scalability Concerns: Although the paper claims that the proposed method is computationally efficient, it does not provide a detailed analysis of how GRAPHITE performs on very large-scale graphs. The addition of feature nodes and edges might still become a bottleneck in cases with millions of nodes, limiting the method's practical applicability for large datasets. More empirical evaluation on large graphs or an analysis of the method’s time complexity would be valuable.

3)	Narrow Focus on Discrete Features: The proposed method mainly focuses on graphs with discrete node features. However, many real-world applications (e.g., social networks, biological networks) involve continuous features. The paper does not address how GRAPHITE would perform in such scenarios or whether the approach could be generalized to handle graphs with continuous or mixed-type features. This limits the generalizability of the method to broader domains.

### Questions
1、The paper repeatedly claims to be the first to propose a graph transformation method for improving performance on heterophilic datasets. However, there are numerous prior works, such as those by [1] and [2], that have also modified graph structures by introducing virtual nodes to enhance performance on heterophilic graphs. Unfortunately, the paper does not adequately discuss or compare these related studies

[1]Dong Y, Dupty M H, Deng L, et al. Differentiable cluster graph neural network. ICLR 2025.
[2]Zhang A, Li P, Chen G. Steering graph neural networks with pinning control[J]. arXiv preprint arXiv:2303.01265, 2023.


2、The feature of a feature node is defined $V_{x}$ as the " we define its node feature as the average feature vector among the graph nodes $v_i$, that are connected to feature node $x_k$:" (Equation 5). However, the rationale behind this design has not been sufficiently justified. If the features of the graph nodes $v_i$, connected to the same feature node $x_k$ exhibit significant differences (for example, some nodes may share the feature k while having completely distinct other features), the average feature may not effectively represent the feature node and could even introduce noise. Has the author compared the effectiveness of other feature construction methods for characteristic nodes, such as majority voting or weighted averaging? Why has "mean" been selected as the optimal choice?

3、The ablation experiment did not analyze the sensitivity of hyperparameters such as "feature edge weights w_x" and "temperature parameter τ" to performance. Do the values of these parameters have significant dataset dependencies? If the dataset is changed, is the cost of hyperparameter tuning too high?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GRAPHITE, a novel graph transformation framework that addresses heterophily in GNNs by explicitly increasing graph homophily through the creation of feature nodes. Unlike existing methods that focus on architectural modifications, GRAPHITE directly transforms the graph structure by introducing feature nodes that serve as hubs to connect nodes with similar discrete features. The method provides theoretical guarantees for homophily improvement with minimal graph size increase and demonstrates strong empirical performance across multiple heterophilic benchmarks, outperforming state-of-the-art methods while maintaining competitive performance on homophilic graphs.

### Strengths
1. First work to explicitly transform graph structure to increase homophily, offering a fundamentally different approach from architectural GNN modifications.
2. Provides formal proofs showing GRAPHITE guarantees homophily improvement (Theorem 3) with only linear growth in graph size.
3. Consistently outperforms 25 baselines across 4 heterophilic datasets with improvements up to 5.35%, while maintaining competitive performance on homophilic graphs.

### Weaknesses
1. Method is specifically designed for discrete node features, limiting applicability to graphs with continuous features without discretization.
2. While theoretical complexity is linear, practical implementation with 8 GNN layers and 512 hidden dimensions may be computationally expensive for very large graphs.
3. Feature node representations (Equation 5) use simple averaging, potentially overlooking more sophisticated feature aggregation strategies.
4. No exploration of how the transformation affects graph properties like diameter, clustering coefficients, or other structural metrics beyond homophily.

### Questions
1. How can GRAPHITE be extended to handle continuous node features while maintaining theoretical guarantees?

2. What is the impact of the transformation on graph structural properties beyond homophily, and could certain properties be adversely affected?

3. Have the authors considered alternative feature node representations beyond averaging, such as learned embeddings or attention-weighted aggregations?

4. How does the method scale to extremely large graphs in practice, considering the increased node count from feature node addition?

### Soundness
2

### Presentation
2

### Contribution
2
