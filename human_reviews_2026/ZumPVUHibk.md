# SUBRead: Clustering Sub-Graphs for Graph-Level Readout

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Graph Neural Networks (GNNs) have transformed graph representation learning tasks across domains from bioinformatics and social networks to engineering applications. In graph classification, the readout function is an important component of the GNN architecture as it aggregates node features into a compact graph-level representation. Standard readouts such as sum, mean, and max often fail on complex graphs, as they cannot capture structural dependencies or contextual relationships among nodes due to the over-compressive nature of these functions, which leads to information loss. To address these challenges, we propose SUBRead, an expressive readout function which integrates subgraph clustering with attention-based weighting to produce a graph-level representation that preserves local structural information while capturing global dependencies. SUBRead is fully differentiable and compatible with various GNN architectures. Experiments on bioinformatics and social network benchmarks demonstrate that SUBRead consistently outperforms existing readouts, improving accuracy and interpretability. We further evaluate SUBRead on a real-world automotive engineering problem, where the task is to classify vibration responses of structures, referred to as structural mode shapes, using attributed graphs derived from simulation results. Unlike common graph benchmarks where graphs vary in topology, the mode shape graphs share a similar topology but significantly differ in node features, making sub-graph essential and providing a unique benchmark for readouts. The analysis demonstrates that SUBRead not only outperforms existing readouts but also provides meaningful substructures comparable to expert reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SUBRead, a graph neural network readout function that clusters nodes into subgraphs and integrates attention mechanisms to generate graph-level representations. The method outperforms baseline methods on multiple benchmark datasets from bioinformatics and social networks, and further demonstrates its effectiveness on a real-world vehicle structural vibration dataset.

### Strengths
The method is evaluated on several benchmark datasets (PROTEINS, MUTAG, NCI1) and a real-world engineering dataset, providing evidence of its effectiveness.

### Weaknesses
1. The paper demonstrates limited novelty, as graph pooling operators beyond simple average or sum pooling has already been studied extensively in the literature. in particular, sub-graph clustering and attention based graph pooling has been explored in the reference [1], and its differences with the proposed method should be discussed. 

2. The comparison with existing methods appears limited. It is recommended that the authors include additional baselines using more advanced readout techniques (e.g., Janossy GRU/MLP [2], kerRead[3]) to better demonstrate the effectiveness of the proposed approach.

3. The paper lacks sufficient parameter sensitivity and ablation studies, making it difficult to assess the contribution of each component in the framework.

4 The writing of the paper and the figures should be improved. Many technical details and key motivations are missing. 


[1] Gaoqi He, Shun Liu, Kai Zhang, Honglin Li. Prototype-based Contrastive Substructure Identification for Molecular Property Prediction. Briefings in Bioinformatics

[2] Buterez, David, et al. "Graph neural networks with adaptive readouts." Advances in Neural Information Processing Systems 35 (2022): 19746-19758.

[3] Yu, Jiajun, et al. "Kernel readout for graph neural networks." Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24. 2024.

### Questions
(1) the subgraph clustering step should guarantee that the segmentation of the initial graph should be physically feasible, i.e., avoiding isolated node, subgraph sizes should be balanced, etc., which requires considering both similarity-between sub-graphs and topological relations, which this is largely ignored.

(2) The paper uses learnable prototypes to perform subgraph clustering of nodes. Could the authors provide visualizations or other analyses to demonstrate the relationship between the clustering results and the underlying graph structure, thereby providing stronger evidence for the effectiveness of the method? 

(3) how do you choose the number of clusters, k in practice? do you need to used validation set to tune this hyper-parameter? 

(3) is the equation in line 243 a self-attention?

### Soundness
2

### Presentation
1

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
This paper proposes SUBRead, a novel graph readout function for Graph Neural Networks (GNNs). Instead of aggregating node representations using simple global pooling operations (e.g., sum/mean/max), SUBRead first clusters a graph into local subgraphs, then applies subgraph-level aggregation, followed by an attention mechanism to combine these subgraph embeddings into a final graph-level representation. The method is fully differentiable and can be integrated with standard GNN architectures. The authors evaluate SUBRead on benchmark datasets as well as a real-world structural vibration classification problem, where graphs share identical topology but differ substantially in node features. Across experiments, SUBRead outperforms existing readout techniques and yields interpretable substructures aligned with expert reasoning.

### Strengths
1. Novel Readout Perspective: The focus on subgraph-level aggregation is a meaningful and well-motivated departure from commonly used global pooling strategies, addressing known limitations of over-compressive readouts.

2. Strong Empirical Results: SUBRead demonstrates consistent improvements across benchmarks and shows particular effectiveness in the mode shape classification task, where structural interpretability matters.

3. Interpretability Component: The revealed substructures are not merely performance-driven—they appear aligned with domain insights, which is valuable for real-world engineering applications.

### Weaknesses
1. Clustering Method Justification: The choice of clustering mechanism could be explained more rigorously. It is not fully clear how sensitive the performance is to the number of clusters or selection of the clustering algorithm.

2. Computational Complexity: The added clustering and attention layers introduce overhead relative to simpler readouts. The paper would benefit from a more explicit complexity analysis.

3. Limited Analysis of Failure Cases: The paper focuses on positive performance outcomes but provides little discussion of where SUBRead might underperform (e.g., graphs with no meaningful substructure).

### Questions
1. How should practitioners choose or tune the number of clusters? Is there a heuristic or learning mechanism planned for future work?

2. How does SUBRead behave when the graph size varies significantly across samples? Does clustering remain stable?

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
The paper proposes a novel readout operation for graph neural networks. Similar to related works, it considers learnable prototype vectors. But instead of using them to attend to the entire graph, they use them to compute "subgraphs" (nodes clustered based on similarity to the prototypes) and aggregate the subgraphs using attention. The concatenated output vectors (i.e., one per subgraph) represent the graph-level representation.

### Strengths
- The paper is well-written and organized
- The approach sounds interesting and novel (but the related work discussion seems to be lacking, see below).
- The qualitative analysis for the vehicle data is very interesting and could be expanded to other datasets.

### Weaknesses
- The paper misses to discuss/compare to subgraph-related works, where the subgraphs are applied during message passing (e.g., [1], [2]). Given that the authors perform many experiments over molecular data, it would be good to understand similarities or differences between these approaches.  There also seems to be a connection to virtual nodes, which might be interesting to discuss.

- My main criticism is the empirical evaluation. The baselines in Tables 1&2 are rather simple. The really interesting numbers on improving existing pooling methods, presented thereafter, are only given over a single dataset. So it's open if this generalizes.


[1] Frasca et al. Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries. Neurips'22.

[2] Lee et al. Graph Convolutional Networks with Motif-based Attention. CIKM'19.

### Questions
---------------------------------

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SUBRead, a graph readout that (i) assigns nodes to one of k learned centroids via distance, producing subgraphs, (ii) aggregates each subgraph, (iii) applies self-attention across the subgraph vectors, and (iv) concatenates them as the final graph embedding. Experiments span several TU datasets and a small real-world “vehicle mode-shape” dataset. The idea is simple and potentially useful, but there are some concerns around differentiability of the clustering step, permutation invariance of the readout, fairness/strength of baselines, statistical rigor, and scaling.

### Strengths
The authors propose a straightforward and modular approach that could improve on existing graph readout methods and easily drop into existing GNNs.

The authors compare against a number of reasonable baselines across two architectures (GCN/GIN) and a number of readouts (Sort-Pool, Set2Set, Attention, Covariance, SOPool, and GMT). 

They compare over 11 datasets from the TU datasets collection and one small real world vehicle mode shape classification dataset.

### Weaknesses
The method forms a binary assignment matrix over distances to learned centroids (hard, 0/1). The text references N3Net to motivate a “learnable relaxation,” but the presented rule is a hard winner-take-all mapping (argmin to one-hot), which as described seems non-differentiable. It is thus currently unclear how gradients propagate from the loss to node assignments and centroids e.g., via straight-through estimator, Gumbel-softmax/concrete? Please clarify. 

What is the time/memory overhead vs. baselines? Complexity appears O(nk) for the distance matrix plus O(k^2) attention... not large for k≤5 but results on larger graphs/benchmarks are absent. Please report wall-clock, peak memory, and parameter counts, and include a large-graph stress test.

Are we supposed to compare the visual clusters in Fig. 3 and find the SUBRead and Expert Cluster similar? They do not look all that similar to me... am I to just believe the authors these are 'meaningful physical parts'? Some more description (and better quantification) would be apprecaited here.

The authors seem to neglect some recent relevant literature on related learnable graph methods (e.g., https://neurips.cc/virtual/2024/poster/94335) and all
their baselines are from 2020 or before, potentially neglecting some like: 
https://arxiv.org/abs/2209.07817
https://www.ijcai.org/proceedings/2024/0277.pdf
https://www.nature.com/articles/s41467-025-60252-z

### Questions
Concatenating the attended subgraph vectors makes the final embedding depend on the ordering of clusters... is this an issue?

Results sweep k∈{2,3,4,5} and sometimes pick best-k, while several baselines are fixed (e.g., GMT seeds fixed to 4). Please confirm that baseline hyperparameters were comparably tuned/matched to avoid search budget asymmetry.

Are the OOM results unavoidable? Please provide per-method memory settings or explain why you can't scale to a common feasible setting.

### Soundness
2

### Presentation
3

### Contribution
2
