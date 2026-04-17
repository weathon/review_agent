# GNNUpdater: Adaptive Self-Triggered Training Framework on Dynamic Graphs

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Adapting Graph Neural Networks (GNNs) to evolving, dynamic graph data presents a significant operational challenge. A critical yet understudied question is determining **when** to update these models to balance model freshness against computational training costs. This problem is particularly difficult in graph settings due to two key issues: **label delay**, where ground truth arrives long after predictions are made, and **hidden drift**, where structural dependencies propagate changes through multiple hops, causing unexpected performance degradation. We propose GNNUpdater, an adaptive framework that decides when to trigger GNN training. It overcomes the aforementioned challenges through two innovations: (1) a performance predictor that estimates model quality by measuring shifts in node embeddings, eliminating dependence on immediate ground-truth labels, and (2) a graph-aware update trigger that uses label propagation to detect widespread performance degradation across the graph. We implement GNNUpdater as a high-performance distributed streaming-GNN library for billion-edge dynamic graphs. Extensive experiments demonstrate that GNNUpdater either exceeds the performance of periodic, performance-based, and drift-detection baselines at comparable training cost or matches their performance with significantly reduced computational effort. The implementation can be found in the anonymous link: https://anonymous.4open.science/r/GNNUpdater-B47D/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies an interesting question: when should the graph model be updated as the graph evolves? The proposed method first predicts the performance of the target nodes by measuring the embedding shifts and mapping the shift score and other graph features to a performance score. Based on the expected performance score, a threshold is set to determine whether the node is problematic. These problematic labels are then diffused by label propagation. It is finally determined whether to update the graph model based on the final ratio of "problematic" nodes.

### Strengths
S1. The topic is interesting and applicable to real-world applications.
S2. The update method is simple, intuitive, and effective.

### Weaknesses
- W1. The proposed method is not end-to-end and relies on numerous predefined hyperparameters—such as $\epsilon$, the $0.5$ term in $r_t$, and $\phi$. These choices are pivotal to the model’s performance, yet selecting them can be costly. Moreover, it is unclear whether fixed values will remain valid as the graph evolves, since the underlying dynamics may change.

- W2. The method should be compared with approaches that explicitly model graph evolution, such as JODIE, TGN, and DyGFormer. These methods naturally incorporate recent neighbors and typically do not require retraining for updates; although JODIE and TGN maintain memories, updating memory is efficient.

- W3. Another relevant line of work focuses on discrete-time dynamic graphs and employs update mechanisms similar to those proposed here (e.g., InstantGNN [R1]). Such methods should be included as baselines.

[R1] Instant Graph Neural Networks for Dynamic Graphs. KDD'22.

### Questions
Q1. In Eq (3), how do the four features {log(num_nodes), log(num_edges), t, deg(v)} affect the performance of the mapping function?

### Soundness
2

### Presentation
2

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
This work presents GNNUpdater, an adaptive framework for deciding when to update GNNs on dynamic graphs. It addresses label delay and hidden drift by introducing a performance predictor that estimates model degradation without ground-truth labels, using shifts in node embeddings as an indicator. A graph-aware trigger then propagates “problematic” node signals via graph propagation and triggers updates when degradation becomes widespread. Notably, they declare that they implement their work in a distributed streaming-GNN library, including a custom dynamic graph storage system to reduce operational overhead.

### Strengths
According to the authors, this is the first work to solve the problem of when to update models in the dynamic graph domain. Meanwhile, the approach features a simple yet well-motivated design tailored to graph data, and the explanations are sound. In addition, They release a distributed streaming-GNN library, which provides a tangible code contribution to the community.

### Weaknesses
1. The evaluation part compares against only three categories of existing update-trigger methods (and their straightforward variants) from previous work, which may not be sufficient. There are many new works of update triggers like:

[1] Wan, Ke, Yi Liang, and Susik Yoon. "Online drift detection with maximum concept discrepancy." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

[2] Lu, Pengqian, et al. "Early concept drift detection via prediction uncertainty." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 18. 2025.

[3] Florence, Regol, et al. "When to retrain a machine learning model." arXiv preprint arXiv:2505.14903 (2025).

2. The work lacks theoretical or mathematical analysis to justify the proposed approach.

3. The evaluation omits several related baselines such as InstantGNN, EvolveGCN, and ROLAND. Despite authors’ claim of orthogonality, it is difficult to assess whether the proposed approach provides genuine added value to the field, and the work lacks any attempt to combine it with these existing methods.

### Questions
Please refer to the weaknesses.

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
This paper introduces GNNUpdater, an adaptive training triggering framework for dynamic graphs, designed to address the "when to update" challenge for Graph Neural Networks (GNNs) in continual learning environments. The framework employs a performance predictor based on node embedding drift to estimate model performance, along with a graph-aware update trigger to detect performance degradation.

### Strengths
This study addresses the "when to update" problem in dynamic graph learning by proposing a performance predictor and a graph-aware update trigger. The former estimates task performance without ground-truth labels. The latter monitors model quality across the entire graph. 

This work develops a block-based streaming graph storage system and CUDA-accelerated GPU neighbor finder to support incremental updates and fast neighborhood sampling.

This study conducts a comprehensive evaluation on multiple real-world temporal graph benchmarks to validate the effectiveness of the proposed method.

### Weaknesses
In GNNUpdater, the calculation of global drift relies on the reference embeddings, $\mathbf{H}_{ref}$. However, these references are generated from a full-graph inference right after the last model update. If the interval between two updates is too long, this reference point itself could become outdated. What would happen to the performance in this case?

In Section 3.2, label propagation is used based on the graph structure. This implicitly assumes that performance degradation propagates through the graph like a label. However, the way a GNN's aggregation mechanism works might not match how actual performance issues spread. How can the validity of this operation be justified? For instance, a node connected to a problematic node isn't necessarily problematic itself. This seems somewhat questionable.

Appendix D.4 shows that different datasets require different parameter tuning ranges, which affects the generalizability of the parameter settings.

Some notations in the paper are not clearly defined, making them hard to understand. Examples include p(x) and p(y|x) in Section 2, and $C_{train, t}$ in Equation 1.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to address the problem of deciding when to update a GNN to balance model freshness and computational cost. Existing methods struggle with label delay (ground truth arrives late) and hidden drift (structural changes propagate across multi-hop neighbors). GNNUpdater addressed the problem by introducing two core components: (1) a performance predictor that estimates model quality without labels by tracking embedding shifts between current and reference node representations using a neighbor-aware drift metric; and (2) a graph-aware update trigger that applies label propagation to detect widespread degradation before major performance drops.

### Strengths
1. The paper tackles a practically critical question of when to update GNNs. The motivation is solid and strongly grounded in real-world industrial scenarios, where retraining is expensive and labels arrive with delay.
2. The paper is well-structured and easy to follow. Problem formulation, methods, and experiments are clear, figures effectively support the narrative.
3. Empirical results justified the effectiveness of proposed approach.

### Weaknesses
All experiments are on the node affinity prediction. It would be better to also explore whether the method works for other common tasks on graphs.

### Questions
1. On “label delay” — is this phenomenon specific to graph data, or also common in other data modalities? If it also exists in non-graph domains, how is it typically handled there and can the solutions be adapted directlt to graphs? If it is more severe in graphs, what graph-specific properties cause label delay (e.g., multi-hop dependency, verification latency, fraud propagation)?

2. In your drift formulation, node drift is computed using v and its neighbors N(v). But on a dynamic graph, nodes and edges appear/disappear, so N(v) changes over time. Is N(v) taken from the current graph structure, or the reference graph at last update? How do you handle cases where a previous neighbor disappears?

### Soundness
3

### Presentation
4

### Contribution
3
