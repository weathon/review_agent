# Adaptive node feature selection for graph neural networks

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
We propose an adaptive node feature selection approach for graph neural networks (GNNs) that identifies and removes unnecessary features during training. The ability to measure how features contribute to model output is key for interpreting decisions, reducing dimensionality, and even improving performance by eliminating unhelpful variables. However, graph-structured data introduces complex dependencies that may not be amenable to classical feature importance metrics. Inspired by this challenge, we present a model- and task-agnostic method that determines relevant features during training based on changes in validation performance upon permuting feature values. We theoretically motivate our intervention-based approach by characterizing how GNN performance depends on the relationships between node data and graph structure. Not only do we return feature importance scores once training concludes, we also track how relevance evolves as features are successively dropped. We can therefore monitor if features are eliminated effectively and also evaluate other metrics with this technique. Our empirical results verify the flexibility of our approach to different graph architectures as well as its adaptability to more challenging graph learning settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an adaptive node feature selection method for graph neural networks that estimates each feature’s importance during training by measuring the drop in validation performance when that feature’s values are randomly permuted, then progressively prunes low-value features via a simple routine. The authors first analyze how graph structure and node attributes jointly affect GCN behavior, deriving a bound that ties error to cross-class edges and feature similarity, and then justify permutation-based importance by showing how permuting decouples features from labels and edges in the bound, making the resulting score a proxy for influence. The approach is model- and task-agnostic, can substitute other objectives (e.g., fairness metrics) for accuracy in the importance calculation, and tracks how feature relevance evolves over time. Experiments on eight benchmarks with GCN, GIN, and TAGCN compare against mutual information, a topological feature informativeness metric, homophily scores, and random selection; results include dynamic importance heatmaps that show stable rankings over training and case studies illustrating when graph structure or attributes dominate.

### Strengths
The paper’s strengths are mostly conceptual and methodological. On originality, it proposes a simple, model- and task-agnostic way to score and prune node features during training via permutation tests that are explicitly tied to a GNN’s current behavior, rather than relying on graph-agnostic or architecture-specific heuristics. In quality, the work motivates the approach with analysis that relates prediction error to how labels, edges, and features interact, and then justifies permutation-based importance by showing how permuting breaks those dependencies—together with a concrete training-time procedure for adaptive pruning. Clarity is solid: the problem setup and notation are explicit, the algorithm is given in pseudocode, and the method’s “monitor as you train” design is easy to follow, including how importance is recomputed on checkpoints.

### Weaknesses
The empirical validation is narrow and leaves the practical value unclear: experiments cover only eight small, legacy node-classification datasets (Cora, Citeseer, PubMed; Amazon Photo/Computers; Cornell/Texas/Wisconsin) with three simple backbones (GCN, GIN, TAGConv), and five random-split runs, which is a thin basis for claims of generality; stronger evidence would add larger/modern benchmarks and architectures, more seeds, and fixed public splits. The feature-selection baselines are mostly simple filters (mutual information, topological feature informativeness, homophily metrics) plus a mask variant of the authors’ own method, omitting model-based or Shapley/attribution-style GNN selection methods that the paper cites in related work; including such baselines would better position the contribution. Results are also noisy and sometimes unconvincing: standard deviations are large on WebKB (e.g., Texas and Wisconsin), and accuracy often drops materially relative to using all features (e.g., −6–7 points on Cora after selection), so it is hard to conclude the method reliably helps. Finally, applicability is limited to node classification and the authors themselves note that correlated features can bias permutation importance; extending to link prediction/graph classification and using conditional permutation or correlation-aware corrections would strengthen the case.

### Questions
- Empirical setup and statistical support: the current evaluation uses five random splits with resampled masks and a fixed pruning schedule, but many results degrade or have large variance—could you expand to more seeds (e.g., 10–20), and report the training/inference overhead of the permutation procedure?
- Your evaluation uses eight small, legacy node-classification datasets with random splits—can you add modern, large-scale benchmarks (e.g., OGB, Heterophilous Node Classification) using their official splits, and report runtime/memory overhead to demonstrate scalability and practical cost?

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
3

### Summary
The paper proposes a model- and task-agnostic method to identify and prune unhelpful node features for GNNs during training. The core idea is a permutation-based, performance-driven importance score: periodically permute one feature column across nodes, re-evaluate validation accuracy, and treat the accuracy drop as that feature’s importance. This directly ties selection to the end task and naturally accounts for graph–feature interactions. The authors provide theory showing how graph structure and feature distributions jointly affect GCN outputs, motivating why permutation tests reveal feature utility in graphs, and then demonstrate the method’s flexibility across homophilic/heterophilic settings and different GNN backbones.

### Strengths
good theoretical analysis

### Weaknesses
computation complexity analysis

### Questions
1. For the study of inter-class vs. intra-class node distinguishability, you can refer to [1].

2. Why is your define Z* being the idealized embedding? Do you mean there is no one better than it?

3. "measuring feature importance based solely on dependencies between y and X may not be sufficient" This is not a surprise, as we need to consider the synergy of feature, graph structure and label distribution together, which is well developed in [2].

4. What is the computational complexity of your proposed adaptive node feature selection in Algorithm 1?

5. What is the relationship between your proposed node feature selection and your analysis in Section 2?

6. The selected features doesn't seem to improve the performance.



[1] When do graph neural networks help with node classification? investigating the homophily principle on node distinguishability. Advances in Neural Information Processing Systems. 2024 Feb 13;36.


[2] What is missing for graph homophily? disentangling graph homophily for graph neural networks. Advances in Neural Information Processing Systems, 37, 68406-68452.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an adaptive node feature selection method for graph neural networks (GNNs) that dynamically identifies and removes unnecessary features during training. The approach uses permutation-based importance scores to measure feature relevance by evaluating changes in validation performance when feature values are permuted. The authors provide theoretical analysis showing how graph structure and node features jointly influence GCN performance, and demonstrate their method across multiple benchmark datasets with different graph properties (homophilic and heterophilic).

### Strengths
- The paper provides theoretical analysis through Theorems 1 and 2, establishing clear bounds on how graph structure and node features jointly affect GCN performance. This theoretical foundation effectively motivates why permutation-based importance scores are particularly suitable for graph-structured data, where traditional feature selection methods may fail to account for the interplay between features and graph topology.
- the proposed method is model-agnostic and task-agnostic, requiring no assumptions about graph properties or specific GNN architectures. This flexibility is demonstrated through experiments on diverse datasets, including homophilic graphs (citation networks), heterophilic graphs (webpage networks), and co-purchase graphs, using different architectures (GCN, GIN, TAGCN) for each setting.
- The algorithm's ability to progressively eliminate features during training while monitoring importance score evolution is valuable. Figure 3's heatmaps effectively show that feature importance rankings stabilize early in training, suggesting the method can identify relevant features before full convergence, potentially saving computational resources.
- Overall, the experimental evaluation is thorough, including multiple datasets with varying properties, comparison against numerous baselines (including graph-specific metrics like homophily-based scores), and analysis of the adaptive selection process over time. The inclusion of heterophilic graphs is particularly valuable as these represent challenging cases for standard GNNs.

### Weaknesses
- The improvements over baseline methods are often marginal. In Table 2, NPT achieves 79.19% on Cora compared to 85.83% with all features. On some datasets like Photo and Computers, even random feature selection performs competitively, raising questions about the method's practical utility when simpler approaches suffice.
- The method requires computing permutation-based scores every T epochs by evaluating the model K times per feature, which could be computationally expensive for high-dimensional datasets. The paper doesn't provide runtime comparisons or discuss the computational trade-off between feature reduction and the cost of computing importance scores.
- While the paper shows that features can be selected effectively, it doesn't provide an analysis of what types of features are being selected or removed. Understanding which features are deemed important could provide valuable insights for domain experts and help validate the method's decisions.
- The method introduces several hyperparameters (burn-in period T_burn, interval T, retention rate r, number of permutations K) but provides limited analysis of sensitivity to these choices. The fixed choice of r=0.5 (dropping half the features at each step) seems arbitrary and may not be optimal for all settings.
- The paper doesn't adequately discuss when the method fails or performs poorly. For instance, on heterophilic graphs, the performance drops are substantial, but there's limited analysis of why permutation-based importance might be less effective in these settings compared to simpler alternatives like mutual information.

### Questions
1. How does the computational cost of the adaptive feature selection scale with the number of features and nodes? Could you provide runtime comparisons with training using all features?
2. Can you provide an analysis of which types of features are consistently selected or removed across different datasets? This would help validate that the method is making sensible choices.
3. How sensitive is the method to the choice of hyperparameters, particularly the retention rate r? Have you experimented with adaptive scheduling of r based on the importance score distribution?
4. Why does the method perform relatively poorly on heterophilic graphs compared to mutual information? Is there a theoretical explanation for this limitation?
5. Could the permutation test be made more efficient by using importance sampling or other variance reduction techniques rather than uniform random permutations?

### Soundness
3

### Presentation
3

### Contribution
2
