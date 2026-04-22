# MDL-Pool: Adaptive Multilevel Graph Pooling Based on Minimum Description Length

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Graph pooling compresses graphs and summarises their topological properties and features in a vectorial representation. 
It is an essential part of deep graph representation learning for graph-level tasks like classification or regression.
Current approaches pool hierarchical structures in graphs by iteratively applying shallow pooling operators up to a fixed depth.
However, they disregard the interdependencies between structures at different hierarchical levels and do not adapt to datasets that contain graphs with different sizes that may require pooling with various depths.
To address these issues, we propose MDL-Pool, a pooling operator based on the minimum description length (MDL) principle, whose loss formulation explicitly models the interdependencies between different hierarchical levels and facilitates a direct comparison between multiple pooling alternatives with different depths.
MDL-Pool builds on the map equation, an information-theoretic objective function for community detection, which naturally implements Occam's razor and balances between model complexity and goodness-of-fit via the MDL.
We demonstrate MDL-Pool's competitive performance in an empirical evaluation against various baselines across standard graph classification datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MDL-Pool, a differentiable graph pooling method that automatically selects the optimal pooling depth for each graph by minimizing the description length—computed via the map equation—of hierarchical cluster assignments.
The method adaptively compresses graph structures and demonstrates effectiveness on both community detection and graph classification tasks.

### Strengths
- Explicit modeling of interdependencies across different hierarchical levels, as opposed to stacking independent pooling layers.

### Weaknesses
- Limited performance improvement
The authors note that MDL-Pool achieves state-of-the-art results only “in two of the eleven scenarios” and explicitly acknowledge that “we do not find a clear winner for graph classification.” This suggests that the overall empirical gain is modest.
- Lack of dataset-specific performance analysis
The paper does not explore which dataset characteristics (e.g., average graph size) affect the method’s relative performance.
A correlation study between these properties and MDL-Pool’s performance would strengthen the empirical section.
- Unclear practical advantage of adaptive depth
Adaptive depth is a central contribution, yet the paper reports that “the maximum chosen depth was two” and that “graphs in the classification datasets are small enough that two layers suffice”.
For single-graph tasks (community detection) or datasets where most graphs use depth 0 or 1, the advantage over fixed-depth configurations remains unclear, especially given that the model performance is comparative to other fixed-depth baselines.
 
- Computational cost and parameter-free claim
The paper emphasizes that MDL-Pool is “parameter-free,” yet it is also “limited to at most two pooling operations”
The actual computational savings compared to conventional hyperparameter tuning are therefore uncertain, given that MDL
Reporting runtime or complexity comparisons with fixed-depth GNN baselines would make the claim more convincing.

### Questions
- Could you provide a quantitative analysis correlating dataset characteristics (e.g., average node/edge count, modularity) with the relative performance of MDL-Pool?
- Do the graph-pooling results align with domain knowledge? For example, in the D&D dataset, do amino acids belonging to the same secondary structure fall within the same cluster?

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
The paper introduces MDL-Pool, a new adaptive, multilevel graph pooling operator grounded in information theory. It applies the map equation to jointly optimize cluster assignments across all hierarchical levels, explicitly modeling interdependencies between them. Unlike existing hierarchical pooling methods that fix the number of pooling layers, MDL-Pool dynamically selects the optimal pooling depth per graph using the MDL principle.

### Strengths
- The integration of the MDL principle and map equation into deep graph pooling is well-motivated. It provides a principled way to address overfitting and model complexity while enhancing interpretability. 
- The proposed multilevel loss seamlessly integrates hierarchical information, overcoming optimization issues caused by layer-wise independence in stacked pooling.
- The MDL framework naturally implements Occam’s razor, removing the need for hyperparameter tuning for cluster count or levels.

### Weaknesses
- The MDL-based loss focuses on topological structure and does not fully leverage node features in evaluating community quality, which might reduce performance on feature-dominant tasks.
- Experiments show most graphs select only one or two pooling levels; it remains unclear whether MDL-Pool is beneficial in tasks with truly deep hierarchies.
- The computation of multilevel flow matrices has quadratic cost in graph size, which may hinder scalability to very large graphs. No experiments on large-scale datasets are shown.

### Questions
See the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a minimum-description-based graph pooling model to learn strong graph representations for different-sized networks. With the mapping function, the model encodes the vertices' interdependencies. in different hierarchical levels that help cluster the network according to its volume and make graph learning effective.

### Strengths
The map equation is beneficial for observing the overall networks and for relevant clustering. 
The clustering helps for balanced training to fit the model in downstream graph analytics tasks. 
MDL is beneficial because it detects the depth of the input graph, which assists in effective hierarchical graph learning. 
The comprehensive result is better than other baselines

### Weaknesses
In the experiment, the authors did not mention the hyperparameter's impact on the model. 
The manuscript does not provide runtime details. Is minimum description length feasible on large volume datasets? 
The optimization of map equations involves nested matrix operations, which can result in a computationally heavy model. Please check the model's runtime with respect to simpler pooling operations like Top-kPool and SAGPool. 
In the case of community detection, the datasets are very sparse. Is the model suitable to cluster denser graphs (like Amazon Photo and Physics)? In this case, how does it perform over the other baselines?

### Questions
Is minimum description length feasible on large volume datasets?  Please observe the community detection on Amazon Photos, Physics.
How much more efficient is the model compared to simpler pooling operations like Top-kPool, SAGPool, GMT, etc.? Does MDLPool outperform these models?  
How does the technique provide expressivity in the model ? could you please show some formal reasoning or visualization?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MDL-Pool, an adaptive multilevel graph pooling operator based on the Minimum Description Length (MDL) principle. The method integrates the map equation into graph neural networks (GNNs) to model hierarchical dependencies between pooling levels. Unlike conventional stacked pooling approaches that fix depth and ignore inter-level dependencies, MDL-Pool formulates a joint loss to optimize clusters across all hierarchical levels and automatically selects the optimal depth per graph. The authors evaluate MDL-Pool on both community detection and graph classification benchmarks.

### Strengths
1. The method automatically determines the optimal pooling depth per graph instance, addressing a long-standing hyperparameter issue in hierarchical pooling.
2. The paper provides experiments on both synthetic and real-world datasets, including ablations on architecture variants and pooling depths.

### Weaknesses
Limited performance gain: In Tables 2 and 3, MDL-Pool does not consistently outperform baselines. For community detection, results are comparable or even worse than baselines on several datasets. Similarly, in graph classification, MDL-Pool’s average accuracy is not higher than several baselines, indicating limited empirical advantage.

Insufficient justification of benefits: While the motivation is sound, the claimed benefits (interdependency modeling and adaptive depth) are not strongly supported by quantitative evidence. The paper should include ablation or visualization explicitly demonstrating that modeling interdependencies leads to measurable improvement.

Unclear parameter selection: Section 4.1 mentions “up to l levels,” but the criterion for selecting the number of levels is not clearly described. How the model avoids overfitting or underfitting different depths needs more elaboration.

Choice of cmax = 50: In Table 2, the authors fix the number of clusters to 50, which seems far from the ground-truth number of communities (e.g., 3–7). This may bias the results. The paper should report results when cmax is closer to the true number (e.g., 10) to assess robustness.

Lack of comparative summary metrics: For Table 3, it would be informative to include an overall metric, such as the average rank or mean relative improvement across datasets, to better illustrate general trends rather than per-dataset fluctuations.

Questionable realization of motivation: The introduction claims that previous works ignore interdependencies between hierarchical structures, but it remains unclear whether MDL-Pool effectively learns such interdependencies rather than simply aggregating multi-level losses. Experimental evidence (e.g., hierarchical attention visualization or gradient correlation analysis) is lacking.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2
