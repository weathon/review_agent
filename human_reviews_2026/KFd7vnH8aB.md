# Lightweight Graph-Free Condensation with MLP-Driven Optimization

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Graph condensation aims to compress large-scale graph data into a small-scale one, enabling efficient training of graph neural networks (GNNs) while preserving strong test performance and minimizing storage demands. Despite the promising performance of existing graph condensation methods, they still face two-fold challenges, i.e., bi-level optimization inefficiency \& rigid condensed node label design, significantly limiting both efficiency and adaptability. To address such challenges, in this work, we propose a novel approach: Lightweight Graph-Free Condensation with MLP-driven optimization, named LightGFC, which condenses large-scale graph data into a structure-free node set in a simple, accurate, yet highly efficient manner. Specifically, our proposed LightGFC contains three essential stages: (S1) Proto-structural aggregation, which first embeds the structural information of the original graph into a proto-graph-free data through multi-hop neighbor aggregation; (S2) MLP-driven structural-free pretraining, which takes the proto-graph-free data as input to train an MLP model, aligning the structural condensed representations with node labels of the original graph; (S3) Lightweight class-to-node condensation, which condenses semantic and class information into representative nodes via a class-to-node projection algorithm with a lightweight projector, resulting in the final graph-free data. Extensive experiments show that the proposed LightGFC achieves state-of-the-art accuracy across multiple benchmarks while requiring minimal training time (as little as 2.0s), highlighting both its effectiveness and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel approach, LIGHTGFC, to address challenges in bilevel optimization and rigid node label distribution for graph condensation. Experiments on five standard benchmarks demonstrate that LIGHTGFC achieves superior performance compared to existing methods.

### Strengths
1. The proposed strategy to break the rigid label distribution constraint is a novel contribution to the field of graph condensation.
2. The paper provides a comprehensive empirical evaluation, comparing both performance and efficiency against a wide range of baselines.

### Weaknesses
1. The analysis in Figure 2(c) is confusing. The trend indicates that performance improves as $\alpha$ and $\beta$ decrease, suggesting the components S2 and S3 (which $\alpha$ and $\beta$ weight) may be unnecessary or even detrimental.
    - Can the authors explain this observation?
    - Why was only the Flickr dataset used for this specific analysis?
2. The paper claims, "Nodes with high $w_c^i$ capture richer structural information... implying that they are more central." This assumption is not sufficiently justified. While high *degree* is intuitively linked to centrality, the relationship between $w_c^i$ and centrality is not obvious. Please provide a theoretical explanation or experimental results (e.g., a correlation study between $w_c^i$ and node degree) to support this claim.
3. For a method that emphasizes efficiency, the evaluation lacks large-scale datasets. To robustly validate the scalability claims, it is essential to include benchmarks like Ogbn-products.
4. Given the significant performance gains reported, the paper must provide the source code or, at minimum, detailed experimental settings and the best hyperparameter configurations to ensure reproducibility.
5. Using 'Dist' (distance) to measure similarity is counter-intuitive, as distance and similarity are inversely related. Using $1 - \text{Dist}$ or an alternative formulation would be clearer.
6. The notation is unclear in places. The authors must explicitly define the subscripts $sc$ in $H_{sc}$ and $cg$ in $H_{cg}$.

## Minor:
1. **Terminology:** The terms "structural-free" and "graph-free" are used interchangeably. Please unify this terminology throughout the manuscript.
2. **Citation Errors:**
    - Line 102: The citation for GCPA appears to be incorrect.
    - Line 106: "SNTK" should be corrected to its official name, "KIDD."
3. **Clarity (Line 323):** Please clarify whether "10 rounds" refers to 10 *condensation* rounds or 10 *downstream GNN training* rounds.
4. **Typo (Line 255):** A space is missing after G'.
5. **Missing References:** The related work section should include and discuss the provided references [1]-[5], which cover relevant surveys, benchmarks, and other lightweight condensation methods.

## References:

[1] A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation, In IJCAI 2024

[2] GC-Bench: An Open and Unified Benchmark for Graph Condensation, In NeurIPS 2024

[3] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights, In NeurIPS 2025

[4] Scalable graph condensation with evolving capabilities, arxiv 2025

[5] Simple Graph Condensation, In PKDD 2024

### Questions
See Weaknesses above.

### Soundness
2

### Presentation
2

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
LIGHTGFC proposes a graph-free, three-stage condensation pipeline that embeds structure into features via multi-hop aggregation, trains an MLP expert, and then allocates condensed nodes per class using class-aware similarity with a lightweight projector. The objective combines label-adaptation (using the pretrained MLP) and prototype-alignment to preserve semantics and class discriminability without bi-level optimization over graphs and models. On transductive and inductive node-classification benchmarks, the method reports strong accuracy with very low training time (as little as seconds) and up to notable gains under tested ratios, highlighting efficiency without explicit condensed graph structures

### Strengths
1.  Efficient MLP-driven condensation that avoids nested optimization.
2.  Class-aware node allocation that better reflects class informativeness.
3.  Solid coverage of transductive and inductive datasets with promising results.

### Weaknesses
1. Potential loss of higher-order structure in graph-free condensation requires deeper validation.
2.  Need ablations on K-hop aggregation breadth, projector design, and class weighting robustness.
3.  Clarify fair-compute settings vs.\ recent baselines to substantiate SOTA claims.

### Questions
Scaling behavior and resource profiles across condensation ratios and graph sizes.
 Cross-backbone and inductive generalization with stability over seeds.
Analyze thresholds where structural loss limits downstream performance.

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
The authors propose a novel graph condensation algorithm using a three staged pipeline which involves an aggregation step over the graph to generate a structure free proto graph and then using that to train an MLP, learn an informative distribution of labels and then do label to node condensation to generate the final graph using a learnable projector matrix.
This technique is very light weight relying on MLPs, provokes some interesting thoughts on using different label distributions and results in an extremely fast and lightweight state-of-the-art condensation algorithm with amazing results on basic evaluation benchmark.

### Strengths
The algorithm uses very few hyper-parameters, is stable with those hyperparamters and also consumes significantly less time and memory than other techniques.

I believe the concept of experimenting with different label distributions and finding “better” has a lot of value for condensation. But this process needs metrics, justification and robustness analysis.

Components S2 and S3 stimulate interesting thoughts for the broader graph community and how simple experiments can lead to strong results as shown by table 1 and table 2.

The numbers in table 1 and 2 are impressive and improvements are significant

### Weaknesses
Lack of code raises serious reproducibility concerns. Number of things need to be specified for benchmarking time and accuracy. The hyperparameters of the base GNN class used, number of cores on which the process is running/ parallelism exploited, libraries and environment used are needed to check the stats.

Not all the techniques mentioned in related works are benchmarked. In fact, few techniques mentioned in table 1 are missing in table 2 and timing and memory plots.

Table 2 does not have standard deviations. This is not a proper presentation method. Numbers have to be reliable and statistically significant.

Reasoning needs to be a “bigger component” than experimentation. The statistics in table 1 and 2 are very interesting. For eg: 
LightGFC outperforms the full dataset on cora and citeseer and very close to full in reddit but not in others, what can be the reason?
Multiple techniques like Bonsai, protostack, lightgfc etc are outperforming the full dataset, is it correct to then rank them solely on basis of performance on these datasets? Are these small datasets then even worth benchmarking?
In GIN for cora, the performance suddenly drops drastically, the only difference between gcn and gin is aggregation, this performance I believe is even worse than randomly selecting nodes. How is that possible?

The challenge C2 is not “intrinsic” to these algorithms. In fact, any label distribution can be defined at start and the final distribution can be made to mimic it. The most faithful choice however,the occams razor, is the original distribution as it would most closely resemble real world data. I believe this is one reason why a random sample also usually gives good condensation results (although this benchmark is absent in your tables)

I think the ProtoAggregation phase is a very common precompute strategy. It does not feel like a novel component you have introduced. The major strength is in finding “better label distributions” and label to node condensation. But again how is your label distribution better is nowhere written. A comparison is needed with original distribution where better results are received.

I believe in conclusion a lot of experimentation is not strongly evaluated and lot of experimentation is missing. There is also not a lot of theory developed as to why this method should be able to outperform full training and achieve the numbers presented in table 1 and 2

Minor typo, is M of line 242 same as P of line 244? Otherwise where is M used? How is P learnt?

### Questions
Can there be an algorithm section which formally lists the steps of the algorithm, the paper organisation makes it hard to keep track of where various things are happening and what is used where. For Eg: What is the use of L_total and where is used?

Why are there no numbers for the full dataset when running different models like GIN,SAGE etc?

Why are different compression ratios taken for different datasets? This reference “Bonsai: Gradient-free Graph Condensation for Node Classification” mentions use of size compression metric as opposed to node compression used by GCond. Different sized compressed outputs would trivially store different information and hence show different performance. Can we have a size based comparison

What does - - - in Flickr for Protostack mean?

Can we also include analysis on larger sized datasets like MAG240M and ogbn-papers?

Some important and recent techniques like GDEM and EXGC are excluded, can they be included?
1) Graph Distillation with Eigenbasis Matching
2)EXGC: Bridging Efficiency and Explainability in Graph Condensation

For the different datasets, can we look at how different the output label distributions of your model are as compared to training data

Can some more clarity be given on ablation techniques? How are the metric computed when various components are disabled? For eg, how is compressed data formed with only PAlign. Also you mention Idx1-3 disable each of the components once but your ticks and crosses show something else?

Can we have a runtime and memory analysis of the different stages as well? This can show if there is scope of improvement via parallelism

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LIGHTGFC (LIGHTweight Graph-Free Condensation with MLP-driven optimization), a novel method that condenses large-scale graph data into a structure-free node set in a simple, accurate, yet highly efficient manner. The approach consists of three stages—proto-structural aggregation, MLP-driven structural-free pretraining, and lightweight class-to-node condensation—to embed structural information, align representations, and generate representative nodes. Extensive experiments demonstrate that LIGHTGFC achieves state-of-the-art accuracy across multiple benchmarks while requiring minimal training time (as little as 2.0s), highlighting both its effectiveness and efficiency.

### Strengths
1. The proposed method effectively eliminates bi-level optimization inefficiency through an MLP-driven structure-free condensation process.
2. It introduces a clear three-stage framework that preserves structural and semantic information while maintaining lightweight computation.
3. Extensive experiments confirm strong performance gains and remarkable efficiency, achieving state-of-the-art accuracy with minimal training time.

### Weaknesses
1. Although this paper outperforms the baselines, many of the main contributions mentioned (especially in the abstract) resemble ideas already proposed in existing methods such as SimGC [1], GCPA [2], and CGC [3]. Given the strong performance, the authors should further clarify the novelty of their approach compared to these works.
2. Some highly related baselines, such as [1], are not discussed.
3. It is unclear why the baseline CGC-X is not included in Figure 2 for comparison.

[1] Zhenbang Xiao, Yu Wang, Shunyu Liu, Huiqiong Wang, Mingli Song, and Tongya Zheng. "Simple graph condensation." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 53-71. Cham: Springer Nature Switzerland, 2024.
[2] Yuan Li, Jun Hu, Zemin Liu, Bryan Hooi, Jia Chen, and Bingsheng He. "Adapting Precomputed Features for Efficient Graph Condensation." In Forty-second International Conference on Machine Learning.
[3] Xinyi Gao, Guanhua Ye, Tong Chen, Wentao Zhang, Junliang Yu, and Hongzhi Yin. "Rethinking and accelerating graph condensation: A training-free approach with class partition." In Proceedings of the ACM on Web Conference 2025, pp. 4359-4373. 2025

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
