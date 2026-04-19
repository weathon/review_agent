# Diss-l-ECT: Dissecting Graph Data with local Euler Characteristic Transforms

- Decision: Reject
- Scores: 3, 6, 6, 8

## Abstract
The Euler Characteristic Transform (ECT) is an efficiently-computable
    geometrical-topological invariant that characterizes the global shape of data. 
    In this paper, we introduce the Local Euler Characteristic Transform (l-ECT), a novel extension of the ECT particularly designed to enhance expressivity and interpretability in graph representation learning.
    Unlike traditional Graph Neural Networks (GNNs), which may lose critical local details through aggregation, the l-ECT provides a lossless representation of local neighborhoods.
    This approach addresses key limitations in GNNs by preserving nuanced local structures while maintaining global interpretability.
    Moreover, we construct a rotation-invariant metric based on l-ECTs for spatial alignment of data spaces.
    Our method exhibits superior performance than standard GNNs on a variety of node classification tasks, particularly in graphs with high heterophily.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a local Euler characteristic transform for enhancing feature representation for graph learning. This approach addresses key limitations in GNNs by preserving nuanced local structures while maintaining global interpretability.

### Strengths
Using local Euler characteristic transform for graph representation is novel to me.

### Weaknesses
1. To compute ECT or l-ECT, one needs to embed a simplicial complex in a Euclidean space. The authors propose to embed using node features. However, I don't think this is a genuine embedding. For example, if the feature space is \mathbb{R}^2, then even if the nodes are embedded to the place in a 1-1 fashion, the edges may cross each other. Therefore, only talking about vertex embedding is insufficient as a graph or a simplicial complex has additional structures. 
2. Related to 1. The author should be more specific on ``embedding'', whether it is metrical embedding, differential embedding, or topological embedding (or something else?). 
3. The proofs are poorly written. The statements are vague and imprecise. Many details are missing. It is hard to assess the correctness of the results. 
4. It seems to me that the proposed l-ECTs capture local structural information. They are used as node features, but not used to guide feature aggregation. I fail to get the intuition of why they can be useful for the node classification task. However, on the other hand, they might be useful for the graph classification task. 
5. The compared benchmarks are very limited (only GCN and GAT). From my own experience, the results are not very impressive, e.g., for the Actor, Squirrel, and Chameleon datasets, there are more recent benchmarks (e.g., ACM-GCN) whose performance is at least 5%-10% higher than those reported by the authors. 
6. Ablation study is missing. It is hard to assess whether l-ECTs play an important role in the reults shown.

### Questions
See weaknesses.

### Soundness
1

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
5

### Summary
The authors introduce a new topological feature extraction methods, Local Euler Characteristic Transform (l-ECT), extending the Euler Characteristic Transform (ECT) to provide a lossless, interpretable representation of local graph neighborhoods, addressing limitations in traditional Graph Neural Networks (GNNs). This novel approach improves performance in node classification tasks, especially in heterophilous graphs, by preserving both local and global structural details.

### Strengths
1. **Novel l-ECT Framework**: Extending the Euler Characteristic Transform to capture local graph details in embedded simplicial complexes is impactful, with theoretical insights enhancing its expressivity, especially for featured graphs.

2. **Extracting  Key Information from Node Neighborhoods from Attribute Space**: The l-ECT enables to obtain node neighborhood information by effectively utilizing the information from attribute space.

3. **Experimental Validation**: The l-ECT consistently outperforms traditional GNNs in node classification tasks, particularly in high-heterophily settings, highlighting its interpretability and effectiveness.

4. **Presentation:** The presentation is very good.

### Weaknesses
1. **Limited Applicability:** The proposed approach is constrained to graphs with node feature vectors in $\mathbb{R}^n$, limiting its applicability to datasets that fit this specific structure.

2. **Effectiveness of Approach:** While the concept of embedding the graph into an attribute space using node attribute vectors is promising, the subsequent steps for extracting meaningful information appear less effective. The method could be enhanced by exploring simpler and more efficient ways to utilize the geometry (rather than topology) of ego networks induced within the attribute space.

3. **Feasibility in High Dimensions:** As the dimension $n$ of the feature space increases, the number $m$ of representative vectors on $S^{n-1}$ must grow nearly exponentially. Furthermore, the feature vector range impacts the number of intervals {$t_i$} needed. For high-dimensional and wide-range data, this results in a very high-dimensional $l$-ECT vector, making the approach impractical for real-world applications. Dimension reduction could help by reducing feature dimensionality to three (as two dimensions may be insufficient for graph embedding) and normalizing feature vectors (e.g. total diameter of feature vector space to 2), allowing for "end-to-end" a fixed-size feature extraction for nodes. Without this, selecting vectors and thresholds can be challenging, particularly for new users.

4. **Theoretical Contributions vs. Practical Applications:** While the rotation-invariant metric is mathematically appealing, it may lack practical relevance since it relies on the infimum over all rotations. Also, the discussion of graph isomorphism seems tangential, as Definition 2 is highly restrictive, applicable only to isomorphic graphs with identical feature vectors.

5. **Experimental Results:** The presented results are uninformative and potentially misleading. The models used, GCN and GAT, are older and are known to perform poorly in heterophilic settings. The authors should consider comparing their approach with newer GNN models that perform well on heterophilic datasets and include more homophilic datasets (other than Computers and Photo) to provide a comprehensive performance assessment. Also, exploring the integration of $l$-ECT vectors with a more recent GNN model may yield interesting insights into performance enhancement.

### Questions
See weaknesess.

### Soundness
3

### Presentation
4

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
This paper introduces the Local Euler Characteristic Transform ($l$-ECT), an extension of the Euler Characteristic Transform (ECT) designed to enhance expressivity and interpretability in graph representation learning. It provides a lossless representation of local neighborhoods and addresses key limitations in GNNs by preserving nuanced local structures while maintaining global interpretability. Their method demonstrates superior performance over standard GNNs on node classification tasks, particularly in graphs with heterophily.

### Strengths
1. Innovative Use of Euler Characteristic Transform: Employing the ECT to enhance graph representation learning, especially in settings with heterophily, is a novel and interesting approach.

2. Solid Theoretical Foundation: The work is thorough, with strong theoretical results that effectively support the proposed method.

### Weaknesses
Missing Important Related Works & Limited Experimental Comparisons: The quantitative experiments focus mainly on node classification tasks in heterophilic graphs but compare the proposed method only with basic models like GCN and GAT. While the authors acknowledge related works on GNNs designed for heterophily in Section 3, the coverage is still limited. It is suggested that the authors include more related works such as [1-5] and select appropriate GNNs for experimental comparison to strengthen the validation of their method.

[1] Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

[2] Graph Neural Networks with Heterophily

[3] Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily

[4] ES-GNN: Generalizing Graph Neural Networks Beyond Homophily With Edge Splitting

[5] GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily

### Questions
See weaknesses.

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
3

### Summary
The paper introduces the Local Euler Characteristic Transform (L-ECT), an extension of the Euler Characteristic Transform (ECT) designed for graph representation learning. Unlike traditional Graph Neural Networks (GNNs), which can obscure local details through node aggregation, the L-ECT maintains local structural data, thus enhancing interpretability and performance, especially in heterogeneous (high heterophily) graphs. By capturing spatial and structural characteristics of local neighborhoods, the L-ECT provides a rotation-invariant metric for data alignment, showcasing improved performance over GNNs in node classification tasks. The method’s compatibility with machine learning models enables use cases beyond standard GNN architectures, offering more accessible and interpretable models, such as tree-based classifiers. Empirical results demonstrate that L-ECT outperforms GNNs in heterogeneous datasets and facilitates robust spatial alignment in both synthetic and high-dimensional data. This research suggests future exploration into scaling L-ECT and integrating global and local information in complex graph structures.

### Strengths
The paper presents the Local Euler Characteristic Transform (L-ECT) as an extension of the traditional Euler Characteristic Transform, enabling a lossless representation of local graph structures and addressing key limitations of Graph Neural Networks (GNNs) such as oversmoothing and loss of local detail in high heterophily graphs. This novel transformation preserves intricate topological information, allowing for more nuanced node representations by capturing both structural and spatial data and offering an alternative to GNN message-passing frameworks. Additionally, the authors introduce a rotation-invariant metric that enables robust spatial alignment of data in Euclidean space, enhancing the method’s applicability in graph-structured data and increasing resilience to coordinate transformations. Empirical results underscore L-ECT’s effectiveness, showing superior performance over standard GNNs in high-heterophily datasets like WebKB, Roman Empire, and Amazon Ratings. Furthermore, L-ECT’s model-agnostic nature facilitates integration with interpretable machine learning models, such as XGBoost, making it ideal for use in regulated fields like healthcare and finance where transparency is paramount. Beyond graph representation, L-ECT extends to point clouds and other high-dimensional data, proving robust to noise and outliers and enabling efficient spatial alignment without the need for exhaustive pairwise distance computations.

The methods section is detailed yet readable, presenting L-ECT’s mathematical foundation and integrating a rotation-invariant metric for spatial alignment, which adds to the paper’s originality. While the experiments section is robust and results are well-presented through tables and figures, additional visual aids could further clarify data characteristics and enhance accessibility.

the discussion on the limitations of the approaches proposed in the paper is appreciated

### Weaknesses
The paper would benefit, both in making more persuasive the novelty of the work with respect to contemporary literature as well as clarity of the work itself, with a more robust background and related works section 

Including a more robust and explicit comparison to related works, which also addresses the novelty of the work being proposed, would be appreciated.

The L-ECT approach, while innovative, faces several limitations and lacks certain aspects of novelty. Its computational complexity scales with graph size and density, making it less efficient for very large or dense graphs and primarily feasible for medium-sized datasets. Although L-ECT emphasizes local information preservation, similar topology-aware or geometric GNN approaches also capture neighborhood-specific details, reducing the uniqueness of this feature. Additionally, traditional GNNs perform comparably well on low-heterophily datasets, indicating that L-ECT may not consistently outperform them across all types of graph data. The approach’s scalability is further limited by sampling trade-offs, as its accuracy depends on carefully chosen parameters, such as direction and filtration steps, which challenge fidelity and computational efficiency at scale. Moreover, despite its model-agnostic design, L-ECT’s interpretability hinges on pre-defined features, potentially restricting its flexibility for complex, dynamic graphs. Finally, L-ECT does not support end-to-end learning as GNNs do; instead, it relies on external classifiers (e.g., XGBoost), which may limit its integration into more comprehensive, end-to-end pipelines.

The authors should include comparison other works which construct topological representations of graphs and graphs neighborhoods and include reference to those related methods such as “graph filtration learning” by Hofer et. al. and other approaches as discussed in survey works such as  “A Survey of Topological Machine Learning Methods” by Hensel et. al.

The authors provide comparative experimental analysis to a number of datasets. It may be misleading, however, to not include other models as discussed in “A critical look at the evaluation of GNNs under heterophily: Are we really making progress?” by Platonov et. al.

### Questions
Would it be possible to include experimental results for the other datasets offered in “A critical look at the evaluation of GNNs under heterophily: Are we really making progress?” by Platonov et. al. or an argument as to why this is done?

### Soundness
3

### Presentation
3

### Contribution
3
